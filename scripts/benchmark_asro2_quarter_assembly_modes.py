#!/usr/bin/env python3
"""Benchmark corrected vs regular native Metal assembly on the ASRO2 WG quarter mesh."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterator

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "hornlab-plots"))

from hornlab_plots import save_directivity_plot  # noqa: E402
from hornlab_solver import (  # noqa: E402
    ObservationConfig,
    ObservationFrame,
    SolveConfig,
    load_mesh,
    solve_frequencies,
)
from hornlab_solver._constants import SPEED_OF_SOUND  # noqa: E402
from hornlab_solver.bie import (  # noqa: E402
    _build_driver_neumann_coeffs,
    _compute_impedance,
    _setup_function_spaces,
    compute_surface_pressure_avg,
)
from hornlab_solver.metal.geometry import build_metal_geometry_buffers  # noqa: E402
from hornlab_solver.metal.native import MetalNativeStandardSession  # noqa: E402
from hornlab_solver.sweep import _read_complex_f32, _solve_dense_direct  # noqa: E402


CORRECTED_RUN_DIR = (
    WORKSPACE_ROOT
    / "runs/canonical-validation/260602-asro2-quarter-directivity-matched-resolution-backplate-sector"
)
DEFAULT_MESH = CORRECTED_RUN_DIR / "wg_hornlab_quarter_snapped.msh"
DEFAULT_CORRECTED_CASE = CORRECTED_RUN_DIR / "wg_hornlab_quarter_snapped_native"
DEFAULT_OUTPUT_DIR = (
    WORKSPACE_ROOT
    / "runs/canonical-validation/260602-asro2-quarter-assembly-mode-benchmark"
)
PLANES = ["horizontal", "vertical", "diagonal"]


def log(message: str) -> None:
    print(f"[asro2-assembly-mode] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--corrected-case-dir", type=Path, default=DEFAULT_CORRECTED_CASE)
    parser.add_argument(
        "--rerun-corrected",
        action="store_true",
        help="rerun corrected assembly instead of reusing the existing corrected result",
    )
    parser.add_argument(
        "--skip-existing-regular",
        action="store_true",
        help="reuse an existing optimized_regular/result.npz in the output folder",
    )
    parser.add_argument("--angle-count", type=int, default=37)
    parser.add_argument(
        "--matrix-probe-count",
        type=int,
        default=3,
        help="number of representative frequencies to probe directly",
    )
    return parser.parse_args()


def axial_throat_frame(loaded: Any) -> ObservationFrame:
    vertices = np.asarray(loaded.grid.vertices.T, dtype=np.float64)
    elements = np.asarray(loaded.grid.elements.T, dtype=np.int32)
    source_mask = np.asarray(loaded.physical_tags, dtype=np.int32) == 2
    source_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if np.any(source_mask):
        source_elems = elements[source_mask]
        p0 = vertices[source_elems[:, 0]]
        p1 = vertices[source_elems[:, 1]]
        p2 = vertices[source_elems[:, 2]]
        areas = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        centroids = (p0 + p1 + p2) / 3.0
        valid = areas > 1.0e-15
        if np.any(valid):
            weighted = np.average(centroids[valid], weights=areas[valid], axis=0)
            source_center[2] = float(weighted[2])
    mouth_center = np.array([0.0, 0.0, float(np.max(vertices[:, 2]))], dtype=np.float64)
    return ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        origin=source_center.copy(),
        u=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        v=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        mouth_center=mouth_center,
        source_center=source_center,
    )


def inspect_mesh(path: Path) -> dict[str, Any]:
    loaded = load_mesh(path, scale=0.001, repair_normals=True)
    bb_min, bb_max = loaded.info.bounding_box_m
    unique, counts = np.unique(loaded.physical_tags, return_counts=True)
    return {
        "path": str(path),
        "vertices": int(loaded.info.n_vertices),
        "triangles": int(loaded.info.n_triangles),
        "physical_groups": {str(k): v for k, v in loaded.info.physical_groups.items()},
        "physical_tag_counts": {
            str(int(tag)): int(count) for tag, count in zip(unique, counts, strict=True)
        },
        "bbox_m": {
            "min": [float(v) for v in bb_min.tolist()],
            "max": [float(v) for v in bb_max.tolist()],
        },
    }


def frequencies_from_npz(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as data:
        return np.asarray(data["frequencies_hz"], dtype=np.float64)


def load_npz(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {
            "frequencies_hz": np.asarray(data["frequencies_hz"], dtype=np.float64),
            "angles_deg": np.asarray(data["angles_deg"], dtype=np.float64),
            "spl_db": np.asarray(data["spl_db"], dtype=np.float64),
            "pressure_complex": np.asarray(data["pressure_complex"], dtype=np.complex128),
            "impedance": np.asarray(data["impedance"], dtype=np.complex128),
            "planes": [str(v) for v in data["planes"].tolist()],
        }


def save_result_npz(path: Path, result: Any) -> None:
    np.savez_compressed(
        path,
        frequencies_hz=result.frequencies_hz,
        angles_deg=result.observation_angles_deg,
        spl_db=result.spl_db,
        pressure_complex=result.pressure_complex,
        impedance=result.impedance,
        planes=np.asarray(result.observation_planes, dtype=object),
    )


def render_heatmap(path: Path, result_data: dict[str, Any]) -> None:
    directivity: dict[str, list[list[list[float]]]] = {}
    for plane_idx, plane in enumerate(result_data["planes"]):
        patterns: list[list[list[float]]] = []
        for freq_idx in range(len(result_data["frequencies_hz"])):
            patterns.append(
                [
                    [
                        float(angle),
                        float(result_data["spl_db"][freq_idx, plane_idx, angle_idx]),
                    ]
                    for angle_idx, angle in enumerate(result_data["angles_deg"])
                ]
            )
        directivity[plane] = patterns
    save_directivity_plot(path, result_data["frequencies_hz"].tolist(), directivity)


def base_config(
    *,
    frame: ObservationFrame,
    assembly_mode: str,
    angle_count: int,
) -> SolveConfig:
    return SolveConfig(
        assembly_backend="metal",
        experimental_metal_backend=True,
        metal_backend_fallback="error",
        native_symmetry_plane="yz+xz",
        metal_native_assembly_mode=assembly_mode,  # type: ignore[arg-type]
        observation=ObservationConfig(
            planes=list(PLANES),
            distance_m=2.0,
            angle_min_deg=0.0,
            angle_max_deg=180.0,
            angle_count=angle_count,
            origin="throat",
        ),
        frame_override=frame,
    )


def metadata_for_result(
    *,
    name: str,
    mesh_info: dict[str, Any],
    frame: ObservationFrame,
    result_npz: Path,
    heatmap_png: Path,
    result_data: dict[str, Any],
    timings_s: dict[str, float],
    reused_from: Path | None,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "mesh": mesh_info,
        "symmetry_plane": "yz+xz",
        "native_assembly_mode": "corrected" if "corrected" in name else "optimized",
        "result_npz": str(result_npz),
        "heatmap_png": str(heatmap_png),
        "frequency_count": int(len(result_data["frequencies_hz"])),
        "angle_count": int(len(result_data["angles_deg"])),
        "planes": list(result_data["planes"]),
        "frame": {
            "mode": "axial_z_throat",
            "axis": [float(v) for v in frame.axis.tolist()],
            "origin": [float(v) for v in frame.origin.tolist()],
            "u": [float(v) for v in frame.u.tolist()],
            "v": [float(v) for v in frame.v.tolist()],
        },
        "timings_s": timings_s,
    }
    if reused_from is not None:
        payload["reused_from"] = str(reused_from)
    return payload


def reuse_corrected_case(
    source_dir: Path,
    dest_dir: Path,
    mesh_info: dict[str, Any],
    frame: ObservationFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_npz = source_dir / "result.npz"
    source_metadata = source_dir / "metadata.json"
    if not source_npz.exists() or not source_metadata.exists():
        raise FileNotFoundError(f"corrected source result missing under {source_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_npz = dest_dir / "result.npz"
    dest_heatmap = dest_dir / "directivity_heatmap.png"
    shutil.copy2(source_npz, dest_npz)
    if (source_dir / "directivity_heatmap.png").exists():
        shutil.copy2(source_dir / "directivity_heatmap.png", dest_heatmap)
    else:
        render_heatmap(dest_heatmap, load_npz(dest_npz))

    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    data = load_npz(dest_npz)
    metadata = metadata_for_result(
        name="corrected_reused",
        mesh_info=mesh_info,
        frame=frame,
        result_npz=dest_npz,
        heatmap_png=dest_heatmap,
        result_data=data,
        timings_s=dict(source["timings_s"]),
        reused_from=source_dir,
    )
    (dest_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata, data


def solve_mode(
    *,
    name: str,
    mesh_path: Path,
    frequencies: np.ndarray,
    output_dir: Path,
    assembly_mode: str,
    angle_count: int,
    skip_existing: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_dir = output_dir / name
    npz_path = case_dir / "result.npz"
    metadata_path = case_dir / "metadata.json"
    if skip_existing and npz_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata, load_npz(npz_path)

    case_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_mesh(mesh_path, scale=0.001, repair_normals=True)
    frame = axial_throat_frame(loaded)
    config = base_config(
        frame=frame,
        assembly_mode=assembly_mode,
        angle_count=angle_count,
    )

    log(f"solving {name} with native Metal assembly mode={assembly_mode}")
    started = time.perf_counter()
    result = solve_frequencies(loaded, frequencies, config)
    wall_s = time.perf_counter() - started

    save_result_npz(npz_path, result)
    data = load_npz(npz_path)
    heatmap_path = case_dir / "directivity_heatmap.png"
    render_heatmap(heatmap_path, data)

    mesh_info = inspect_mesh(mesh_path)
    metadata = metadata_for_result(
        name=name,
        mesh_info=mesh_info,
        frame=frame,
        result_npz=npz_path,
        heatmap_png=heatmap_path,
        result_data=data,
        timings_s={
            "assembly": float(result.timings.get("assembly_s", math.nan)),
            "dense_solve": float(result.timings.get("dense_solve_s", math.nan)),
            "field": float(result.timings.get("directivity_s", math.nan)),
            "solve": float(result.timings.get("solve_s", math.nan)),
            "total": float(result.timings.get("total_s", wall_s)),
            "wall": float(wall_s),
        },
        reused_from=None,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata, data


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    if denom == 0.0:
        return 0.0 if float(np.linalg.norm(a)) == 0.0 else math.inf
    return float(np.linalg.norm(a - b) / denom)


def compare_directivity(corrected: dict[str, Any], regular: dict[str, Any]) -> dict[str, Any]:
    if corrected["planes"] != regular["planes"]:
        raise ValueError("plane lists differ")
    diff = regular["spl_db"] - corrected["spl_db"]
    report: dict[str, Any] = {}
    for plane_idx, plane in enumerate(corrected["planes"]):
        plane_diff = diff[:, plane_idx, :]
        report[plane] = {
            "max_abs_db": float(np.max(np.abs(plane_diff))),
            "rms_db": rms(plane_diff),
        }
    report["all_planes"] = {
        "max_abs_db": float(np.max(np.abs(diff))),
        "rms_db": rms(diff),
    }
    return report


def compare_complex_observation(
    corrected: dict[str, Any],
    regular: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for plane_idx, plane in enumerate(corrected["planes"]):
        report[plane] = {
            "relative_l2": relative_l2(
                regular["pressure_complex"][:, plane_idx, :],
                corrected["pressure_complex"][:, plane_idx, :],
            )
        }
    report["all_planes"] = {
        "relative_l2": relative_l2(
            regular["pressure_complex"],
            corrected["pressure_complex"],
        )
    }
    return report


def compare_impedance(corrected: dict[str, Any], regular: dict[str, Any]) -> dict[str, Any]:
    diff = regular["impedance"] - corrected["impedance"]
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rms_abs": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
        "relative_l2": relative_l2(regular["impedance"], corrected["impedance"]),
    }


def timing_compare(corrected: dict[str, Any], regular: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("assembly", "dense_solve", "field", "solve", "total", "wall"):
        corrected_s = float(corrected["timings_s"].get(key, math.nan))
        regular_s = float(regular["timings_s"].get(key, math.nan))
        out[key] = {
            "corrected_s": corrected_s,
            "regular_s": regular_s,
            "delta_s": regular_s - corrected_s,
            "regular_speedup": corrected_s / regular_s if regular_s > 0.0 else None,
        }
    return out


def representative_indices(frequencies: np.ndarray, count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [0]
    raw = np.linspace(0, len(frequencies) - 1, count)
    return sorted({int(round(v)) for v in raw})


@contextmanager
def temporary_env(values: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def read_assembly(system: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = _read_complex_f32(
        Path(system.matrix_real_f32),
        Path(system.matrix_imag_f32),
        tuple(system.matrix_shape),
    )
    rhs = _read_complex_f32(
        Path(system.rhs_real_f32),
        Path(system.rhs_imag_f32),
        tuple(system.rhs_shape),
    )
    return matrix, rhs


def matrix_probe(
    *,
    mesh_path: Path,
    frequencies: np.ndarray,
    output_dir: Path,
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    loaded = load_mesh(mesh_path, scale=0.001, repair_normals=True)
    p1_space, dp0_space = _setup_function_spaces(loaded.grid)
    buffers = build_metal_geometry_buffers(
        loaded.grid,
        loaded.physical_tags,
        p1_space,
        dp0_space,
    )
    config = SolveConfig()
    source_tags = list(config.velocity_sources.keys())
    probe_dir = output_dir / "matrix_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with MetalNativeStandardSession.create_session(
        geometry_buffers=buffers,
        work_dir=probe_dir / "native-session",
        session_id="asro2-quarter-assembly-mode-probe",
        symmetry_plane="yz+xz",
    ) as session:
        for idx in representative_indices(frequencies, count):
            freq = float(frequencies[idx])
            omega = 2.0 * math.pi * freq
            k_real = omega / SPEED_OF_SOUND
            neumann = _build_driver_neumann_coeffs(
                dp0_space,
                loaded.physical_tags,
                omega,
                config,
                np.complex64,
            )
            systems: dict[str, Any] = {}
            for mode in ("corrected", "optimized"):
                with temporary_env(
                    {
                        "HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE": mode,
                        "HORNLAB_SOLVER_METAL_NATIVE_DUFFY_MODE": "gpu_blocks",
                    }
                ):
                    systems[mode] = session.assemble_standard_neumann(
                        freq,
                        k_real,
                        neumann,
                        operation_id=f"{mode}-{idx:04d}-{freq:.6g}hz",
                    )
            corrected_matrix, corrected_rhs = read_assembly(systems["corrected"])
            regular_matrix, regular_rhs = read_assembly(systems["optimized"])
            corrected_pressure = _solve_dense_direct(corrected_matrix, corrected_rhs)
            regular_pressure = _solve_dense_direct(regular_matrix, regular_rhs)

            p_surface_corrected = type(
                "PressureSurface",
                (),
                {"coefficients": corrected_pressure},
            )()
            p_surface_regular = type(
                "PressureSurface",
                (),
                {"coefficients": regular_pressure},
            )()
            corrected_impedance = _compute_impedance(
                loaded.grid,
                p_surface_corrected,
                loaded.physical_tags,
                p1_space,
                source_tag=min(config.velocity_sources.keys(), default=2),
            )
            regular_impedance = _compute_impedance(
                loaded.grid,
                p_surface_regular,
                loaded.physical_tags,
                p1_space,
                source_tag=min(config.velocity_sources.keys(), default=2),
            )
            corrected_pavg = compute_surface_pressure_avg(
                loaded.grid,
                p_surface_corrected,
                loaded.physical_tags,
                p1_space,
                source_tags,
            )
            regular_pavg = compute_surface_pressure_avg(
                loaded.grid,
                p_surface_regular,
                loaded.physical_tags,
                p1_space,
                source_tags,
            )
            rows.append(
                {
                    "frequency_index": int(idx),
                    "frequency_hz": freq,
                    "matrix_relative_l2": relative_l2(
                        regular_matrix,
                        corrected_matrix,
                    ),
                    "rhs_relative_l2": relative_l2(regular_rhs, corrected_rhs),
                    "surface_pressure_relative_l2": relative_l2(
                        regular_pressure,
                        corrected_pressure,
                    ),
                    "impedance_abs_delta": float(
                        abs(regular_impedance - corrected_impedance)
                    ),
                    "surface_pressure_avg_relative_l2": {
                        str(tag): relative_l2(
                            np.asarray([regular_pavg[tag]], dtype=np.complex128),
                            np.asarray([corrected_pavg[tag]], dtype=np.complex128),
                        )
                        for tag in source_tags
                    },
                    "assembly_files": {
                        mode: {
                            "matrix_real_f32": str(systems[mode].matrix_real_f32),
                            "matrix_imag_f32": str(systems[mode].matrix_imag_f32),
                            "rhs_real_f32": str(systems[mode].rhs_real_f32),
                            "rhs_imag_f32": str(systems[mode].rhs_imag_f32),
                        }
                        for mode in ("corrected", "optimized")
                    },
                }
            )
    return rows


def verdict(
    directivity: dict[str, Any],
    impedance: dict[str, Any],
    probes: list[dict[str, Any]],
) -> dict[str, Any]:
    max_db = float(directivity["all_planes"]["max_abs_db"])
    rms_db = float(directivity["all_planes"]["rms_db"])
    imp_rel = float(impedance["relative_l2"])
    max_surface_rel = max(
        (float(row["surface_pressure_relative_l2"]) for row in probes),
        default=math.nan,
    )
    acceptable = (
        max_db <= 0.25
        and rms_db <= 0.05
        and imp_rel <= 1.0e-3
        and (math.isnan(max_surface_rel) or max_surface_rel <= 1.0e-3)
    )
    return {
        "regular_assembly_acceptable_for_this_mesh": acceptable,
        "scope": "corrected ASRO2 WG native quarter mesh only",
        "criteria": {
            "directivity_all_planes_max_abs_db_lte": 0.25,
            "directivity_all_planes_rms_db_lte": 0.05,
            "impedance_relative_l2_lte": 1.0e-3,
            "surface_pressure_probe_relative_l2_lte": 1.0e-3,
        },
        "observed": {
            "directivity_all_planes_max_abs_db": max_db,
            "directivity_all_planes_rms_db": rms_db,
            "impedance_relative_l2": imp_rel,
            "max_surface_pressure_probe_relative_l2": max_surface_rel,
        },
    }


def validate_same_grid(corrected: dict[str, Any], regular: dict[str, Any]) -> None:
    checks = (
        ("frequencies_hz", 0.0, 0.0),
        ("angles_deg", 0.0, 0.0),
    )
    for key, rtol, atol in checks:
        if not np.allclose(corrected[key], regular[key], rtol=rtol, atol=atol):
            raise ValueError(f"{key} differs between corrected and regular results")
    if corrected["planes"] != regular["planes"]:
        raise ValueError("observation planes differ between corrected and regular results")


def main() -> int:
    args = parse_args()
    mesh_path = args.mesh.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    corrected_case_dir = args.corrected_case_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_info = inspect_mesh(mesh_path)
    loaded_for_frame = load_mesh(mesh_path, scale=0.001, repair_normals=True)
    frame = axial_throat_frame(loaded_for_frame)

    corrected_source_npz = corrected_case_dir / "result.npz"
    if args.rerun_corrected or not corrected_source_npz.exists():
        frequencies = (
            frequencies_from_npz(corrected_source_npz)
            if corrected_source_npz.exists()
            else np.geomspace(100.0, 20_000.0, 40, dtype=np.float64)
        )
        corrected_metadata, corrected_data = solve_mode(
            name="corrected",
            mesh_path=mesh_path,
            frequencies=frequencies,
            output_dir=output_dir,
            assembly_mode="corrected",
            angle_count=args.angle_count,
            skip_existing=False,
        )
    else:
        log(f"reusing corrected baseline from {corrected_case_dir}")
        corrected_metadata, corrected_data = reuse_corrected_case(
            corrected_case_dir,
            output_dir / "corrected_reused",
            mesh_info,
            frame,
        )
        frequencies = corrected_data["frequencies_hz"]

    regular_metadata, regular_data = solve_mode(
        name="optimized_regular",
        mesh_path=mesh_path,
        frequencies=frequencies,
        output_dir=output_dir,
        assembly_mode="optimized",
        angle_count=args.angle_count,
        skip_existing=args.skip_existing_regular,
    )
    validate_same_grid(corrected_data, regular_data)

    log("running representative matrix/RHS probes")
    probes = matrix_probe(
        mesh_path=mesh_path,
        frequencies=frequencies,
        output_dir=output_dir,
        count=args.matrix_probe_count,
    )

    directivity_delta = compare_directivity(corrected_data, regular_data)
    impedance_delta = compare_impedance(corrected_data, regular_data)
    summary = {
        "schema": "hornlab.asro2_quarter_assembly_mode_benchmark.v1",
        "created_local_date": date.today().isoformat(),
        "scope": "corrected ASRO2 WG native quarter mesh",
        "mesh": mesh_info,
        "frequency_grid_hz": [float(v) for v in frequencies.tolist()],
        "cases": {
            "corrected": corrected_metadata,
            "optimized_regular": regular_metadata,
        },
        "timing_s": timing_compare(corrected_metadata, regular_metadata),
        "directivity_delta_regular_vs_corrected_db": directivity_delta,
        "observation_pressure_delta_regular_vs_corrected": compare_complex_observation(
            corrected_data,
            regular_data,
        ),
        "impedance_delta_regular_vs_corrected": impedance_delta,
        "matrix_rhs_probe_regular_vs_corrected": probes,
        "verdict": verdict(directivity_delta, impedance_delta, probes),
        "notes": [
            "Corrected baseline is reused when the existing 40-frequency result is present.",
            "No ATH meshes are solved by this benchmark.",
            "Full surface-pressure vectors are compared only for representative direct assembly probes.",
        ],
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log(f"wrote {summary_path}")
    print(json.dumps(summary["verdict"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

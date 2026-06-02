#!/usr/bin/env python3
"""Compare ATH ASRO68 full/quarter meshes with a WG-generated quarter mesh."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "hornlab-mesher"))
sys.path.insert(0, str(WORKSPACE_ROOT / "hornlab-plots"))

from hornlab_plots import save_directivity_plot  # noqa: E402
from hornlab_solver import (  # noqa: E402
    ObservationConfig,
    ObservationFrame,
    SolveConfig,
    load_mesh,
    solve_frequencies,
)


ASRO_ROOT = Path(
    "/Users/magnus/IM Dropbox/Magnus Andersen/DOCS/code/misc/"
    "ATH results 0 degree norm"
)
ATH_FULL_DIR = ASRO_ROOT / "250917asro68"
ATH_QUARTER_DIR = ASRO_ROOT / "asro2"
ATH_FULL_MESH = ATH_FULL_DIR / "ABEC_FreeStanding/250917asro68.msh"
ATH_QUARTER_MESH = ATH_QUARTER_DIR / "ABEC_FreeStanding/asro2.msh"
ATH_FULL_CONFIG = ATH_FULL_DIR / "config.txt"
ATH_QUARTER_CONFIG = ATH_QUARTER_DIR / "config.txt"

DEFAULT_OUTPUT_DIR = (
    WORKSPACE_ROOT / "runs/canonical-validation/260602-asro2-quarter-directivity"
)
GEOMETRY_CLI = WORKSPACE_ROOT / "hornlab-geometry/bin/geometry-cli.js"


@dataclass(frozen=True)
class Case:
    name: str
    mesh_path: Path
    symmetry_plane: str | None
    mesh_scale: float = 0.001
    repair_normals: bool = True
    notes: tuple[str, ...] = ()


ASRO_PARAMS: dict[str, Any] = {
    "type": "R-OSSE",
    "R": "160 * (abs(cos(p)/1.8)^3 + abs(sin(p)/1)^4)^(-1/7)",
    "r": 0.35,
    "b": 0.4,
    "m": 0.84,
    "tmax": 1.0,
    "a": "22 * (abs(cos(p)/1.2)^8 + abs(sin(p)/1)^4)^(-1/4)",
    "a0": 15.5,
    "r0": 12.7,
    "k": "4 * (abs(cos(p)/1.2)^8 + abs(sin(p)/1)^4)^(-1/4)",
    "q": 4.0,
    "angularSegments": 50,
    "lengthSegments": 20,
    "throatResolution": 5.0,
    "mouthResolution": 8.0,
    "quadrants": "1",
    "wallThickness": 6.0,
    "rearResolution": 25.0,
    "encDepth": 0.0,
    "sourceShape": 1,
    "sourceRadius": -1.0,
    "sourceCurv": 0,
    "athParitySampling": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freq-count", type=int, default=40)
    parser.add_argument("--freq-min", type=float, default=100.0)
    parser.add_argument("--freq-max", type=float, default=20_000.0)
    parser.add_argument("--angle-count", type=int, default=37)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="reuse existing per-case result.npz files when present",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[asro2-compare] {message}", flush=True)


def require_inputs() -> None:
    missing = [
        path
        for path in (
            ATH_FULL_MESH,
            ATH_QUARTER_MESH,
            ATH_FULL_CONFIG,
            ATH_QUARTER_CONFIG,
        )
        if not path.exists()
    ]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"ASRO input file(s) missing:\n{joined}")


def read_config_quadrants(path: Path) -> str | None:
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("Mesh.Quadrants"):
            _, value = stripped.split("=", maxsplit=1)
            return value.strip()
    return None


def build_point_grid(params: dict[str, Any]) -> dict[str, Any]:
    message = {
        "id": "asro2-grid",
        "op": "build_point_grid",
        "params": {"params": params},
    }
    proc = subprocess.run(
        ["node", str(GEOMETRY_CLI)],
        input=json.dumps(message) + "\n",
        cwd=WORKSPACE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"geometry-cli exited {proc.returncode}")
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        if payload.get("id") == "asro2-grid":
            return dict(payload["result"])
    raise RuntimeError("geometry-cli did not return a point grid")


def reshape_grid(raw: Any, n_phi: int, n_length: int, name: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    expected = n_phi * (n_length + 1) * 3
    if arr.size != expected:
        raise ValueError(f"{name} has {arr.size} values; expected {expected}")
    return arr.reshape(n_phi, n_length + 1, 3)


def write_wg_quarter_mesh(output_dir: Path) -> Path:
    from hornlab_mesher import MeshDensity, PointGridHornGeometry, build_mesh
    from hornlab_mesher import load_mesh as inspect_generated_mesh

    out_path = output_dir / "wg_hornlab_quarter.msh"
    grid = build_point_grid(ASRO_PARAMS)
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner_points = reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")
    outer_points_raw = grid.get("outer_points")
    outer_points = (
        reshape_grid(outer_points_raw, n_phi, n_length, "outer_points")
        if outer_points_raw is not None
        else None
    )
    build_mesh(
        PointGridHornGeometry(
            inner_points=inner_points,
            preserve_grid=False,
            closed=bool(grid["full_circle"]),
            outer_points=outer_points,
            wall_thickness_mm=float(ASRO_PARAMS["wallThickness"]),
            source_shape=int(ASRO_PARAMS["sourceShape"]),
            source_radius_mm=float(ASRO_PARAMS["sourceRadius"]),
            source_curv=int(ASRO_PARAMS["sourceCurv"]),
            source_auto_angle_deg=float(ASRO_PARAMS["a0"]),
            ath_parity_topology=True,
        ),
        MeshDensity(
            throat_res_mm=float(ASRO_PARAMS["throatResolution"]),
            mouth_res_mm=float(ASRO_PARAMS["mouthResolution"]),
            rear_res_mm=float(ASRO_PARAMS["rearResolution"]),
        ),
        out_path,
        scale_to_metres=False,
    )
    info = inspect_generated_mesh(out_path)
    payload = {
        "vertices": int(info.n_vertices),
        "triangles": int(info.n_triangles),
        "metadata": {
            "generator": "hornlab-mesher-point-grid-occ",
            "grid_n_phi": n_phi,
            "grid_n_length": n_length,
            "full_circle": bool(grid["full_circle"]),
            "has_outer_points": outer_points is not None,
            "angle_list_rad": [float(v) for v in (grid.get("angle_list") or [])],
            "slice_map": [float(v) for v in (grid.get("slice_map") or [])],
            "units": info.units,
            "mesh_density": {
                "throat_res_mm": float(ASRO_PARAMS["throatResolution"]),
                "mouth_res_mm": float(ASRO_PARAMS["mouthResolution"]),
                "rear_res_mm": float(ASRO_PARAMS["rearResolution"]),
            },
            "physical_groups": {str(k): v for k, v in info.physical_groups.items()},
            "bbox": {
                "min": [float(v) for v in info.bounding_box[0].tolist()],
                "max": [float(v) for v in info.bounding_box[1].tolist()],
            },
        },
    }
    (output_dir / "wg_hornlab_quarter_payload.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def _triangle_cell_key(mesh: Any) -> str:
    if "triangle" in mesh.cells_dict:
        return "triangle"
    if "triangle3" in mesh.cells_dict:
        return "triangle3"
    raise ValueError("mesh has no triangle cells")


def _physical_tags(mesh: Any, tri_key: str) -> np.ndarray:
    for key, by_type in mesh.cell_data_dict.items():
        if "physical" in key and tri_key in by_type:
            return np.asarray(by_type[tri_key], dtype=np.int32)
    raise ValueError("mesh has no triangle physical tags")


def inspect_mesh(path: Path, *, scale: float = 0.001, repair_normals: bool = True) -> dict[str, Any]:
    loaded = load_mesh(path, scale=scale, repair_normals=repair_normals)
    bb_min, bb_max = loaded.info.bounding_box_m
    unique, counts = np.unique(loaded.physical_tags, return_counts=True)
    return {
        "path": str(path),
        "vertices": int(loaded.info.n_vertices),
        "triangles": int(loaded.info.n_triangles),
        "physical_tag_counts": {
            str(int(tag)): int(count) for tag, count in zip(unique, counts, strict=True)
        },
        "physical_groups": {str(k): v for k, v in loaded.info.physical_groups.items()},
        "bbox_m": {
            "min": [float(v) for v in bb_min.tolist()],
            "max": [float(v) for v in bb_max.tolist()],
        },
    }


def snap_positive_quadrant_mesh(in_path: Path, out_path: Path, *, tolerance_mm: float) -> dict[str, Any]:
    import meshio

    mesh = meshio.read(in_path)
    points = np.asarray(mesh.points, dtype=np.float64).copy()
    before_min = points.min(axis=0)
    snap_mask_x = (points[:, 0] < 0.0) & (points[:, 0] >= -tolerance_mm)
    snap_mask_y = (points[:, 1] < 0.0) & (points[:, 1] >= -tolerance_mm)
    too_negative = np.flatnonzero(
        (points[:, 0] < -tolerance_mm) | (points[:, 1] < -tolerance_mm)
    )
    if too_negative.size:
        first = int(too_negative[0])
        raise ValueError(
            f"{in_path} has vertex {first} outside snap tolerance: "
            f"{points[first].tolist()} mm"
        )
    points[snap_mask_x, 0] = 0.0
    points[snap_mask_y, 1] = 0.0
    after_min = points.min(axis=0)
    snapped = meshio.Mesh(
        points=points,
        cells=mesh.cells,
        cell_data=mesh.cell_data,
        field_data=mesh.field_data,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(out_path, snapped, file_format="gmsh22", binary=False)
    return {
        "source": str(in_path),
        "path": str(out_path),
        "tolerance_mm": float(tolerance_mm),
        "snapped_x_vertices": int(np.count_nonzero(snap_mask_x)),
        "snapped_y_vertices": int(np.count_nonzero(snap_mask_y)),
        "min_before_mm": [float(v) for v in before_min.tolist()],
        "min_after_mm": [float(v) for v in after_min.tolist()],
    }


def expand_quarter_mesh_xy(in_path: Path, out_path: Path) -> dict[str, Any]:
    import meshio

    mesh = meshio.read(in_path)
    tri_key = _triangle_cell_key(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    triangles = np.asarray(mesh.cells_dict[tri_key], dtype=np.int64)
    tags = _physical_tags(mesh, tri_key)
    if np.min(points[:, 0]) < -1.0e-9 or np.min(points[:, 1]) < -1.0e-9:
        raise ValueError("quarter mesh must be snapped into X>=0, Y>=0 before expansion")

    vertex_map: dict[tuple[int, int, int], int] = {}
    vertices: list[np.ndarray] = []
    expanded_tris: list[np.ndarray] = []
    expanded_tags: list[np.ndarray] = []
    tol = 1.0e-9

    def key_for(point: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.round(point / tol).astype(np.int64).tolist())

    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        remap = np.empty(points.shape[0], dtype=np.int64)
        for idx, point in enumerate(points):
            mirrored = np.array([sx * point[0], sy * point[1], point[2]], dtype=np.float64)
            key = key_for(mirrored)
            out_idx = vertex_map.get(key)
            if out_idx is None:
                out_idx = len(vertices)
                vertex_map[key] = out_idx
                vertices.append(mirrored)
            remap[idx] = out_idx
        mapped = remap[triangles]
        if sx * sy < 0:
            mapped = mapped[:, [0, 2, 1]]
        expanded_tris.append(mapped)
        expanded_tags.append(tags.copy())

    used_tags = sorted({int(tag) for tag in tags.tolist()})
    names = {
        1: np.array([1, 2], dtype=np.int32),
        2: np.array([2, 2], dtype=np.int32),
        3: np.array([3, 2], dtype=np.int32),
        4: np.array([4, 2], dtype=np.int32),
    }
    field_data = {
        ("SD1G0" if tag == 1 else "SD1D1001" if tag == 2 else f"SD{tag}"):
        names.get(tag, np.array([tag, 2], dtype=np.int32))
        for tag in used_tags
    }
    out_mesh = meshio.Mesh(
        points=np.asarray(vertices, dtype=np.float64),
        cells=[("triangle", np.vstack(expanded_tris).astype(np.int64))],
        cell_data={
            "gmsh:physical": [np.concatenate(expanded_tags).astype(np.int32)],
            "gmsh:geometrical": [np.concatenate(expanded_tags).astype(np.int32)],
        },
        field_data=field_data,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(out_path, out_mesh, file_format="gmsh22", binary=False)
    return {
        "source": str(in_path),
        "path": str(out_path),
        "vertices": int(len(vertices)),
        "triangles": int(sum(block.shape[0] for block in expanded_tris)),
    }


def frequencies(args: argparse.Namespace) -> np.ndarray:
    return np.geomspace(args.freq_min, args.freq_max, args.freq_count).astype(np.float64)


def solve_case(case: Case, freqs: np.ndarray, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    npz_path = case_dir / "result.npz"
    heatmap_path = case_dir / "directivity_heatmap.png"
    metadata_path = case_dir / "metadata.json"

    if args.skip_existing and npz_path.exists() and metadata_path.exists():
        log(f"reusing {case.name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata

    mesh_info = inspect_mesh(
        case.mesh_path,
        scale=case.mesh_scale,
        repair_normals=case.repair_normals,
    )
    log(
        f"solving {case.name}: {mesh_info['vertices']} verts, "
        f"{mesh_info['triangles']} tris, symmetry={case.symmetry_plane or 'none'}"
    )
    loaded = load_mesh(
        case.mesh_path,
        scale=case.mesh_scale,
        repair_normals=case.repair_normals,
    )
    frame = axial_throat_frame(loaded)
    config = SolveConfig(
        freq_min_hz=float(freqs[0]),
        freq_max_hz=float(freqs[-1]),
        freq_count=len(freqs),
        freq_spacing="log",
        assembly_backend="metal",
        experimental_metal_backend=True,
        metal_backend_fallback="error",
        native_symmetry_plane=case.symmetry_plane,
        observation=ObservationConfig(
            planes=["horizontal", "vertical", "diagonal"],
            distance_m=2.0,
            angle_min_deg=0.0,
            angle_max_deg=180.0,
            angle_count=args.angle_count,
            origin="throat",
        ),
        frame_override=frame,
    )
    started = time.perf_counter()
    result = solve_frequencies(loaded, freqs, config)
    wall_s = time.perf_counter() - started

    np.savez_compressed(
        npz_path,
        frequencies_hz=result.frequencies_hz,
        angles_deg=result.observation_angles_deg,
        spl_db=result.spl_db,
        pressure_complex=result.pressure_complex,
        impedance=result.impedance,
        planes=np.asarray(result.observation_planes, dtype=object),
    )
    render_heatmap(heatmap_path, result)

    metadata = {
        "name": case.name,
        "mesh": mesh_info,
        "symmetry_plane": case.symmetry_plane,
        "notes": list(case.notes),
        "frame": {
            "mode": "axial_z_throat",
            "axis": [float(v) for v in frame.axis.tolist()],
            "origin": [float(v) for v in frame.origin.tolist()],
            "u": [float(v) for v in frame.u.tolist()],
            "v": [float(v) for v in frame.v.tolist()],
        },
        "result_npz": str(npz_path),
        "heatmap_png": str(heatmap_path),
        "frequency_count": int(len(result.frequencies_hz)),
        "angle_count": int(len(result.observation_angles_deg)),
        "planes": list(result.observation_planes),
        "timings_s": {
            "assembly": float(result.timings.get("assembly_s", math.nan)),
            "dense_solve": float(result.timings.get("dense_solve_s", math.nan)),
            "field": float(result.timings.get("directivity_s", math.nan)),
            "solve": float(result.timings.get("solve_s", math.nan)),
            "total": float(result.timings.get("total_s", wall_s)),
            "wall": float(wall_s),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(
        f"finished {case.name}: total={metadata['timings_s']['total']:.2f}s, "
        f"assembly={metadata['timings_s']['assembly']:.2f}s, "
        f"solve={metadata['timings_s']['dense_solve']:.2f}s, "
        f"field={metadata['timings_s']['field']:.2f}s"
    )
    return metadata


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


def render_heatmap(path: Path, result: Any) -> None:
    plane_names = list(result.observation_planes)
    directivity: dict[str, list[list[list[float]]]] = {}
    for plane_idx, plane in enumerate(plane_names):
        patterns: list[list[list[float]]] = []
        for freq_idx in range(len(result.frequencies_hz)):
            patterns.append(
                [
                    [
                        float(angle),
                        float(result.spl_db[freq_idx, plane_idx, angle_idx]),
                    ]
                    for angle_idx, angle in enumerate(result.observation_angles_deg)
                ]
            )
        directivity[plane] = patterns
    save_directivity_plot(path, result.frequencies_hz.tolist(), directivity)


def load_result(metadata: dict[str, Any]) -> dict[str, Any]:
    data = np.load(metadata["result_npz"], allow_pickle=True)
    return {
        "frequencies_hz": np.asarray(data["frequencies_hz"], dtype=np.float64),
        "angles_deg": np.asarray(data["angles_deg"], dtype=np.float64),
        "spl_db": np.asarray(data["spl_db"], dtype=np.float64),
        "planes": [str(v) for v in data["planes"].tolist()],
    }


def compare_directivity(reference: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    ref = load_result(reference)
    cur = load_result(other)
    if ref["spl_db"].shape != cur["spl_db"].shape:
        return {"error": f"shape mismatch {ref['spl_db'].shape} != {cur['spl_db'].shape}"}
    if not np.allclose(ref["frequencies_hz"], cur["frequencies_hz"]):
        return {"error": "frequency grids differ"}
    if not np.allclose(ref["angles_deg"], cur["angles_deg"]):
        return {"error": "angle grids differ"}
    out: dict[str, Any] = {}
    for plane_idx, plane in enumerate(ref["planes"]):
        diff = cur["spl_db"][:, plane_idx, :] - ref["spl_db"][:, plane_idx, :]
        finite = np.isfinite(diff)
        if not np.any(finite):
            out[plane] = {"max_abs_db": None, "rms_db": None}
            continue
        out[plane] = {
            "max_abs_db": float(np.max(np.abs(diff[finite]))),
            "rms_db": float(np.sqrt(np.mean(np.square(diff[finite])))),
        }
    diff_all = cur["spl_db"] - ref["spl_db"]
    finite_all = np.isfinite(diff_all)
    out["all_planes"] = {
        "max_abs_db": float(np.max(np.abs(diff_all[finite_all]))),
        "rms_db": float(np.sqrt(np.mean(np.square(diff_all[finite_all])))),
    }
    return out


def timing_speedups(reference: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    ref_t = reference["timings_s"]
    cur_t = other["timings_s"]
    out: dict[str, Any] = {}
    for key in ("assembly", "dense_solve", "field", "total", "wall"):
        ref_value = float(ref_t.get(key, math.nan))
        cur_value = float(cur_t.get(key, math.nan))
        out[key] = {
            "reference_s": ref_value,
            "case_s": cur_value,
            "speedup": ref_value / cur_value if cur_value > 0.0 else None,
            "delta_s": cur_value - ref_value,
        }
    return out


def main() -> None:
    args = parse_args()
    require_inputs()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    freqs = frequencies(args)

    shutil.copyfile(ATH_FULL_CONFIG, output_dir / "ath_full_config.txt")
    shutil.copyfile(ATH_QUARTER_CONFIG, output_dir / "ath_quarter_asro2_config.txt")
    wg_quarter = write_wg_quarter_mesh(output_dir)
    wg_snapped = output_dir / "wg_hornlab_quarter_snapped.msh"
    snap_meta = snap_positive_quadrant_mesh(wg_quarter, wg_snapped, tolerance_mm=1.0)
    ath_quarter_full = output_dir / "ath_quarter_asro2_expanded_full.msh"
    ath_expansion_meta = expand_quarter_mesh_xy(ATH_QUARTER_MESH, ath_quarter_full)
    wg_full = output_dir / "wg_hornlab_quarter_snapped_expanded_full.msh"
    expansion_meta = expand_quarter_mesh_xy(wg_snapped, wg_full)

    cases = [
        Case(
            "ath_full",
            ATH_FULL_MESH,
            None,
            notes=("ATH Mesh.Quadrants=1234 full mesh from 250917asro68.",),
        ),
        Case(
            "ath_quarter_asro2_native",
            ATH_QUARTER_MESH,
            "yz+xz",
            notes=("ATH Mesh.Quadrants=1 quarter mesh from asro2 with native image symmetry.",),
        ),
        Case(
            "ath_quarter_asro2_expanded_full",
            ath_quarter_full,
            None,
            notes=(
                "Exact four-image full expansion of the ATH asro2 quarter mesh.",
                "Included to separate ATH full-vs-quarter mesh-generation differences from native reduced-domain symmetry.",
            ),
        ),
        Case(
            "wg_hornlab_quarter_snapped_native",
            wg_snapped,
            "yz+xz",
            notes=(
                "WG/HornLab-generated quarter mesh from the same public ASRO config.",
                "Sub-millimeter negative X/Y seam coordinates were snapped to satisfy native positive-domain symmetry.",
            ),
        ),
        Case(
            "wg_hornlab_quarter_snapped_expanded_full",
            wg_full,
            None,
            notes=(
                "Exact four-image full expansion of the snapped WG/HornLab quarter mesh.",
                "Included to separate mesh-generation differences from native reduced-domain timing.",
            ),
        ),
    ]

    summary: dict[str, Any] = {
        "created_utc_note": "local run on 2026-06-02",
        "output_dir": str(output_dir),
        "frequency_grid_hz": [float(v) for v in freqs.tolist()],
        "angle_count": int(args.angle_count),
        "source_configs": {
            "ath_full": str(ATH_FULL_CONFIG),
            "ath_full_quadrants": read_config_quadrants(ATH_FULL_CONFIG),
            "ath_quarter": str(ATH_QUARTER_CONFIG),
            "ath_quarter_quadrants": read_config_quadrants(ATH_QUARTER_CONFIG),
        },
        "wg_generation": {
            "params": ASRO_PARAMS,
            "quarter_mesh": str(wg_quarter),
            "ath_quarter_expanded_full": ath_expansion_meta,
            "snap": snap_meta,
            "expanded_full": expansion_meta,
        },
        "cases": {},
        "directivity_delta_vs_ath_full_db": {},
        "timing_vs_ath_full": {},
        "native_symmetry_parity_db": {},
        "native_symmetry_timing_vs_expanded_full": {},
    }

    for case in cases:
        try:
            summary["cases"][case.name] = solve_case(case, freqs, args, output_dir)
        except Exception as exc:
            summary["cases"][case.name] = {
                "name": case.name,
                "mesh_path": str(case.mesh_path),
                "symmetry_plane": case.symmetry_plane,
                "notes": list(case.notes),
                "error": f"{type(exc).__name__}: {exc}",
            }
            log(f"failed {case.name}: {type(exc).__name__}: {exc}")
            raise

    reference = summary["cases"]["ath_full"]
    for name, metadata in summary["cases"].items():
        if name == "ath_full" or "result_npz" not in metadata:
            continue
        summary["directivity_delta_vs_ath_full_db"][name] = compare_directivity(
            reference,
            metadata,
        )
        summary["timing_vs_ath_full"][name] = timing_speedups(reference, metadata)

    parity_pairs = {
        "ath_quarter_asro2": (
            "ath_quarter_asro2_expanded_full",
            "ath_quarter_asro2_native",
        ),
        "wg_hornlab_quarter_snapped": (
            "wg_hornlab_quarter_snapped_expanded_full",
            "wg_hornlab_quarter_snapped_native",
        ),
    }
    for label, (expanded_name, native_name) in parity_pairs.items():
        expanded = summary["cases"].get(expanded_name, {})
        native = summary["cases"].get(native_name, {})
        if "result_npz" not in expanded or "result_npz" not in native:
            continue
        summary["native_symmetry_parity_db"][label] = compare_directivity(
            expanded,
            native,
        )
        summary["native_symmetry_timing_vs_expanded_full"][label] = timing_speedups(
            expanded,
            native,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

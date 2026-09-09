#!/usr/bin/env python3
"""Compare CircSym with a matched quarter-domain full-3D Metal solve.

The harness intentionally accepts an external mesher config and quarter mesh so
the solver package does not acquire a mesher dependency. Mesh generation is not
part of either timed arm. Runs are paired and alternate order after one excluded
warm-up per arm.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAG_SOURCE = 2


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _azimuth_int(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 8:
        raise argparse.ArgumentTypeError("must be at least 8")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--quarter-mesh", type=Path, required=True)
    parser.add_argument("--f1", type=_positive_float, default=100.0)
    parser.add_argument("--f2", type=_positive_float, default=20_000.0)
    parser.add_argument("--frequencies", type=_positive_int, default=40)
    parser.add_argument("--angles", type=_positive_int, default=37)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument(
        "--axisym-backend",
        choices=("cpu", "metal"),
        default="cpu",
        help="CircSym assembly and field executor (default: %(default)s)",
    )
    parser.add_argument(
        "--cpu-field",
        choices=("numpy", "numba"),
        default="numba",
        help="CircSym portable field executor (default: %(default)s)",
    )
    parser.add_argument(
        "--qualification-ratio",
        type=_positive_float,
        default=0.5,
        help="required warm median CircSym/quarter ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--azimuth-min",
        type=_azimuth_int,
        default=64,
        metavar="N",
        help="experimental CircSym azimuth-order floor (default: %(default)s)",
    )
    parser.add_argument("--max-pressure-rel-l2", type=_positive_float, default=0.02)
    parser.add_argument("--max-directivity-error-db", type=_positive_float, default=0.5)
    parser.add_argument("--max-phase-rms-deg", type=_positive_float, default=5.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.f2 < args.f1:
        parser.error("--f2 must be greater than or equal to --f1")
    if args.angles < 2:
        parser.error("--angles must be at least 2")
    if args.qualification_ratio >= 1.0:
        parser.error("--qualification-ratio must be less than 1")
    return args


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imaginary": value.imag}
    return value


def _build_inputs(config_path: Path, quarter_mesh_path: Path, angle_count: int):
    import hornlab_mesher as hm

    import hornlab_metal_bem as mb

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    meridian = hm.build_meridian(raw_config).as_metal_meridian()
    quarter = mb.load_mesh(
        quarter_mesh_path,
        scale=1.0,
        native_symmetry_plane="yz+xz",
    )
    _validate_matched_inputs(raw_config, meridian, quarter)
    frame = mb.ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.zeros(3),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=np.zeros(3),
        source_center=np.zeros(3),
    )
    observation = mb.ObservationConfig(
        planes=["horizontal", "vertical"],
        distance_m=2.0,
        angle_min_deg=0.0,
        angle_max_deg=180.0,
        angle_count=angle_count,
    )
    common = {
        "velocity_sources": {TAG_SOURCE: 1.0},
        "observation": observation,
        "frame_override": frame,
        "formulation": "complex_k",
        "complex_k_shift": 0.005,
    }
    axisym_config = mb.SolveConfig(**common)
    quarter_config = mb.SolveConfig(
        **common,
        mesh_scale=1.0,
        native_symmetry_plane="yz+xz",
        metal_native_assembly_mode="corrected",
        dense_solve_implementation="cgetrf_cgetrs",
    )
    return meridian, quarter, axisym_config, quarter_config


def _validate_matched_inputs(
    raw_config: dict[str, Any], meridian: Any, quarter: Any
) -> None:
    if str(raw_config.get("mode", "")).strip().lower() != "freestanding":
        raise ValueError("matched benchmark currently requires mode='freestanding'")
    cross_section = dict(raw_config.get("cross_section") or {})
    morph = dict(raw_config.get("morph") or {})
    if not np.isclose(float(cross_section.get("exponent", 2.0)), 2.0) or not np.isclose(
        float(cross_section.get("aspectRatio", 1.0)), 1.0
    ):
        raise ValueError("matched benchmark requires a circular cross section")
    if not np.isclose(float(morph.get("morphTarget", 0.0)), 0.0):
        raise ValueError("matched benchmark does not support morphed geometry")

    meridian_tags = {int(tag) for tag in np.unique(meridian.physical_tags)}
    quarter_tags = {int(tag) for tag in np.unique(quarter.physical_tags)}
    if meridian_tags != {1, TAG_SOURCE} or quarter_tags != {1, TAG_SOURCE}:
        raise ValueError(
            "matched benchmark requires only rigid-wall tag 1 and source tag 2; "
            f"got meridian={sorted(meridian_tags)}, quarter={sorted(quarter_tags)}"
        )

    vertices = np.asarray(quarter.grid.vertices, dtype=np.float64).T
    if np.min(vertices[:, :2]) < -1.0e-6:
        raise ValueError("quarter mesh must lie in the non-negative X/Y quadrant")
    meridian_rho_max = float(np.max(meridian.nodes[:, 0]))
    quarter_rho_max = float(np.max(np.hypot(vertices[:, 0], vertices[:, 1])))
    if not np.isclose(quarter_rho_max, meridian_rho_max, rtol=0.01, atol=1.0e-5):
        raise ValueError("quarter mesh radial extent does not match the meridian")
    if not np.allclose(
        [np.min(vertices[:, 2]), np.max(vertices[:, 2])],
        [np.min(meridian.nodes[:, 1]), np.max(meridian.nodes[:, 1])],
        rtol=0.01,
        atol=5.0e-4,
    ):
        raise ValueError("quarter mesh axial extent does not match the meridian")

    triangles = np.asarray(quarter.grid.elements, dtype=np.int64).T
    corners = vertices[triangles]
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    quarter_source_area = 4.0 * float(
        np.sum(triangle_areas[np.asarray(quarter.physical_tags) == TAG_SOURCE])
    )
    meridian_geom = meridian.segment_geometry()
    meridian_source_area = float(
        np.sum(
            meridian_geom.area_weights[np.asarray(meridian.physical_tags) == TAG_SOURCE]
        )
    )
    if not np.isclose(
        quarter_source_area, meridian_source_area, rtol=0.05, atol=1.0e-10
    ):
        raise ValueError("quarter mesh source area does not match the meridian")


def _timed(call: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - started
    if not np.all(np.isfinite(result.pressure_complex)):
        raise RuntimeError("solver returned non-finite pressure")
    return result, {
        "wall_seconds": elapsed,
        "timings": _jsonable(dict(result.timings)),
    }


def _accuracy(axisym: Any, quarter: Any) -> dict[str, float]:
    axisym_pressure = np.asarray(axisym.pressure_complex, dtype=np.complex128)
    quarter_pressure = np.asarray(quarter.pressure_complex, dtype=np.complex128)
    if axisym_pressure.shape != quarter_pressure.shape:
        raise RuntimeError(
            "matched comparison returned different pressure shapes: "
            f"{axisym_pressure.shape} versus {quarter_pressure.shape}"
        )
    scale = max(float(np.linalg.norm(quarter_pressure)), 1.0e-30)
    pressure_rel_l2 = float(np.linalg.norm(axisym_pressure - quarter_pressure) / scale)
    magnitude_scale = max(float(np.linalg.norm(np.abs(quarter_pressure))), 1.0e-30)
    pressure_magnitude_rel_l2 = float(
        np.linalg.norm(np.abs(axisym_pressure) - np.abs(quarter_pressure))
        / magnitude_scale
    )
    axisym_db = np.asarray(axisym.directivity_db, dtype=np.float64)
    quarter_db = np.asarray(quarter.directivity_db, dtype=np.float64)
    directivity_max_abs_db = float(np.max(np.abs(axisym_db - quarter_db)))
    directivity_mask = (axisym_db >= -40.0) & (quarter_db >= -40.0)
    directivity_max_abs_db_above_minus_40 = float(
        np.max(np.abs(axisym_db[directivity_mask] - quarter_db[directivity_mask]))
    )
    mask = np.abs(quarter_pressure) >= 1.0e-4 * float(np.max(np.abs(quarter_pressure)))
    phase_delta = np.angle(axisym_pressure[mask] / quarter_pressure[mask])
    phase_rms_deg = float(np.sqrt(np.mean(np.square(np.rad2deg(phase_delta)))))
    return {
        "pressure_relative_l2": pressure_rel_l2,
        "pressure_magnitude_relative_l2": pressure_magnitude_rel_l2,
        "directivity_max_abs_db": directivity_max_abs_db,
        "directivity_max_abs_db_above_minus_40": (
            directivity_max_abs_db_above_minus_40
        ),
        "phase_rms_degrees_above_floor": phase_rms_deg,
    }


def _axisym_parity(candidate: Any, reference: Any) -> dict[str, float]:
    metrics = _accuracy(candidate, reference)
    candidate_impedance = np.asarray(candidate.impedance, dtype=np.complex128)
    reference_impedance = np.asarray(reference.impedance, dtype=np.complex128)
    if candidate_impedance.shape != reference_impedance.shape:
        raise RuntimeError("axisymmetric parity returned different impedance shapes")
    scale = max(float(np.linalg.norm(reference_impedance)), 1.0e-30)
    metrics["impedance_relative_l2"] = float(
        np.linalg.norm(candidate_impedance - reference_impedance) / scale
    )
    return metrics


def _median_wall(records: list[dict[str, Any]]) -> float:
    return float(statistics.median(float(record["wall_seconds"]) for record in records))


def _passes_overall_qualification(
    *,
    speed_ratio: bool,
    half_second: bool,
    compact_cpu_parity: bool,
    numerical_gate: bool,
) -> bool:
    return speed_ratio and half_second and compact_cpu_parity and numerical_gate


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["HORNLAB_CIRCSYM_ASSEMBLY_BACKEND"] = args.axisym_backend
    os.environ["HORNLAB_CIRCSYM_FIELD_BACKEND"] = args.axisym_backend
    os.environ["HORNLAB_CIRCSYM_CPU_FIELD_BACKEND"] = args.cpu_field
    os.environ["HORNLAB_CIRCSYM_AZIMUTH_POINTS_MIN"] = str(args.azimuth_min)

    import hornlab_metal_bem as mb

    meridian, quarter, axisym_config, quarter_config = _build_inputs(
        args.config,
        args.quarter_mesh,
        args.angles,
    )
    frequencies = np.geomspace(args.f1, args.f2, args.frequencies)

    def axisym_call():
        return mb.solve_circsym_frequencies(meridian, frequencies, axisym_config)

    def quarter_call():
        return mb.solve_frequencies(quarter, frequencies, quarter_config)

    axisym_warmup, axisym_cold = _timed(axisym_call)
    quarter_warmup, quarter_cold = _timed(quarter_call)
    records: dict[str, list[dict[str, Any]]] = {"axisymmetric": [], "quarter_3d": []}
    last_results = {"axisymmetric": axisym_warmup, "quarter_3d": quarter_warmup}
    for repeat_index in range(args.repeats):
        order = ("axisymmetric", "quarter_3d")
        if repeat_index % 2:
            order = tuple(reversed(order))
        for name in order:
            call = axisym_call if name == "axisymmetric" else quarter_call
            result, record = _timed(call)
            record["repeat_index"] = repeat_index
            record["order_in_pair"] = order.index(name)
            records[name].append(record)
            last_results[name] = result

    axisym_median = _median_wall(records["axisymmetric"])
    quarter_median = _median_wall(records["quarter_3d"])
    ratio = axisym_median / quarter_median
    axisym_candidate = last_results["axisymmetric"]
    os.environ["HORNLAB_CIRCSYM_ASSEMBLY_BACKEND"] = "cpu"
    os.environ["HORNLAB_CIRCSYM_FIELD_BACKEND"] = "cpu"
    compact_cpu_reference = axisym_call()
    os.environ["HORNLAB_CIRCSYM_ASSEMBLY_BACKEND"] = args.axisym_backend
    os.environ["HORNLAB_CIRCSYM_FIELD_BACKEND"] = args.axisym_backend
    os.environ["HORNLAB_CIRCSYM_AZIMUTH_POINTS_MIN"] = "64"
    axisym_reference = axisym_call()
    os.environ["HORNLAB_CIRCSYM_AZIMUTH_POINTS_MIN"] = str(args.azimuth_min)
    cross_solver_difference = _accuracy(
        last_results["axisymmetric"], last_results["quarter_3d"]
    )
    numerical_gate = {
        "max_pressure_relative_l2": args.max_pressure_rel_l2,
        "max_directivity_error_db_above_minus_40": args.max_directivity_error_db,
        "max_phase_rms_degrees_above_floor": args.max_phase_rms_deg,
    }
    passes_numerical_gate = (
        cross_solver_difference["pressure_relative_l2"] <= args.max_pressure_rel_l2
        and cross_solver_difference["directivity_max_abs_db_above_minus_40"]
        <= args.max_directivity_error_db
        and cross_solver_difference["phase_rms_degrees_above_floor"]
        <= args.max_phase_rms_deg
    )
    passes_speed_ratio = ratio < args.qualification_ratio
    passes_half_second = axisym_median <= 0.5
    compact_cpu_parity = _axisym_parity(axisym_candidate, compact_cpu_reference)
    passes_compact_cpu_parity = (
        compact_cpu_parity["pressure_relative_l2"] < 5.0e-5
        and compact_cpu_parity["directivity_max_abs_db"] < 0.01
        and compact_cpu_parity["impedance_relative_l2"] < 5.0e-5
    )
    try:
        package_version = importlib.metadata.version("hornlab-metal-bem")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source-tree"
    from hornlab_metal_bem.metal.native import discover_native_runtime

    native_runtime = discover_native_runtime(run_smoke_test=False)
    payload = {
        "benchmark": "axisymmetric_vs_quarter_3d",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "package_version": package_version,
        "process_load_average": os.getloadavg(),
        "native_runtime": {
            "helper_source": native_runtime.helper_source,
            "helper_name": (
                native_runtime.helper_executable_path.name
                if native_runtime.helper_executable_path is not None
                else None
            ),
            "apple_silicon": native_runtime.is_apple_silicon,
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "HORNLAB_CIRCSYM_ASSEMBLY_BACKEND",
                "HORNLAB_CIRCSYM_FIELD_BACKEND",
                "HORNLAB_CIRCSYM_CPU_FIELD_BACKEND",
                "HORNLAB_CIRCSYM_CPU_REMAINDER_BACKEND",
                "HORNLAB_CIRCSYM_AZIMUTH_POINTS_MIN",
                "HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE",
                "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE",
                "HORNLAB_METAL_BEM_NATIVE_SOLVE_CONCURRENCY",
                "NUMBA_THREADING_LAYER",
                "NUMBA_NUM_THREADS",
            )
        },
        "inputs": {
            "config": str(args.config.resolve()),
            "quarter_mesh": str(args.quarter_mesh.resolve()),
            "frequency_count": args.frequencies,
            "frequencies_hz": frequencies,
            "angle_count": args.angles,
            "meridian_segments": meridian.segment_count,
            "quarter_triangles": quarter.info.n_triangles,
            "quarter_symmetry": "yz+xz",
            "axisymmetric_azimuth_min": args.azimuth_min,
        },
        "axisymmetric_backend": {
            "assembly": args.axisym_backend,
            "field": (
                f"cpu-{args.cpu_field}"
                if args.axisym_backend == "cpu"
                else "metal-frequency-batch"
            ),
        },
        "warmup_excluded": {
            "axisymmetric": axisym_cold,
            "quarter_3d": quarter_cold,
        },
        "runs": records,
        "summary": {
            "axisymmetric_median_seconds": axisym_median,
            "quarter_3d_median_seconds": quarter_median,
            "axisymmetric_over_quarter_ratio": ratio,
            "faster_than_quarter": ratio < 1.0,
            "qualification_ratio": args.qualification_ratio,
            "passes_speed_ratio": passes_speed_ratio,
            "passes_half_second_target": passes_half_second,
            "passes_compact_cpu_parity": passes_compact_cpu_parity,
            "passes_numerical_gate": passes_numerical_gate,
            "passes_overall_qualification": _passes_overall_qualification(
                speed_ratio=passes_speed_ratio,
                half_second=passes_half_second,
                compact_cpu_parity=passes_compact_cpu_parity,
                numerical_gate=passes_numerical_gate,
            ),
        },
        "numerical_gate": numerical_gate,
        "cross_solver_difference": cross_solver_difference,
        "candidate_vs_64_point_axisymmetric": _accuracy(
            last_results["axisymmetric"], axisym_reference
        ),
        "candidate_vs_compact_cpu": compact_cpu_parity,
        "compact_cpu_parity_gate": {
            "max_pressure_relative_l2": 5.0e-5,
            "max_directivity_error_db": 0.01,
            "max_impedance_relative_l2": 5.0e-5,
        },
    }
    if args.json:
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    else:
        summary = payload["summary"]
        print(
            "Axisymmetric median "
            f"{summary['axisymmetric_median_seconds']:.3f} s; quarter 3-D median "
            f"{summary['quarter_3d_median_seconds']:.3f} s; ratio "
            f"{summary['axisymmetric_over_quarter_ratio']:.3f}"
        )
        print(
            "qualification: "
            + ("PASS" if summary["passes_overall_qualification"] else "FAIL")
        )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())

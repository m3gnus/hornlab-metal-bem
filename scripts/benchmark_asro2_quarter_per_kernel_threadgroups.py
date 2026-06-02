#!/usr/bin/env python3
"""Benchmark per-kernel native Metal threadgroups on the corrected ASRO2 quarter mesh."""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_asro2_quarter_assembly_modes import (  # noqa: E402
    DEFAULT_CORRECTED_CASE,
    DEFAULT_MESH,
    PLANES,
    axial_throat_frame,
    inspect_mesh,
    load_npz,
    relative_l2,
    save_result_npz,
)
from hornlab_solver import ObservationConfig, SolveConfig, load_mesh, solve_frequencies  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    WORKSPACE_ROOT
    / "runs/canonical-validation/260602-asro2-quarter-per-kernel-threadgroup-tuning"
)
GLOBAL_FALLBACK_THREADS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--corrected-case-dir", type=Path, default=DEFAULT_CORRECTED_CASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--angle-count", type=int, default=37)
    parser.add_argument("--global-threadgroup", type=int, default=GLOBAL_FALLBACK_THREADS)
    return parser.parse_args()


def config_for_case(
    *,
    frame: Any,
    angle_count: int,
    global_threads: int,
    kernel: str | None,
    threads: int | None,
) -> SolveConfig:
    kwargs: dict[str, Any] = {
        "metal_native_threads_per_group": global_threads,
    }
    if kernel is not None:
        kwargs[f"metal_native_{kernel}_threads_per_group"] = threads
    return SolveConfig(
        assembly_backend="metal",
        experimental_metal_backend=True,
        metal_backend_fallback="error",
        native_symmetry_plane="yz+xz",
        metal_native_assembly_mode="corrected",
        observation=ObservationConfig(
            planes=list(PLANES),
            distance_m=2.0,
            angle_min_deg=0.0,
            angle_max_deg=180.0,
            angle_count=angle_count,
            origin="throat",
        ),
        frame_override=frame,
        **kwargs,
    )


def directivity_delta_db(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    diff = candidate["spl_db"] - reference["spl_db"]
    return {
        "max_abs_db": float(np.max(np.abs(diff))),
        "rms_db": float(np.sqrt(np.mean(np.square(diff)))),
    }


def impedance_delta(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    diff = candidate["impedance"] - reference["impedance"]
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rms_abs": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
        "relative_l2": relative_l2(candidate["impedance"], reference["impedance"]),
    }


def timings(result: Any, wall_s: float) -> dict[str, float]:
    return {
        "assembly": float(result.timings.get("assembly_s", math.nan)),
        "dense_solve": float(result.timings.get("dense_solve_s", math.nan)),
        "field": float(result.timings.get("directivity_s", math.nan)),
        "solve": float(result.timings.get("solve_s", math.nan)),
        "total": float(result.timings.get("total_s", wall_s)),
        "wall": float(wall_s),
    }


def case_definitions() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for value in (32, 64, 128):
        cases.append({"name": f"matrix_tg{value}", "kernel": "matrix", "threads": value})
    for value in (64, 128, 256, 448):
        cases.append({"name": f"duffy_tg{value}", "kernel": "duffy", "threads": value})
    for value in (32, 64, 128):
        cases.append({"name": f"field_tg{value}", "kernel": "field", "threads": value})
    return cases


def main() -> int:
    args = parse_args()
    mesh_path = args.mesh.expanduser().resolve()
    corrected_case_dir = args.corrected_case_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_npz = corrected_case_dir / "result.npz"
    baseline_metadata_path = corrected_case_dir / "metadata.json"
    if not baseline_npz.exists():
        raise FileNotFoundError(f"corrected baseline result missing: {baseline_npz}")
    baseline = load_npz(baseline_npz)
    baseline_metadata = (
        json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
        if baseline_metadata_path.exists()
        else {}
    )

    loaded = load_mesh(mesh_path, scale=0.001, repair_normals=True)
    frame = axial_throat_frame(loaded)
    frequencies = np.asarray(baseline["frequencies_hz"], dtype=np.float64)

    runs: list[dict[str, Any]] = []
    for case in case_definitions():
        case_dir = output_dir / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[asro2-per-kernel-threadgroups] "
            f"running {case['name']} with global fallback {args.global_threadgroup}",
            flush=True,
        )
        config = config_for_case(
            frame=frame,
            angle_count=args.angle_count,
            global_threads=args.global_threadgroup,
            kernel=case["kernel"],
            threads=case["threads"],
        )
        started = time.perf_counter()
        result = solve_frequencies(loaded, frequencies, config)
        wall_s = time.perf_counter() - started
        result_npz = case_dir / "result.npz"
        save_result_npz(result_npz, result)
        data = load_npz(result_npz)
        run = {
            "name": case["name"],
            "kernel": case["kernel"],
            "threads_per_group": case["threads"],
            "global_fallback_threads_per_group": args.global_threadgroup,
            "timings_s": timings(result, wall_s),
            "pressure_relative_l2": relative_l2(
                data["pressure_complex"],
                baseline["pressure_complex"],
            ),
            "directivity_delta_db": directivity_delta_db(baseline, data),
            "impedance_delta": impedance_delta(baseline, data),
            "backend": sorted(
                {
                    str(entry.get("backend"))
                    for entry in result.solver_log
                    if entry.get("backend") is not None
                }
            ),
            "result_npz": str(result_npz),
        }
        runs.append(run)
        (case_dir / "metadata.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    best_metric_by_kernel = {
        "matrix": "assembly",
        "duffy": "assembly",
        "field": "field",
    }
    best_by_kernel: dict[str, Any] = {}
    for kernel in ("matrix", "duffy", "field"):
        kernel_runs = [run for run in runs if run["kernel"] == kernel]
        metric = best_metric_by_kernel[kernel]
        best_by_kernel[kernel] = min(
            kernel_runs,
            key=lambda run: (
                float(run["timings_s"][metric]),
                float(run["timings_s"]["wall"]),
            ),
        )

    summary = {
        "schema": "hornlab.asro2_quarter_per_kernel_threadgroup_tuning.v1",
        "created_local_date": date.today().isoformat(),
        "scope": "corrected ASRO2 WG native quarter mesh; corrected assembly only",
        "mesh": inspect_mesh(mesh_path),
        "baseline": {
            "case_dir": str(corrected_case_dir),
            "result_npz": str(baseline_npz),
            "metadata": baseline_metadata,
        },
        "frequency_grid_hz": [float(v) for v in frequencies.tolist()],
        "angle_count": int(len(baseline["angles_deg"])),
        "planes": list(baseline["planes"]),
        "global_fallback_threads_per_group": args.global_threadgroup,
        "sweeps": {
            "matrix": [32, 64, 128],
            "duffy": [64, 128, 256, 448],
            "field": [32, 64, 128],
            "rhs": "global fallback only",
        },
        "runs": runs,
        "best_by_kernel": {
            kernel: {
                "name": run["name"],
                "selection_metric": best_metric_by_kernel[kernel],
                "threads_per_group": run["threads_per_group"],
                "timings_s": run["timings_s"],
            }
            for kernel, run in best_by_kernel.items()
        },
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["best_by_kernel"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

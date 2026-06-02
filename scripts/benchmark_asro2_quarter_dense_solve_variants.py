#!/usr/bin/env python3
"""Benchmark native Accelerate dense-solve variants on corrected ASRO2 quarter."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

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
from benchmark_asro2_quarter_per_kernel_threadgroups import (  # noqa: E402
    directivity_delta_db,
    impedance_delta,
    timings,
)
from hornlab_solver import ObservationConfig, SolveConfig, load_mesh, solve_frequencies  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    WORKSPACE_ROOT
    / "runs/canonical-validation/260602-asro2-quarter-dense-solve-variants"
)
DENSE_SOLVE_ENV = "HORNLAB_SOLVER_METAL_NATIVE_DENSE_SOLVE_IMPL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--corrected-case-dir", type=Path, default=DEFAULT_CORRECTED_CASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--angle-count", type=int, default=37)
    parser.add_argument("--threadgroup", type=int, default=64)
    return parser.parse_args()


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


def config_for_case(*, frame: Any, angle_count: int, threadgroup: int) -> SolveConfig:
    return SolveConfig(
        assembly_backend="metal",
        experimental_metal_backend=True,
        metal_backend_fallback="error",
        native_symmetry_plane="yz+xz",
        metal_native_assembly_mode="corrected",
        metal_native_threads_per_group=threadgroup,
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


def main() -> int:
    args = parse_args()
    mesh_path = args.mesh.expanduser().resolve()
    corrected_case_dir = args.corrected_case_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_npz = corrected_case_dir / "result.npz"
    if not baseline_npz.exists():
        raise FileNotFoundError(f"corrected baseline result missing: {baseline_npz}")
    baseline = load_npz(baseline_npz)
    frequencies = np.asarray(baseline["frequencies_hz"], dtype=np.float64)

    loaded = load_mesh(mesh_path, scale=0.001, repair_normals=True)
    frame = axial_throat_frame(loaded)
    config = config_for_case(
        frame=frame,
        angle_count=args.angle_count,
        threadgroup=args.threadgroup,
    )

    runs: list[dict[str, Any]] = []
    for name, env_value in (("cgesv", None), ("cgetrf_cgetrs", "cgetrf_cgetrs")):
        case_dir = output_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[asro2-dense-solve-variants] running {name}", flush=True)
        with temporary_env({DENSE_SOLVE_ENV: env_value}):
            started = time.perf_counter()
            result = solve_frequencies(loaded, frequencies, config)
            wall_s = time.perf_counter() - started
        result_npz = case_dir / "result.npz"
        save_result_npz(result_npz, result)
        data = load_npz(result_npz)
        run = {
            "name": name,
            "dense_solve_env": env_value or "default",
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

    summary = {
        "schema": "hornlab.asro2_quarter_dense_solve_variants.v1",
        "created_local_date": date.today().isoformat(),
        "scope": "corrected ASRO2 WG native quarter mesh; corrected assembly only",
        "mesh": inspect_mesh(mesh_path),
        "baseline_result_npz": str(baseline_npz),
        "frequency_grid_hz": [float(v) for v in frequencies.tolist()],
        "threadgroup": args.threadgroup,
        "runs": runs,
        "timing_delta_cgetrf_cgetrs_vs_cgesv_s": {
            key: float(runs[1]["timings_s"][key]) - float(runs[0]["timings_s"][key])
            for key in ("assembly", "dense_solve", "field", "solve", "wall")
        },
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["timing_delta_cgetrf_cgetrs_vs_cgesv_s"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

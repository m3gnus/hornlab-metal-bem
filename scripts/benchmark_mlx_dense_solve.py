#!/usr/bin/env python3
"""Benchmark SciPy and MLX dense solves on a native Metal ASRO68 system."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import scipy.linalg

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from hornlab_solver import SolveConfig, load_mesh  # noqa: E402
from hornlab_solver._constants import SPEED_OF_SOUND  # noqa: E402
from hornlab_solver.bie import _setup_function_spaces  # noqa: E402
from hornlab_solver.metal.geometry import build_metal_geometry_buffers  # noqa: E402
from hornlab_solver.metal.native import (  # noqa: E402
    MetalNativeStandardSession,
    discover_native_runtime,
)
from hornlab_solver.sweep import _build_driver_neumann_coeffs  # noqa: E402

ASRO68_MESH_CANDIDATES = (
    Path(
        "/Users/magnus/IM Dropbox/Magnus Andersen/DOCS/code/misc/"
        "ATH results 0 degree norm/250917asro68/ABEC_FreeStanding/"
        "250917asro68.msh"
    ),
    WORKSPACE_ROOT
    / "MEH-Lab/projects/bigmeh/ATH results 0 degree norm/250917asro68/"
    / "ABEC_FreeStanding/250917asro68.msh",
)
DEFAULT_ASRO68_MESH = next(
    (path for path in ASRO68_MESH_CANDIDATES if path.exists()),
    ASRO68_MESH_CANDIDATES[0],
)


def read_complex(real_path: Path, imag_path: Path, shape: tuple[int, ...]) -> np.ndarray:
    real = np.fromfile(real_path, dtype="<f4").reshape(shape)
    imag = np.fromfile(imag_path, dtype="<f4").reshape(shape)
    return real.astype(np.complex64) + 1j * imag.astype(np.complex64)


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    ref = np.linalg.norm(reference)
    diff = np.linalg.norm(candidate - reference)
    return float(diff if ref == 0.0 else diff / ref)


def relative_residual(matrix: np.ndarray, solution: np.ndarray, rhs: np.ndarray) -> float:
    numerator = np.linalg.norm(matrix @ solution - rhs)
    denominator = np.linalg.norm(matrix) * np.linalg.norm(solution) + np.linalg.norm(rhs)
    return float(numerator if denominator == 0.0 else numerator / denominator)


def scipy_solve(matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, float]:
    started = perf_counter()
    solution = scipy.linalg.solve(matrix, rhs, assume_a="gen", check_finite=False)
    return solution.astype(np.complex64, copy=False), perf_counter() - started


def mlx_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    repeats: int,
    *,
    device: Any | None = None,
) -> dict[str, Any]:
    import mlx.core as mx

    transfer_started = perf_counter()
    mx_matrix = mx.array(matrix)
    mx_rhs = mx.array(rhs)
    mx.eval(mx_matrix, mx_rhs)
    transfer_s = perf_counter() - transfer_started

    times: list[float] = []
    solution_np: np.ndarray | None = None
    for _ in range(repeats):
        started = perf_counter()
        solution = mx.linalg.solve(mx_matrix, mx_rhs, stream=device)
        mx.eval(solution)
        solve_s = perf_counter() - started
        times.append(solve_s)
        solution_np = np.array(solution).astype(np.complex64, copy=False)

    if solution_np is None:
        raise RuntimeError("MLX solve produced no result")

    return {
        "available": True,
        "version": getattr(mx, "__version__", "unknown"),
        "transfer_s": transfer_s,
        "solve_times_s": times,
        "solve_best_s": min(times),
        "solve_mean_s": float(np.mean(times)),
        "solution": solution_np,
    }


def complex_system_as_real_block(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    real = np.asarray(matrix.real, dtype=np.float32)
    imag = np.asarray(matrix.imag, dtype=np.float32)
    top = np.concatenate((real, -imag), axis=1)
    bottom = np.concatenate((imag, real), axis=1)
    block_matrix = np.concatenate((top, bottom), axis=0)
    block_rhs = np.concatenate(
        (
            np.asarray(rhs.real, dtype=np.float32),
            np.asarray(rhs.imag, dtype=np.float32),
        ),
        axis=0,
    )
    return block_matrix, block_rhs


def mlx_real_block_cpu_solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    import mlx.core as mx

    build_started = perf_counter()
    block_matrix, block_rhs = complex_system_as_real_block(matrix, rhs)
    build_s = perf_counter() - build_started

    transfer_started = perf_counter()
    mx_matrix = mx.array(block_matrix)
    mx_rhs = mx.array(block_rhs)
    mx.eval(mx_matrix, mx_rhs)
    transfer_s = perf_counter() - transfer_started

    n = rhs.shape[0]
    times: list[float] = []
    solution_np: np.ndarray | None = None
    for _ in range(repeats):
        started = perf_counter()
        block_solution = mx.linalg.solve(mx_matrix, mx_rhs, stream=mx.cpu)
        mx.eval(block_solution)
        solve_s = perf_counter() - started
        times.append(solve_s)
        block_np = np.array(block_solution).astype(np.float32, copy=False)
        solution_np = (
            block_np[:n].astype(np.complex64)
            + 1j * block_np[n:].astype(np.complex64)
        )

    if solution_np is None:
        raise RuntimeError("MLX real-block solve produced no result")

    return {
        "available": True,
        "version": getattr(mx, "__version__", "unknown"),
        "block_matrix_shape": list(block_matrix.shape),
        "build_block_s": build_s,
        "transfer_s": transfer_s,
        "solve_times_s": times,
        "solve_best_s": min(times),
        "solve_mean_s": float(np.mean(times)),
        "solution": solution_np,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, default=DEFAULT_ASRO68_MESH)
    parser.add_argument("--frequency-hz", type=float, default=500.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT
        / "runs/canonical-validation/260602-mlx-dense-solve-benchmark",
    )
    parser.add_argument("--mlx-repeats", type=int, default=3)
    parser.add_argument("--threadgroup-size", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = args.mesh.expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    runtime = discover_native_runtime(run_smoke_test=True)
    if not runtime.available:
        raise RuntimeError("; ".join(runtime.unavailable_reasons))

    previous_mode = os.environ.get("HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE")
    previous_tg = os.environ.get("HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP")
    os.environ["HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE"] = "corrected"
    if args.threadgroup_size is None:
        os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP", None)
    else:
        os.environ["HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP"] = str(
            args.threadgroup_size
        )

    try:
        loaded = load_mesh(mesh_path, scale=0.001)
        p1_space, dp0_space = _setup_function_spaces(loaded.grid)
        buffers = build_metal_geometry_buffers(
            loaded.grid,
            loaded.physical_tags,
            p1_space,
            dp0_space,
        )
        frequency_hz = float(args.frequency_hz)
        omega = 2.0 * math.pi * frequency_hz
        neumann = _build_driver_neumann_coeffs(
            dp0_space,
            loaded.physical_tags,
            omega,
            SolveConfig(),
            np.complex64,
        )

        with MetalNativeStandardSession.create_session(
            geometry_buffers=buffers,
            work_dir=output_dir / f"asro68-{frequency_hz:g}hz",
            session_id=f"asro68-{frequency_hz:g}hz-mlx",
            keep_artifacts=True,
        ) as session:
            assembly_started = perf_counter()
            system = session.assemble_standard_neumann(
                frequency_hz,
                omega / SPEED_OF_SOUND,
                neumann,
                operation_id="assembly",
            )
            assembly_wall_s = perf_counter() - assembly_started
            assembly_manifest = json.loads(
                (
                    session.info.work_dir / "assembly" / "assembly-result.json"
                ).read_text(encoding="utf-8")
            )

        matrix = read_complex(
            Path(system.matrix_real_f32),
            Path(system.matrix_imag_f32),
            tuple(system.matrix_shape),
        )
        rhs = read_complex(
            Path(system.rhs_real_f32),
            Path(system.rhs_imag_f32),
            tuple(system.rhs_shape),
        )

        scipy_solution, scipy_s = scipy_solve(matrix, rhs)
        try:
            mlx_default_result = mlx_solve(matrix, rhs, args.mlx_repeats)
            mlx_solution = mlx_default_result.pop("solution")
            mlx_default_result["relative_l2_vs_scipy"] = relative_l2(
                mlx_solution,
                scipy_solution,
            )
            mlx_default_result["relative_residual"] = relative_residual(
                matrix,
                mlx_solution,
                rhs,
            )
        except Exception as exc:
            mlx_default_result = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        try:
            import mlx.core as mx

            mlx_cpu_result = mlx_solve(
                matrix,
                rhs,
                args.mlx_repeats,
                device=mx.cpu,
            )
            mlx_solution = mlx_cpu_result.pop("solution")
            mlx_cpu_result["relative_l2_vs_scipy"] = relative_l2(
                mlx_solution,
                scipy_solution,
            )
            mlx_cpu_result["relative_residual"] = relative_residual(
                matrix,
                mlx_solution,
                rhs,
            )
        except Exception as exc:
            mlx_cpu_result = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        try:
            mlx_real_block_cpu_result = mlx_real_block_cpu_solve(
                matrix,
                rhs,
                args.mlx_repeats,
            )
            mlx_solution = mlx_real_block_cpu_result.pop("solution")
            mlx_real_block_cpu_result["relative_l2_vs_scipy"] = relative_l2(
                mlx_solution,
                scipy_solution,
            )
            mlx_real_block_cpu_result["relative_residual"] = relative_residual(
                matrix,
                mlx_solution,
                rhs,
            )
        except Exception as exc:
            mlx_real_block_cpu_result = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        summary = {
            "mesh": str(mesh_path),
            "frequency_hz": frequency_hz,
            "matrix_shape": list(matrix.shape),
            "rhs_shape": list(rhs.shape),
            "matrix_dtype": str(matrix.dtype),
            "rhs_dtype": str(rhs.dtype),
            "assembly_wall_s": assembly_wall_s,
            "assembly_manifest": assembly_manifest,
            "scipy": {
                "solve_s": scipy_s,
                "relative_residual": relative_residual(matrix, scipy_solution, rhs),
            },
            "mlx_default": mlx_default_result,
            "mlx_cpu": mlx_cpu_result,
            "mlx_real_block_cpu": mlx_real_block_cpu_result,
            "artifacts": {
                "matrix_real_f32": str(system.matrix_real_f32),
                "matrix_imag_f32": str(system.matrix_imag_f32),
                "rhs_real_f32": str(system.rhs_real_f32),
                "rhs_imag_f32": str(system.rhs_imag_f32),
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        if previous_mode is None:
            os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE", None)
        else:
            os.environ["HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE"] = previous_mode
        if previous_tg is None:
            os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP", None)
        else:
            os.environ["HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP"] = previous_tg

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

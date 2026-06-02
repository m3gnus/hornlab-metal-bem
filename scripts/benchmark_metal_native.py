#!/usr/bin/env python3
"""Benchmark the package-owned native Metal assembly helper.

Generated artifacts are written under ``runs/canonical-validation``. The
script benchmarks the promoted native dense assembly slice and optional CPU
direct solves from the assembled dense systems. Corrected native assembly uses
Metal Duffy block evaluation by default, with the legacy CPU Duffy path
available through ``HORNLAB_SOLVER_METAL_NATIVE_DUFFY_MODE=cpu``. GPU solve
acceleration and production routing remain outside this runner.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from hornlab_solver import SolveConfig, load_mesh, solve_frequencies  # noqa: E402
from hornlab_solver._constants import SPEED_OF_SOUND  # noqa: E402
from hornlab_solver.bie import (  # noqa: E402
    _build_neumann_data,
    _evaluate_far_field,
    _operator_kwargs,
    _setup_function_spaces,
)
from hornlab_solver.observation import infer_frame  # noqa: E402
from hornlab_solver.metal.geometry import build_metal_geometry_buffers  # noqa: E402
from hornlab_solver.metal.native import (  # noqa: E402
    MetalNativeStandardSession,
    discover_native_runtime,
)

ASRO68_MESH_CANDIDATES = (
    Path(
        "/Users/magnus/IM Dropbox/Magnus Andersen/DOCS/code/misc/"
        "ATH results 0 degree norm/250917asro68/ABEC_FreeStanding/"
        "250917asro68.msh"
    ),
    WORKSPACE_ROOT / (
        "MEH-Lab/projects/bigmeh/ATH results 0 degree norm/250917asro68/"
        "ABEC_FreeStanding/250917asro68.msh"
    ),
)
DEFAULT_ASRO68_MESH = next(
    (path for path in ASRO68_MESH_CANDIDATES if path.exists()),
    ASRO68_MESH_CANDIDATES[0],
)


def tiny_geometry_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([1, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def tiny_yz_half_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0], [1], [2]], dtype=np.int64),
        number_of_elements=1,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2]], dtype=np.int64),
        global_dof_count=3,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=1),
    )


def tiny_yz_full_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 3], [1, 5], [2, 4]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [3, 5, 4]], dtype=np.int64),
        global_dof_count=6,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def read_complex(real_path: Path, imag_path: Path, shape: tuple[int, ...]) -> np.ndarray:
    real = np.fromfile(real_path, dtype="<f4").reshape(shape)
    imag = np.fromfile(imag_path, dtype="<f4").reshape(shape)
    return real.astype(np.complex64) + 1j * imag.astype(np.complex64)


def write_f32(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(values, dtype="<f4").tofile(path)
    return str(path)


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    diff = np.linalg.norm(candidate - reference)
    ref = np.linalg.norm(reference)
    return float(diff if ref == 0.0 else diff / ref)


def horizontal_arc_points(
    *,
    npoints: int = 181,
    radius_m: float = 1.0,
    center_z_m: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, npoints, dtype=np.float32)
    points = np.vstack(
        [
            radius_m * np.sin(angles),
            np.zeros_like(angles),
            center_z_m + radius_m * np.cos(angles),
        ]
    ).astype(np.float32)
    return points, angles * np.float32(180.0 / math.pi)


def hemisphere_points(
    *,
    frame: Any,
    n_polar: int = 91,
    n_azimuth: int = 181,
    radius_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    polar_deg = np.linspace(0.0, 90.0, n_polar, dtype=np.float32)
    azimuth_deg = np.linspace(0.0, 360.0, n_azimuth, endpoint=False, dtype=np.float32)
    polar = np.deg2rad(polar_deg.astype(np.float64))
    azimuth = np.deg2rad(azimuth_deg.astype(np.float64))
    points = np.empty((3, n_polar * n_azimuth), dtype=np.float32)
    idx = 0
    axis = np.asarray(frame.axis, dtype=np.float64)
    u = np.asarray(frame.u, dtype=np.float64)
    v = np.asarray(frame.v, dtype=np.float64)
    origin = np.asarray(frame.origin, dtype=np.float64)
    for theta in polar:
        forward = math.cos(float(theta)) * axis
        transverse_scale = math.sin(float(theta))
        for phi in azimuth:
            direction = (
                forward
                + transverse_scale
                * (math.cos(float(phi)) * u + math.sin(float(phi)) * v)
            )
            points[:, idx] = (origin + radius_m * direction).astype(np.float32)
            idx += 1
    return points, polar_deg, azimuth_deg


def field_grid_points(
    *,
    loaded: Any,
    config: SolveConfig,
    grid_name: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    if grid_name == "horizontal":
        points, angles_deg = horizontal_arc_points(npoints=181)
        return (
            points,
            {
                "name": "horizontal_arc",
                "n_points": int(points.shape[1]),
                "radius_m": 1.0,
                "center_z_m": 0.08,
            },
            {"horizontal_angles_deg_f32": angles_deg},
        )
    if grid_name == "hemisphere":
        frame = infer_frame(
            loaded.grid,
            loaded.physical_tags,
            source_tag=min(config.velocity_sources.keys(), default=2),
            origin_at=config.observation.origin,
        )
        points, polar_deg, azimuth_deg = hemisphere_points(frame=frame)
        return (
            points,
            {
                "name": "front_hemisphere",
                "n_points": int(points.shape[1]),
                "n_polar": int(polar_deg.size),
                "n_azimuth": int(azimuth_deg.size),
                "radius_m": 1.0,
                "origin": config.observation.origin,
            },
            {
                "hemisphere_polar_deg_f32": polar_deg,
                "hemisphere_azimuth_deg_f32": azimuth_deg,
            },
        )
    raise ValueError(f"unsupported field grid: {grid_name}")


def directivity_db_1d(field: np.ndarray) -> np.ndarray:
    magnitude = np.abs(field).astype(np.float32)
    reference = float(np.max(magnitude))
    if reference <= 0.0:
        reference = np.finfo(np.float32).eps
    return 20.0 * np.log10(np.maximum(magnitude / reference, 1.0e-12)).astype(
        np.float32
    )


def db_error(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = np.asarray(candidate - reference, dtype=np.float32)
    return {
        "rms_db": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_db": float(np.max(np.abs(diff))),
    }


def parse_threadgroup_sweep(raw: str | None) -> list[int]:
    if raw is None or not raw.strip():
        return []
    sizes: list[int] = []
    for piece in raw.split(","):
        text = piece.strip()
        if not text:
            continue
        value = int(text)
        if value <= 0:
            raise ValueError("--threadgroup-sweep values must be positive integers")
        sizes.append(value)
    return sizes


def run_native_case(
    *,
    case_name: str,
    buffers: Any,
    frequency_hz: float,
    neumann: np.ndarray,
    output_dir: Path,
    mode: str,
    threadgroup_size: int | None = None,
    symmetry_plane: str | None = None,
) -> dict[str, Any]:
    previous_mode = os.environ.get("HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE")
    previous_threadgroup = os.environ.get(
        "HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP"
    )
    os.environ["HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE"] = mode
    if threadgroup_size is None:
        os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP", None)
    else:
        os.environ["HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP"] = str(
            threadgroup_size
        )
    try:
        with MetalNativeStandardSession.create_session(
            geometry_buffers=buffers,
            work_dir=output_dir
            / f"{case_name}-{mode}"
            / (
                f"tg{threadgroup_size}"
                if threadgroup_size is not None
                else "tg-default"
            ),
            session_id=f"{case_name}-{mode}",
            keep_artifacts=True,
            symmetry_plane=symmetry_plane,
        ) as session:
            started = perf_counter()
            result = session.assemble_standard_neumann(
                frequency_hz,
                2.0 * math.pi * frequency_hz / SPEED_OF_SOUND,
                neumann,
                operation_id="assembly",
            )
            wall_s = perf_counter() - started
            manifest = json.loads(
                (
                    session.info.work_dir / "assembly" / "assembly-result.json"
                ).read_text(encoding="utf-8")
            )
    finally:
        if previous_mode is None:
            os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE", None)
        else:
            os.environ["HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE"] = previous_mode
        if previous_threadgroup is None:
            os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP", None)
        else:
            os.environ[
                "HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP"
            ] = previous_threadgroup

    matrix = read_complex(
        result.matrix_real_f32,
        result.matrix_imag_f32,
        result.matrix_shape,
    )
    rhs = read_complex(result.rhs_real_f32, result.rhs_imag_f32, result.rhs_shape)
    return {
        "mode": mode,
        "requested_threadgroup_size": threadgroup_size,
        "symmetry_plane": manifest.get("symmetry_plane"),
        "implementation": manifest.get("implementation"),
        "wall_seconds": wall_s,
        "assembly_seconds": manifest.get("assembly_seconds"),
        "regular_assembly_seconds": manifest.get("regular_assembly_seconds"),
        "metal_dispatch": manifest.get("metal_dispatch"),
        "matrix_shape": list(result.matrix_shape),
        "rhs_shape": list(result.rhs_shape),
        "matrix_real_f32": str(result.matrix_real_f32),
        "matrix_imag_f32": str(result.matrix_imag_f32),
        "rhs_real_f32": str(result.rhs_real_f32),
        "rhs_imag_f32": str(result.rhs_imag_f32),
        "matrix_norm": float(np.linalg.norm(matrix)),
        "rhs_norm": float(np.linalg.norm(rhs)),
        "reference_parity": manifest.get("reference_parity"),
        "duffy_corrections": manifest.get("duffy_corrections"),
    }


def run_native_field_case(
    *,
    case_name: str,
    field_name: str,
    buffers: Any,
    frequency_hz: float,
    neumann: np.ndarray,
    pressure: np.ndarray,
    observation_points_3xn: np.ndarray,
    output_dir: Path,
    mode: str = "optimized",
    threadgroup_size: int | None = None,
    symmetry_plane: str | None = None,
) -> dict[str, Any]:
    previous_mode = os.environ.get("HORNLAB_SOLVER_METAL_NATIVE_FIELD_MODE")
    previous_threadgroup = os.environ.get(
        "HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP"
    )
    os.environ["HORNLAB_SOLVER_METAL_NATIVE_FIELD_MODE"] = mode
    if threadgroup_size is None:
        os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP", None)
    else:
        os.environ["HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP"] = str(
            threadgroup_size
        )
    try:
        with MetalNativeStandardSession.create_session(
            geometry_buffers=buffers,
            work_dir=output_dir
            / f"{case_name}-field"
            / (
                f"tg{threadgroup_size}"
                if threadgroup_size is not None
                else "tg-default"
            ),
            session_id=f"{case_name}-field",
            keep_artifacts=True,
            symmetry_plane=symmetry_plane,
        ) as session:
            started = perf_counter()
            result = session.evaluate_standard_exterior(
                frequency_hz,
                2.0 * math.pi * frequency_hz / SPEED_OF_SOUND,
                pressure,
                neumann,
                observation_points_3xn,
                batch_id=field_name,
                operation_id=f"field-{field_name}",
            )
            wall_s = perf_counter() - started
            manifest = json.loads(
                (
                    session.info.work_dir
                    / f"field-{field_name}"
                    / "field-result.json"
                ).read_text(encoding="utf-8")
            )
    finally:
        if previous_mode is None:
            os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_FIELD_MODE", None)
        else:
            os.environ["HORNLAB_SOLVER_METAL_NATIVE_FIELD_MODE"] = previous_mode
        if previous_threadgroup is None:
            os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP", None)
        else:
            os.environ[
                "HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP"
            ] = previous_threadgroup
    field = read_complex(
        result.pressure_real_f32,
        result.pressure_imag_f32,
        result.shape,
    )
    return {
        "implementation": manifest.get("implementation"),
        "field_mode": manifest.get("field_mode"),
        "symmetry_plane": manifest.get("symmetry_plane"),
        "requested_threadgroup_size": threadgroup_size,
        "wall_seconds": wall_s,
        "field_seconds": manifest.get("field_seconds"),
        "metal_dispatch": manifest.get("metal_dispatch"),
        "reference_parity": manifest.get("reference_parity"),
        "shape": list(result.shape),
        "pressure_real_f32": str(result.pressure_real_f32),
        "pressure_imag_f32": str(result.pressure_imag_f32),
        "field_norm": float(np.linalg.norm(field)),
        "field": field,
    }


def assemble_bempp_opencl(
    *,
    p1_space: Any,
    dp0_space: Any,
    frequency_hz: float,
    config: SolveConfig,
    neumann_coefficients: np.ndarray,
) -> dict[str, Any]:
    try:
        import bempp_cl.api as bempp_api
    except Exception as exc:  # pragma: no cover - depends on local runtime.
        return {"available": False, "blocker": f"bempp-cl unavailable: {exc}"}

    try:
        op_kwargs = _operator_kwargs(
            "opencl",
            "single",
            opencl_device=config.opencl_device,
            quadrature_order=4,
        )
        k = 2.0 * math.pi * frequency_hz / SPEED_OF_SOUND
        started = perf_counter()
        identity = bempp_api.operators.boundary.sparse.identity(
            p1_space,
            p1_space,
            p1_space,
        )
        dlp = bempp_api.operators.boundary.helmholtz.double_layer(
            p1_space,
            p1_space,
            p1_space,
            k,
            **op_kwargs,
        )
        slp = bempp_api.operators.boundary.helmholtz.single_layer(
            dp0_space,
            p1_space,
            p1_space,
            k,
            **op_kwargs,
        )
        matrix = bempp_api.as_matrix((dlp - 0.5 * identity).weak_form())
        slp_matrix = bempp_api.as_matrix(slp.weak_form())
        rhs = slp_matrix @ neumann_coefficients
        wall_s = perf_counter() - started
    except Exception as exc:  # pragma: no cover - depends on local runtime.
        return {"available": False, "blocker": f"OpenCL/Bempp assembly failed: {exc}"}

    return {
        "available": True,
        "wall_seconds": wall_s,
        "matrix": np.asarray(matrix, dtype=np.complex64),
        "rhs": np.asarray(rhs, dtype=np.complex64),
    }


def evaluate_bempp_field_opencl(
    *,
    p1_space: Any,
    dp0_space: Any,
    pressure_coefficients: np.ndarray,
    neumann_fun: Any,
    observation_points_3xn: np.ndarray,
    frequency_hz: float,
    config: SolveConfig,
) -> dict[str, Any]:
    try:
        import bempp_cl.api as bempp_api
    except Exception as exc:  # pragma: no cover - depends on local runtime.
        return {"available": False, "blocker": f"bempp-cl unavailable: {exc}"}

    try:
        op_kwargs = _operator_kwargs(
            "opencl",
            "single",
            opencl_device=config.opencl_device,
            quadrature_order=4,
        )
        k = 2.0 * math.pi * frequency_hz / SPEED_OF_SOUND
        surface = bempp_api.GridFunction(
            p1_space,
            coefficients=np.asarray(pressure_coefficients, dtype=np.complex64),
        )
        started = perf_counter()
        field = _evaluate_far_field(
            p1_space,
            dp0_space,
            surface,
            neumann_fun,
            k,
            observation_points_3xn.T.astype(np.float64),
            op_kwargs,
        )
        wall_s = perf_counter() - started
    except Exception as exc:  # pragma: no cover - depends on local runtime.
        return {"available": False, "blocker": f"OpenCL/Bempp field failed: {exc}"}

    return {
        "available": True,
        "wall_seconds": wall_s,
        "field": np.asarray(field, dtype=np.complex64),
        "field_norm": float(np.linalg.norm(field)),
    }


def solve_dense_direct(matrix: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    started = perf_counter()
    implementation = "numpy.linalg.solve"
    try:
        import scipy.linalg

        pressure = scipy.linalg.solve(
            np.asarray(matrix, dtype=np.complex64),
            np.asarray(rhs, dtype=np.complex64),
            assume_a="gen",
            check_finite=False,
        )
        implementation = "scipy.linalg.solve"
    except Exception as exc:
        if "scipy" not in str(type(exc)).lower() and not isinstance(exc, ImportError):
            return {"available": False, "blocker": f"dense direct solve failed: {exc}"}
        try:
            pressure = np.linalg.solve(
                np.asarray(matrix, dtype=np.complex64),
                np.asarray(rhs, dtype=np.complex64),
            )
        except Exception as solve_exc:
            return {
                "available": False,
                "blocker": f"dense direct solve failed: {solve_exc}",
            }
    wall_s = perf_counter() - started
    residual = matrix @ pressure - rhs
    rhs_norm = np.linalg.norm(rhs)
    return {
        "available": True,
        "implementation": implementation,
        "wall_seconds": wall_s,
        "pressure": np.asarray(pressure, dtype=np.complex64),
        "pressure_norm": float(np.linalg.norm(pressure)),
        "relative_residual_l2": float(
            np.linalg.norm(residual) if rhs_norm == 0.0 else np.linalg.norm(residual) / rhs_norm
        ),
    }


def solve_report(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("available"):
        return result
    return {
        "available": True,
        "implementation": result["implementation"],
        "wall_seconds": result["wall_seconds"],
        "pressure_norm": result["pressure_norm"],
        "relative_residual_l2": result["relative_residual_l2"],
    }


def run_synthetic_yz_symmetry(output_dir: Path) -> dict[str, Any]:
    frequency_hz = 100.0
    full_buffers = tiny_yz_full_buffers()
    half_buffers = tiny_yz_half_buffers()
    points = np.array(
        [[0.25, 0.2, 1.0], [-0.25, 0.2, 1.0]],
        dtype=np.float32,
    )
    cases = {
        "full_domain": {
            "buffers": full_buffers,
            "neumann": np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64),
            "symmetry_plane": None,
        },
        "reduced_yz": {
            "buffers": half_buffers,
            "neumann": np.array([1.0 + 0.0j], dtype=np.complex64),
            "symmetry_plane": "yz",
        },
    }
    reports: dict[str, Any] = {}
    pressures: dict[str, np.ndarray] = {}
    fields: dict[str, np.ndarray] = {}
    for name, case in cases.items():
        started = perf_counter()
        assembly = run_native_case(
            case_name=f"synthetic-yz-{name}",
            buffers=case["buffers"],
            frequency_hz=frequency_hz,
            neumann=case["neumann"],
            output_dir=output_dir,
            mode="optimized",
            symmetry_plane=case["symmetry_plane"],
        )
        matrix = read_complex(
            Path(assembly["matrix_real_f32"]),
            Path(assembly["matrix_imag_f32"]),
            tuple(assembly["matrix_shape"]),
        )
        rhs = read_complex(
            Path(assembly["rhs_real_f32"]),
            Path(assembly["rhs_imag_f32"]),
            tuple(assembly["rhs_shape"]),
        )
        if name == "full_domain":
            real_dofs = np.array([0, 1, 2], dtype=np.int64)
            mirror_dofs = np.array([3, 4, 5], dtype=np.int64)
            solve_matrix = (
                matrix[np.ix_(real_dofs, real_dofs)]
                + matrix[np.ix_(real_dofs, mirror_dofs)]
            )
            solve_rhs = rhs[real_dofs]
            solve = solve_dense_direct(solve_matrix, solve_rhs)
            if solve.get("available"):
                expanded = np.zeros(matrix.shape[0], dtype=np.complex64)
                expanded[real_dofs] = solve["pressure"]
                expanded[mirror_dofs] = solve["pressure"]
                solve["reduced_pressure"] = solve["pressure"]
                solve["pressure"] = expanded
        else:
            solve = solve_dense_direct(matrix, rhs)
        if not solve.get("available"):
            reports[name] = {
                "assembly": assembly,
                "direct_solve": solve_report(solve),
                "available": False,
            }
            continue
        field = run_native_field_case(
            case_name=f"synthetic-yz-{name}",
            field_name="two_points",
            buffers=case["buffers"],
            frequency_hz=frequency_hz,
            neumann=case["neumann"],
            pressure=solve["pressure"],
            observation_points_3xn=points,
            output_dir=output_dir,
            mode="optimized",
            symmetry_plane=case["symmetry_plane"],
        )
        total_s = perf_counter() - started
        pressures[name] = solve["pressure"]
        fields[name] = field["field"]
        reports[name] = {
            "available": True,
            "dof_count": int(matrix.shape[0]),
            "triangle_count": int(case["buffers"].n_triangles),
            "assembly": {
                "wall_seconds": assembly["wall_seconds"],
                "assembly_seconds": assembly["assembly_seconds"],
                "regular_assembly_seconds": assembly["regular_assembly_seconds"],
                "symmetry_plane": assembly["symmetry_plane"],
            },
            "direct_solve": {
                **solve_report(solve),
                "system_dimension": (
                    3 if name == "full_domain" else int(matrix.shape[0])
                ),
                "full_matrix_dimension": int(matrix.shape[0]),
            },
            "field": field_report(field),
            "total_wall_seconds": total_s,
        }

    comparison: dict[str, Any] = {}
    if "full_domain" in pressures and "reduced_yz" in pressures:
        full_positive = pressures["full_domain"][[0, 1, 2]]
        full_mirror = pressures["full_domain"][[3, 4, 5]]
        reduced = pressures["reduced_yz"]
        comparison = {
            "pressure_positive_vs_reduced_relative_l2": relative_l2(
                full_positive,
                reduced,
            ),
            "pressure_mirror_vs_reduced_relative_l2": relative_l2(
                full_mirror,
                reduced,
            ),
            "field_full_vs_reduced_relative_l2": relative_l2(
                fields["full_domain"],
                fields["reduced_yz"],
            ),
        }

    dof_ratio = (
        reports["reduced_yz"]["dof_count"] / reports["full_domain"]["dof_count"]
        if reports.get("full_domain", {}).get("available")
        and reports.get("reduced_yz", {}).get("available")
        else None
    )
    return {
        "available": all(report.get("available") for report in reports.values()),
        "frequency_hz": frequency_hz,
        "observation_points": points.astype(float).tolist(),
        "reports": reports,
        "comparison": comparison,
        "solve_time_scaling_estimate": {
            "basis": "dense direct solve O(N^3), labeled estimate",
            "dof_ratio": dof_ratio,
            "expected_reduced_solve_fraction": (
                dof_ratio ** 3 if dof_ratio is not None else None
            ),
        },
    }


def field_report(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("available", True):
        return result
    report = {key: value for key, value in result.items() if key != "field"}
    report["available"] = True
    return report


def bempp_report(assembly: dict[str, Any]) -> dict[str, Any]:
    if not assembly.get("available"):
        return assembly
    matrix = assembly["matrix"]
    rhs = assembly["rhs"]
    return {
        "available": True,
        "wall_seconds": assembly["wall_seconds"],
        "matrix_shape": list(matrix.shape),
        "rhs_shape": list(rhs.shape),
        "matrix_norm": float(np.linalg.norm(matrix)),
        "rhs_norm": float(np.linalg.norm(rhs)),
    }


def run_tiny(output_dir: Path) -> dict[str, Any]:
    buffers = tiny_geometry_buffers()
    neumann = np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64)
    return {
        "mesh": {
            "name": "tiny-two-triangle",
            "n_vertices": buffers.n_vertices,
            "n_triangles": buffers.n_triangles,
            "p1_dof_count": buffers.p1_dof_count,
            "dp0_dof_count": buffers.dp0_dof_count,
        },
        "native_reference": run_native_case(
            case_name="tiny",
            buffers=buffers,
            frequency_hz=100.0,
            neumann=neumann,
            output_dir=output_dir,
            mode="reference",
        ),
        "native_optimized": run_native_case(
            case_name="tiny",
            buffers=buffers,
            frequency_hz=100.0,
            neumann=neumann,
            output_dir=output_dir,
            mode="parity",
        ),
        "native_corrected": run_native_case(
            case_name="tiny",
            buffers=buffers,
            frequency_hz=100.0,
            neumann=neumann,
            output_dir=output_dir,
            mode="corrected",
        ),
    }


def run_asro68(
    output_dir: Path,
    mesh_path: Path,
    native_mode: str,
    threadgroup_size: int | None,
    field_grid: str,
) -> dict[str, Any]:
    if not mesh_path.exists():
        return {"available": False, "blocker": f"ASRO68 mesh not found: {mesh_path}"}

    loaded = load_mesh(mesh_path, scale=0.001)
    p1_space, dp0_space = _setup_function_spaces(loaded.grid)
    buffers = build_metal_geometry_buffers(
        loaded.grid,
        loaded.physical_tags,
        p1_space,
        dp0_space,
    )
    config = SolveConfig()
    omega = 2.0 * math.pi * 100.0
    neumann_fun = _build_neumann_data(
        dp0_space,
        loaded.physical_tags,
        omega,
        config,
        precision="single",
    )
    native_neumann = np.asarray(neumann_fun.coefficients, dtype=np.complex64)
    native = run_native_case(
        case_name="asro68-100hz",
        buffers=buffers,
        frequency_hz=100.0,
        neumann=native_neumann,
        output_dir=output_dir,
        mode=native_mode,
        threadgroup_size=threadgroup_size,
    )
    bempp_assembly = assemble_bempp_opencl(
        p1_space=p1_space,
        dp0_space=dp0_space,
        frequency_hz=100.0,
        config=config,
        neumann_coefficients=np.asarray(neumann_fun.coefficients, dtype=np.complex64),
    )
    bempp = bempp_report(bempp_assembly)
    if bempp_assembly.get("available"):
        native_matrix = read_complex(
            Path(native["matrix_real_f32"]),
            Path(native["matrix_imag_f32"]),
            tuple(native["matrix_shape"]),
        )
        native_rhs = read_complex(
            Path(native["rhs_real_f32"]),
            Path(native["rhs_imag_f32"]),
            tuple(native["rhs_shape"]),
        )
        relative = {
            "matrix": relative_l2(native_matrix, bempp_assembly["matrix"]),
            "rhs": relative_l2(native_rhs, bempp_assembly["rhs"]),
        }
        native_solve = solve_dense_direct(native_matrix, native_rhs)
        bempp_solve = solve_dense_direct(
            bempp_assembly["matrix"],
            bempp_assembly["rhs"],
        )
        solve_comparison: dict[str, Any] = {
            "native_direct": solve_report(native_solve),
            "bempp_direct": solve_report(bempp_solve),
        }
        if native_solve.get("available") and bempp_solve.get("available"):
            relative["pressure"] = relative_l2(
                native_solve["pressure"],
                bempp_solve["pressure"],
            )
            obs_points, grid_meta, grid_arrays = field_grid_points(
                loaded=loaded,
                config=config,
                grid_name=field_grid,
            )
            field_name = str(grid_meta["name"]).replace("_", "-")
            field_output_dir = output_dir / "asro68-100hz-field" / field_name
            grid_report = dict(grid_meta)
            for key, values in grid_arrays.items():
                grid_report[key] = write_f32(
                    field_output_dir / f"{key}.bin",
                    values,
                )
            native_field = run_native_field_case(
                case_name="asro68-100hz-native-pressure",
                field_name=field_name,
                buffers=buffers,
                frequency_hz=100.0,
                neumann=native_neumann,
                pressure=native_solve["pressure"],
                observation_points_3xn=obs_points,
                output_dir=output_dir,
                threadgroup_size=threadgroup_size,
            )
            helper_bempp_field = run_native_field_case(
                case_name="asro68-100hz-bempp-pressure",
                field_name=field_name,
                buffers=buffers,
                frequency_hz=100.0,
                neumann=native_neumann,
                pressure=bempp_solve["pressure"],
                observation_points_3xn=obs_points,
                output_dir=output_dir,
                threadgroup_size=threadgroup_size,
            )
            bempp_field = evaluate_bempp_field_opencl(
                p1_space=p1_space,
                dp0_space=dp0_space,
                pressure_coefficients=bempp_solve["pressure"],
                neumann_fun=neumann_fun,
                observation_points_3xn=obs_points,
                frequency_hz=100.0,
                config=config,
            )
            field_relative: dict[str, float] = {
                "native_pressure_vs_bempp_pressure_same_helper": relative_l2(
                    native_field["field"],
                    helper_bempp_field["field"],
                ),
            }
            native_db = directivity_db_1d(native_field["field"])
            helper_bempp_db = directivity_db_1d(helper_bempp_field["field"])
            directivity_error = {
                "native_pressure_vs_bempp_pressure_same_helper": db_error(
                    native_db,
                    helper_bempp_db,
                ),
            }
            field_report_payload: dict[str, Any] = {
                "grid": grid_report,
                "native_helper_native_pressure": field_report(native_field),
                "native_helper_bempp_pressure": field_report(helper_bempp_field),
                "relative_l2": field_relative,
                "directivity_db_error": directivity_error,
                "directivity_outputs": {
                    "native_helper_native_pressure_db_f32": write_f32(
                        field_output_dir / "native_pressure_db_f32.bin",
                        native_db,
                    ),
                    "native_helper_bempp_pressure_db_f32": write_f32(
                        field_output_dir / "bempp_pressure_helper_db_f32.bin",
                        helper_bempp_db,
                    ),
                },
            }
            if bempp_field.get("available"):
                bempp_db = directivity_db_1d(bempp_field["field"])
                field_relative["native_helper_native_pressure_vs_bempp_potential"] = (
                    relative_l2(native_field["field"], bempp_field["field"])
                )
                field_relative["native_helper_bempp_pressure_vs_bempp_potential"] = (
                    relative_l2(helper_bempp_field["field"], bempp_field["field"])
                )
                directivity_error["native_helper_native_pressure_vs_bempp_potential"] = (
                    db_error(native_db, bempp_db)
                )
                directivity_error["native_helper_bempp_pressure_vs_bempp_potential"] = (
                    db_error(helper_bempp_db, bempp_db)
                )
                field_report_payload["bempp_opencl_q4_bempp_pressure"] = field_report(
                    bempp_field
                )
                field_report_payload["directivity_outputs"][
                    "bempp_opencl_q4_bempp_pressure_db_f32"
                ] = write_f32(
                    field_output_dir / "bempp_potential_db_f32.bin",
                    bempp_db,
                )
            else:
                field_report_payload["bempp_opencl_q4_bempp_pressure"] = bempp_field
            bempp[f"field_{grid_meta['name']}"] = field_report_payload
        bempp["native_relative_l2"] = relative
        bempp["direct_solve"] = solve_comparison
    return {
        "available": True,
        "mesh": {
            "path": str(mesh_path),
            "n_vertices": int(loaded.info.n_vertices),
            "n_triangles": int(loaded.info.n_triangles),
            "p1_dof_count": int(buffers.p1_dof_count),
            "dp0_dof_count": int(buffers.dp0_dof_count),
        },
        "native_mode": native_mode,
        "requested_threadgroup_size": threadgroup_size,
        "field_grid": field_grid,
        "neumann_input": {
            "native_uses": "bempp_dp0_coefficients",
            "coefficients_norm": float(np.linalg.norm(neumann_fun.coefficients)),
            "projections_norm": float(np.linalg.norm(neumann_fun.projections())),
        },
        "native_assembly": native,
        "bempp_opencl_q4": bempp,
    }


def run_asro68_threadgroup_sweep(
    output_dir: Path,
    mesh_path: Path,
    native_mode: str,
    threadgroup_sizes: list[int],
) -> dict[str, Any]:
    if not threadgroup_sizes:
        return {"available": False, "blocker": "no threadgroup sizes requested"}
    if not mesh_path.exists():
        return {"available": False, "blocker": f"ASRO68 mesh not found: {mesh_path}"}

    loaded = load_mesh(mesh_path, scale=0.001)
    p1_space, dp0_space = _setup_function_spaces(loaded.grid)
    buffers = build_metal_geometry_buffers(
        loaded.grid,
        loaded.physical_tags,
        p1_space,
        dp0_space,
    )
    config = SolveConfig()
    neumann_fun = _build_neumann_data(
        dp0_space,
        loaded.physical_tags,
        2.0 * math.pi * 100.0,
        config,
        precision="single",
    )
    neumann = np.asarray(neumann_fun.coefficients, dtype=np.complex64)
    cases: list[dict[str, Any]] = []
    for size in threadgroup_sizes:
        cases.append(
            run_native_case(
                case_name="asro68-100hz-threadgroup-sweep",
                buffers=buffers,
                frequency_hz=100.0,
                neumann=neumann,
                output_dir=output_dir,
                mode=native_mode,
                threadgroup_size=size,
            )
        )
    return {
        "available": True,
        "mesh": {
            "path": str(mesh_path),
            "n_vertices": int(loaded.info.n_vertices),
            "n_triangles": int(loaded.info.n_triangles),
            "p1_dof_count": int(buffers.p1_dof_count),
            "dp0_dof_count": int(buffers.dp0_dof_count),
        },
        "frequency_hz": 100.0,
        "native_mode": native_mode,
        "cases": cases,
    }


def _metal_sweep_config(*, force_per_frequency_helper: bool) -> SolveConfig:
    config = SolveConfig()
    config.assembly_backend = "metal"
    config.experimental_metal_backend = True
    config.metal_backend_fallback = "error"
    config.observation.planes = ["horizontal", "vertical"]
    config.observation.angle_count = 37
    config.observation.angle_min_deg = 0.0
    config.observation.angle_max_deg = 180.0
    if force_per_frequency_helper:
        config.on_frequency_result = lambda _idx, _freq, _entry: True
    return config


def _sweep_report(result: Any, wall_seconds: float) -> dict[str, Any]:
    return {
        "wall_seconds": wall_seconds,
        "result_shape": list(result.pressure_complex.shape),
        "frequency_count": int(result.frequencies_hz.size),
        "timings": {
            key: float(value)
            for key, value in result.timings.items()
        },
        "solver_log_sums": {
            "assembly_s": float(
                sum(entry.get("assembly_s", 0.0) for entry in result.solver_log)
            ),
            "dense_solve_s": float(
                sum(entry.get("dense_solve_s", 0.0) for entry in result.solver_log)
            ),
            "field_s": float(
                sum(entry.get("field_s", 0.0) for entry in result.solver_log)
            ),
            "timing_s": float(
                sum(entry.get("timing_s", 0.0) for entry in result.solver_log)
            ),
        },
    }


def run_asro68_sweep_compare(
    output_dir: Path,
    mesh_path: Path,
    freq_count: int,
) -> dict[str, Any]:
    if not mesh_path.exists():
        return {"available": False, "blocker": f"ASRO68 mesh not found: {mesh_path}"}
    if freq_count <= 0:
        return {"available": False, "blocker": "freq_count must be positive"}

    loaded = load_mesh(mesh_path, scale=0.001)
    frequencies = np.geomspace(100.0, 20_000.0, freq_count, dtype=np.float64)
    reports: dict[str, Any] = {}
    for name, force_per_frequency in (
        ("per_frequency_helper", True),
        ("resident_batch_helper", False),
    ):
        started = perf_counter()
        result = solve_frequencies(
            loaded,
            frequencies,
            _metal_sweep_config(force_per_frequency_helper=force_per_frequency),
        )
        reports[name] = _sweep_report(result, perf_counter() - started)

    resident_total = reports["resident_batch_helper"]["wall_seconds"]
    per_frequency_total = reports["per_frequency_helper"]["wall_seconds"]
    return {
        "available": True,
        "mesh": {
            "path": str(mesh_path),
            "n_vertices": int(loaded.info.n_vertices),
            "n_triangles": int(loaded.info.n_triangles),
        },
        "frequency_hz": [float(v) for v in frequencies],
        "reports": reports,
        "speedup_vs_per_frequency": (
            per_frequency_total / resident_total
            if resident_total > 0.0
            else None
        ),
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "runs/canonical-validation/260601-metal-native-regular",
    )
    parser.add_argument("--asro68-mesh", type=Path, default=DEFAULT_ASRO68_MESH)
    parser.add_argument(
        "--native-mode",
        choices=("optimized", "corrected"),
        default="optimized",
        help=(
            "ASRO native helper mode. 'optimized' is regular quadrature only; "
            "'corrected' applies Duffy matrix/RHS corrections."
        ),
    )
    parser.add_argument(
        "--threadgroup-size",
        type=int,
        default=None,
        help=(
            "Optional native Metal threads-per-threadgroup override for the "
            "main ASRO assembly and field runs."
        ),
    )
    parser.add_argument(
        "--threadgroup-sweep",
        default=None,
        help=(
            "Comma-separated native Metal threadgroup sizes for an ASRO "
            "native-only assembly sweep, for example 32,64,128,256."
        ),
    )
    parser.add_argument(
        "--field-grid",
        choices=("horizontal", "hemisphere"),
        default="horizontal",
        help=(
            "ASRO field validation grid: the existing 181-point horizontal "
            "arc or a 91x181 forward hemisphere."
        ),
    )
    parser.add_argument("--skip-asro68", action="store_true")
    parser.add_argument(
        "--run-sweep-compare",
        action="store_true",
        help=(
            "Run the full native Metal ASRO68 sweep twice: old per-frequency "
            "helper route and new resident batch route."
        ),
    )
    parser.add_argument("--sweep-freq-count", type=int, default=40)
    args = parser.parse_args()
    if args.threadgroup_size is not None and args.threadgroup_size <= 0:
        parser.error("--threadgroup-size must be a positive integer")
    try:
        threadgroup_sweep = parse_threadgroup_sweep(args.threadgroup_sweep)
    except ValueError as exc:
        parser.error(str(exc))

    args.output_dir = args.output_dir.resolve()
    args.asro68_mesh = args.asro68_mesh.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = discover_native_runtime(run_smoke_test=True)
    report: dict[str, Any] = {
        "schema": "hornlab.metal.native_benchmark.v1",
        "scope": (
            "dense_assembly_with_optional_cpu_duffy_cpu_direct_solve_"
            "and_selectable_field_grid"
        ),
        "native_runtime": {
            "available": runtime.available,
            "platform_system": runtime.platform_system,
            "platform_machine": runtime.platform_machine,
            "helper_executable_path": str(runtime.helper_executable_path)
            if runtime.helper_executable_path
            else None,
            "swift_path": runtime.swift_path,
            "reasons": list(runtime.unavailable_reasons),
        },
    }
    if not runtime.available:
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    report["tiny"] = run_tiny(args.output_dir)
    report["synthetic_yz_symmetry"] = run_synthetic_yz_symmetry(args.output_dir)
    if args.skip_asro68:
        report["asro68_100hz"] = {"available": False, "blocker": "--skip-asro68"}
    else:
        report["asro68_100hz"] = run_asro68(
            args.output_dir,
            args.asro68_mesh,
            args.native_mode,
            args.threadgroup_size,
            args.field_grid,
        )
        if threadgroup_sweep:
            report["asro68_100hz_threadgroup_sweep"] = (
                run_asro68_threadgroup_sweep(
                    args.output_dir,
                    args.asro68_mesh,
                    args.native_mode,
                    threadgroup_sweep,
                )
            )
        if args.run_sweep_compare:
            report["asro68_sweep_compare"] = run_asro68_sweep_compare(
                args.output_dir,
                args.asro68_mesh,
                args.sweep_freq_count,
            )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

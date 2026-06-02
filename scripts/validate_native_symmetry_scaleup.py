#!/usr/bin/env python3
"""Validate native Metal quarter-domain symmetry on generated WG meshes."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "hornlab-mesher"))

from hornlab_solver import SolveConfig, load_mesh  # noqa: E402
from hornlab_solver._constants import SPEED_OF_SOUND  # noqa: E402
from hornlab_solver.bie import _build_neumann_data, _setup_function_spaces  # noqa: E402
from hornlab_solver.metal.geometry import build_metal_geometry_buffers  # noqa: E402
from hornlab_solver.metal.native import (  # noqa: E402
    MetalNativeStandardSession,
    discover_native_runtime,
)
from hornlab_solver.validation.native_symmetry import (  # noqa: E402
    build_local2global_xy_mirror_orbits,
    classify_orbits_by_size,
    expand_quarter_mesh_xy,
    expand_reduced_pressure,
    orbit_reduce_matrix_rhs,
)


GEOMETRY_CLI = WORKSPACE_ROOT / "hornlab-geometry/bin/geometry-cli.js"
DEFAULT_OUTPUT_DIR = (
    WORKSPACE_ROOT / "runs/canonical-validation/260602-native-symmetry-scaleup"
)


@dataclass(frozen=True)
class MeshCase:
    name: str
    angular_segments: int
    length_segments: int
    throat_res_mm: float
    mouth_res_mm: float
    rear_res_mm: float
    profile: str = "osse"


DEFAULT_CASES = (
    MeshCase("small", 16, 8, 12.0, 18.0, 24.0),
    MeshCase("medium", 24, 12, 9.0, 14.0, 20.0),
    MeshCase("larger", 32, 16, 7.0, 11.0, 18.0),
    # Historical public ASRO68 R-OSSE parameters at the M2 density from the
    # WG mesh-resolution study. The imported ABEC ASRO68 full mesh is not an
    # individual X/Y mirror oracle, so this case builds a reduced WG mesh first
    # and expands it exactly with shared seam vertices.
    MeshCase("asro-rosse-m2", 64, 20, 2.8, 11.2, 17.0, "asro_rosse"),
)


def base_wg_params(case: MeshCase) -> dict[str, Any]:
    common = {
        "angularSegments": case.angular_segments,
        "lengthSegments": case.length_segments,
        "wallThickness": 0.0,
        "encDepth": 0.0,
        "quadrants": "1",
        "throatResolution": case.throat_res_mm,
        "mouthResolution": case.mouth_res_mm,
        "rearResolution": case.rear_res_mm,
        "profileSystem": {
            "crossSection": {
                "exponent": 2.0,
                "aspectRatio": 1.0,
            },
        },
    }
    if case.profile == "asro_rosse":
        return {
            **common,
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
        }
    if case.profile != "osse":
        raise ValueError(f"unknown mesh profile: {case.profile}")
    return {
        **common,
        "type": "OSSE",
        "r0": 12.7,
        "a": 55.0,
        "a0": 15.5,
        "k": 1.0,
        "q": 0.995,
        "L": 90.0,
        "n": 4.0,
        "s": 0.0,
    }


def build_point_grid(params: dict[str, Any]) -> dict[str, Any]:
    message = {
        "id": "scaleup-grid",
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
        if payload.get("id") == "scaleup-grid":
            return dict(payload["result"])
    raise RuntimeError("geometry-cli did not return a point grid")


def reshape_grid(raw: Any, n_phi: int, n_length: int, name: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    expected = n_phi * (n_length + 1) * 3
    if arr.size != expected:
        raise ValueError(f"{name} has {arr.size} values; expected {expected}")
    return arr.reshape(n_phi, n_length + 1, 3)


def generate_quarter_mesh(case: MeshCase, case_dir: Path) -> dict[str, Any]:
    from hornlab_mesher import MeshDensity, PointGridHornGeometry, build_mesh
    from hornlab_mesher import load_mesh as inspect_mesh

    params = base_wg_params(case)
    grid = build_point_grid(params)
    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner_points = reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")

    mesh_path = case_dir / f"{case.name}-quarter.msh"
    build_mesh(
        PointGridHornGeometry(
            inner_points=inner_points,
            preserve_grid=True,
            closed=True,
            outer_points=None,
            wall_thickness_mm=0.0,
        ),
        MeshDensity(
            throat_res_mm=case.throat_res_mm,
            mouth_res_mm=case.mouth_res_mm,
            rear_res_mm=case.rear_res_mm,
        ),
        mesh_path,
        scale_to_metres=True,
    )
    info = inspect_mesh(mesh_path)
    config = {
        "formula": "ASRO68 R-OSSE" if case.profile == "asro_rosse" else "OSSE",
        "mode": "bare",
        "profile": params,
        "mesh": {
            "quadrants": "1",
            "angularSegments": case.angular_segments,
            "lengthSegments": case.length_segments,
            "preserve_grid": True,
            "grid_closed": True,
            "throatResolution": case.throat_res_mm,
            "mouthResolution": case.mouth_res_mm,
            "rearResolution": case.rear_res_mm,
        },
        "cross_section": {"aspect_ratio": 1.0, "exponent": 2.0},
    }
    (case_dir / "quarter-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(mesh_path),
        "n_vertices": int(info.n_vertices),
        "n_triangles": int(info.n_triangles),
        "units": info.units,
        "physical_groups": {str(k): v for k, v in info.physical_groups.items()},
        "bounds": {
            "min": info.bounding_box[0].astype(float).tolist(),
            "max": info.bounding_box[1].astype(float).tolist(),
        },
        "point_grid": {
            "grid_n_phi": n_phi,
            "grid_n_length": n_length,
            "full_circle": bool(grid["full_circle"]),
            "angle_list": grid.get("angle_list"),
        },
    }


def write_expanded_full_mesh(quarter_path: Path, full_path: Path) -> dict[str, Any]:
    import meshio

    mesh = meshio.read(quarter_path)
    triangles = np.asarray(mesh.cells_dict["triangle"], dtype=np.int64)
    tags = np.asarray(mesh.cell_data_dict["gmsh:physical"]["triangle"], dtype=np.int32)
    expanded = expand_quarter_mesh_xy(mesh.points, triangles, tags)
    used_tags = sorted({int(tag) for tag in expanded.physical_tags.tolist()})
    field_data = {
        ("SD1G0" if tag == 1 else "SD1D1001" if tag == 2 else f"SD1D{1000 + tag - 1}"): np.array(
            [tag, 2],
            dtype=np.int32,
        )
        for tag in used_tags
    }
    out_mesh = meshio.Mesh(
        points=expanded.vertices_nx3,
        cells=[("triangle", expanded.triangles_nx3)],
        cell_data={
            "gmsh:physical": [expanded.physical_tags],
            "gmsh:geometrical": [expanded.physical_tags],
        },
        field_data=field_data,
    )
    meshio.write(full_path, out_mesh, file_format="gmsh22", binary=False)
    return {
        "path": str(full_path),
        "n_vertices": int(expanded.vertices_nx3.shape[0]),
        "n_triangles": int(expanded.triangles_nx3.shape[0]),
        "image_counts": {
            f"{int(sx)},{int(sy)}": int(np.sum(np.all(expanded.triangle_image_signs == [sx, sy], axis=1)))
            for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1))
        },
    }


def read_complex(real_path: Path, imag_path: Path, shape: tuple[int, ...]) -> np.ndarray:
    real = np.fromfile(real_path, dtype="<f4").reshape(shape)
    imag = np.fromfile(imag_path, dtype="<f4").reshape(shape)
    return real.astype(np.complex64) + 1j * imag.astype(np.complex64)


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    diff = np.linalg.norm(candidate - reference)
    ref = np.linalg.norm(reference)
    return float(diff if ref == 0.0 else diff / ref)


def max_abs(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(candidate - reference))) if candidate.size else 0.0


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
        if not isinstance(exc, ImportError) and "scipy" not in str(type(exc)).lower():
            return {"available": False, "blocker": f"dense solve failed: {exc}"}
        try:
            pressure = np.linalg.solve(
                np.asarray(matrix, dtype=np.complex64),
                np.asarray(rhs, dtype=np.complex64),
            )
        except Exception as solve_exc:
            return {"available": False, "blocker": f"dense solve failed: {solve_exc}"}
    residual = matrix @ pressure - rhs
    rhs_norm = np.linalg.norm(rhs)
    return {
        "available": True,
        "implementation": implementation,
        "wall_seconds": perf_counter() - started,
        "pressure": np.asarray(pressure, dtype=np.complex64),
        "relative_residual_l2": float(
            np.linalg.norm(residual) if rhs_norm == 0.0 else np.linalg.norm(residual) / rhs_norm
        ),
    }


def directivity_points(n_angles: int = 37, radius_m: float = 2.0) -> tuple[np.ndarray, dict[str, slice], np.ndarray]:
    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, n_angles, dtype=np.float64)
    cuts: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    start = 0
    for name in ("horizontal", "vertical", "diagonal"):
        s = np.sin(angles)
        c = np.cos(angles)
        if name == "horizontal":
            points = np.vstack([radius_m * s, np.zeros_like(s), radius_m * c])
        elif name == "vertical":
            points = np.vstack([np.zeros_like(s), radius_m * s, radius_m * c])
        else:
            points = np.vstack([radius_m * s / math.sqrt(2.0), radius_m * s / math.sqrt(2.0), radius_m * c])
        cuts.append(points)
        slices[name] = slice(start, start + n_angles)
        start += n_angles
    fixed = np.array(
        [
            [0.0, 0.25, -0.25, 0.35, -0.35, 0.2, -0.2, 0.45],
            [0.0, 0.2, 0.2, -0.35, -0.35, 0.45, -0.45, 0.1],
            [1.0, 1.2, 1.2, 1.4, 1.4, 0.9, 0.9, 1.7],
        ],
        dtype=np.float64,
    )
    slices["fixed_points"] = slice(start, start + fixed.shape[1])
    points = np.hstack([*cuts, fixed]).astype(np.float32)
    return points, slices, (angles * 180.0 / math.pi).astype(np.float32)


def directivity_db(values: np.ndarray) -> np.ndarray:
    mag = np.abs(values).astype(np.float64)
    ref = float(np.max(mag))
    if ref <= 0.0:
        ref = np.finfo(np.float64).eps
    return 20.0 * np.log10(np.maximum(mag / ref, 1.0e-12))


def comparison_report(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {
        "relative_l2": relative_l2(candidate, reference),
        "max_abs": max_abs(candidate, reference),
    }


def run_native_field(
    session: MetalNativeStandardSession,
    *,
    frequency_hz: float,
    pressure: np.ndarray,
    neumann: np.ndarray,
    points_3xn: np.ndarray,
    operation_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    k_real = 2.0 * math.pi * frequency_hz / SPEED_OF_SOUND
    started = perf_counter()
    result = session.evaluate_standard_exterior(
        frequency_hz,
        k_real,
        pressure,
        neumann,
        points_3xn,
        batch_id="scaleup-points",
        operation_id=operation_id,
    )
    wall = perf_counter() - started
    manifest = json.loads(
        (session.info.work_dir / operation_id / "field-result.json").read_text(encoding="utf-8")
    )
    field = read_complex(result.pressure_real_f32, result.pressure_imag_f32, result.shape)
    return field, {
        "wall_seconds": wall,
        "field_seconds": manifest.get("field_seconds"),
        "field_mode": manifest.get("field_mode"),
        "symmetry_plane": manifest.get("symmetry_plane"),
    }


def run_mesh_case(
    case: MeshCase,
    *,
    output_dir: Path,
    frequencies_hz: np.ndarray,
    native_mode: str,
    include_directivity: bool,
) -> dict[str, Any]:
    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    quarter_summary = generate_quarter_mesh(case, case_dir)
    full_summary = write_expanded_full_mesh(
        Path(quarter_summary["path"]),
        case_dir / f"{case.name}-expanded-full.msh",
    )

    loaded_quarter = load_mesh(quarter_summary["path"], scale=1.0)
    loaded_full = load_mesh(full_summary["path"], scale=1.0)
    q_p1, q_dp0 = _setup_function_spaces(loaded_quarter.grid)
    f_p1, f_dp0 = _setup_function_spaces(loaded_full.grid)
    q_buffers = build_metal_geometry_buffers(
        loaded_quarter.grid,
        loaded_quarter.physical_tags,
        q_p1,
        q_dp0,
    )
    f_buffers = build_metal_geometry_buffers(
        loaded_full.grid,
        loaded_full.physical_tags,
        f_p1,
        f_dp0,
    )
    p1_orbits = build_local2global_xy_mirror_orbits(
        q_buffers.p1_local2global_i32,
        f_buffers.p1_local2global_i32,
    )
    row_classes = classify_orbits_by_size(p1_orbits)
    row_counts = {
        name: int(np.sum(row_classes == name))
        for name in ("interior", "single_seam", "double_seam")
    }

    points, cut_slices, angles_deg = directivity_points()
    np.asarray(points, dtype="<f4").tofile(case_dir / "observation_points_3xn_f32.bin")
    np.asarray(angles_deg, dtype="<f4").tofile(case_dir / "directivity_angles_deg_f32.bin")

    previous_mode = os.environ.get("HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE")
    os.environ["HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE"] = native_mode
    try:
        with MetalNativeStandardSession.create_session(
            geometry_buffers=f_buffers,
            work_dir=case_dir / "full-session",
            session_id=f"{case.name}-expanded-full",
            keep_artifacts=True,
        ) as full_session, MetalNativeStandardSession.create_session(
            geometry_buffers=q_buffers,
            work_dir=case_dir / "quarter-session",
            session_id=f"{case.name}-native-quarter",
            keep_artifacts=True,
            symmetry_plane="yz+xz",
        ) as quarter_session:
            frequency_reports = []
            for idx, frequency_hz in enumerate(frequencies_hz):
                frequency_reports.append(
                    run_frequency(
                        idx,
                        float(frequency_hz),
                        full_session=full_session,
                        quarter_session=quarter_session,
                        full_loaded=loaded_full,
                        quarter_loaded=loaded_quarter,
                        full_dp0=f_dp0,
                        quarter_dp0=q_dp0,
                        p1_orbits=p1_orbits,
                        row_classes=row_classes,
                        points=points,
                        cut_slices=cut_slices if include_directivity else {},
                    )
                )
    finally:
        if previous_mode is None:
            os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE", None)
        else:
            os.environ["HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE"] = previous_mode

    return {
        "case": case.name,
        "mesh_parameters": {
            "profile": case.profile,
            "angular_segments": case.angular_segments,
            "length_segments": case.length_segments,
            "throat_res_mm": case.throat_res_mm,
            "mouth_res_mm": case.mouth_res_mm,
            "rear_res_mm": case.rear_res_mm,
        },
        "quarter_mesh": quarter_summary,
        "expanded_full_mesh": full_summary,
        "dofs": {
            "quarter_p1": int(q_buffers.p1_dof_count),
            "quarter_dp0": int(q_buffers.dp0_dof_count),
            "full_p1": int(f_buffers.p1_dof_count),
            "full_dp0": int(f_buffers.dp0_dof_count),
            "row_classes": row_counts,
            "orbit_sizes": {
                "min": int(min(len(o) for o in p1_orbits)),
                "max": int(max(len(o) for o in p1_orbits)),
                "mean": float(np.mean([len(o) for o in p1_orbits])),
            },
        },
        "observation": {
            "n_points": int(points.shape[1]),
            "cuts": {
                key: [int(value.start), int(value.stop)]
                for key, value in cut_slices.items()
            },
            "angles_deg": angles_deg.astype(float).tolist(),
        },
        "frequencies": frequency_reports,
        "summary": summarize_frequency_reports(frequency_reports),
    }


def run_frequency(
    idx: int,
    frequency_hz: float,
    *,
    full_session: MetalNativeStandardSession,
    quarter_session: MetalNativeStandardSession,
    full_loaded: Any,
    quarter_loaded: Any,
    full_dp0: Any,
    quarter_dp0: Any,
    p1_orbits: list[np.ndarray],
    row_classes: np.ndarray,
    points: np.ndarray,
    cut_slices: dict[str, slice],
) -> dict[str, Any]:
    omega = 2.0 * math.pi * frequency_hz
    k_real = 2.0 * math.pi * frequency_hz / SPEED_OF_SOUND
    config = SolveConfig()
    full_neumann_fun = _build_neumann_data(
        full_dp0,
        full_loaded.physical_tags,
        omega,
        config,
        precision="single",
    )
    quarter_neumann_fun = _build_neumann_data(
        quarter_dp0,
        quarter_loaded.physical_tags,
        omega,
        config,
        precision="single",
    )
    full_neumann = np.asarray(full_neumann_fun.coefficients, dtype=np.complex64)
    quarter_neumann = np.asarray(quarter_neumann_fun.coefficients, dtype=np.complex64)

    full_op = f"assembly-{idx:04d}-full"
    quarter_op = f"assembly-{idx:04d}-quarter"
    full_started = perf_counter()
    full_assembly = full_session.assemble_standard_neumann(
        frequency_hz,
        k_real,
        full_neumann,
        operation_id=full_op,
    )
    full_wall = perf_counter() - full_started
    quarter_started = perf_counter()
    quarter_assembly = quarter_session.assemble_standard_neumann(
        frequency_hz,
        k_real,
        quarter_neumann,
        operation_id=quarter_op,
    )
    quarter_wall = perf_counter() - quarter_started

    full_manifest = json.loads(
        (full_session.info.work_dir / full_op / "assembly-result.json").read_text(encoding="utf-8")
    )
    quarter_manifest = json.loads(
        (quarter_session.info.work_dir / quarter_op / "assembly-result.json").read_text(encoding="utf-8")
    )
    full_matrix = read_complex(
        full_assembly.matrix_real_f32,
        full_assembly.matrix_imag_f32,
        full_assembly.matrix_shape,
    )
    full_rhs = read_complex(
        full_assembly.rhs_real_f32,
        full_assembly.rhs_imag_f32,
        full_assembly.rhs_shape,
    )
    quarter_matrix = read_complex(
        quarter_assembly.matrix_real_f32,
        quarter_assembly.matrix_imag_f32,
        quarter_assembly.matrix_shape,
    )
    quarter_rhs = read_complex(
        quarter_assembly.rhs_real_f32,
        quarter_assembly.rhs_imag_f32,
        quarter_assembly.rhs_shape,
    )

    reduced_full_matrix, reduced_full_rhs = orbit_reduce_matrix_rhs(
        full_matrix,
        full_rhs,
        p1_orbits,
    )
    orbit_solve = solve_dense_direct(reduced_full_matrix, reduced_full_rhs)
    quarter_solve = solve_dense_direct(quarter_matrix, quarter_rhs)
    full_solve = solve_dense_direct(full_matrix, full_rhs)

    report: dict[str, Any] = {
        "frequency_hz": frequency_hz,
        "assembly": {
            "expanded_full": assembly_report(full_manifest, full_wall),
            "native_quarter": assembly_report(quarter_manifest, quarter_wall),
        },
        "matrix": comparison_report(quarter_matrix, reduced_full_matrix),
        "rhs": comparison_report(quarter_rhs, reduced_full_rhs),
        "row_type_metrics": row_type_metrics(
            quarter_matrix,
            reduced_full_matrix,
            quarter_rhs,
            reduced_full_rhs,
            row_classes,
        ),
        "solve": {
            "orbit_reduced_full": solve_report(orbit_solve),
            "native_quarter": solve_report(quarter_solve),
            "expanded_full": solve_report(full_solve),
        },
    }

    field_seconds: dict[str, Any] = {}
    if orbit_solve.get("available") and quarter_solve.get("available"):
        report["solution_native_vs_orbit_reduced"] = comparison_report(
            quarter_solve["pressure"],
            orbit_solve["pressure"],
        )
    if full_solve.get("available") and quarter_solve.get("available"):
        expanded_quarter_pressure = expand_reduced_pressure(
            quarter_solve["pressure"],
            full_matrix.shape[0],
            p1_orbits,
        )
        report["solution_expanded_native_vs_full_solve"] = comparison_report(
            expanded_quarter_pressure,
            full_solve["pressure"],
        )
        full_field, full_field_report = run_native_field(
            full_session,
            frequency_hz=frequency_hz,
            pressure=full_solve["pressure"],
            neumann=full_neumann,
            points_3xn=points,
            operation_id=f"field-{idx:04d}-full-solve",
        )
        quarter_field, quarter_field_report = run_native_field(
            quarter_session,
            frequency_hz=frequency_hz,
            pressure=quarter_solve["pressure"],
            neumann=quarter_neumann,
            points_3xn=points,
            operation_id=f"field-{idx:04d}-quarter-solve",
        )
        field_seconds = {
            "expanded_full": full_field_report,
            "native_quarter": quarter_field_report,
        }
        report["field_native_vs_full_solve"] = comparison_report(
            quarter_field,
            full_field,
        )
        if cut_slices:
            report["directivity_db_error"] = {
                name: comparison_report(
                    directivity_db(quarter_field[slc]),
                    directivity_db(full_field[slc]),
                )
                for name, slc in cut_slices.items()
                if name != "fixed_points"
            }
    report["field"] = field_seconds
    report["timing_seconds"] = timing_report(report)
    report["diagnosis"] = diagnose_frequency(report)
    return report


def assembly_report(manifest: dict[str, Any], wall_seconds: float) -> dict[str, Any]:
    return {
        "wall_seconds": wall_seconds,
        "assembly_seconds": manifest.get("assembly_seconds"),
        "regular_assembly_seconds": manifest.get("regular_assembly_seconds"),
        "duffy_corrections": manifest.get("duffy_corrections"),
        "symmetry_plane": manifest.get("symmetry_plane"),
        "implementation": manifest.get("implementation"),
    }


def solve_report(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("available"):
        return result
    return {
        "available": True,
        "implementation": result["implementation"],
        "wall_seconds": result["wall_seconds"],
        "relative_residual_l2": result["relative_residual_l2"],
    }


def row_type_metrics(
    candidate_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    candidate_rhs: np.ndarray,
    reference_rhs: np.ndarray,
    row_classes: np.ndarray,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row_type in ("interior", "single_seam", "double_seam"):
        mask = row_classes == row_type
        if not np.any(mask):
            continue
        out.append(
            {
                "type": row_type,
                "count": int(np.sum(mask)),
                "matrix": comparison_report(candidate_matrix[mask, :], reference_matrix[mask, :]),
                "rhs": comparison_report(candidate_rhs[mask], reference_rhs[mask]),
            }
        )
    return out


def timing_report(report: dict[str, Any]) -> dict[str, Any]:
    full_assembly = float(report["assembly"]["expanded_full"].get("wall_seconds") or 0.0)
    quarter_assembly = float(report["assembly"]["native_quarter"].get("wall_seconds") or 0.0)
    full_solve = float(report["solve"]["expanded_full"].get("wall_seconds") or 0.0)
    quarter_solve = float(report["solve"]["native_quarter"].get("wall_seconds") or 0.0)
    full_field = float(report.get("field", {}).get("expanded_full", {}).get("wall_seconds") or 0.0)
    quarter_field = float(report.get("field", {}).get("native_quarter", {}).get("wall_seconds") or 0.0)
    return {
        "expanded_full": {
            "assembly": full_assembly,
            "solve": full_solve,
            "field": full_field,
            "total": full_assembly + full_solve + full_field,
        },
        "native_quarter": {
            "assembly": quarter_assembly,
            "solve": quarter_solve,
            "field": quarter_field,
            "total": quarter_assembly + quarter_solve + quarter_field,
        },
    }


def diagnose_frequency(report: dict[str, Any]) -> dict[str, Any]:
    matrix_rel = float(report["matrix"]["relative_l2"])
    rhs_rel = float(report["rhs"]["relative_l2"])
    solution_rel = float(report.get("solution_expanded_native_vs_full_solve", {}).get("relative_l2", math.inf))
    field_rel = float(report.get("field_native_vs_full_solve", {}).get("relative_l2", math.inf))
    seam_rows = [
        item for item in report["row_type_metrics"]
        if item["type"] in {"single_seam", "double_seam"}
    ]
    seam_matrix_rel = max(
        (float(item["matrix"]["relative_l2"]) for item in seam_rows),
        default=0.0,
    )
    likely: list[str] = []
    if matrix_rel < 5.0e-5 and rhs_rel < 5.0e-5 and field_rel < 5.0e-5:
        likely.append("numerical precision/conditioning only")
    if seam_matrix_rel > max(5.0 * matrix_rel, 1.0e-4):
        likely.append("seam row/orbit weighting or image-adjacent Duffy correction")
    if rhs_rel > max(5.0 * matrix_rel, 1.0e-4):
        likely.append("source/RHS scaling")
    if matrix_rel > 1.0e-4 and seam_matrix_rel <= max(5.0 * matrix_rel, 1.0e-4):
        likely.append("normal reflection signs or mesh local-to-global mapping")
    if field_rel > max(5.0 * solution_rel, 1.0e-4):
        likely.append("field evaluation weights")
    return {
        "parity_holds": bool(matrix_rel < 1.0e-4 and rhs_rel < 1.0e-4 and field_rel < 1.0e-4),
        "likely_sources": likely or ["undetermined"],
    }


def summarize_frequency_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {
        "matrix_relative_l2": ("matrix", "relative_l2"),
        "rhs_relative_l2": ("rhs", "relative_l2"),
        "solution_native_vs_orbit_relative_l2": ("solution_native_vs_orbit_reduced", "relative_l2"),
        "solution_expanded_native_vs_full_relative_l2": ("solution_expanded_native_vs_full_solve", "relative_l2"),
        "field_relative_l2": ("field_native_vs_full_solve", "relative_l2"),
    }
    summary: dict[str, Any] = {
        "frequency_count": len(reports),
        "parity_holds_all": all(report.get("diagnosis", {}).get("parity_holds", False) for report in reports),
    }
    for out_key, path in keys.items():
        values = []
        for report in reports:
            current: Any = report
            for part in path:
                current = current.get(part, {}) if isinstance(current, dict) else {}
            if isinstance(current, (int, float)) and math.isfinite(float(current)):
                values.append(float(current))
        if values:
            summary[out_key] = {
                "max": max(values),
                "mean": float(np.mean(values)),
            }
    directivity_values: dict[str, list[float]] = {
        "relative_l2": [],
        "max_abs_db": [],
    }
    for report in reports:
        for item in report.get("directivity_db_error", {}).values():
            rel = item.get("relative_l2")
            max_abs_db = item.get("max_abs")
            if isinstance(rel, (int, float)) and math.isfinite(float(rel)):
                directivity_values["relative_l2"].append(float(rel))
            if isinstance(max_abs_db, (int, float)) and math.isfinite(float(max_abs_db)):
                directivity_values["max_abs_db"].append(float(max_abs_db))
    if directivity_values["relative_l2"]:
        summary["directivity_db_relative_l2"] = {
            "max": max(directivity_values["relative_l2"]),
            "mean": float(np.mean(directivity_values["relative_l2"])),
        }
    if directivity_values["max_abs_db"]:
        summary["directivity_db_max_abs"] = {
            "max": max(directivity_values["max_abs_db"]),
            "mean": float(np.mean(directivity_values["max_abs_db"])),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freq-count", type=int, default=5)
    parser.add_argument("--min-frequency", type=float, default=250.0)
    parser.add_argument("--max-frequency", type=float, default=2000.0)
    parser.add_argument(
        "--native-mode",
        choices=("optimized", "corrected"),
        default="optimized",
    )
    parser.add_argument("--skip-directivity", action="store_true")
    parser.add_argument(
        "--cases",
        default="small,medium,larger",
        help="Comma-separated subset of small,medium,larger,asro-rosse-m2.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.freq_count <= 0:
        raise SystemExit("--freq-count must be positive")
    selected = {name.strip() for name in args.cases.split(",") if name.strip()}
    cases = [case for case in DEFAULT_CASES if case.name in selected]
    if not cases:
        raise SystemExit("--cases selected no known mesh cases")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = discover_native_runtime(run_smoke_test=True)
    frequencies = np.geomspace(
        float(args.min_frequency),
        float(args.max_frequency),
        int(args.freq_count),
        dtype=np.float64,
    )
    report: dict[str, Any] = {
        "schema": "hornlab.native_symmetry_scaleup.v1",
        "scope": "generated_bare_wg_quarter_mesh_vs_exact_expanded_full_orbit_reference",
        "native_runtime": {
            "available": runtime.available,
            "platform_system": runtime.platform_system,
            "platform_machine": runtime.platform_machine,
            "helper_executable_path": str(runtime.helper_executable_path) if runtime.helper_executable_path else None,
            "swift_path": runtime.swift_path,
            "reasons": list(runtime.unavailable_reasons),
        },
        "native_mode": args.native_mode,
        "frequencies_hz": [float(v) for v in frequencies],
        "uses_matrix_or_hermitian_shortcuts": False,
        "directivity_cuts": not args.skip_directivity,
        "cases": [],
    }
    if not runtime.available:
        (output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    for case in cases:
        print(f"[scaleup] running {case.name} ({case.angular_segments}x{case.length_segments})")
        case_report = run_mesh_case(
            case,
            output_dir=output_dir,
            frequencies_hz=frequencies,
            native_mode=args.native_mode,
            include_directivity=not args.skip_directivity,
        )
        report["cases"].append(case_report)
        (output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    report["summary"] = {
        "parity_holds_all": all(case["summary"]["parity_holds_all"] for case in report["cases"]),
        "case_count": len(report["cases"]),
        "frequency_count": int(args.freq_count),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

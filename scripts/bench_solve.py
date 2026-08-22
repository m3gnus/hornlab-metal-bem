"""Benchmark the native resident assembly, solve, and field batch path."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREQUENCY_COUNT = 8
DEFAULT_F1_HZ = 500.0
DEFAULT_F2_HZ = 4000.0
DEFAULT_REPEAT = 3
DEFAULT_SOURCE_TAG = 2
BUILTIN_APERTURE_TAG = 7

# These variables select materially different native code paths or precision
# and quadrature policies. Recording unset values is intentional: the helper's
# defaults can change independently of this harness, and an A/B result without
# evidence that a knob was inherited rather than pinned is not reproducible.
RELEVANT_ENV = (
    "HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE",
    "HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL",
    "HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE",
    "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE",
    "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_IMPL",
    "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_REFINE",
    "HORNLAB_METAL_BEM_NATIVE_SOLVE_CONCURRENCY",
    "HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE",
    "HORNLAB_METAL_BEM_NATIVE_FIELD_MODE",
    "HORNLAB_METAL_BEM_SPHERE_SYMMETRY_DEDUPE",
    "HORNLAB_METAL_BEM_NATIVE_COUPLED_IB_APERTURE_ASSEMBLY",
    "HORNLAB_METAL_BEM_NATIVE_COUPLED_IB_SOLVE",
)


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


def _sphere_grid(value: str) -> tuple[int, int]:
    try:
        raw_theta, raw_phi = value.split(",", maxsplit=1)
        n_theta = _positive_int(raw_theta)
        n_phi = _positive_int(raw_phi)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            "must be two positive integers formatted n_theta,n_phi"
        ) from exc
    if n_theta < 2 or n_phi < 3:
        raise argparse.ArgumentTypeError("requires n_theta >= 2 and n_phi >= 3")
    return n_theta, n_phi


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Gmsh surface mesh; omit to use the built-in coupled test box",
    )
    parser.add_argument(
        "--frequencies",
        type=_positive_int,
        default=DEFAULT_FREQUENCY_COUNT,
        metavar="N",
        help="number of log-spaced solve cases (default: %(default)s)",
    )
    parser.add_argument(
        "--f1",
        type=_positive_float,
        default=DEFAULT_F1_HZ,
        metavar="HZ",
        help="first frequency (default: %(default)s)",
    )
    parser.add_argument(
        "--f2",
        type=_positive_float,
        default=DEFAULT_F2_HZ,
        metavar="HZ",
        help="last frequency (default: %(default)s)",
    )
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=DEFAULT_REPEAT,
        metavar="R",
        help="total batch runs; the first is reported as warm-up (default: %(default)s)",
    )
    parser.add_argument(
        "--source-tag",
        type=int,
        default=DEFAULT_SOURCE_TAG,
        help="driven physical tag (default: %(default)s)",
    )
    parser.add_argument(
        "--sphere-grid",
        type=_sphere_grid,
        default=None,
        metavar="N_THETA,N_PHI",
        help="append a frame-relative full-sphere field grid",
    )
    parser.add_argument(
        "--native-symmetry-plane",
        choices=("yz", "xz", "xy", "yz+xz"),
        default=None,
        help="native mirror plane for an already reduced input mesh",
    )
    parser.add_argument(
        "--disable-sphere-symmetry-dedupe",
        action="store_true",
        help="evaluate every sphere-grid point for a symmetry A/B",
    )
    parser.add_argument(
        "--native-allow-open-rim",
        action="store_true",
        help="allow real off-plane free edges on a mirror-reduced open shell",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args(argv)


def _built_in_box():
    """Return the small coupled box used by the native aperture tests.

    The topology and inward cavity orientation match ``_ib_box_geometry_buffers``
    in ``tests/test_metal_native.py``. Scaling it to a 100 mm square keeps the
    default acoustic fixture plausible without changing its deliberately tiny
    eight-DOF system, which is useful for smoke runs and dispatch-overhead A/Bs.
    """
    from hornlab_metal_bem.mesh import LoadedMesh, make_pure_grid
    from hornlab_metal_bem.result import MeshInfo

    scale = 0.05
    vertices = scale * np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            # Aperture -Z, rear cap +Z, and side-wall normals all point into
            # the box cavity, as required by the coupled infinite-baffle path.
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 5, 4],
            [0, 1, 5],
            [1, 6, 5],
            [1, 2, 6],
            [2, 7, 6],
            [2, 3, 7],
            [3, 4, 7],
            [3, 0, 4],
        ],
        dtype=np.int32,
    )
    tags = np.array(
        [BUILTIN_APERTURE_TAG, BUILTIN_APERTURE_TAG, DEFAULT_SOURCE_TAG, DEFAULT_SOURCE_TAG]
        + [1] * 8,
        dtype=np.int32,
    )
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    return LoadedMesh(
        grid=make_pure_grid(vertices, triangles),
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=int(vertices.shape[0]),
            n_triangles=int(triangles.shape[0]),
            physical_groups={
                1: "walls",
                DEFAULT_SOURCE_TAG: "source",
                BUILTIN_APERTURE_TAG: "mouth_aperture",
            },
            bounding_box_m=(lower, upper),
        ),
        coupled_ib_aperture_tag=BUILTIN_APERTURE_TAG,
    )


def _observation_points(
    mesh: Any,
    *,
    source_tag: int,
    sphere_grid: tuple[int, int] | None,
    native_symmetry_plane: str | None,
    disable_sphere_symmetry_dedupe: bool,
) -> tuple[np.ndarray, int, int]:
    lower, upper = mesh.info.bounding_box_m
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    center = 0.5 * (lower + upper)
    span = max(float(np.linalg.norm(upper - lower)), 1.0e-3)
    z = upper[2] + span
    # A small fixed field workload keeps this a solve benchmark while still
    # exercising the exact assemble+solve+evaluate operation used by the public
    # wrapper. Every point is above the bounding box, including for the coupled
    # box whose physical exterior is the +Z half-space.
    points = np.array(
        [
            [center[0], center[1], z],
            [center[0] - 0.25 * span, center[1], z],
            [center[0] + 0.25 * span, center[1], z],
            [center[0], center[1] + 0.25 * span, z],
        ],
        dtype=np.float64,
    )
    sphere_total = 0
    sphere_evaluated = 0
    if sphere_grid is not None:
        from hornlab_metal_bem.config import ObservationConfig
        from hornlab_metal_bem.observation import (
            build_mirror_evaluation_classes,
            build_sphere_grid_points,
            infer_frame,
        )

        frame = infer_frame(
            mesh.grid,
            mesh.physical_tags,
            source_tag=source_tag,
            origin_at="mouth",
            symmetry_plane=native_symmetry_plane,
        )
        sphere_points, _theta, _phi = build_sphere_grid_points(
            frame,
            ObservationConfig(
                distance_m=span,
                sphere_grid=sphere_grid,
            ),
        )
        sphere_total = int(sphere_points.shape[0])
        evaluation_points = sphere_points
        if (
            native_symmetry_plane in {"yz", "xz", "yz+xz"}
            and not disable_sphere_symmetry_dedupe
        ):
            evaluation_points, _inverse = build_mirror_evaluation_classes(
                sphere_points,
                native_symmetry_plane,
            )
        sphere_evaluated = int(evaluation_points.shape[0])
        points = np.vstack([points, evaluation_points])
    return (
        np.ascontiguousarray(points.T, dtype=np.float32),
        sphere_total,
        sphere_evaluated,
    )


def _environment_header() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in RELEVANT_ENV}


def _dense_dtype_handshake(environment: dict[str, str | None]) -> str:
    # The method argument checks that a float64-capable helper acknowledged the
    # request; the environment still performs the actual routing. Mirror an
    # explicitly inherited float64 value here without injecting or changing it.
    value = environment.get("HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE")
    return "float64" if value == "float64" else "float32"


def _run_record(
    session: Any,
    frequencies: np.ndarray,
    k_real: np.ndarray,
    neumann_rows: np.ndarray,
    observation_points: np.ndarray,
    *,
    source_tag: int,
    system_order: int,
    repeat_index: int,
    dense_solve_dtype: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_completed_s: float | None = None
    first_callback_has_batch_diagnostics = False

    def _on_case_result(_index: int, result: Any) -> None:
        nonlocal first_completed_s, first_callback_has_batch_diagnostics
        if first_completed_s is None:
            first_completed_s = time.perf_counter() - started
            diagnostics = dict(getattr(result, "diagnostics", {}) or {})
            first_callback_has_batch_diagnostics = "batch" in diagnostics

    systems = session.assemble_solve_evaluate_standard_neumann_batch(
        frequencies,
        k_real,
        neumann_rows,
        observation_points,
        batch_id=f"bench-solve-{repeat_index}",
        operation_id=f"bench-solve-{repeat_index}",
        source_tags=[source_tag],
        impedance_source_tag=source_tag,
        write_surface_pressure=False,
        on_case_result=_on_case_result,
        dense_solve_dtype=dense_solve_dtype,
    )
    wall_s = time.perf_counter() - started
    if len(systems) != len(frequencies):
        raise RuntimeError(
            f"native solve benchmark completed {len(systems)} of {len(frequencies)} cases"
        )

    regular_assembly_s = 0.0
    corrections_s = 0.0
    dense_solve_s = 0.0
    field_s = 0.0
    duffy_s = 0.0
    near_s = 0.0
    saw_duffy_seconds = False
    saw_near_seconds = False
    case_diagnostics: list[dict[str, Any]] = []
    for frequency_hz, system in zip(frequencies, systems):
        diagnostics = dict(getattr(system, "diagnostics", {}) or {})
        assembly_s = float(system.assembly_s)
        regular_s = float(diagnostics.get("regular_assembly_seconds", assembly_s))
        regular_assembly_s += regular_s
        # Native assembly_seconds includes its Duffy and near-quadrature work;
        # regular_assembly_seconds excludes both. Their difference is therefore
        # the complete correction stage even when an older helper omits one of
        # the nested detail reports, and it avoids double-counting assembly in
        # the reported-stage sum.
        corrections_s += assembly_s - regular_s
        dense_solve_s += float(system.dense_solve_s)
        field_s += float(system.field_s)

        duffy = diagnostics.get("duffy_corrections")
        if isinstance(duffy, dict) and "correction_seconds" in duffy:
            duffy_s += float(duffy["correction_seconds"])
            saw_duffy_seconds = True
        near = diagnostics.get("near_quadrature")
        if isinstance(near, dict) and "seconds" in near:
            near_s += float(near["seconds"])
            saw_near_seconds = True

        selected = {"frequency_hz": float(frequency_hz)}
        for key in (
            "ib_aperture_rcond",
            "ib_aperture_dof_count",
            "near_quadrature_level",
            "near_quadrature_kh",
        ):
            if key in diagnostics:
                selected[key] = diagnostics[key]
        case_diagnostics.append(selected)

    stages = {
        "regular_assembly_seconds": regular_assembly_s,
        "duffy_and_near_corrections_seconds": corrections_s,
        "dense_solve_seconds": dense_solve_s,
        "field_seconds": field_s,
    }
    stages_sum_s = sum(stages.values())
    correction_detail: dict[str, float] = {}
    if saw_duffy_seconds:
        correction_detail["duffy_correction_seconds"] = duffy_s
    if saw_near_seconds:
        correction_detail["near_quadrature_seconds"] = near_s

    if first_completed_s is None or first_callback_has_batch_diagnostics:
        # Current helpers stream one case manifest as soon as it finishes. An
        # older helper falls back to invoking the callback from its final batch
        # result, in which case no honest first-case latency is available and
        # the complete batch wall time is the required proxy.
        first_latency_s = wall_s
        first_latency_source = "batch_total_proxy"
    else:
        first_latency_s = first_completed_s
        first_latency_source = "streamed_case_callback"

    record: dict[str, Any] = {
        "repeat_index": repeat_index,
        "warmup": repeat_index == 0,
        "wall_seconds": wall_s,
        "stages": stages,
        "reported_stages_sum_seconds": stages_sum_s,
        "unaccounted_process_serialization_dispatch_seconds": wall_s - stages_sum_s,
        "first_result_latency_seconds": first_latency_s,
        "first_result_latency_source": first_latency_source,
        "system_order_dofs": system_order,
        "field_share_of_wall": field_s / wall_s if wall_s > 0.0 else 0.0,
        "case_diagnostics": case_diagnostics,
    }
    if correction_detail:
        record["correction_detail"] = correction_detail
    return record


def _print_plain(payload: dict[str, Any]) -> None:
    fixture = payload["fixture"]
    print(f"solve benchmark fixture: {fixture['name']}")
    print(f"helper: {payload['runtime']['helper_path']}")
    print(f"frequencies Hz: {', '.join(f'{v:.6g}' for v in payload['frequencies_hz'])}")
    print(f"system order: {fixture['system_order_dofs']} DOFs")
    print("environment:")
    for name, value in payload["environment"].items():
        print(f"  {name}={value if value is not None else '<unset>'}")

    records = [payload["warmup"], *payload["runs"]]
    for record in records:
        label = "warm-up" if record["warmup"] else f"run {record['repeat_index']}"
        print(
            f"{label}: wall={record['wall_seconds']:.6f}s "
            f"stages={record['reported_stages_sum_seconds']:.6f}s "
            "unaccounted(process/serialization/dispatch)="
            f"{record['unaccounted_process_serialization_dispatch_seconds']:.6f}s "
            f"first={record['first_result_latency_seconds']:.6f}s "
            f"({record['first_result_latency_source']})"
        )
        print(
            "  "
            + "  ".join(
                f"{name}={seconds:.6f}s"
                for name, seconds in record["stages"].items()
            )
        )
        print(f"  field_share_of_wall={100.0 * record['field_share_of_wall']:.2f}%")
        for diagnostics in record["case_diagnostics"]:
            values = [
                f"{name}={value}"
                for name, value in diagnostics.items()
                if name != "frequency_hz"
            ]
            if values:
                print(
                    f"  {diagnostics['frequency_hz']:.6g} Hz: "
                    + "  ".join(values)
                )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sys.path.insert(0, str(ROOT))

    from hornlab_metal_bem import load_mesh, native_config
    from hornlab_metal_bem._constants import SPEED_OF_SOUND
    from hornlab_metal_bem.field_traces import _native_field_env_overrides
    from hornlab_metal_bem.mesh import make_pure_function_spaces
    from hornlab_metal_bem.metal import discover_native_runtime
    from hornlab_metal_bem.metal.geometry import build_metal_geometry_buffers
    from hornlab_metal_bem.metal.native import MetalNativeStandardSession
    from hornlab_metal_bem.sweep import _build_neumann_rows

    environment = _environment_header()
    runtime = discover_native_runtime(run_smoke_test=True)
    if not runtime.available:
        reason = "; ".join(runtime.unavailable_reasons)
        skipped = {
            "benchmark": "native_solve_batch",
            "skipped": True,
            "reason": reason,
            "environment": environment,
        }
        if args.json:
            print(json.dumps(skipped, indent=2, sort_keys=True))
        else:
            print(f"solve benchmark skipped: {reason}", file=sys.stderr)
        return 0

    mesh = (
        load_mesh(
            args.mesh,
            native_symmetry_plane=args.native_symmetry_plane,
        )
        if args.mesh is not None
        else _built_in_box()
    )
    source_tag = int(args.source_tag)
    available_tags = {int(value) for value in np.unique(mesh.physical_tags)}
    if source_tag not in available_tags:
        raise ValueError(
            f"source tag {source_tag} is absent from the mesh; available tags: "
            f"{sorted(available_tags)}"
        )

    frequencies = np.geomspace(args.f1, args.f2, args.frequencies, dtype=np.float64)
    k_real = np.ascontiguousarray(
        2.0 * np.pi * frequencies / SPEED_OF_SOUND,
        dtype=np.float32,
    )
    p1_space, dp0_space = make_pure_function_spaces(mesh.grid)
    geometry = build_metal_geometry_buffers(
        mesh.grid,
        mesh.physical_tags,
        p1_space,
        dp0_space,
    )
    # Reuse the wrapper's Neumann builder so the benchmark input has the same
    # velocity/acceleration convention and per-face layout as a public solve.
    # The SolveConfig is not passed to the session, which is important: doing so
    # through sweep._native_env_overrides would replace inherited A/B knobs with
    # config defaults instead of benchmarking the caller's environment.
    input_config = native_config(velocity_sources={source_tag: 1.0})
    neumann_rows = _build_neumann_rows(
        dp0_space,
        mesh.physical_tags,
        frequencies,
        input_config,
        {},
    )
    aperture_tag = getattr(mesh, "coupled_ib_aperture_tag", None)
    fixture_name = str(args.mesh) if args.mesh is not None else "built-in-coupled-box"
    helper_path = runtime.helper_executable_path

    records: list[dict[str, Any]] = []
    with MetalNativeStandardSession.create_session(
        geometry_buffers=geometry,
        symmetry_plane=args.native_symmetry_plane,
        aperture_tag=aperture_tag,
        velocity_source_tags=[source_tag],
        check_open_edges=not args.native_allow_open_rim,
        runtime_status=runtime,
        # Only the field-kernel default is normalized, exactly as the production
        # sweep does: an unset HORNLAB_METAL_BEM_NATIVE_FIELD_MODE would make the
        # helper fall back to the serial CPU reference evaluator, which once
        # reported a 4 s balloon that the Metal kernel evaluates in 13 ms. Every
        # other A/B knob is still inherited from os.environ verbatim.
        extra_env=_native_field_env_overrides(),
    ) as session:
        points, sphere_total, sphere_evaluated = _observation_points(
            mesh,
            source_tag=source_tag,
            sphere_grid=args.sphere_grid,
            native_symmetry_plane=args.native_symmetry_plane,
            disable_sphere_symmetry_dedupe=(
                args.disable_sphere_symmetry_dedupe
            ),
        )
        for repeat_index in range(args.repeat):
            records.append(
                _run_record(
                    session,
                    frequencies,
                    k_real,
                    neumann_rows,
                    points,
                    source_tag=source_tag,
                    system_order=geometry.p1_dof_count,
                    repeat_index=repeat_index,
                    dense_solve_dtype=_dense_dtype_handshake(environment),
                )
            )

    payload = {
        "benchmark": "native_solve_batch",
        "skipped": False,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "runtime": {
            "helper_path": str(helper_path) if helper_path is not None else None,
            "helper_source": runtime.helper_source,
        },
        "environment": environment,
        "fixture": {
            "name": fixture_name,
            "source_tag": source_tag,
            "aperture_tag": int(aperture_tag) if aperture_tag is not None else None,
            "vertices": geometry.n_vertices,
            "triangles": geometry.n_triangles,
            "system_order_dofs": geometry.p1_dof_count,
        },
        "frequency_count": int(frequencies.size),
        "frequencies_hz": frequencies.tolist(),
        "repeat_count": args.repeat,
        "observation": {
            "sphere_grid": list(args.sphere_grid) if args.sphere_grid else None,
            "sphere_targets": sphere_total,
            "sphere_evaluation_targets": sphere_evaluated,
            "native_symmetry_plane": args.native_symmetry_plane,
            "sphere_symmetry_dedupe": bool(
                args.sphere_grid is not None
                and args.native_symmetry_plane in {"yz", "xz", "yz+xz"}
                and not args.disable_sphere_symmetry_dedupe
            ),
        },
        "timing_notes": {
            "reported_stages_sum_seconds": (
                "regular assembly + Duffy/near corrections + dense solve + field"
            ),
            "unaccounted_process_serialization_dispatch_seconds": (
                "wall - reported stages; includes process startup, serialization, "
                "dispatch, and may be negative when native stages overlap"
            ),
            "first_result_latency_seconds": (
                "streamed callback latency, or batch wall time when an older helper "
                "cannot surface a completed case before the batch result"
            ),
        },
        "warmup": records[0],
        "runs": records[1:],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_plain(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark post-solve native field evaluation from retained traces."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from uuid import uuid4

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREQUENCIES = (250.0, 500.0, 1000.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="Gmsh surface mesh to solve")
    parser.add_argument(
        "--frequencies",
        type=float,
        nargs=3,
        metavar=("F1", "F2", "F3"),
        default=DEFAULT_FREQUENCIES,
        help="exactly three benchmark frequencies in Hz",
    )
    parser.add_argument("--source-tag", type=int, default=2)
    parser.add_argument(
        "--symmetry-plane",
        choices=("yz", "xz", "xy", "yz+xz"),
        default=None,
    )
    parser.add_argument(
        "--allow-open-edges",
        action="store_true",
        help="disable the symmetry-cut open-edge check for open shells",
    )
    return parser.parse_args()


def _field_plane(mesh, count: int) -> np.ndarray:
    lower, upper = mesh.info.bounding_box_m
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    extent = upper - lower
    margin = max(float(np.max(extent)), 1.0e-3) * 0.25
    x = np.linspace(lower[0] - margin, upper[0] + margin, count)
    y = np.linspace(lower[1] - margin, upper[1] + margin, count)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    z = np.full_like(xx, upper[2] + max(float(np.max(extent)), 1.0e-3))
    return np.column_stack((xx.ravel(), yy.ravel(), z.ravel()))


def _seconds(value: float) -> str:
    return f"{value:.6f}"


def main() -> int:
    args = _parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("field-trace benchmark skipped: requires Apple Silicon", file=sys.stderr)
        return 0

    sys.path.insert(0, str(ROOT))
    import hornlab_metal_bem as metal_bem
    from hornlab_metal_bem._constants import SPEED_OF_SOUND
    from hornlab_metal_bem.field_traces import _native_field_env_overrides
    from hornlab_metal_bem.mesh import make_pure_function_spaces
    from hornlab_metal_bem.metal import discover_native_runtime
    from hornlab_metal_bem.metal.geometry import (
        _build_metal_geometry_buffers_with_max_edge,
    )
    from hornlab_metal_bem.metal.native import MetalNativeStandardSession
    from hornlab_metal_bem.sweep import _read_complex_f32

    runtime = discover_native_runtime(run_smoke_test=True)
    helper = runtime.helper_executable_path
    if not runtime.available or helper is None or not helper.is_file():
        reason = "; ".join(runtime.unavailable_reasons) or "helper executable missing"
        print(f"field-trace benchmark skipped: {reason}", file=sys.stderr)
        return 0

    load_start = time.perf_counter()
    mesh = metal_bem.load_mesh(
        args.mesh,
        native_symmetry_plane=args.symmetry_plane,
    )
    mesh_load_s = time.perf_counter() - load_start

    frequencies = np.asarray(args.frequencies, dtype=np.float64)
    solve_config = metal_bem.native_config(
        velocity_sources={args.source_tag: 1.0},
        native_symmetry_plane=args.symmetry_plane,
        native_check_open_edges=not args.allow_open_edges,
        return_surface_traces=True,
    )
    solve_start = time.perf_counter()
    solved = metal_bem.solve_frequencies(mesh, frequencies, solve_config)
    solve_s = time.perf_counter() - solve_start
    assert solved.surface_pressure_complex is not None
    assert solved.surface_neumann_complex is not None
    k_real = (2.0 * np.pi * frequencies / SPEED_OF_SOUND).astype(np.float32)

    session_start = time.perf_counter()
    p1_space, dp0_space = make_pure_function_spaces(mesh.grid)
    geometry_buffers, _ = _build_metal_geometry_buffers_with_max_edge(
        mesh.grid,
        mesh.physical_tags,
        p1_space,
        dp0_space,
    )
    session = MetalNativeStandardSession.create_session(
        geometry_buffers=geometry_buffers,
        symmetry_plane=args.symmetry_plane,
        check_open_edges=not args.allow_open_edges,
        runtime_status=runtime,
        extra_env=_native_field_env_overrides(),
    )
    session_build_s = time.perf_counter() - session_start

    rows: list[tuple[str, ...]] = []
    try:
        for grid_size in (32, 128):
            points = _field_plane(mesh, grid_size)
            points_3xn = np.ascontiguousarray(points.T, dtype=np.float32)
            for index, frequency_hz in enumerate(frequencies):
                operation_id = f"bench-field-{grid_size}-{uuid4().hex}"
                helper_start = time.perf_counter()
                field = session.evaluate_standard_exterior(
                    float(frequency_hz),
                    float(k_real[index]),
                    solved.surface_pressure_complex[index],
                    solved.surface_neumann_complex[index],
                    points_3xn,
                    batch_id=f"grid-{grid_size}",
                    operation_id=operation_id,
                )
                helper_wall_s = time.perf_counter() - helper_start
                manifest = json.loads(
                    (
                        session.info.work_dir / operation_id / "field-result.json"
                    ).read_text(encoding="utf-8")
                )
                native_field_s = float(manifest["field_seconds"])

                readback_start = time.perf_counter()
                pressure = _read_complex_f32(
                    field.pressure_real_f32,
                    field.pressure_imag_f32,
                    tuple(field.shape),
                    dtype=np.complex128,
                )
                python_readback_s = time.perf_counter() - readback_start
                if pressure.shape != (points.shape[0],) or not np.all(
                    np.isfinite(pressure)
                ):
                    raise RuntimeError("field benchmark returned invalid pressure")
                rows.append(
                    (
                        f"{grid_size}x{grid_size}",
                        str(points.shape[0]),
                        f"{frequency_hz:.1f}",
                        _seconds(helper_wall_s),
                        _seconds(native_field_s),
                        _seconds(max(0.0, helper_wall_s - native_field_s)),
                        _seconds(python_readback_s),
                    )
                )
    finally:
        session.close()

    print(f"helper: {helper}")
    print(f"mesh load: {_seconds(mesh_load_s)} s")
    print(f"trace solve (3 frequencies): {_seconds(solve_s)} s")
    print(f"session/geometry build: {_seconds(session_build_s)} s")
    headings = (
        "grid",
        "points",
        "Hz",
        "helper+kernel s",
        "native field s",
        "startup/IPC s",
        "Python readback s",
    )
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in rows))
        for index in range(len(headings))
    ]
    print("  ".join(value.rjust(widths[index]) for index, value in enumerate(headings)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.rjust(widths[index]) for index, value in enumerate(row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

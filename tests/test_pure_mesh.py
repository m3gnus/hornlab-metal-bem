from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hornlab_solver
from hornlab_solver import sweep
from hornlab_solver.config import SolveConfig
from hornlab_solver.mesh import make_pure_function_spaces, make_pure_grid


def test_pure_grid_exposes_bempp_shaped_geometry():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
        ],
        dtype=np.int32,
    )

    grid = make_pure_grid(vertices, triangles)

    assert grid.vertices.shape == (3, 4)
    assert grid.elements.shape == (3, 2)
    assert grid.number_of_elements == 2
    np.testing.assert_allclose(grid.volumes, [0.5, 0.5])


def test_pure_function_spaces_use_vertex_p1_and_triangle_dp0():
    vertices = np.eye(3, dtype=np.float64)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    grid = make_pure_grid(vertices, triangles)

    p1, dp0 = make_pure_function_spaces(grid)

    np.testing.assert_array_equal(p1.local2global, [[0, 1, 2]])
    assert p1.global_dof_count == 3
    assert dp0.global_dof_count == 1


def test_native_fallback_converts_to_bempp_at_dispatch_boundary(monkeypatch):
    pure_mesh = SimpleNamespace(grid="pure")
    bempp_mesh = SimpleNamespace(grid="bempp")
    sentinel = object()

    monkeypatch.setattr(sweep, "to_bempp_loaded_mesh", lambda mesh: bempp_mesh)

    def fake_run_sweep_serial(mesh, frequencies, frame, config):
        assert mesh is bempp_mesh
        return sentinel

    monkeypatch.setattr(sweep, "run_sweep_serial", fake_run_sweep_serial)

    result = sweep._run_bempp_fallback(
        pure_mesh,
        np.asarray([1000.0], dtype=np.float64),
        frame=object(),
        config=SolveConfig(),
    )

    assert result is sentinel


def test_solve_loads_pure_grid_before_native_dispatch_with_fallback(monkeypatch):
    loaded = SimpleNamespace(grid="pure", physical_tags=np.asarray([2], dtype=np.int32))
    sentinel = object()
    calls = {}

    def fake_load_mesh(mesh, *, scale, grid_backend):
        calls["mesh"] = mesh
        calls["scale"] = scale
        calls["grid_backend"] = grid_backend
        return loaded

    monkeypatch.setattr(hornlab_solver, "load_mesh", fake_load_mesh)
    monkeypatch.setattr(hornlab_solver, "_resolve_frame", lambda mesh, config: object())
    monkeypatch.setattr(
        sweep,
        "run_sweep_native_metal",
        lambda mesh, frequencies, frame, config: sentinel,
    )

    config = SolveConfig(
        assembly_backend="metal",
        experimental_metal_backend=True,
        metal_backend_fallback="opencl",
    )

    result = hornlab_solver.solve("waveguide.msh", config)

    assert result is sentinel
    assert calls == {
        "mesh": "waveguide.msh",
        "scale": 1.0,
        "grid_backend": "pure",
    }

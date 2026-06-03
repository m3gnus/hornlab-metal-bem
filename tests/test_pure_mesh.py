from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hornlab_metal_bem
from hornlab_metal_bem import sweep
from hornlab_metal_bem.mesh import make_pure_function_spaces, make_pure_grid


def test_pure_grid_exposes_metal_shaped_geometry():
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


def test_solve_loads_pure_grid_before_native_dispatch(monkeypatch):
    loaded = SimpleNamespace(grid="pure", physical_tags=np.asarray([2], dtype=np.int32))
    sentinel = object()
    calls = {}

    def fake_load_mesh(mesh, *, scale):
        calls["mesh"] = mesh
        calls["scale"] = scale
        return loaded

    monkeypatch.setattr(hornlab_metal_bem, "load_mesh", fake_load_mesh)
    monkeypatch.setattr(hornlab_metal_bem, "_resolve_frame", lambda mesh, config: object())
    monkeypatch.setattr(
        sweep,
        "run_sweep_native_metal",
        lambda mesh, frequencies, frame, config: sentinel,
    )

    result = hornlab_metal_bem.solve("waveguide.msh")

    assert result is sentinel
    assert calls == {
        "mesh": "waveguide.msh",
        "scale": 1.0,
    }

from __future__ import annotations

import numpy as np

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

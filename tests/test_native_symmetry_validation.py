from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hornlab_solver.validation.native_symmetry import (
    build_local2global_xy_mirror_orbits,
    build_xy_mirror_orbits,
    classify_orbits_by_size,
    classify_xy_reduced_dofs,
    expand_quarter_mesh_xy,
    expand_reduced_pressure,
    orbit_reduce_matrix_rhs,
    p1_dof_coordinates,
)


def test_expand_quarter_mesh_xy_shares_seams_and_preserves_orbits():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    tags = np.array([2], dtype=np.int32)

    expanded = expand_quarter_mesh_xy(vertices, triangles, tags)

    assert expanded.vertices_nx3.shape == (5, 3)
    assert expanded.triangles_nx3.shape == (4, 3)
    assert expanded.physical_tags.tolist() == [2, 2, 2, 2]

    reduced_grid = SimpleNamespace(
        vertices=vertices.T,
        elements=triangles.T,
    )
    reduced_p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2]], dtype=np.int64),
        global_dof_count=3,
    )
    full_grid = SimpleNamespace(
        vertices=expanded.vertices_nx3.T,
        elements=expanded.triangles_nx3.T,
    )
    full_p1 = SimpleNamespace(
        local2global=expanded.triangles_nx3.astype(np.int64),
        global_dof_count=5,
    )
    reduced_coords = p1_dof_coordinates(reduced_grid, reduced_p1)
    full_coords = p1_dof_coordinates(full_grid, full_p1)
    orbits = build_xy_mirror_orbits(reduced_coords, full_coords)
    local_orbits = build_local2global_xy_mirror_orbits(
        reduced_p1.local2global,
        full_p1.local2global,
    )

    assert [len(orbit) for orbit in orbits] == [1, 2, 2]
    assert [orbit.tolist() for orbit in local_orbits] == [orbit.tolist() for orbit in orbits]
    assert classify_orbits_by_size(local_orbits).tolist() == [
        "double_seam",
        "single_seam",
        "single_seam",
    ]
    assert classify_xy_reduced_dofs(reduced_coords).tolist() == [
        "double_seam",
        "single_seam",
        "single_seam",
    ]

    full_matrix = np.arange(25, dtype=np.float32).reshape(5, 5).astype(np.complex64)
    full_rhs = np.arange(5, dtype=np.float32).astype(np.complex64)
    reduced_matrix, reduced_rhs = orbit_reduce_matrix_rhs(
        full_matrix,
        full_rhs,
        orbits,
    )
    assert reduced_matrix.shape == (3, 3)
    assert reduced_rhs.tolist() == [
        full_rhs[orbits[0]].sum(),
        full_rhs[orbits[1]].sum(),
        full_rhs[orbits[2]].sum(),
    ]

    expanded_pressure = expand_reduced_pressure(
        np.array([1.0, 2.0, 3.0], dtype=np.complex64),
        5,
        orbits,
    )
    assert expanded_pressure[orbits[0]].tolist() == [1.0 + 0.0j]
    assert all(expanded_pressure[idx] == 2.0 + 0.0j for idx in orbits[1])
    assert all(expanded_pressure[idx] == 3.0 + 0.0j for idx in orbits[2])

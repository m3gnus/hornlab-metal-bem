from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import hornlab_metal_bem
from hornlab_metal_bem import sweep
from hornlab_metal_bem.mesh import (
    detect_reduced_symmetry_plane,
    load_mesh,
    make_pure_function_spaces,
    make_pure_grid,
)


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

    def fake_load_mesh(
        mesh,
        *,
        scale,
        validate=True,
        merge_tol=1e-9,
        repair_normals=False,
        native_symmetry_plane=None,
    ):
        calls["mesh"] = mesh
        calls["scale"] = scale
        calls["native_symmetry_plane"] = native_symmetry_plane
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
        "native_symmetry_plane": None,
    }


def test_detect_reduced_symmetry_plane_finds_half_mesh_cut():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.2, 0.3],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )

    assert detect_reduced_symmetry_plane(vertices, triangles) == "yz"


def test_detect_reduced_symmetry_plane_finds_quarter_mesh_cut():
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
            [1, 3, 2],
        ],
        dtype=np.int32,
    )

    assert detect_reduced_symmetry_plane(vertices, triangles) == "yz+xz"


def test_detect_reduced_symmetry_plane_finds_xy_half_mesh_cut():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.3, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )

    assert detect_reduced_symmetry_plane(vertices, triangles) == "xy"


def test_load_mesh_warns_when_reduced_mesh_has_no_native_symmetry(monkeypatch, tmp_path):
    mesh_path = tmp_path / "half.msh"
    mesh_path.write_text("$MeshFormat\n", encoding="utf-8")
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.2, 0.3],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    fake_mesh = SimpleNamespace(
        cells_dict={"triangle": triangles},
        points=vertices,
        cell_data_dict={"gmsh:physical": {"triangle": np.ones(3, dtype=np.int32)}},
    )

    import meshio

    monkeypatch.setattr(meshio, "read", lambda path: fake_mesh)

    with pytest.warns(RuntimeWarning, match="native_symmetry_plane='yz'"):
        load_mesh(mesh_path, validate=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_mesh(mesh_path, validate=False, native_symmetry_plane="yz")
    assert caught == []

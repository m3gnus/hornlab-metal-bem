from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_metal_bem.metal.geometry import (
    MetalGeometryError,
    build_metal_geometry_buffers,
)


def _mock_grid(
    vertices_3xn: np.ndarray | None = None,
    triangles_3xm: np.ndarray | None = None,
) -> SimpleNamespace:
    if vertices_3xn is None:
        vertices_3xn = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if triangles_3xm is None:
        triangles_3xm = np.array(
            [
                [0, 0],
                [1, 2],
                [2, 3],
            ],
            dtype=np.int64,
        )
    return SimpleNamespace(
        vertices=vertices_3xn,
        elements=triangles_3xm,
        number_of_elements=triangles_3xm.shape[1],
    )


def _mock_p1(local2global: np.ndarray | None = None) -> SimpleNamespace:
    if local2global is None:
        local2global = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return SimpleNamespace(
        local2global=local2global,
        global_dof_count=4,
    )


def _mock_dp0(count: int = 2) -> SimpleNamespace:
    return SimpleNamespace(global_dof_count=count)


def test_build_metal_geometry_buffers_exports_scratch_shapes_and_dtypes():
    grid = _mock_grid()
    p1_space = _mock_p1()
    physical_tags = np.array([1, 2], dtype=np.int64)

    buffers = build_metal_geometry_buffers(
        grid,
        physical_tags,
        p1_space,
        _mock_dp0(),
    )

    assert buffers.vertices_3xn_f32.shape == (3, 4)
    assert buffers.vertices_3xn_f32.dtype == np.float32
    assert buffers.vertices_3xn_f32.flags.c_contiguous
    assert buffers.triangles_3xm_i32.shape == (3, 2)
    assert buffers.triangles_3xm_i32.dtype == np.int32
    assert buffers.physical_tags_i32.tolist() == [1, 2]
    assert buffers.p1_local2global_i32.shape == (2, 3)
    assert buffers.p1_local2global_i32.dtype == np.int32
    assert buffers.p1_dof_count == 4
    assert buffers.dp0_dof_count == 2
    assert buffers.n_vertices == 4
    assert buffers.n_triangles == 2

    np.testing.assert_array_equal(
        buffers.triangles_nx3_i32,
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    np.testing.assert_allclose(buffers.triangle_areas_f32, [0.5, 0.5])
    np.testing.assert_allclose(
        buffers.triangle_normals_3xm_f32,
        np.array(
            [
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_metal_geometry_buffers_accepts_tag_column_vector():
    buffers = build_metal_geometry_buffers(
        _mock_grid(),
        np.array([[1], [2]], dtype=np.int32),
        _mock_p1(),
    )

    np.testing.assert_array_equal(
        buffers.physical_tags_i32,
        np.array([1, 2], dtype=np.int32),
    )


def test_build_metal_geometry_buffers_rejects_one_based_triangles():
    one_based_triangles = np.array(
        [
            [1, 1],
            [2, 3],
            [3, 4],
        ],
        dtype=np.int32,
    )

    with pytest.raises(MetalGeometryError, match="zero-based"):
        build_metal_geometry_buffers(
            _mock_grid(triangles_3xm=one_based_triangles),
            np.array([1, 2], dtype=np.int32),
            _mock_p1(),
        )


def test_build_metal_geometry_buffers_rejects_local2global_shape_mismatch():
    with pytest.raises(MetalGeometryError, match="local2global"):
        build_metal_geometry_buffers(
            _mock_grid(),
            np.array([1, 2], dtype=np.int32),
            _mock_p1(np.array([[0, 1, 2]], dtype=np.int32)),
        )


def test_build_metal_geometry_buffers_rejects_one_based_local2global():
    with pytest.raises(MetalGeometryError, match="minimum 0"):
        build_metal_geometry_buffers(
            _mock_grid(),
            np.array([1, 2], dtype=np.int32),
            _mock_p1(np.array([[1, 2, 3], [1, 3, 4]], dtype=np.int32)),
        )


def test_build_metal_geometry_buffers_rejects_physical_tag_count_mismatch():
    with pytest.raises(MetalGeometryError, match="one value per triangle"):
        build_metal_geometry_buffers(
            _mock_grid(),
            np.array([1], dtype=np.int32),
            _mock_p1(),
        )


def test_build_metal_geometry_buffers_rejects_int32_overflow():
    triangles = np.array(
        [
            [0],
            [1],
            [np.iinfo(np.int32).max + 1],
        ],
        dtype=np.int64,
    )

    with pytest.raises(MetalGeometryError, match="int32"):
        build_metal_geometry_buffers(
            _mock_grid(triangles_3xm=triangles),
            np.array([1], dtype=np.int32),
            _mock_p1(np.array([[0, 1, 2]], dtype=np.int32)),
        )


def test_build_metal_geometry_buffers_rejects_zero_area_triangle():
    vertices = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0], [1], [2]], dtype=np.int32)

    with pytest.raises(MetalGeometryError, match="zero area"):
        build_metal_geometry_buffers(
            _mock_grid(vertices, triangles),
            np.array([1], dtype=np.int32),
            _mock_p1(np.array([[0, 1, 2]], dtype=np.int32)),
            _mock_dp0(1),
        )


def test_build_metal_geometry_buffers_rejects_dp0_count_mismatch():
    with pytest.raises(MetalGeometryError, match="dp0_space"):
        build_metal_geometry_buffers(
            _mock_grid(),
            np.array([1, 2], dtype=np.int32),
            _mock_p1(),
            _mock_dp0(3),
        )


def test_build_metal_geometry_buffers_snaps_near_zero_coordinates():
    # Near-plane CAD vertices must land exactly on 0.0 so they cannot fall
    # between Python plane validation (1e-7) and the native helper's 1e-6
    # image-pair coordinate keys; coordinates beyond the tolerance survive.
    vertices = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -5.0e-7],
            [5.0e-7, 2.0e-6, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    buffers = build_metal_geometry_buffers(
        _mock_grid(vertices),
        np.array([1, 2], dtype=np.int32),
        _mock_p1(),
        _mock_dp0(),
    )

    assert buffers.vertices_3xn_f32[2, 0] == 0.0
    assert buffers.vertices_3xn_f32[1, 3] == 0.0
    assert buffers.vertices_3xn_f32[2, 1] == np.float32(2.0e-6)
    # The caller's array is untouched.
    assert vertices[2, 0] == 5.0e-7

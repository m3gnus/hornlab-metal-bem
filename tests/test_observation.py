"""Unit tests for hornlab_metal_bem.observation — no bempp needed.

Tests custom_points validation, infer_frame geometry heuristics,
and observation point construction.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from hornlab_metal_bem.config import ObservationConfig
from hornlab_metal_bem.observation import (
    ObservationFrame,
    build_observation_points,
    infer_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(**kwargs) -> ObservationFrame:
    defaults = dict(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.array([0.0, 0.0, 0.0]),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=np.array([0.0, 0.0, 1.0]),
        source_center=np.array([0.0, 0.0, 0.0]),
    )
    defaults.update(kwargs)
    return ObservationFrame(**defaults)


def _mock_grid(vertices: np.ndarray, elements: np.ndarray):
    """Create a mock grid with (3, N_verts) vertices and (3, N_elems) elements."""
    grid = MagicMock()
    grid.vertices = vertices
    grid.elements = elements
    grid.number_of_elements = elements.shape[1]
    return grid


# ---------------------------------------------------------------------------
# build_observation_points — custom_points happy path
# ---------------------------------------------------------------------------

class TestCustomPointsValid:

    def test_single_plane_returns_correct_shape(self):
        pts = np.random.randn(10, 3)
        cfg = ObservationConfig(
            planes=["horizontal"],
            custom_points={"horizontal": pts},
            angle_min_deg=0.0,
            angle_max_deg=180.0,
            angle_count=10,
        )
        frame = _make_frame()
        points, angles = build_observation_points(frame, cfg)
        assert points.shape == (1, 10, 3)
        assert len(angles) == 10
        np.testing.assert_allclose(points[0], pts)

    def test_two_planes_same_count(self):
        h_pts = np.random.randn(7, 3)
        v_pts = np.random.randn(7, 3)
        cfg = ObservationConfig(
            planes=["horizontal", "vertical"],
            custom_points={"horizontal": h_pts, "vertical": v_pts},
            angle_count=7,
        )
        frame = _make_frame()
        points, angles = build_observation_points(frame, cfg)
        assert points.shape == (2, 7, 3)
        assert len(angles) == 7

    def test_angle_count_overridden_when_mismatched(self):
        pts = np.random.randn(15, 3)
        cfg = ObservationConfig(
            planes=["horizontal"],
            custom_points={"horizontal": pts},
            angle_count=37,  # differs from actual 15
        )
        frame = _make_frame()
        points, angles = build_observation_points(frame, cfg)
        assert points.shape == (1, 15, 3)
        assert len(angles) == 15
        assert angles[0] == cfg.angle_min_deg
        assert angles[-1] == cfg.angle_max_deg


# ---------------------------------------------------------------------------
# build_observation_points — custom_points validation errors
# ---------------------------------------------------------------------------

class TestCustomPointsValidation:

    def test_missing_plane_raises(self):
        cfg = ObservationConfig(
            planes=["horizontal", "vertical"],
            custom_points={"horizontal": np.zeros((5, 3))},
        )
        frame = _make_frame()
        with pytest.raises(ValueError, match="custom_points missing plane 'vertical'"):
            build_observation_points(frame, cfg)

    def test_wrong_ndim_raises(self):
        cfg = ObservationConfig(
            planes=["horizontal"],
            custom_points={"horizontal": np.zeros(15)},  # 1D
        )
        frame = _make_frame()
        with pytest.raises(ValueError, match=r"must be \(N, 3\)"):
            build_observation_points(frame, cfg)

    def test_wrong_columns_raises(self):
        cfg = ObservationConfig(
            planes=["horizontal"],
            custom_points={"horizontal": np.zeros((10, 2))},  # (N, 2)
        )
        frame = _make_frame()
        with pytest.raises(ValueError, match=r"must be \(N, 3\)"):
            build_observation_points(frame, cfg)

    def test_mismatched_plane_sizes_raises(self):
        cfg = ObservationConfig(
            planes=["horizontal", "vertical"],
            custom_points={
                "horizontal": np.zeros((10, 3)),
                "vertical": np.zeros((8, 3)),
            },
        )
        frame = _make_frame()
        with pytest.raises(ValueError, match="same number of points"):
            build_observation_points(frame, cfg)


# ---------------------------------------------------------------------------
# build_observation_points — standard polar arc construction
# ---------------------------------------------------------------------------

class TestStandardObservationPoints:

    def test_on_axis_point_is_along_frame_axis(self):
        frame = _make_frame()
        cfg = ObservationConfig(
            planes=["horizontal"],
            angle_min_deg=0.0,
            angle_max_deg=180.0,
            angle_count=3,
            distance_m=1.0,
        )
        points, angles = build_observation_points(frame, cfg)
        # angle=0 should be along +axis from origin
        on_axis = points[0, 0]
        expected = frame.origin + 1.0 * frame.axis
        np.testing.assert_allclose(on_axis, expected, atol=1e-12)

    def test_angle_count_matches_output(self):
        frame = _make_frame()
        cfg = ObservationConfig(planes=["vertical"], angle_count=91)
        points, angles = build_observation_points(frame, cfg)
        assert points.shape == (1, 91, 3)
        assert len(angles) == 91

    def test_unknown_plane_raises(self):
        frame = _make_frame()
        cfg = ObservationConfig(planes=["foobar"])
        with pytest.raises(ValueError, match="Unknown plane"):
            build_observation_points(frame, cfg)


# ---------------------------------------------------------------------------
# infer_frame — enclosed geometry detection
# ---------------------------------------------------------------------------

def _build_simple_horn_mesh(source_z: float, mouth_z: float):
    """Build a minimal triangulated mesh with source elements at source_z
    and body elements spanning to mouth_z.

    Returns (vertices_3xN, elements_3xM, physical_tags).
    Source elements get tag=2, body gets tag=1.
    """
    # Source: a small square at z=source_z, facing +z
    src_verts = np.array([
        [-0.01, -0.01, source_z],
        [0.01, -0.01, source_z],
        [0.01, 0.01, source_z],
        [-0.01, 0.01, source_z],
    ])
    # Body: vertices at mouth end
    body_verts = np.array([
        [-0.05, -0.05, mouth_z],
        [0.05, -0.05, mouth_z],
        [0.05, 0.05, mouth_z],
        [-0.05, 0.05, mouth_z],
    ])
    # Connect them with some body triangles along the side
    mid_z = (source_z + mouth_z) / 2
    mid_verts = np.array([
        [-0.03, -0.03, mid_z],
        [0.03, -0.03, mid_z],
        [0.03, 0.03, mid_z],
        [-0.03, 0.03, mid_z],
    ])

    vertices = np.vstack([src_verts, mid_verts, body_verts])  # 12 vertices

    # Source triangles (at z=source_z, normals pointing +z)
    # Winding order so cross product points +z
    src_elems = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    # Body triangles (connecting mid to mouth)
    body_elems = np.array([
        [4, 5, 9],
        [5, 6, 10],
        [6, 7, 11],
        [7, 4, 8],
    ])

    elements = np.vstack([src_elems, body_elems])

    tags = np.array([2, 2, 1, 1, 1, 1], dtype=np.int32)

    return vertices.T, elements.T, tags


class TestInferFrameEnclosed:

    def test_source_at_min_uses_normal_direction(self):
        """Source near min of projection span — normal already forward."""
        verts, elems, tags = _build_simple_horn_mesh(source_z=0.0, mouth_z=0.3)
        grid = _mock_grid(verts, elems)

        frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")

        # axis should point from source toward mouth (+z direction)
        assert frame.axis[2] > 0.9

    def test_source_at_max_flips_axis(self):
        """Source near max of projection span — axis should be flipped."""
        verts, elems, tags = _build_simple_horn_mesh(source_z=0.3, mouth_z=0.0)
        grid = _mock_grid(verts, elems)

        frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")

        # Source is at z=0.3 (max), mouth at z=0.0 (min). Axis should
        # point from source toward mouth = -z direction.
        assert frame.axis[2] < -0.9

    def test_enclosed_geometry_trusts_source_normal(self):
        """Source at midpoint of span (enclosed) — trusts source normal."""
        # Source at z=0.15 with mesh spanning 0.0 to 0.3
        # Source sits at 50% of span, both source_from_min and source_from_max > 0.25
        src_verts = np.array([
            [-0.01, -0.01, 0.15],
            [0.01, -0.01, 0.15],
            [0.01, 0.01, 0.15],
            [-0.01, 0.01, 0.15],
        ])
        # Enclosure walls from z=0.0 to z=0.3
        wall_lo = np.array([
            [-0.05, -0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.05, 0.05, 0.0],
            [-0.05, 0.05, 0.0],
        ])
        wall_hi = np.array([
            [-0.05, -0.05, 0.3],
            [0.05, -0.05, 0.3],
            [0.05, 0.05, 0.3],
            [-0.05, 0.05, 0.3],
        ])

        vertices = np.vstack([src_verts, wall_lo, wall_hi])  # 12 verts

        # Source tris: winding so normal points +z
        src_elems = np.array([[0, 1, 2], [0, 2, 3]])
        # Body tris at extremes
        body_elems = np.array([
            [4, 5, 6], [6, 7, 4],
            [8, 9, 10], [10, 11, 8],
        ])
        elements = np.vstack([src_elems, body_elems])
        tags = np.array([2, 2, 1, 1, 1, 1], dtype=np.int32)

        grid = _mock_grid(vertices.T, elements.T)
        frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")

        # Enclosed: should trust source normal (+z)
        assert frame.axis[2] > 0.9

    def test_no_source_tag_raises(self):
        verts, elems, tags = _build_simple_horn_mesh(0.0, 0.3)
        grid = _mock_grid(verts, elems)
        tags_no_source = np.ones_like(tags)  # all tag=1

        with pytest.raises(ValueError, match="No elements with tag 99"):
            infer_frame(grid, tags_no_source, source_tag=99)


# ---------------------------------------------------------------------------
# infer_frame — area-weighted normal correctness
# ---------------------------------------------------------------------------

class TestInferFrameAreaWeighting:

    def test_large_triangle_dominates_normal(self):
        """A single large triangle should dominate the average normal
        over many small triangles, thanks to area weighting via np.sum."""
        # Large triangle at z=0 with normal +z, area ~1
        large = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        # Two tiny triangles with normal -z, area ~0.0001 each
        tiny1 = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.01, 0.0, 0.0],
        ])
        tiny2 = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
        ])

        # The large triangle: cross((1,0,0)-(0,0,0), (0,1,0)-(0,0,0)) = (0,0,1)
        # tiny1: cross((0,0.01,0), (0.01,0,0)) = (0,0,-0.0001)
        # tiny2: cross((0.01,0,0), (0,0.01,0)) = (0,0,0.0001)
        # Net sum should strongly point +z

        vertices = np.vstack([large, tiny1, tiny2])  # 9 verts

        # All source elements
        elements = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        tags = np.array([2, 2, 2], dtype=np.int32)

        # Need a body element to give the mesh some extent
        body_vert = np.array([[0.0, 0.0, 0.5]])
        vertices = np.vstack([vertices, body_vert])  # 10 verts
        body_elem = np.array([[0, 1, 9]])  # random body tri
        elements = np.vstack([elements, body_elem])
        tags = np.append(tags, 1)

        grid = _mock_grid(vertices.T, elements.T)
        frame = infer_frame(grid, tags, source_tag=2)

        # Large triangle has normal +z and dominates; axis should be +z
        assert frame.axis[2] > 0.99


# ---------------------------------------------------------------------------
# infer_frame — origin_at selection
# ---------------------------------------------------------------------------

class TestInferFrameOrigin:

    def test_origin_mouth(self):
        verts, elems, tags = _build_simple_horn_mesh(0.0, 0.3)
        grid = _mock_grid(verts, elems)
        frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")
        # mouth_center should be near z=0.3
        assert frame.origin[2] > 0.2
        np.testing.assert_allclose(frame.origin, frame.mouth_center)

    def test_origin_throat(self):
        verts, elems, tags = _build_simple_horn_mesh(0.0, 0.3)
        grid = _mock_grid(verts, elems)
        frame = infer_frame(grid, tags, source_tag=2, origin_at="throat")
        # source_center should be near z=0
        assert abs(frame.origin[2]) < 0.05
        np.testing.assert_allclose(frame.origin, frame.source_center)

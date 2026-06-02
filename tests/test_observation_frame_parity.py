"""Parity tests for ``hornlab_solver.observation.infer_frame``.

Exercises the cases WG's ``infer_observation_frame`` historically
handled "more robustly":

- enclosed waveguide (source disc sits in the middle of the mesh)
- freestanding horn (source at one extreme)
- BIGMEH-style cabinet with multiple source-tagged elements
- mixed source-element winding (sign-aligned normal sum)
- defensive handling of stale element indices
- symmetry-plane projection (yz / xy)

These are the behaviours the WG-preference fallback in
``hornlab_bridge._infer_frame_from_wg`` and
``Optimizer-Dashboard/bem_optimizer/solver/pipeline.py`` used to import
from WG. They now live in canonical ``infer_frame``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from hornlab_solver.observation import infer_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_grid(vertices: np.ndarray, elements: np.ndarray):
    grid = MagicMock()
    grid.vertices = vertices  # (3, N)
    grid.elements = elements  # (3, M)
    grid.number_of_elements = elements.shape[1]
    return grid


# ---------------------------------------------------------------------------
# Freestanding horn — source at one extreme of the span
# ---------------------------------------------------------------------------

def test_freestanding_horn_axis_points_from_throat_to_mouth():
    """Classic horn: source disc at z=0, mouth at z=1. Forward axis = +z."""
    # Source disc (4 verts, 2 triangles, normal +z)
    src_verts = np.array([
        [-0.02, -0.02, 0.0], [0.02, -0.02, 0.0],
        [0.02, 0.02, 0.0], [-0.02, 0.02, 0.0],
    ])
    # Horn body extending forward to z=1
    body_verts = np.array([
        [-0.2, -0.2, 1.0], [0.2, -0.2, 1.0],
        [0.2, 0.2, 1.0], [-0.2, 0.2, 1.0],
    ])
    vertices = np.vstack([src_verts, body_verts])
    src_elems = np.array([[0, 1, 2], [0, 2, 3]])
    body_elems = np.array([[4, 5, 6], [4, 6, 7]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")

    # Axis points strongly +z
    assert frame.axis[2] > 0.9
    # Mouth origin near z=1, source centre near z=0
    assert frame.origin[2] > 0.8
    assert abs(frame.source_center[2]) < 0.05


# ---------------------------------------------------------------------------
# Enclosed waveguide — source disc in the middle of the mesh
# ---------------------------------------------------------------------------

def test_enclosed_waveguide_trusts_source_normal_over_extent():
    """When ``enc_depth > 2 * horn_length`` the source sits ~midway
    between the horn mouth and the enclosure back wall. The extent
    heuristic would flip the axis; the canonical implementation must
    trust the source normal instead.
    """
    # Source disc at z=0.5 with normal +z
    src_verts = np.array([
        [-0.02, -0.02, 0.5], [0.02, -0.02, 0.5],
        [0.02, 0.02, 0.5], [-0.02, 0.02, 0.5],
    ])
    # Horn mouth ahead at z=0.7
    mouth_verts = np.array([
        [-0.2, -0.2, 0.7], [0.2, -0.2, 0.7],
        [0.2, 0.2, 0.7], [-0.2, 0.2, 0.7],
    ])
    # Enclosure back wall far behind at z=0.0
    back_verts = np.array([
        [-0.3, -0.3, 0.0], [0.3, -0.3, 0.0],
        [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0],
    ])
    vertices = np.vstack([src_verts, mouth_verts, back_verts])

    src_elems = np.array([[0, 1, 2], [0, 2, 3]])
    mouth_elems = np.array([[4, 5, 6], [4, 6, 7]])
    back_elems = np.array([[8, 9, 10], [8, 10, 11]])
    elements = np.vstack([src_elems, mouth_elems, back_elems])
    tags = np.array([2, 2, 1, 1, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")

    # Source normal says +z; back wall at z=0 is further than mouth at z=0.7,
    # so the naive extent test (mouth_at_max) would point the axis at -z.
    # Canonical implementation must trust the normal.
    assert frame.axis[2] > 0.9


# ---------------------------------------------------------------------------
# BIGMEH cabinet — multiple source-tagged elements (multi-driver)
# ---------------------------------------------------------------------------

def test_bigmeh_cabinet_multiple_source_elements():
    """A BIGMEH-style cabinet may carry many source-tagged elements
    across multiple drivers. The area-weighted normal sum should still
    converge to the cabinet-forward axis.
    """
    # Two driver discs, both facing +y, at slightly different positions
    src_disc_1 = np.array([
        [-0.05, 0.0, -0.1], [0.05, 0.0, -0.1],
        [0.05, 0.0, 0.1], [-0.05, 0.0, 0.1],
    ])
    src_disc_2 = np.array([
        [-0.05, 0.0, -0.4], [0.05, 0.0, -0.4],
        [0.05, 0.0, -0.2], [-0.05, 0.0, -0.2],
    ])
    # Cabinet front face at y=0.5
    front_verts = np.array([
        [-0.3, 0.5, -0.5], [0.3, 0.5, -0.5],
        [0.3, 0.5, 0.5], [-0.3, 0.5, 0.5],
    ])
    vertices = np.vstack([src_disc_1, src_disc_2, front_verts])

    # Source tris (winding chosen so normals point +y)
    src_elems = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
    ])
    body_elems = np.array([[8, 9, 10], [8, 10, 11]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 2, 2, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")

    # Axis should point +y toward the cabinet front
    assert frame.axis[1] > 0.9


# ---------------------------------------------------------------------------
# Mixed source winding — sign-aligned normal sum must not cancel
# ---------------------------------------------------------------------------

def test_mixed_winding_does_not_cancel_axis():
    """If some source triangles are wound CW and others CCW (legacy gmsh
    quirk), the raw sum of cross-products would partially cancel. WG's
    robust path sign-aligns normals into one hemisphere first.
    """
    # Four source tris at z=0; two with +z normal, two with -z normal
    base = np.array([
        [-0.01, -0.01, 0.0], [0.01, -0.01, 0.0],
        [0.01, 0.01, 0.0], [-0.01, 0.01, 0.0],
    ])
    body_vert = np.array([[0.0, 0.0, 0.5]])
    vertices = np.vstack([base, body_vert])

    # Tris 0-1: CCW winding → normal +z
    # Tris 2-3: CW winding → normal -z
    src_elems = np.array([
        [0, 1, 2], [0, 2, 3],   # +z
        [0, 2, 1], [0, 3, 2],   # -z
    ])
    body_elem = np.array([[0, 1, 4]])
    elements = np.vstack([src_elems, body_elem])
    tags = np.array([2, 2, 2, 2, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")

    # With sign-alignment the axis magnitude must be well-defined;
    # source-at-min ⇒ axis +z; body vert pulls mouth to +z.
    axis_norm = float(np.linalg.norm(frame.axis))
    assert abs(axis_norm - 1.0) < 1e-6
    assert frame.axis[2] > 0.9


# ---------------------------------------------------------------------------
# Defensive: stale element indices
# ---------------------------------------------------------------------------

def test_stale_element_indices_are_skipped():
    """Legacy meshes occasionally carry element rows that index past the
    vertex array. The canonical implementation must drop these rows
    rather than IndexError.
    """
    src_verts = np.array([
        [-0.02, -0.02, 0.0], [0.02, -0.02, 0.0],
        [0.02, 0.02, 0.0], [-0.02, 0.02, 0.0],
    ])
    body_verts = np.array([
        [-0.2, -0.2, 1.0], [0.2, -0.2, 1.0],
    ])
    vertices = np.vstack([src_verts, body_verts])

    # First two source tris valid; third indexes a non-existent vertex 99.
    src_elems = np.array([
        [0, 1, 2], [0, 2, 3], [0, 1, 99],
    ])
    body_elems = np.array([[4, 5, 0]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 2, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    # Should not raise, should converge to +z axis from the valid tris.
    frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")
    assert frame.axis[2] > 0.9


# ---------------------------------------------------------------------------
# Symmetry-plane projection
# ---------------------------------------------------------------------------

def test_yz_symmetry_projects_origin_x_to_zero():
    """Half-mesh in X>=0 (yz symmetry) — origin x-coord must collapse
    to 0 so observation points sit on the symmetry plane.
    """
    # Source at X>0 (half mesh)
    src_verts = np.array([
        [0.05, -0.02, 0.0], [0.15, -0.02, 0.0],
        [0.15, 0.02, 0.0], [0.05, 0.02, 0.0],
    ])
    mouth_verts = np.array([
        [0.05, -0.2, 1.0], [0.25, -0.2, 1.0],
        [0.25, 0.2, 1.0], [0.05, 0.2, 1.0],
    ])
    vertices = np.vstack([src_verts, mouth_verts])
    src_elems = np.array([[0, 1, 2], [0, 2, 3]])
    body_elems = np.array([[4, 5, 6], [4, 6, 7]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)

    frame_default = infer_frame(grid, tags, source_tag=2, origin_at="mouth")
    frame_yz = infer_frame(
        grid, tags, source_tag=2, origin_at="mouth", symmetry_plane="yz",
    )

    # Without symmetry projection, origin.x > 0
    assert frame_default.origin[0] > 0.0
    # With yz symmetry, origin.x = 0
    assert abs(frame_yz.origin[0]) < 1e-12
    # Other coords preserved
    np.testing.assert_allclose(frame_yz.origin[1:], frame_default.origin[1:])


def test_xy_symmetry_projects_origin_z_to_zero():
    """Half-mesh in Z>=0 (xy symmetry)."""
    src_verts = np.array([
        [-0.02, 0.0, 0.05], [0.02, 0.0, 0.05],
        [0.02, 0.0, 0.15], [-0.02, 0.0, 0.15],
    ])
    mouth_verts = np.array([
        [-0.2, 1.0, 0.05], [0.2, 1.0, 0.05],
        [0.2, 1.0, 0.25], [-0.2, 1.0, 0.25],
    ])
    vertices = np.vstack([src_verts, mouth_verts])
    src_elems = np.array([[0, 1, 2], [0, 2, 3]])
    body_elems = np.array([[4, 5, 6], [4, 6, 7]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame_default = infer_frame(grid, tags, source_tag=2, origin_at="mouth")
    frame_xy = infer_frame(
        grid, tags, source_tag=2, origin_at="mouth", symmetry_plane="xy",
    )

    assert frame_default.origin[2] > 0.0
    assert abs(frame_xy.origin[2]) < 1e-12
    np.testing.assert_allclose(frame_xy.origin[:2], frame_default.origin[:2])


def test_symmetry_plane_none_is_passthrough():
    """``symmetry_plane=None`` must return the same origin as omitting it."""
    src_verts = np.array([
        [0.05, -0.02, 0.0], [0.15, -0.02, 0.0],
        [0.15, 0.02, 0.0], [0.05, 0.02, 0.0],
    ])
    mouth_verts = np.array([
        [0.05, -0.2, 1.0], [0.25, -0.2, 1.0],
        [0.25, 0.2, 1.0], [0.05, 0.2, 1.0],
    ])
    vertices = np.vstack([src_verts, mouth_verts])
    src_elems = np.array([[0, 1, 2], [0, 2, 3]])
    body_elems = np.array([[4, 5, 6], [4, 6, 7]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame_default = infer_frame(grid, tags, source_tag=2, origin_at="mouth")
    frame_none = infer_frame(
        grid, tags, source_tag=2, origin_at="mouth", symmetry_plane=None,
    )

    np.testing.assert_allclose(frame_none.origin, frame_default.origin)

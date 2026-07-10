from __future__ import annotations

import numpy as np
import pytest

from hornlab_metal_bem.mesh import (
    MeshError,
    _merge_duplicate_vertices,
    _signed_mesh_volume_indicator,
    _validate_outward_normals,
)


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    outward_tris = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    return verts, outward_tris


def test_validate_outward_normals_accepts_canonical_winding():
    verts, tris = _tetrahedron()

    _validate_outward_normals(verts, tris)

    assert _signed_mesh_volume_indicator(verts, tris) > 0


def test_validate_outward_normals_rejects_inward_winding_by_default():
    verts, outward = _tetrahedron()
    inward = outward[:, [0, 2, 1]].copy()

    with pytest.raises(MeshError, match="Canonical meshes"):
        _validate_outward_normals(verts, inward)

    assert _signed_mesh_volume_indicator(verts, inward) < 0


def test_validate_outward_normals_repairs_only_when_explicit():
    verts, outward = _tetrahedron()
    inward = outward[:, [0, 2, 1]].copy()

    _validate_outward_normals(verts, inward, repair=True)

    assert _signed_mesh_volume_indicator(verts, inward) > 0


def test_open_surface_winding_verdict_is_translation_invariant():
    verts = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    translated = verts + np.array([0.0, 0.0, -2.0])
    assert _signed_mesh_volume_indicator(verts, tris) > 0.0
    assert _signed_mesh_volume_indicator(translated, tris) < 0.0

    original = tris.copy()
    _validate_outward_normals(verts, tris, repair=True)
    _validate_outward_normals(translated, tris, repair=True)
    np.testing.assert_array_equal(tris, original)


def test_duplicate_merge_uses_actual_euclidean_distance():
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    farther_than_tol = np.array(
        [[0.49, 0.49, 0.49], [-0.49, -0.49, -0.49], [5.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    merged_verts, merged_tris, count = _merge_duplicate_vertices(
        farther_than_tol, triangles, 1.0
    )
    assert count == 0
    assert len(merged_verts) == 3
    np.testing.assert_array_equal(merged_tris, triangles)

    closer_than_tol = np.array(
        [[0.49, 0.0, 0.0], [0.51, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    merged_verts, merged_tris, count = _merge_duplicate_vertices(
        closer_than_tol, triangles, 1.0
    )
    assert count == 1
    assert len(merged_verts) == 2
    assert merged_tris[0, 0] == merged_tris[0, 1]

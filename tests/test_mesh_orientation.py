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


def _reference_merge_duplicate_vertices(
    verts: np.ndarray,
    tris: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pre-cKDTree spatial-hash merger, retained as an equivalence oracle."""
    cells = np.floor(verts / tol).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, cells)):
        buckets.setdefault(key, []).append(index)

    parent = np.arange(len(verts), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    tol_sq = float(tol) ** 2
    for key, indices in buckets.items():
        neighbours = [
            neighbour
            for dx, dy, dz in offsets
            for neighbour in buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), ())
        ]
        for left in indices:
            for right in neighbours:
                if right <= left:
                    continue
                delta = verts[right] - verts[left]
                if float(delta @ delta) > tol_sq:
                    continue
                root_left = find(left)
                root_right = find(right)
                if root_left != root_right:
                    parent[max(root_left, root_right)] = min(root_left, root_right)

    roots = np.fromiter(
        (find(index) for index in range(len(verts))),
        dtype=np.int64,
        count=len(verts),
    )
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    if len(unique_roots) == len(verts):
        return verts, tris, 0
    return (
        verts[unique_roots],
        inverse[tris].astype(np.int32, copy=False),
        len(verts) - len(unique_roots),
    )


def test_duplicate_merge_matches_spatial_hash_reference_on_edge_fixtures():
    """The accelerated pair search preserves the hardened merge contract."""
    vertices = np.array(
        [
            [30.9, 0.0, 0.0],  # transitive chain, deliberately interleaved
            [10.49, 10.49, 10.49],  # diagonal neighbour-cell non-merge
            [0.0, 0.0, 0.0],  # IEEE-boundary pair at distance exactly one
            [40.0, 0.0, 0.0],  # exact duplicate
            [21.0, 0.0, 0.0],  # close pair across cells
            [31.8, 0.0, 0.0],
            [-0.6369326152038236, -0.7154417306587971, -0.28715844706635957],
            [40.0, 0.0, 0.0],
            [20.49, 0.0, 0.0],
            [9.51, 9.51, 9.51],
            [30.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [[2, 0, 4], [8, 7, 5], [1, 9, 3], [10, 6, 4]],
        dtype=np.int32,
    )

    expected = _reference_merge_duplicate_vertices(vertices, triangles, 1.0)
    actual = _merge_duplicate_vertices(vertices, triangles, 1.0)

    assert actual[2] == expected[2]
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])

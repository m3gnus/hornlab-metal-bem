from __future__ import annotations

import warnings

import numpy as np
import pytest

from hornlab_metal_bem.mesh import (
    MeshError,
    _merge_duplicate_vertices,
    _signed_mesh_volume_indicator,
    _validate_outward_normals,
    _warn_if_inverted_open_shell,
    open_shell_bore_alignment,
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


def _cone_horn(
    n_phi: int = 24,
    n_axial: int = 6,
    r_throat: float = 0.0127,
    r_mouth: float = 0.15,
    length: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed conical horn: wall (tag 1), throat cap (tag 2), mouth cap (tag 3).

    Wound outward, so the solid's exterior is the positive side. The bare
    variants below are derived from it rather than wound by hand, which keeps
    "into the bore" a consequence of the construction instead of an assertion.
    """
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    radii = r_throat + (r_mouth - r_throat) * np.arange(n_axial + 1) / n_axial
    z = length * np.arange(n_axial + 1) / n_axial
    verts = np.array(
        [
            [radius * np.cos(angle), radius * np.sin(angle), height]
            for radius, height in zip(radii, z, strict=True)
            for angle in phi
        ],
        dtype=np.float64,
    )
    throat_centre = len(verts)
    mouth_centre = throat_centre + 1
    verts = np.vstack([verts, [0.0, 0.0, 0.0], [0.0, 0.0, length]])

    def node(ring: int, k: int) -> int:
        return ring * n_phi + (k % n_phi)

    tris: list[list[int]] = []
    tags: list[int] = []
    for ring in range(n_axial):
        for k in range(n_phi):
            a, b = node(ring, k), node(ring, k + 1)
            c, d = node(ring + 1, k + 1), node(ring + 1, k)
            tris += [[a, b, c], [a, c, d]]
            tags += [1, 1]
    for k in range(n_phi):
        tris.append([throat_centre, node(0, k + 1), node(0, k)])
        tags.append(2)
        tris.append([mouth_centre, node(n_axial, k), node(n_axial, k + 1)])
        tags.append(3)
    return (
        verts,
        np.asarray(tris, dtype=np.int32),
        np.asarray(tags, dtype=np.int32),
    )


def _bare_cone_horn(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop the mouth cap and reverse: wall and throat cap now face the bore."""
    verts, tris, tags = _cone_horn(**kwargs)
    keep = tags != 3
    return verts, tris[keep][:, [0, 2, 1]].copy(), tags[keep].copy()


def test_open_shell_bore_alignment_separates_the_two_windings():
    verts, tris, tags = _bare_cone_horn()

    assert open_shell_bore_alignment(verts, tris, tags) == 1.0

    wall = tags == 1
    inverted = tris.copy()
    inverted[wall] = inverted[wall][:, [0, 2, 1]]
    assert open_shell_bore_alignment(verts, inverted, tags) == 0.0


def test_open_shell_bore_alignment_survives_rigid_motion():
    """The measure takes both references from the mesh, so it must be intrinsic."""
    verts, tris, tags = _bare_cone_horn()
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    moved = verts @ rotation.T + np.array([0.4, -1.3, 2.2])

    assert open_shell_bore_alignment(moved, tris, tags) == 1.0


def test_inverted_open_shell_warns():
    verts, tris, tags = _bare_cone_horn()
    tris[tags == 1] = tris[tags == 1][:, [0, 2, 1]]

    with pytest.warns(RuntimeWarning, match="acoustic domain"):
        _warn_if_inverted_open_shell(
            verts, tris, tags, coupled_ib_aperture_tag=None,
        )


def test_correct_open_shell_does_not_warn():
    verts, tris, tags = _bare_cone_horn()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_inverted_open_shell(
            verts, tris, tags, coupled_ib_aperture_tag=None,
        )


def test_closed_horn_is_not_judged_as_an_open_shell():
    """A closed body's outer wall legitimately faces away from the bore."""
    verts, tris, tags = _cone_horn()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_inverted_open_shell(
            verts, tris, tags, coupled_ib_aperture_tag=None,
        )


def test_mirror_reduced_closed_horn_is_not_judged_as_an_open_shell():
    """Reduced bodies are open only on their cut plane; skip, do not warn."""
    verts, tris, tags = _cone_horn()
    keep = np.all(verts[tris][:, :, 1] >= -1.0e-12, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_inverted_open_shell(
            verts, tris[keep], tags[keep], coupled_ib_aperture_tag=None,
        )


def test_coupled_ib_meshes_keep_their_own_contract():
    verts, tris, tags = _bare_cone_horn()
    tris[tags == 1] = tris[tags == 1][:, [0, 2, 1]]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_inverted_open_shell(
            verts, tris, tags, coupled_ib_aperture_tag=12,
        )

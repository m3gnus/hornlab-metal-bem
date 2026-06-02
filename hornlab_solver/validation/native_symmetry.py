from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ExpandedMesh:
    vertices_nx3: NDArray[np.float64]
    triangles_nx3: NDArray[np.int64]
    physical_tags: NDArray[np.int32]
    triangle_image_signs: NDArray[np.int32]
    quarter_to_full_vertices: dict[tuple[int, int, int], NDArray[np.int64]]


def expand_quarter_mesh_xy(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.integer],
    physical_tags: NDArray[np.integer],
    *,
    tolerance: float = 1.0e-9,
) -> ExpandedMesh:
    """Expand an X>=0, Y>=0 reduced mesh into four X/Y mirror images.

    Coincident seam vertices are shared. Triangle winding is reversed for
    exactly one reflection so the mirrored normals preserve the source mesh's
    outward orientation.
    """
    vertices = np.asarray(vertices_nx3, dtype=np.float64)
    triangles = np.asarray(triangles_nx3, dtype=np.int64)
    tags = np.asarray(physical_tags, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_nx3 must have shape (n_vertices, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles_nx3 must have shape (n_triangles, 3)")
    if tags.shape != (triangles.shape[0],):
        raise ValueError("physical_tags must have one value per triangle")
    if np.min(vertices[:, 0]) < -tolerance or np.min(vertices[:, 1]) < -tolerance:
        raise ValueError("quarter mesh must lie in X>=0 and Y>=0")

    vertex_map: dict[tuple[int, int, int], int] = {}
    expanded_vertices: list[NDArray[np.float64]] = []
    q_to_full: dict[tuple[int, int, int], list[int]] = {}

    def key_for(point: NDArray[np.float64]) -> tuple[int, int, int]:
        return tuple(np.round(point / tolerance).astype(np.int64).tolist())

    image_signs = ((1, 1), (-1, 1), (1, -1), (-1, -1))
    image_vertex_maps: list[NDArray[np.int64]] = []
    for sx, sy in image_signs:
        remap = np.empty(vertices.shape[0], dtype=np.int64)
        for idx, point in enumerate(vertices):
            mirrored = np.array([sx * point[0], sy * point[1], point[2]], dtype=np.float64)
            key = key_for(mirrored)
            out_idx = vertex_map.get(key)
            if out_idx is None:
                out_idx = len(expanded_vertices)
                vertex_map[key] = out_idx
                expanded_vertices.append(mirrored)
            remap[idx] = out_idx
            q_to_full.setdefault((idx, sx, sy), []).append(out_idx)
        image_vertex_maps.append(remap)

    expanded_triangles: list[NDArray[np.int64]] = []
    expanded_tags: list[NDArray[np.int32]] = []
    triangle_signs: list[tuple[int, int]] = []
    for (sx, sy), remap in zip(image_signs, image_vertex_maps, strict=True):
        mapped = remap[triangles]
        if (sx * sy) < 0:
            mapped = mapped[:, [0, 2, 1]]
        expanded_triangles.append(mapped)
        expanded_tags.append(tags.copy())
        triangle_signs.extend((sx, sy) for _ in range(triangles.shape[0]))

    return ExpandedMesh(
        vertices_nx3=np.asarray(expanded_vertices, dtype=np.float64),
        triangles_nx3=np.vstack(expanded_triangles).astype(np.int64),
        physical_tags=np.concatenate(expanded_tags).astype(np.int32),
        triangle_image_signs=np.asarray(triangle_signs, dtype=np.int32),
        quarter_to_full_vertices={
            key: np.unique(np.asarray(value, dtype=np.int64))
            for key, value in q_to_full.items()
        },
    )


def p1_dof_coordinates(grid: object, p1_space: object) -> NDArray[np.float64]:
    """Resolve P1 global dof coordinates from grid vertices and local2global."""
    vertices = np.asarray(grid.vertices, dtype=np.float64).T
    elements = np.asarray(grid.elements, dtype=np.int64).T
    local2global = np.asarray(p1_space.local2global, dtype=np.int64)
    if local2global.shape != elements.shape:
        raise ValueError("p1 local2global and grid elements must have matching shape")
    n_dofs = int(p1_space.global_dof_count)
    coords = np.full((n_dofs, 3), np.nan, dtype=np.float64)
    for elem_idx in range(elements.shape[0]):
        for local_idx in range(3):
            dof = int(local2global[elem_idx, local_idx])
            point = vertices[int(elements[elem_idx, local_idx])]
            if np.any(np.isnan(coords[dof])):
                coords[dof] = point
            elif np.linalg.norm(coords[dof] - point) > 1.0e-8:
                raise ValueError(f"P1 dof {dof} maps to inconsistent coordinates")
    if np.any(np.isnan(coords)):
        missing = np.flatnonzero(np.any(np.isnan(coords), axis=1))
        raise ValueError(f"P1 dof coordinates missing for dofs {missing[:8].tolist()}")
    return coords


def build_xy_mirror_orbits(
    reduced_dof_coords_nx3: NDArray[np.float64],
    full_dof_coords_nx3: NDArray[np.float64],
    *,
    tolerance: float = 1.0e-7,
) -> list[NDArray[np.int64]]:
    """Map each reduced P1 dof to all matching full-domain X/Y mirror dofs."""
    reduced = np.asarray(reduced_dof_coords_nx3, dtype=np.float64)
    full = np.asarray(full_dof_coords_nx3, dtype=np.float64)
    full_lookup: dict[tuple[int, int, int], list[int]] = {}
    for idx, point in enumerate(full):
        full_lookup.setdefault(_coord_key(point, tolerance), []).append(idx)

    orbits: list[NDArray[np.int64]] = []
    for point in reduced:
        members: set[int] = set()
        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            mirrored = np.array([sx * point[0], sy * point[1], point[2]], dtype=np.float64)
            members.update(full_lookup.get(_coord_key(mirrored, tolerance), []))
        if not members:
            raise ValueError(f"no full-domain orbit found for reduced dof at {point.tolist()}")
        orbits.append(np.asarray(sorted(members), dtype=np.int64))
    return orbits


def classify_xy_reduced_dofs(
    reduced_dof_coords_nx3: NDArray[np.float64],
    *,
    tolerance: float = 1.0e-7,
) -> NDArray[np.str_]:
    coords = np.asarray(reduced_dof_coords_nx3, dtype=np.float64)
    on_x = np.abs(coords[:, 0]) <= tolerance
    on_y = np.abs(coords[:, 1]) <= tolerance
    out = np.full(coords.shape[0], "interior", dtype="<U16")
    out[np.logical_xor(on_x, on_y)] = "single_seam"
    out[np.logical_and(on_x, on_y)] = "double_seam"
    return out


def build_local2global_xy_mirror_orbits(
    reduced_local2global: NDArray[np.integer],
    full_local2global: NDArray[np.integer],
) -> list[NDArray[np.int64]]:
    """Build P1 dof orbits from exact four-image triangle block ordering.

    This is useful for Bempp spaces whose P1 dofs are not uniquely represented
    by a single grid vertex coordinate on closed reduced-domain surfaces. The
    expanded mesh writer emits image blocks in this order:

    ``(+X,+Y), (-X,+Y), (+X,-Y), (-X,-Y)``.

    Exactly one reflection reverses triangle winding, so the local basis index
    mapping is inverted for those image blocks.
    """
    reduced = np.asarray(reduced_local2global, dtype=np.int64)
    full = np.asarray(full_local2global, dtype=np.int64)
    if reduced.ndim != 2 or reduced.shape[1] != 3:
        raise ValueError("reduced_local2global must have shape (n_triangles, 3)")
    if full.shape != (4 * reduced.shape[0], 3):
        raise ValueError(
            "full_local2global must have shape (4 * n_reduced_triangles, 3)"
        )
    n_reduced_dofs = int(reduced.max()) + 1
    orbit_sets: list[set[int]] = [set() for _ in range(n_reduced_dofs)]
    image_perms = (
        (0, 1, 2),
        (0, 2, 1),
        (0, 2, 1),
        (0, 1, 2),
    )
    for elem_idx in range(reduced.shape[0]):
        for local_idx in range(3):
            reduced_dof = int(reduced[elem_idx, local_idx])
            for image_idx, perm in enumerate(image_perms):
                full_row = image_idx * reduced.shape[0] + elem_idx
                full_local_idx = int(perm.index(local_idx))
                orbit_sets[reduced_dof].add(int(full[full_row, full_local_idx]))
    return [np.asarray(sorted(members), dtype=np.int64) for members in orbit_sets]


def classify_orbits_by_size(orbits: Iterable[NDArray[np.integer]]) -> NDArray[np.str_]:
    classes: list[str] = []
    for orbit in orbits:
        size = len(np.asarray(orbit, dtype=np.int64))
        if size >= 4:
            classes.append("interior")
        elif size == 2:
            classes.append("single_seam")
        elif size == 1:
            classes.append("double_seam")
        else:
            raise ValueError(f"unexpected empty orbit of size {size}")
    return np.asarray(classes, dtype="<U16")


def orbit_reduce_matrix_rhs(
    full_matrix: NDArray[np.complexfloating],
    full_rhs: NDArray[np.complexfloating],
    row_orbits: Iterable[NDArray[np.integer]],
    col_orbits: Iterable[NDArray[np.integer]] | None = None,
) -> tuple[NDArray[np.complex64], NDArray[np.complex64]]:
    rows = [np.asarray(row, dtype=np.int64) for row in row_orbits]
    cols = rows if col_orbits is None else [np.asarray(col, dtype=np.int64) for col in col_orbits]
    reduced_matrix = np.empty((len(rows), len(cols)), dtype=np.complex64)
    for row_idx, row_dofs in enumerate(rows):
        for col_idx, col_dofs in enumerate(cols):
            reduced_matrix[row_idx, col_idx] = np.asarray(
                full_matrix[np.ix_(row_dofs, col_dofs)].sum(),
                dtype=np.complex64,
            )
    reduced_rhs = np.asarray([full_rhs[row_dofs].sum() for row_dofs in rows], dtype=np.complex64)
    return reduced_matrix, reduced_rhs


def expand_reduced_pressure(
    reduced_pressure: NDArray[np.complexfloating],
    full_dof_count: int,
    col_orbits: Iterable[NDArray[np.integer]],
) -> NDArray[np.complex64]:
    expanded = np.zeros(int(full_dof_count), dtype=np.complex64)
    for value, dofs in zip(reduced_pressure, col_orbits, strict=True):
        expanded[np.asarray(dofs, dtype=np.int64)] = np.asarray(value, dtype=np.complex64)
    return expanded


def _coord_key(point: NDArray[np.float64], tolerance: float) -> tuple[int, int, int]:
    return tuple(np.round(np.asarray(point, dtype=np.float64) / tolerance).astype(np.int64).tolist())

"""Place several bodies into one exterior BEM domain.

The solver itself never needed teaching about multiple cabinets: a boundary
integral operator sees element pairs, not connected components, so N disjoint
closed surfaces in one mesh are coupled through the kernel automatically and
correctly. What was missing is the bookkeeping in front of it -- putting each
body where it belongs, keeping its physical tags distinct from its neighbours',
and remembering which body each element came from so a caller can drive one
cabinet and read another's loading.

That is what :func:`combine_bodies` does, and it is deliberately the only thing
it does. It performs no acoustics, applies no ground plane, and does not merge
vertices between bodies: two cabinets that touch stay two closed surfaces,
because welding them would silently change the geometry the solver sees.

Placement is a rigid motion, rotation then translation. Improper transforms are
refused rather than silently corrected: a reflection turns an outward-wound
surface inward, and the loader's winding check cannot catch it once several
bodies are summed into one signed volume.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .mesh import LoadedMesh, MeshError, make_pure_grid
from .result import MeshInfo

__all__ = ["BodyPlacement", "CombinedMesh", "combine_bodies", "rotation_matrix"]

# A rotation matrix that has drifted further than this from orthonormality is
# treated as an authoring error rather than quietly renormalised, because the
# repair would move geometry the caller believes it positioned exactly.
_ORTHONORMAL_TOL = 1.0e-9


def rotation_matrix(axis: NDArray[np.float64], angle_deg: float) -> NDArray[np.float64]:
    """Right-handed rotation about ``axis`` by ``angle_deg`` degrees."""
    direction = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise MeshError("rotation axis must be finite and non-zero")
    direction = direction / norm
    angle = np.deg2rad(float(angle_deg))
    cross = np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )


@dataclass(frozen=True)
class BodyPlacement:
    """One body and where it sits in the shared exterior domain.

    ``tag_map`` renames that body's physical tags. Give every body its own tag
    numbers when they must be driven or reported separately -- two cabinets
    that both arrive tagged ``{1: rigid, 2: throat}`` are otherwise
    indistinguishable to ``velocity_sources``. Tags left unmapped are kept as
    they are, which is what a caller wants for a genuinely shared tag such as a
    common rigid surface.
    """

    mesh: LoadedMesh
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: NDArray[np.float64] | None = None
    tag_map: dict[int, int] = field(default_factory=dict)
    name: str = ""


@dataclass(frozen=True)
class CombinedMesh:
    """The merged mesh plus the provenance a multi-body caller needs."""

    mesh: LoadedMesh
    # (n_triangles,) index of the placement each triangle came from.
    body_ids: NDArray[np.int32]
    # body index -> {original tag: tag in the combined mesh}
    tag_maps: tuple[dict[int, int], ...]
    body_names: tuple[str, ...]

    def tags_for(self, body: int) -> tuple[int, ...]:
        """Combined-mesh tags belonging to one body, in ascending order."""
        return tuple(sorted(self.tag_maps[int(body)].values()))


def _mesh_arrays(mesh: LoadedMesh) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    vertices = np.asarray(mesh.grid.vertices, dtype=np.float64)
    if vertices.ndim == 2 and vertices.shape[0] == 3 and vertices.shape[1] != 3:
        vertices = vertices.T
    elements = np.asarray(mesh.grid.elements, dtype=np.int32)
    if elements.ndim == 2 and elements.shape[0] == 3 and elements.shape[1] != 3:
        elements = elements.T
    return np.ascontiguousarray(vertices), np.ascontiguousarray(elements)


def _validated_rotation(rotation: object, name: str) -> NDArray[np.float64] | None:
    if rotation is None:
        return None
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise MeshError(f"{name} rotation must be a 3x3 matrix")
    if not np.all(np.isfinite(matrix)):
        raise MeshError(f"{name} rotation must be finite")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=_ORTHONORMAL_TOL):
        raise MeshError(f"{name} rotation must be orthonormal")
    determinant = float(np.linalg.det(matrix))
    if determinant < 0.0:
        raise MeshError(
            f"{name} rotation has determinant {determinant:+.6f}: it is a "
            "reflection, which inverts outward triangle winding. Compose the "
            "placement from proper rotations, or mirror the mesh yourself and "
            "reverse its winding."
        )
    return matrix


def combine_bodies(placements: list[BodyPlacement]) -> CombinedMesh:
    """Merge placed bodies into one mesh for a single exterior solve.

    Raises :class:`MeshError` if two bodies would end up sharing a physical tag
    without being asked to. Silent tag collision is the failure that matters
    here: it does not crash, it just drives the wrong cabinet.
    """
    if not placements:
        raise MeshError("combine_bodies requires at least one placement")

    vertices_parts: list[NDArray[np.float64]] = []
    triangle_parts: list[NDArray[np.int32]] = []
    tag_parts: list[NDArray[np.int32]] = []
    body_id_parts: list[NDArray[np.int32]] = []
    resolved_maps: list[dict[int, int]] = []
    names: list[str] = []
    groups: dict[int, str] = {}
    tag_owner: dict[int, int] = {}
    vertex_offset = 0

    for index, placement in enumerate(placements):
        name = placement.name or f"body{index}"
        names.append(name)
        vertices, triangles = _mesh_arrays(placement.mesh)
        if vertices.shape[0] == 0 or triangles.shape[0] == 0:
            raise MeshError(f"{name} contributes no geometry")

        rotation = _validated_rotation(placement.rotation, name)
        if rotation is not None:
            vertices = vertices @ rotation.T
        translation = np.asarray(placement.translation_m, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(translation)):
            raise MeshError(f"{name} translation must be finite")
        vertices = vertices + translation

        source_tags = np.asarray(placement.mesh.physical_tags, dtype=np.int32)
        if source_tags.shape[0] != triangles.shape[0]:
            raise MeshError(
                f"{name} has {source_tags.shape[0]} physical tags for "
                f"{triangles.shape[0]} triangles"
            )
        mapping = {int(k): int(v) for k, v in placement.tag_map.items()}
        present = {int(tag) for tag in np.unique(source_tags)}
        unknown = sorted(set(mapping) - present)
        if unknown:
            raise MeshError(
                f"{name} tag_map renames tags {unknown} that the body does not "
                f"use; it uses {sorted(present)}"
            )
        resolved = {tag: mapping.get(tag, tag) for tag in sorted(present)}
        if len(set(resolved.values())) != len(resolved):
            raise MeshError(f"{name} tag_map maps two of its own tags onto one")

        for original, combined in resolved.items():
            owner = tag_owner.get(combined)
            if owner is not None and owner != index:
                raise MeshError(
                    f"{name} would use physical tag {combined}, already taken "
                    f"by {names[owner]}. Give each body distinct tags through "
                    "BodyPlacement.tag_map so velocity_sources can address "
                    "them separately, or map them onto each other deliberately."
                )
            tag_owner[combined] = index
            source_names = getattr(placement.mesh.info, "physical_groups", {}) or {}
            label = source_names.get(original, f"tag{original}")
            groups[combined] = f"{name}:{label}"

        lookup = np.zeros(max(present) + 1, dtype=np.int32)
        for original, combined in resolved.items():
            lookup[original] = combined

        vertices_parts.append(vertices)
        triangle_parts.append(triangles + vertex_offset)
        tag_parts.append(lookup[source_tags])
        body_id_parts.append(np.full(triangles.shape[0], index, dtype=np.int32))
        resolved_maps.append(resolved)
        vertex_offset += vertices.shape[0]

    all_vertices = np.ascontiguousarray(np.vstack(vertices_parts))
    all_triangles = np.ascontiguousarray(
        np.vstack(triangle_parts).astype(np.int32, copy=False)
    )
    all_tags = np.ascontiguousarray(np.concatenate(tag_parts).astype(np.int32))
    body_ids = np.ascontiguousarray(np.concatenate(body_id_parts))

    combined = LoadedMesh(
        grid=make_pure_grid(all_vertices, all_triangles),
        physical_tags=all_tags,
        info=MeshInfo(
            n_vertices=int(all_vertices.shape[0]),
            n_triangles=int(all_triangles.shape[0]),
            physical_groups=groups,
            bounding_box_m=(all_vertices.min(axis=0), all_vertices.max(axis=0)),
        ),
    )
    return CombinedMesh(
        mesh=combined,
        body_ids=body_ids,
        tag_maps=tuple(resolved_maps),
        body_names=tuple(names),
    )

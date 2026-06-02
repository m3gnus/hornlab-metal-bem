from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .result import MeshInfo

logger = logging.getLogger(__name__)


@dataclass
class LoadedMesh:
    grid: object  # bempp.api.Grid
    physical_tags: NDArray[np.int32]
    info: MeshInfo


class MeshError(Exception):
    pass


def load_mesh(
    path: str | Path,
    scale: float = 1.0,
    validate: bool = True,
    merge_tol: float = 1e-9,
    repair_normals: bool = False,
) -> LoadedMesh:
    """Load a .msh file into a bempp Grid with physical group tags.

    Gmsh/ABEC surface meshes can contain duplicate seam vertices. Bempp treats
    those as disconnected components unless we stitch them before grid creation.

    Canonical HornLab meshes are expected to arrive with outward-oriented
    triangle winding. Set ``repair_normals=True`` only for explicit
    compatibility with arbitrary external meshes that may use inward winding.
    """
    import bempp_cl.api as bempp_api
    import meshio

    path = Path(path)
    if not path.exists():
        raise MeshError(f"Mesh file not found: {path}")

    mesh = meshio.read(path)
    tri_key = "triangle" if "triangle" in mesh.cells_dict else "triangle3"
    if tri_key not in mesh.cells_dict:
        raise MeshError("No triangles found in mesh")

    triangles = np.asarray(mesh.cells_dict[tri_key], dtype=np.int32)
    verts = np.asarray(mesh.points, dtype=np.float64) * scale
    phys_tags = _extract_physical_tags(mesh, tri_key)
    phys_group_names = _extract_physical_names(path)

    verts, triangles, merged_vertices = _merge_duplicate_vertices(
        verts, triangles, merge_tol,
    )
    if merged_vertices:
        logger.info("Merged %d duplicate seam vertices", merged_vertices)

    # Remove degenerate triangles, including any created by seam merging.
    valid = ~(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    )
    n_degen = np.sum(~valid)
    if n_degen > 0:
        logger.info("Removed %d degenerate triangles", n_degen)
        triangles = triangles[valid]
        phys_tags = phys_tags[valid]

    if validate:
        _validate_outward_normals(
            verts,
            triangles,
            repair=repair_normals,
        )
        _validate_physical_groups(phys_tags)

    grid = bempp_api.Grid(verts.T, triangles.T.astype(np.int32), phys_tags)

    info = MeshInfo(
        n_vertices=len(verts),
        n_triangles=len(triangles),
        physical_groups=phys_group_names,
        bounding_box_m=(verts.min(axis=0), verts.max(axis=0)),
    )

    logger.info(
        "Loaded mesh: %d verts, %d tris, groups=%s",
        info.n_vertices, info.n_triangles, info.physical_groups,
    )

    return LoadedMesh(grid=grid, physical_tags=phys_tags, info=info)


def _extract_physical_tags(mesh, tri_key: str) -> NDArray[np.int32]:
    for key, by_type in mesh.cell_data_dict.items():
        if "physical" in key and tri_key in by_type:
            return np.asarray(by_type[tri_key], dtype=np.int32)
    raise MeshError("Mesh file has no triangle physical-group tags")


def _extract_physical_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    in_block = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if line == "$PhysicalNames":
                    in_block = True
                    continue
                if line == "$EndPhysicalNames":
                    break
                if not in_block:
                    continue
                parts = line.split(maxsplit=2)
                if len(parts) < 3 or not parts[0].isdigit():
                    continue
                dim = int(parts[0])
                tag = int(parts[1])
                if dim == 2:
                    names[tag] = parts[2].strip().strip('"')
    except OSError:
        return names
    return names


def _merge_duplicate_vertices(
    verts: NDArray[np.float64],
    tris: NDArray[np.int32],
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.int32], int]:
    """Merge coincident seam vertices and remap triangle connectivity."""
    if tol <= 0 or len(verts) == 0:
        return verts, tris, 0

    keys = np.round(verts / tol).astype(np.int64)
    _, first_indices, inverse = np.unique(
        keys,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    if len(first_indices) == len(verts):
        return verts, tris, 0

    merged_verts = verts[first_indices]
    merged_tris = inverse[tris].astype(np.int32, copy=False)
    return merged_verts, merged_tris, len(verts) - len(merged_verts)


def _validate_outward_normals(
    verts: NDArray[np.float64],
    tris: NDArray[np.int32],
    *,
    repair: bool = False,
) -> None:
    """Validate outward winding, optionally repairing legacy external meshes."""
    signed_vol = _signed_mesh_volume_indicator(verts, tris)
    if signed_vol >= 0:
        return

    if repair:
        logger.info("Flipping triangle winding (signed volume negative)")
        tris[:, [1, 2]] = tris[:, [2, 1]]
        return

    raise MeshError(
        "Mesh triangle winding appears inward (signed volume negative). "
        "Canonical meshes must be emitted with outward normals by the mesher; "
        "pass repair_normals=True only for explicit external-mesh compatibility."
    )


def _signed_mesh_volume_indicator(
    verts: NDArray[np.float64],
    tris: NDArray[np.int32],
) -> float:
    """Return the signed volume indicator used for closed-surface winding."""
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float(np.sum(p0 * np.cross(p1, p2)))


def _validate_physical_groups(phys_tags: NDArray[np.int32]) -> None:
    unique = np.unique(phys_tags)
    if not np.any(unique >= 2):
        raise MeshError(
            f"No velocity source (tag >= 2) found. Tags: {unique.tolist()}"
        )
    if not np.any(unique == 1):
        logger.warning("No rigid wall (tag 1) in mesh")

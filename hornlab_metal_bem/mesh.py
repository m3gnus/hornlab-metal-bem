from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .result import MeshInfo

logger = logging.getLogger(__name__)

_SYMMETRY_SNAP_TOLERANCE = 1.0e-6
_CANONICAL_COUPLED_IB_APERTURE_NAME = "mouth_aperture"


@dataclass
class LoadedMesh:
    grid: object
    physical_tags: NDArray[np.int32]
    info: MeshInfo
    # Set only when the caller explicitly requested coupled infinite-baffle
    # mode or the mesh declares the canonical ``mouth_aperture`` physical
    # group.  Keeping this on the loaded artifact lets the public solve API
    # preserve mesh semantics instead of silently routing it as free space.
    coupled_ib_aperture_tag: int | None = None


@dataclass(frozen=True)
class PureGrid:
    """Bempp-shaped mesh view backed only by NumPy arrays."""

    vertices: NDArray[np.float64]
    elements: NDArray[np.int32]
    volumes: NDArray[np.float64]

    @property
    def number_of_elements(self) -> int:
        return int(self.elements.shape[1])


@dataclass(frozen=True)
class PureFunctionSpace:
    local2global: NDArray[np.int32]
    global_dof_count: int


class MeshError(Exception):
    pass


def load_mesh(
    path: str | Path,
    scale: float = 1.0,
    validate: bool = True,
    merge_tol: float = 1e-9,
    repair_normals: bool = False,
    native_symmetry_plane: str | None = None,
    aperture_tag: int | None = None,
) -> LoadedMesh:
    """Load a .msh file into a lightweight grid with physical group tags.

    Gmsh/ABEC surface meshes can contain duplicate seam vertices. Bempp treats
    those as disconnected components unless we stitch them before grid creation.

    Canonical HornLab meshes are expected to arrive with outward-oriented
    triangle winding. Set ``repair_normals=True`` only for explicit
    compatibility with arbitrary external meshes that may use inward winding.

    The returned grid is a NumPy-only object shaped like the metadata consumed
    by the native Metal path.
    """
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise MeshError(
            "meshio is required to read .msh files; install hornlab-metal-bem "
            "with mesh support or pass a pre-loaded LoadedMesh."
        ) from exc

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
    for name, raw in getattr(mesh, "field_data", {}).items():
        values = np.asarray(raw).reshape(-1)
        if values.size >= 2 and int(values[1]) == 2:
            phys_group_names[int(values[0])] = str(name)

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

    coupled_ib_aperture_tag = _resolve_coupled_ib_aperture_tag(
        phys_tags,
        phys_group_names,
        aperture_tag=aperture_tag,
    )

    if validate:
        _validate_outward_normals(
            verts,
            triangles,
            repair=repair_normals,
            coupled_ib_aperture_tag=coupled_ib_aperture_tag,
        )
        if coupled_ib_aperture_tag is not None:
            _validate_coupled_ib_aperture_normals(
                verts,
                triangles,
                phys_tags,
                aperture_tag=coupled_ib_aperture_tag,
            )
        _validate_physical_groups(
            phys_tags,
            coupled_ib_aperture_tag=coupled_ib_aperture_tag,
        )

    _warn_if_reduced_symmetry_mesh(
        verts,
        triangles,
        native_symmetry_plane=native_symmetry_plane,
    )

    grid = make_pure_grid(verts, triangles)

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

    return LoadedMesh(
        grid=grid,
        physical_tags=phys_tags,
        info=info,
        coupled_ib_aperture_tag=coupled_ib_aperture_tag,
    )


def make_pure_grid(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
) -> PureGrid:
    vertices = np.ascontiguousarray(vertices_nx3.T, dtype=np.float64)
    elements = np.ascontiguousarray(triangles_nx3.T, dtype=np.int32)
    volumes = _triangle_areas(vertices_nx3, triangles_nx3)
    return PureGrid(vertices=vertices, elements=elements, volumes=volumes)


def make_pure_function_spaces(
    grid: object,
) -> tuple[PureFunctionSpace, PureFunctionSpace]:
    """Create P1 and DP0 spaces for a Bempp-shaped pure grid."""
    if not hasattr(grid, "elements") or not hasattr(grid, "vertices"):
        raise MeshError("Pure function spaces require grid.vertices and grid.elements")
    elements = np.asarray(grid.elements, dtype=np.int32)
    vertices = np.asarray(grid.vertices)
    if elements.ndim != 2 or elements.shape[0] != 3:
        raise MeshError(
            "grid.elements must have shape (3, n_triangles), "
            f"got {elements.shape}"
        )
    if vertices.ndim != 2 or vertices.shape[0] != 3:
        raise MeshError(
            "grid.vertices must have shape (3, n_vertices), "
            f"got {vertices.shape}"
        )
    local2global = np.ascontiguousarray(elements.T, dtype=np.int32)
    p1 = PureFunctionSpace(
        local2global=local2global,
        global_dof_count=int(vertices.shape[1]),
    )
    dp0 = PureFunctionSpace(
        local2global=np.arange(elements.shape[1], dtype=np.int32)[:, None],
        global_dof_count=int(elements.shape[1]),
    )
    return p1, dp0


def open_boundary_edges(
    triangles_nx3: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Return ``(n, 2)`` sorted vertex pairs for edges used by exactly one triangle.

    A closed surface has no open boundary edges; a mirror-reduced mesh has its
    open rim on the cut plane(s).
    """
    tris = np.asarray(triangles_nx3)
    if tris.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    edges = np.sort(
        np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]])),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return np.ascontiguousarray(unique_edges[counts == 1], dtype=np.int32)


def detect_reduced_symmetry_plane(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
    *,
    tolerance: float = _SYMMETRY_SNAP_TOLERANCE,
) -> str | None:
    """Heuristically detect mirror-reduced meshes loaded without symmetry.

    The detector is intentionally conservative: it only reports a candidate
    when the mesh lives on the positive side of a candidate plane, has a
    meaningful set of used vertices on that plane, and every open boundary edge
    is explained by the candidate plane set.
    """
    vertices = np.asarray(vertices_nx3, dtype=np.float64)
    triangles = np.asarray(triangles_nx3, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.size == 0:
        return None

    used_vertices = np.unique(triangles.reshape(-1))
    if used_vertices.size == 0:
        return None

    boundary_edges = open_boundary_edges(triangles)
    if boundary_edges.size == 0:
        return None

    used = vertices[used_vertices]
    candidates: list[tuple[str, int]] = []
    for plane, component in (("yz", 0), ("xz", 1), ("xy", 2)):
        values = used[:, component]
        on_plane = np.abs(values) <= tolerance
        has_positive_side = bool(np.max(values) > tolerance)
        meaningful_count = int(np.count_nonzero(on_plane))
        if (
            np.min(values) >= -tolerance
            and has_positive_side
            and meaningful_count >= 2
            and _count_edges_on_plane(
                vertices,
                boundary_edges,
                component,
                tolerance,
            )
            >= 2
        ):
            candidates.append((plane, component))

    if not candidates:
        return None

    candidate_components = {plane: component for plane, component in candidates}
    for plane, component in candidate_components.items():
        if _all_edges_on_any_plane(vertices, boundary_edges, [component], tolerance):
            return plane
    if (
        "yz" in candidate_components
        and "xz" in candidate_components
        and _all_edges_on_any_plane(
            vertices,
            boundary_edges,
            [candidate_components["yz"], candidate_components["xz"]],
            tolerance,
        )
    ):
        return "yz+xz"
    return None


def _warn_if_reduced_symmetry_mesh(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
    *,
    native_symmetry_plane: str | None,
) -> None:
    if native_symmetry_plane is not None:
        return
    suspected = detect_reduced_symmetry_plane(vertices_nx3, triangles_nx3)
    if suspected is None:
        return
    warnings.warn(
        "Mesh may be a reduced native-symmetry mesh "
        f"(suspected plane {suspected!r}) but native_symmetry_plane is None; "
        "if this is intended as a mirror-reduced mesh, pass "
        f"native_symmetry_plane={suspected!r} in SolveConfig to solve the "
        "free-space mirrored geometry. If the rim is a real open boundary, "
        "ignore this warning.",
        RuntimeWarning,
        stacklevel=3,
    )


def _count_edges_on_plane(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int32],
    component: int,
    tolerance: float,
) -> int:
    return int(
        np.count_nonzero(_edge_on_plane_mask(vertices, edges, component, tolerance))
    )


def _all_edges_on_any_plane(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int32],
    components: list[int],
    tolerance: float,
) -> bool:
    explained = np.zeros(edges.shape[0], dtype=bool)
    for component in components:
        explained |= _edge_on_plane_mask(vertices, edges, component, tolerance)
    return bool(np.all(explained))


def _edge_on_plane_mask(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int32],
    component: int,
    tolerance: float,
) -> NDArray[np.bool_]:
    edge_values = vertices[edges, component]
    return np.all(np.abs(edge_values) <= tolerance, axis=1)


def _triangle_areas(
    verts: NDArray[np.float64],
    tris: NDArray[np.int32],
) -> NDArray[np.float64]:
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return np.ascontiguousarray(
        0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1),
        dtype=np.float64,
    )


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
    """Merge seam vertices within the requested Euclidean tolerance.

    Pairs come from a spatial tree, then union only after an exact
    squared-distance check. This preserves the Euclidean tolerance semantics
    at hash-cell boundaries without making every mesh load Python-loop bound.
    """
    if tol <= 0 or len(verts) == 0:
        return verts, tris, 0

    if not np.isfinite(tol):
        raise MeshError("merge_tol must be finite")

    parent = np.arange(len(verts), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    tol_sq = float(tol) ** 2
    # ``query_pairs(tol)`` can exclude a pair whose squared distance rounds to
    # exactly ``tol_sq``. Search one representable step wider, then retain the
    # pre-existing squared-distance predicate as the semantic authority.
    search_radius = np.nextafter(float(tol), np.inf)
    pairs = cKDTree(verts).query_pairs(search_radius, output_type="ndarray")
    for left, right in pairs:
        left_index = int(left)
        right_index = int(right)
        delta = verts[right_index] - verts[left_index]
        if float(delta @ delta) > tol_sq:
            continue
        root_left = find(left_index)
        root_right = find(right_index)
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

    merged_verts = verts[unique_roots]
    merged_tris = inverse[tris].astype(np.int32, copy=False)
    return merged_verts, merged_tris, len(verts) - len(merged_verts)


def _validate_outward_normals(
    verts: NDArray[np.float64],
    tris: NDArray[np.int32],
    *,
    repair: bool = False,
    coupled_ib_aperture_tag: int | None = None,
) -> None:
    """Validate outward winding, optionally repairing legacy external meshes."""
    signed_vol = _signed_mesh_volume_indicator(verts, tris)
    if coupled_ib_aperture_tag is not None:
        if signed_vol <= 0:
            return
        raise MeshError(
            "Coupled infinite-baffle mesh winding appears inverse "
            f"(aperture_tag={coupled_ib_aperture_tag}, signed volume positive). "
            "Coupled IB meshes are interior-domain surfaces and must keep "
            "negative signed volume with the aperture normals pointing -Z."
        )

    # The signed-volume indicator is translation invariant only for a closed
    # two-manifold. Bare horns and symmetry-reduced meshes are intentionally
    # open; translating one must not reverse the loader's winding verdict.
    if not _is_closed_two_manifold(tris):
        return

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


def _is_closed_two_manifold(triangles_nx3: NDArray[np.int32]) -> bool:
    tris = np.asarray(triangles_nx3, dtype=np.int64)
    if tris.ndim != 2 or tris.shape[1] != 3 or tris.size == 0:
        return False
    edges = np.sort(
        np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]])),
        axis=1,
    )
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(counts.size and np.all(counts == 2))


def _validate_coupled_ib_aperture_normals(
    verts: NDArray[np.float64],
    tris: NDArray[np.int32],
    phys_tags: NDArray[np.int32],
    *,
    aperture_tag: int,
    tolerance: float = 1.0e-6,
) -> None:
    mask = np.asarray(phys_tags, dtype=np.int32) == int(aperture_tag)
    if not np.any(mask):
        return
    aperture_tris = tris[mask]
    p0 = verts[aperture_tris[:, 0]]
    p1 = verts[aperture_tris[:, 1]]
    p2 = verts[aperture_tris[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    if not np.any(valid):
        raise MeshError("Coupled infinite-baffle aperture contains only degenerate triangles")
    unit_z = normals[valid, 2] / lengths[valid]
    bad = unit_z >= -1.0 + float(tolerance)
    if np.any(bad):
        first_local = int(np.flatnonzero(valid)[int(np.flatnonzero(bad)[0])])
        first = int(np.flatnonzero(mask)[first_local])
        raise MeshError(
            "Coupled infinite-baffle aperture normals must point -Z for the "
            "interior-domain BIE; regenerate meshes emitted with the old +Z "
            f"aperture contract (triangle {first}, normal_z={unit_z[bad][0]:.6g})."
        )


def _resolve_coupled_ib_aperture_tag(
    phys_tags: NDArray[np.int32],
    phys_group_names: dict[int, str],
    *,
    aperture_tag: int | None,
) -> int | None:
    """Return the aperture tag when physical tags identify coupled IB topology."""
    present = {int(tag) for tag in np.unique(phys_tags)}
    explicit = _coerce_aperture_tag(aperture_tag)
    named = [
        int(tag)
        for tag, name in phys_group_names.items()
        if str(name).strip().lower() == _CANONICAL_COUPLED_IB_APERTURE_NAME
    ]
    if len(named) > 1:
        raise MeshError(
            "Mesh declares more than one mouth_aperture physical group: "
            f"{sorted(named)}"
        )
    named_tag = named[0] if named else None
    if named_tag is not None and named_tag not in present:
        raise MeshError(
            "Mesh declares mouth_aperture physical tag "
            f"{named_tag}, but no triangle uses that tag"
        )

    if explicit is not None:
        if explicit not in present:
            raise MeshError(
                f"aperture_tag {explicit} is not present in the mesh; "
                f"available physical tags: {sorted(present)}"
            )
        if named_tag is not None and explicit != named_tag:
            raise MeshError(
                f"Explicit aperture_tag {explicit} conflicts with canonical "
                f"mouth_aperture physical tag {named_tag}"
            )
        return explicit

    # Do not infer coupled-IB semantics from a raw numeric tag.  Tag 12 is the
    # mesher's current canonical value, but external meshes are free to use it
    # for an unrelated boundary.  The physical name is the semantic contract.
    return named_tag


def _coerce_aperture_tag(aperture_tag: int | None) -> int | None:
    if aperture_tag is None:
        return None
    if (
        isinstance(aperture_tag, bool)
        or not isinstance(aperture_tag, Integral)
        or int(aperture_tag) <= 0
    ):
        raise MeshError("aperture_tag must be a positive int or None")
    return int(aperture_tag)


def _validate_physical_groups(
    phys_tags: NDArray[np.int32],
    *,
    coupled_ib_aperture_tag: int | None = None,
) -> None:
    unique = np.unique(phys_tags)
    source_candidates = unique
    if coupled_ib_aperture_tag is not None:
        source_candidates = unique[unique != int(coupled_ib_aperture_tag)]
    if not np.any(source_candidates >= 2):
        raise MeshError(
            f"No velocity source (tag >= 2) found. Tags: {unique.tolist()}"
        )
    if not np.any(unique == 1):
        logger.warning("No rigid wall (tag 1) in mesh")

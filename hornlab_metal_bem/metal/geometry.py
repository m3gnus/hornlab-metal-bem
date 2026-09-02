"""Geometry buffer export for the future Metal backend.

This module is deliberately pure Python/NumPy. It does not import Bempp,
Swift, or Metal; callers pass Bempp-like objects that already expose the
metadata needed by the adapter.

All index arrays at this Python boundary are **zero-based**. Production Python
code must not emit one-based triangle or DOF indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import (
    GROUND_PLANE_NORMAL_AXIS,
    NATIVE_GROUND_PLANES,
    NATIVE_SYMMETRY_PLANES,
)
from ..mesh import open_boundary_edges


class MetalGeometryError(ValueError):
    """Raised when grid/space metadata cannot satisfy the Metal data contract."""


# A fixed absolute plane tolerance is the wrong shape for geometry spanning
# roughly 25 mm throats through 1.5 m horns: the coordinate residue left by CAD
# export and unit conversion scales with the model. Compute the actual tolerance
# from every mesh presented to this adapter, after the loader has applied its
# unit conversion, rather than caching a tolerance from some earlier mesh with a
# different scale. The absolute floor only matters for a zero-span input.
#
# Swift's coordinateKey() still owns a fixed 1e-6 quantization grid for matching
# image singularities. This snap does not make that grid looser: it replaces a
# qualifying near-plane component with exact zero *before* Swift constructs its
# key, so both the real vertex and its image quantize identically. For meshes no
# larger than one unit the relative tolerance is no wider than coordinateKey's
# grid; for a metre-scale model it may be a few micrometres, but none of those
# snapped residues reach coordinateKey. The z=5e-7 regression case therefore
# remains an exact on-plane vertex instead of falling between Python's plane
# validation and Swift's image-pair matching.
_PLANE_SNAP_ABSOLUTE_FLOOR = 1.0e-9
_PLANE_SNAP_RELATIVE_FACTOR = 1.0e-6


@dataclass(frozen=True)
class MetalGeometryBuffers:
    """Validated NumPy buffers for Metal dense BEM assembly.

    Arrays intentionally mirror the scratch prototype naming and orientation:

    - ``vertices_3xn_f32`` has shape ``(3, n_vertices)`` and dtype ``float32``.
    - ``triangles_3xm_i32`` has shape ``(3, n_triangles)`` and dtype ``int32``.
    - ``physical_tags_i32`` has shape ``(n_triangles,)`` and dtype ``int32``.
    - ``p1_local2global_i32`` has shape ``(n_triangles, 3)`` and dtype
      ``int32``.
    - ``triangle_areas_f32`` has shape ``(n_triangles,)`` and dtype
      ``float32``.
    - ``triangle_normals_3xm_f32`` has shape ``(3, n_triangles)`` and dtype
      ``float32``.

    Triangle and P1 DOF indices are zero-based at the Python boundary.
    """

    vertices_3xn_f32: NDArray[np.float32]
    triangles_3xm_i32: NDArray[np.int32]
    physical_tags_i32: NDArray[np.int32]
    p1_local2global_i32: NDArray[np.int32]
    triangle_areas_f32: NDArray[np.float32]
    triangle_normals_3xm_f32: NDArray[np.float32]
    p1_dof_count: int
    dp0_dof_count: int

    @property
    def n_vertices(self) -> int:
        return int(self.vertices_3xn_f32.shape[1])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles_3xm_i32.shape[1])

    @property
    def triangles_nx3_i32(self) -> NDArray[np.int32]:
        """Triangle connectivity as ``(n_triangles, 3)`` zero-based rows."""
        return np.ascontiguousarray(self.triangles_3xm_i32.T)


def build_metal_geometry_buffers(
    grid: Any,
    physical_tags: Any,
    p1_space: Any,
    dp0_space: Any | None = None,
) -> MetalGeometryBuffers:
    """Convert Bempp-like grid/space metadata into Metal geometry buffers.

    Parameters
    ----------
    grid:
        Object exposing Bempp-style ``vertices`` with shape ``(3, N)`` and
        ``elements`` with shape ``(3, M)``. ``number_of_elements`` is validated
        when present.
    physical_tags:
        Per-triangle physical tags with shape ``(M,)``. Column/row vectors with
        one singleton dimension are accepted and flattened.
    p1_space:
        Object exposing ``local2global`` with shape ``(M, 3)`` and, when
        present, ``global_dof_count``.
    dp0_space:
        Optional object exposing ``global_dof_count``. When provided, it must
        equal ``M`` because DP0 has one DOF per triangle.

    Returns
    -------
    MetalGeometryBuffers
        C-contiguous arrays with the scratch prototype shapes. All triangle
        vertex indices and P1 DOF indices are zero-based int32 values; this
        function rejects one-based or out-of-range input rather than converting
        it implicitly.
    """
    buffers, _ = _build_metal_geometry_buffers(
        grid,
        physical_tags,
        p1_space,
        dp0_space,
        include_max_edge=False,
    )
    return buffers


def _build_metal_geometry_buffers_with_max_edge(
    grid: Any,
    physical_tags: Any,
    p1_space: Any,
    dp0_space: Any | None = None,
) -> tuple[MetalGeometryBuffers, float]:
    """Build buffers and reuse the triangle gather to find the longest edge."""
    buffers, max_edge_m = _build_metal_geometry_buffers(
        grid,
        physical_tags,
        p1_space,
        dp0_space,
        include_max_edge=True,
    )
    assert max_edge_m is not None
    return buffers, max_edge_m


def _build_metal_geometry_buffers(
    grid: Any,
    physical_tags: Any,
    p1_space: Any,
    dp0_space: Any | None,
    *,
    include_max_edge: bool,
) -> tuple[MetalGeometryBuffers, float | None]:
    input_vertices_f64 = _require_vertices_3xn(grid)
    plane_snap_tolerance = _plane_snap_tolerance(input_vertices_f64)
    vertices_f64 = input_vertices_f64.copy()
    vertices_f64[np.abs(vertices_f64) <= plane_snap_tolerance] = 0.0
    triangles_i32 = _require_triangles_3xm(grid, vertices_f64.shape[1])
    n_triangles = int(triangles_i32.shape[1])

    _validate_grid_element_count(grid, n_triangles)

    physical_tags_i32 = _require_physical_tags(physical_tags, n_triangles)
    p1_local2global_i32 = _require_p1_local2global(
        p1_space,
        n_triangles,
    )
    p1_dof_count = _resolve_p1_dof_count(p1_space, p1_local2global_i32)
    dp0_dof_count = _resolve_dp0_dof_count(dp0_space, n_triangles)

    max_edge_m: float | None = None
    if include_max_edge:
        (
            triangle_areas_f32,
            triangle_normals_3xm_f32,
            max_edge_m,
        ) = _compute_areas_normals_and_max_edge(
            input_vertices_f64,
            triangles_i32,
            plane_snap_tolerance=plane_snap_tolerance,
        )
    else:
        triangle_areas_f32, triangle_normals_3xm_f32 = _compute_areas_normals(
            vertices_f64,
            triangles_i32,
        )

    return (
        MetalGeometryBuffers(
            vertices_3xn_f32=np.ascontiguousarray(vertices_f64, dtype=np.float32),
            triangles_3xm_i32=np.ascontiguousarray(triangles_i32, dtype=np.int32),
            physical_tags_i32=physical_tags_i32,
            p1_local2global_i32=p1_local2global_i32,
            triangle_areas_f32=triangle_areas_f32,
            triangle_normals_3xm_f32=triangle_normals_3xm_f32,
            p1_dof_count=p1_dof_count,
            dp0_dof_count=dp0_dof_count,
        ),
        max_edge_m,
    )


def validate_native_symmetry_plane(
    buffers: MetalGeometryBuffers,
    symmetry_plane: str | None,
    *,
    tolerance: float = 1.0e-7,
    check_open_edges: bool = True,
) -> str | None:
    """Validate the narrow native reduced-domain symmetry contract.

    The native symmetry path supports caller-supplied positive-side reduced
    meshes mirrored across YZ (X symmetry), XZ (Y symmetry), XY (Z symmetry),
    or YZ+XZ. Symmetry planes are not real BEM boundaries, so triangles lying
    entirely on a requested plane are rejected.

    With ``check_open_edges`` (the default) every open boundary edge must lie
    on a requested symmetry plane: a closed surface reduced by mirror cuts has
    its entire rim on the cut planes, so an off-plane open edge means the mesh
    was cut along an unrequested plane and the mirrored solve would be wrong.
    Pass ``check_open_edges=False`` only for open shells whose rim is a real
    free edge of the full (reduced + mirrored) geometry.
    """
    if symmetry_plane is None:
        return None
    plane = str(symmetry_plane).strip().lower()
    if plane not in NATIVE_SYMMETRY_PLANES:
        raise MetalGeometryError(
            "native_symmetry_plane currently supports 'yz', 'xz', 'xy', and 'yz+xz'"
        )

    coords = np.asarray(buffers.vertices_3xn_f32, dtype=np.float64)
    if coords.shape[1] == 0:
        raise MetalGeometryError(f"native_symmetry_plane={plane!r} requires vertices")

    triangles = buffers.triangles_nx3_i32
    used_vertices = np.unique(triangles.reshape(-1))

    def _validate_axis(component: int, name: str, plane_name: str) -> None:
        values = coords[component]
        # Only vertices referenced by triangles constrain the reduced domain;
        # orphan vertices on the negative side are harmless.
        min_value = float(np.min(values[used_vertices]))
        if min_value < -tolerance:
            raise MetalGeometryError(
                f"native_symmetry_plane={plane!r} requires a positive-{name} "
                f"reduced-domain mesh; minimum {name.upper()} is {min_value:.6g}"
            )
        used_values = values[used_vertices]
        if not np.any(np.abs(used_values) <= tolerance):
            raise MetalGeometryError(
                f"native_symmetry_plane={plane!r} requires boundary vertices "
                f"on {name.upper()}=0"
            )
        tri_values = values[triangles]
        cut_faces = np.all(np.abs(tri_values) <= tolerance, axis=1)
        if np.any(cut_faces):
            first = int(np.flatnonzero(cut_faces)[0])
            raise MetalGeometryError(
                f"native_symmetry_plane={plane!r} treats {plane_name} as an "
                f"image plane, not a physical boundary; triangle {first} lies "
                "entirely on the plane"
            )

    if "yz" in plane:
        _validate_axis(0, "x", "X=0")
    if "xz" in plane:
        _validate_axis(1, "y", "Y=0")
    if plane == "xy":
        _validate_axis(2, "z", "Z=0")
    if check_open_edges:
        _validate_open_edges_on_symmetry_planes(coords, triangles, plane)
    return plane


def validate_native_ground_plane(
    buffers: MetalGeometryBuffers,
    ground_plane: str | None,
    *,
    tolerance: float = 1.0e-7,
    min_clearance_m: float = 0.0,
) -> str | None:
    """Validate the rigid half-space contract and return the normalized plane.

    Unlike :func:`validate_native_symmetry_plane`, the mesh here is the whole
    radiating body standing next to a rigid boundary, so it is NOT required to
    reach the plane and its rim is not constrained. What is required is that
    the body lies wholly in the open half space:

    * every vertex used by a triangle is on the non-negative side, and
    * no triangle lies flat on the plane.

    A triangle flat on the plane is exactly coincident with its own image, so
    the real and image surfaces would occupy the same place and the boundary
    integral would be singular there. Remove that face -- an acoustically rigid
    boundary in contact with a rigid floor is not part of the radiating surface
    -- or lift the body clear.

    ``min_clearance_m`` optionally requires a positive gap between the body and
    the plane. Zero (the default) permits contact along an edge or vertex,
    which is geometrically legal and quadrature-safe because the image kernel's
    Duffy correction handles coincident and adjacent pairs.
    """
    if ground_plane is None:
        return None
    plane = str(ground_plane).strip().lower()
    if plane not in NATIVE_GROUND_PLANES:
        raise MetalGeometryError(
            "ground_plane currently supports 'xy', 'yz', and 'xz'"
        )
    if not (np.isfinite(min_clearance_m) and min_clearance_m >= 0.0):
        raise MetalGeometryError(
            "ground_plane min_clearance_m must be finite and non-negative"
        )

    coords = np.asarray(buffers.vertices_3xn_f32, dtype=np.float64)
    if coords.shape[1] == 0:
        raise MetalGeometryError(f"ground_plane={plane!r} requires vertices")

    axis = GROUND_PLANE_NORMAL_AXIS[plane]
    axis_name = "XYZ"[axis]
    triangles = buffers.triangles_nx3_i32
    used_vertices = np.unique(triangles.reshape(-1))
    values = coords[axis]
    used_values = values[used_vertices]

    min_value = float(np.min(used_values))
    if min_value < -tolerance:
        raise MetalGeometryError(
            f"ground_plane={plane!r} is a rigid half-space boundary: the whole "
            f"mesh must lie at {axis_name} >= 0, but the minimum {axis_name} is "
            f"{min_value:.6g}. Translate the body above the plane; the solver "
            "will not clip it."
        )

    tri_values = values[triangles]
    flat_faces = np.all(np.abs(tri_values) <= tolerance, axis=1)
    if np.any(flat_faces):
        first = int(np.flatnonzero(flat_faces)[0])
        raise MetalGeometryError(
            f"ground_plane={plane!r} treats {axis_name}=0 as an image plane, "
            f"not a physical boundary; triangle {first} lies flat on it and "
            "would coincide with its own image. Delete the ground-contact "
            "faces, or lift the body clear of the plane."
        )

    if min_clearance_m > 0.0 and min_value < min_clearance_m:
        raise MetalGeometryError(
            f"ground_plane={plane!r} requires at least "
            f"{min_clearance_m:.6g} m of clearance, but the mesh reaches "
            f"{axis_name}={min_value:.6g}"
        )
    return plane


def validate_native_infinite_baffle_aperture(
    buffers: MetalGeometryBuffers,
    aperture_tag: int | None,
    *,
    velocity_source_tags: Any | None = None,
    symmetry_plane: str | None = None,
    tolerance: float = 1.0e-6,
) -> int | None:
    """Validate the native full-3D coupled infinite-baffle aperture contract."""
    if aperture_tag is None:
        return None
    if (
        isinstance(aperture_tag, bool)
        or not isinstance(aperture_tag, Integral)
        or int(aperture_tag) <= 0
    ):
        raise MetalGeometryError("aperture_tag must be a positive int or None")
    tag = int(aperture_tag)

    plane = None if symmetry_plane is None else str(symmetry_plane).strip().lower()
    if plane == "xy":
        raise MetalGeometryError(
            "aperture_tag coupled infinite-baffle mode does not compose with "
            "native_symmetry_plane='xy'; use None, 'yz', 'xz', or 'yz+xz'"
        )
    if plane is not None and plane not in {"yz", "xz", "yz+xz"}:
        raise MetalGeometryError(
            "aperture_tag coupled infinite-baffle mode requires "
            "native_symmetry_plane to be None, 'yz', 'xz', or 'yz+xz'"
        )

    if velocity_source_tags is None:
        source_tags: set[int] = set()
    else:
        source_tags = set()
        for source_tag in velocity_source_tags:
            if (
                isinstance(source_tag, bool)
                or not isinstance(source_tag, Integral)
                or int(source_tag) <= 0
            ):
                raise MetalGeometryError("velocity_source_tags must be positive ints")
            source_tags.add(int(source_tag))
    if tag in source_tags:
        raise MetalGeometryError(
            "aperture_tag must not also be listed in velocity_sources"
        )

    tags = np.asarray(buffers.physical_tags_i32, dtype=np.int32)
    aperture_mask = tags == tag
    if not np.any(aperture_mask):
        available = sorted(int(value) for value in np.unique(tags))
        raise MetalGeometryError(
            f"aperture_tag {tag} is not present in the mesh; "
            f"available physical tags: {available}"
        )

    triangles = buffers.triangles_nx3_i32[aperture_mask]
    if triangles.shape[0] == 0:
        raise MetalGeometryError("aperture_tag must select at least one triangle")

    coords = np.asarray(buffers.vertices_3xn_f32, dtype=np.float64)
    z_values = coords[2, triangles]
    max_abs_z = float(np.max(np.abs(z_values)))
    if max_abs_z > tolerance:
        raise MetalGeometryError(
            "aperture_tag triangles must be coplanar at Z=0; "
            f"max |Z| is {max_abs_z:.6g}"
        )

    normals = np.asarray(buffers.triangle_normals_3xm_f32, dtype=np.float64)[
        :,
        aperture_mask,
    ]
    normal_tol = max(float(tolerance), 1.0e-5)
    off_axis = np.hypot(normals[0], normals[1])
    bad = (
        (off_axis > normal_tol)
        | (normals[2] > -1.0 + normal_tol)
        | (normals[2] >= 0.0)
    )
    if np.any(bad):
        first = int(np.flatnonzero(aperture_mask)[int(np.flatnonzero(bad)[0])])
        normal = normals[:, int(np.flatnonzero(bad)[0])]
        raise MetalGeometryError(
            "aperture_tag triangles must have normals pointing -Z; "
            f"triangle {first} normal is "
            f"({normal[0]:.6g}, {normal[1]:.6g}, {normal[2]:.6g})"
        )

    _validate_coupled_ib_topology(
        buffers,
        aperture_mask=aperture_mask,
        source_tags=source_tags,
        symmetry_plane=plane,
        tolerance=tolerance,
    )

    return tag


def _validate_coupled_ib_topology(
    buffers: MetalGeometryBuffers,
    *,
    aperture_mask: NDArray[np.bool_],
    source_tags: set[int],
    symmetry_plane: str | None,
    tolerance: float,
) -> None:
    """Validate that the aperture closes one manifold interior domain.

    A full coupled-IB mesh is watertight after the aperture disc is included.
    A symmetry-reduced mesh may be open only along the requested image planes.
    The aperture patch itself must be connected and its physical rim must be
    shared with non-coplanar, non-source wall triangles.  These checks prevent
    a detached disc or a partially tagged mouth from reaching the native
    augmented BIE/Rayleigh system.
    """
    triangles = np.asarray(buffers.triangles_nx3_i32, dtype=np.int32)
    coords = np.asarray(buffers.vertices_3xn_f32, dtype=np.float64).T
    tags = np.asarray(buffers.physical_tags_i32, dtype=np.int32)

    used_vertices = np.unique(triangles.reshape(-1))
    forward = coords[used_vertices, 2] > float(tolerance)
    if np.any(forward):
        vertex = int(used_vertices[int(np.flatnonzero(forward)[0])])
        raise MetalGeometryError(
            "coupled infinite-baffle cavity geometry must lie at or behind "
            f"Z=0; vertex {vertex} has Z={coords[vertex, 2]:.6g}"
        )

    edge_uses: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for tri_index, (a, b, c) in enumerate(triangles):
        for start, end in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (start, end) if start < end else (end, start)
            edge_uses.setdefault(key, []).append((tri_index, start, end))

    requested_components: tuple[int, ...]
    if symmetry_plane is None:
        requested_components = ()
    else:
        requested_components = tuple(
            component
            for token, component in (("yz", 0), ("xz", 1))
            if token in symmetry_plane
        )

    def _on_requested_plane(edge: tuple[int, int]) -> bool:
        return any(
            bool(np.all(np.abs(coords[list(edge), component]) <= tolerance))
            for component in requested_components
        )

    for edge, uses in edge_uses.items():
        if len(uses) > 2:
            raise MetalGeometryError(
                "coupled infinite-baffle mesh must be edge-manifold; "
                f"edge {edge} belongs to {len(uses)} triangles"
            )
        if len(uses) == 1:
            if symmetry_plane is None:
                raise MetalGeometryError(
                    "coupled infinite-baffle full-domain mesh must be watertight; "
                    f"edge {edge} is open"
                )
            if not _on_requested_plane(edge):
                raise MetalGeometryError(
                    "coupled infinite-baffle reduced mesh may have open edges "
                    "only on the requested symmetry plane(s); "
                    f"edge {edge} is off {symmetry_plane!r}"
                )
            continue

        (_, start0, end0), (_, start1, end1) = uses
        if start0 != end1 or end0 != start1:
            raise MetalGeometryError(
                "coupled infinite-baffle mesh has inconsistent triangle winding "
                f"across edge {edge}"
            )

    aperture_indices = set(int(i) for i in np.flatnonzero(aperture_mask))
    aperture_adjacency = {index: set() for index in aperture_indices}
    rim_edges = 0
    for edge, uses in edge_uses.items():
        aperture_uses = [use for use in uses if use[0] in aperture_indices]
        if len(aperture_uses) == 2:
            left, right = aperture_uses[0][0], aperture_uses[1][0]
            aperture_adjacency[left].add(right)
            aperture_adjacency[right].add(left)
            continue
        if len(aperture_uses) != 1:
            continue

        if len(uses) == 1:
            # A patch edge is allowed to be unshared only where the entire
            # reduced domain is cut by a requested image plane.
            if symmetry_plane is None or not _on_requested_plane(edge):
                raise MetalGeometryError(
                    "aperture patch has a detached boundary edge "
                    f"{edge} that is not shared with the cavity wall"
                )
            continue

        other = uses[0] if uses[1][0] in aperture_indices else uses[1]
        other_index = other[0]
        if int(tags[other_index]) in source_tags:
            raise MetalGeometryError(
                "aperture rim must be shared with cavity wall triangles, not "
                f"velocity-source triangle {other_index}"
            )
        other_vertices = triangles[other_index]
        if np.all(np.abs(coords[other_vertices, 2]) <= tolerance):
            raise MetalGeometryError(
                "aperture_tag selects only part of a coplanar mouth patch; "
                f"boundary edge {edge} is shared with coplanar triangle {other_index}"
            )
        rim_edges += 1

    seed = next(iter(aperture_indices))
    visited = {seed}
    pending = [seed]
    while pending:
        current = pending.pop()
        for adjacent in aperture_adjacency[current] - visited:
            visited.add(adjacent)
            pending.append(adjacent)
    if visited != aperture_indices:
        raise MetalGeometryError(
            "aperture_tag triangles must form one edge-connected mouth patch"
        )
    if rim_edges == 0:
        raise MetalGeometryError(
            "aperture patch has no physical rim shared with cavity wall triangles"
        )


def _validate_open_edges_on_symmetry_planes(
    coords_3xn: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
    symmetry_plane: str,
) -> None:
    edges = open_boundary_edges(triangles_nx3)
    if edges.size == 0:
        return

    components: list[int] = []
    labels: list[str] = []
    if "yz" in symmetry_plane:
        components.append(0)
        labels.append("X=0")
    if "xz" in symmetry_plane:
        components.append(1)
        labels.append("Y=0")
    if symmetry_plane == "xy":
        components.append(2)
        labels.append("Z=0")

    # Buffers normally contain exact zeros from the adapter snap, but compute
    # this validation tolerance from the current mesh as well. Reusing a value
    # derived from a previous call would make validation order-dependent when a
    # process handles a millimetre fixture and a metre-scale horn in succession.
    plane_tolerance = _plane_snap_tolerance(coords_3xn)
    on_requested_plane = np.zeros(edges.shape[0], dtype=bool)
    for component in components:
        edge_values = coords_3xn[component, edges]
        on_requested_plane |= np.all(
            np.abs(edge_values) <= plane_tolerance,
            axis=1,
        )

    if np.all(on_requested_plane):
        return

    first = int(np.flatnonzero(~on_requested_plane)[0])
    edge = tuple(int(v) for v in edges[first])
    requested = " or ".join(labels)
    raise MetalGeometryError(
        f"native_symmetry_plane={symmetry_plane!r} requires every open boundary "
        f"edge to lie on {requested}; boundary edge {first} with vertices {edge} "
        "is off the requested symmetry plane(s). If this rim is a real free edge "
        "of the full (reduced + mirrored) geometry (e.g. an open horn mouth), set "
        "SolveConfig.native_check_open_edges=False (or pass check_open_edges=False "
        "to create_session)."
    )


def _require_vertices_3xn(grid: Any) -> NDArray[np.float64]:
    if not hasattr(grid, "vertices"):
        raise MetalGeometryError("grid must expose a vertices array")
    vertices = np.asarray(grid.vertices)
    if vertices.ndim != 2 or vertices.shape[0] != 3:
        raise MetalGeometryError(
            f"grid.vertices must have shape (3, n_vertices), got {vertices.shape}"
        )
    if vertices.shape[1] == 0:
        raise MetalGeometryError("grid.vertices must contain at least one vertex")
    vertices_f64 = np.asarray(vertices, dtype=np.float64)
    if not np.all(np.isfinite(vertices_f64)):
        raise MetalGeometryError("grid.vertices must contain only finite values")
    return vertices_f64


def _plane_snap_tolerance(vertices_3xn: NDArray[np.float64]) -> float:
    """Return the current mesh's scale-relative symmetry-plane tolerance."""
    lower = np.min(vertices_3xn, axis=1)
    upper = np.max(vertices_3xn, axis=1)
    bbox_diagonal = float(np.linalg.norm(upper - lower))
    return max(
        _PLANE_SNAP_ABSOLUTE_FLOOR,
        _PLANE_SNAP_RELATIVE_FACTOR * bbox_diagonal,
    )


def _require_triangles_3xm(
    grid: Any,
    n_vertices: int,
) -> NDArray[np.int32]:
    if not hasattr(grid, "elements"):
        raise MetalGeometryError("grid must expose an elements array")
    elements = np.asarray(grid.elements)
    if elements.ndim != 2 or elements.shape[0] != 3:
        raise MetalGeometryError(
            f"grid.elements must have shape (3, n_triangles), got {elements.shape}"
        )
    if elements.shape[1] == 0:
        raise MetalGeometryError("grid.elements must contain at least one triangle")
    triangles = _as_int32_array("grid.elements", elements)
    _validate_zero_based_indices("grid.elements", triangles, upper_bound=n_vertices)
    tri_rows = triangles.T
    repeated = (
        (tri_rows[:, 0] == tri_rows[:, 1])
        | (tri_rows[:, 1] == tri_rows[:, 2])
        | (tri_rows[:, 0] == tri_rows[:, 2])
    )
    if np.any(repeated):
        first = int(np.flatnonzero(repeated)[0])
        raise MetalGeometryError(
            f"grid.elements triangle {first} repeats a vertex index"
        )
    return np.ascontiguousarray(triangles, dtype=np.int32)


def _require_physical_tags(
    physical_tags: Any,
    n_triangles: int,
) -> NDArray[np.int32]:
    tags = _as_vector("physical_tags", physical_tags)
    if tags.shape != (n_triangles,):
        raise MetalGeometryError(
            "physical_tags must have one value per triangle: "
            f"expected {(n_triangles,)}, got {tags.shape}"
        )
    return np.ascontiguousarray(
        _as_int32_array("physical_tags", tags),
        dtype=np.int32,
    )


def _require_p1_local2global(
    p1_space: Any,
    n_triangles: int,
) -> NDArray[np.int32]:
    if not hasattr(p1_space, "local2global"):
        raise MetalGeometryError("p1_space must expose a local2global array")
    local2global_raw = np.asarray(p1_space.local2global)
    if local2global_raw.shape != (n_triangles, 3):
        raise MetalGeometryError(
            "p1_space.local2global must have shape (n_triangles, 3): "
            f"expected {(n_triangles, 3)}, got {local2global_raw.shape}"
        )
    local2global = _as_int32_array("p1_space.local2global", local2global_raw)
    _validate_zero_based_indices("p1_space.local2global", local2global)
    return np.ascontiguousarray(local2global, dtype=np.int32)


def _resolve_p1_dof_count(
    p1_space: Any,
    local2global: NDArray[np.int32],
) -> int:
    min_required = int(local2global.max()) + 1
    dof_count = getattr(p1_space, "global_dof_count", min_required)
    dof_count = _as_nonnegative_int("p1_space.global_dof_count", dof_count)
    if dof_count < min_required:
        raise MetalGeometryError(
            "p1_space.global_dof_count is smaller than local2global requires: "
            f"{dof_count} < {min_required}"
        )
    return dof_count


def _resolve_dp0_dof_count(dp0_space: Any | None, n_triangles: int) -> int:
    if dp0_space is None:
        return n_triangles
    dof_count = _as_nonnegative_int(
        "dp0_space.global_dof_count",
        getattr(dp0_space, "global_dof_count", n_triangles),
    )
    if dof_count != n_triangles:
        raise MetalGeometryError(
            "dp0_space.global_dof_count must equal n_triangles: "
            f"{dof_count} != {n_triangles}"
        )
    return dof_count


def _compute_areas_normals(
    vertices_3xn: NDArray[np.float64],
    triangles_3xm: NDArray[np.int32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    vertices_nx3 = vertices_3xn.T
    triangles_nx3 = triangles_3xm.T
    p0 = vertices_nx3[triangles_nx3[:, 0]]
    p1 = vertices_nx3[triangles_nx3[:, 1]]
    p2 = vertices_nx3[triangles_nx3[:, 2]]
    return _areas_normals_from_points(p0, p1, p2)


def _compute_areas_normals_and_max_edge(
    input_vertices_3xn: NDArray[np.float64],
    triangles_3xm: NDArray[np.int32],
    *,
    plane_snap_tolerance: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
    """Compute snapped geometry and the original mesh edge metric in one gather."""
    vertices_nx3 = input_vertices_3xn.T
    triangles_nx3 = triangles_3xm.T
    p0 = vertices_nx3[triangles_nx3[:, 0]]
    p1 = vertices_nx3[triangles_nx3[:, 1]]
    p2 = vertices_nx3[triangles_nx3[:, 2]]
    max_edge_m = max(
        float(np.max(np.linalg.norm(p1 - p0, axis=1))),
        float(np.max(np.linalg.norm(p2 - p1, axis=1))),
        float(np.max(np.linalg.norm(p0 - p2, axis=1))),
    )
    for points in (p0, p1, p2):
        points[np.abs(points) <= plane_snap_tolerance] = 0.0
    areas, normals = _areas_normals_from_points(p0, p1, p2)
    return areas, normals, max_edge_m


def _areas_normals_from_points(
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    cross = np.cross(p1 - p0, p2 - p0)
    twice_area = np.linalg.norm(cross, axis=1)
    degenerate = twice_area <= 0.0
    if np.any(degenerate):
        first = int(np.flatnonzero(degenerate)[0])
        raise MetalGeometryError(f"grid.elements triangle {first} has zero area")

    normals_nx3 = cross / twice_area[:, None]
    areas = 0.5 * twice_area
    if not np.all(np.isfinite(normals_nx3)) or not np.all(np.isfinite(areas)):
        raise MetalGeometryError("triangle areas/normals must be finite")
    return (
        np.ascontiguousarray(areas, dtype=np.float32),
        np.ascontiguousarray(normals_nx3.T, dtype=np.float32),
    )


def _validate_grid_element_count(grid: Any, n_triangles: int) -> None:
    if not hasattr(grid, "number_of_elements"):
        return
    count = _as_nonnegative_int("grid.number_of_elements", grid.number_of_elements)
    if count != n_triangles:
        raise MetalGeometryError(
            f"grid.number_of_elements={count} does not match elements shape "
            f"with {n_triangles} triangles"
        )


def _as_vector(name: str, value: Any) -> NDArray[Any]:
    array = np.asarray(value)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)
    raise MetalGeometryError(f"{name} must be a 1D array, got shape {array.shape}")


def _as_int32_array(name: str, value: Any) -> NDArray[np.int32]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise MetalGeometryError(f"{name} must contain integer values")
    if array.size == 0:
        raise MetalGeometryError(f"{name} must not be empty")

    min_value = int(array.min())
    max_value = int(array.max())
    info = np.iinfo(np.int32)
    if min_value < info.min or max_value > info.max:
        raise MetalGeometryError(
            f"{name} values must fit int32, got range [{min_value}, {max_value}]"
        )
    return np.asarray(array, dtype=np.int32)


def _validate_zero_based_indices(
    name: str,
    indices: NDArray[np.int32],
    *,
    upper_bound: int | None = None,
) -> None:
    min_value = int(indices.min())
    max_value = int(indices.max())
    if min_value < 0:
        raise MetalGeometryError(f"{name} indices must be zero-based and nonnegative")
    if min_value != 0:
        raise MetalGeometryError(
            f"{name} indices must be compact zero-based arrays with minimum 0; "
            f"got minimum {min_value}"
        )
    if upper_bound is not None and max_value >= upper_bound:
        raise MetalGeometryError(
            f"{name} indices must be zero-based and less than {upper_bound}; "
            f"got max {max_value}"
        )


def _as_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise MetalGeometryError(f"{name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise MetalGeometryError(f"{name} must be an integer") from exc
    if integer < 0:
        raise MetalGeometryError(f"{name} must be nonnegative")
    return integer

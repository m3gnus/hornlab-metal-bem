"""Geometry buffer export for the future Metal backend.

This module is deliberately pure Python/NumPy. It does not import Bempp,
Swift, or Metal; callers pass Bempp-like objects that already expose the
metadata needed by the adapter.

All index arrays at this Python boundary are **zero-based**. Production Python
code must not emit one-based triangle or DOF indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import NATIVE_SYMMETRY_PLANES


class MetalGeometryError(ValueError):
    """Raised when grid/space metadata cannot satisfy the Metal data contract."""


# Vertex coordinates this close to zero are snapped to exactly 0.0. The Swift
# helper quantizes vertex coordinates with a 1e-6 tolerance when matching
# mirrored image vertices for singular-pair detection, while Python symmetry
# validation uses 1e-7; without snapping, a CAD vertex at e.g. z=5e-7 passes
# neither as on-plane nor mirrors onto itself, so image Duffy pairs silently
# fail to fire. Must stay aligned with coordinateKey() in the native helper.
_PLANE_SNAP_TOLERANCE = 1.0e-6


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
    vertices_f64 = _require_vertices_3xn(grid)
    vertices_f64 = vertices_f64.copy()
    vertices_f64[np.abs(vertices_f64) <= _PLANE_SNAP_TOLERANCE] = 0.0
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

    triangle_areas_f32, triangle_normals_3xm_f32 = _compute_areas_normals(
        vertices_f64,
        triangles_i32,
    )

    return MetalGeometryBuffers(
        vertices_3xn_f32=np.ascontiguousarray(vertices_f64, dtype=np.float32),
        triangles_3xm_i32=np.ascontiguousarray(triangles_i32, dtype=np.int32),
        physical_tags_i32=physical_tags_i32,
        p1_local2global_i32=p1_local2global_i32,
        triangle_areas_f32=triangle_areas_f32,
        triangle_normals_3xm_f32=triangle_normals_3xm_f32,
        p1_dof_count=p1_dof_count,
        dp0_dof_count=dp0_dof_count,
    )


def validate_native_symmetry_plane(
    buffers: MetalGeometryBuffers,
    symmetry_plane: str | None,
    *,
    tolerance: float = 1.0e-7,
) -> str | None:
    """Validate the narrow native reduced-domain symmetry contract.

    The native symmetry path supports caller-supplied positive-side reduced
    meshes mirrored across YZ (X symmetry), XZ (Y symmetry), XY (Z symmetry),
    or YZ+XZ. Symmetry planes are not real BEM boundaries, so triangles lying
    entirely on a requested plane are rejected.
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
    return plane


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

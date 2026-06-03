from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ObservationConfig

logger = logging.getLogger(__name__)


@dataclass
class ObservationFrame:
    """Reference frame for observation point construction.

    axis: unit vector from throat toward mouth (forward radiation direction)
    origin: measurement origin point (mouth or throat centre)
    u: horizontal transverse unit vector
    v: vertical transverse unit vector
    mouth_center: mouth centroid (always computed)
    source_center: throat/driver centroid
    """
    axis: NDArray[np.float64]
    origin: NDArray[np.float64]
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    mouth_center: NDArray[np.float64]
    source_center: NDArray[np.float64]


def _principal_axis(vertices: NDArray[np.float64], center: NDArray[np.float64]) -> NDArray[np.float64]:
    """PCA principal axis of a vertex cloud centred on ``center``.

    Used as a last-resort axis estimate when source-element normals are
    unavailable or degenerate.
    """
    centered = vertices - center[None, :]
    if centered.shape[0] == 0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cov = centered.T @ centered
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return axis / norm


def infer_frame(
    grid,
    physical_tags: NDArray[np.int32],
    source_tag: int = 2,
    origin_at: str = "mouth",
    symmetry_plane: str | None = None,
) -> ObservationFrame:
    """Infer radiation reference frame from mesh geometry.

    Uses source-element normals to determine the forward axis, then
    identifies the mouth as the mesh extreme along that axis.

    Args:
        grid: BEM grid object with ``.vertices`` (3, N) and ``.elements``
            (3, M) attributes.
        physical_tags: per-element tag array (shape (M,)).
        source_tag: physical tag of the driver/source disc elements.
            Raises ``ValueError`` if no element carries this tag.
        origin_at: ``"mouth"`` (default, IEC 60268-5) measures from the
            radiating aperture; ``"throat"`` measures from the source
            centroid.
        symmetry_plane: optional plane identifier. When set, the observation
            origin is projected onto requested image planes (X=0 for ``yz``,
            Y=0 for ``xz``, Z=0 for legacy ``xy``) for image-source physics
            in half/quarter models.

    Returns:
        ObservationFrame with axis/origin/u/v and diagnostic mouth_center
        and source_center.
    """
    vertices = np.array(grid.vertices.T, dtype=np.float64)
    elements = np.array(grid.elements.T, dtype=np.int32)

    # Axis from source element normals
    source_mask = physical_tags == source_tag
    if not np.any(source_mask):
        raise ValueError(f"No elements with tag {source_tag} in mesh")

    source_elems = elements[source_mask]

    # Defensive: drop element rows that index outside the vertex array.
    # Canonical meshes are clean, but defensive validation matches WG's
    # behaviour for legacy/external meshes with stale element indices.
    vertex_count = vertices.shape[0]
    valid_elem_mask = np.all(
        (source_elems >= 0) & (source_elems < vertex_count), axis=1
    )
    source_elems = source_elems[valid_elem_mask]

    avg_normal = None
    source_center = np.mean(vertices, axis=0)
    if source_elems.shape[0] > 0:
        p0 = vertices[source_elems[:, 0]]
        p1 = vertices[source_elems[:, 1]]
        p2 = vertices[source_elems[:, 2]]

        edges1 = p1 - p0
        edges2 = p2 - p0
        normals = np.cross(edges1, edges2)
        areas = np.linalg.norm(normals, axis=1)
        valid_area_mask = areas > 1e-15
        if np.any(valid_area_mask):
            normals = normals[valid_area_mask]
            areas = areas[valid_area_mask]
            centroids = (p0[valid_area_mask] + p1[valid_area_mask] + p2[valid_area_mask]) / 3.0

            # Sign-align normals into one hemisphere so mixed winding does
            # not cancel the axis. Matches WG's robust axis detection.
            ref = normals[0]
            signs = np.sign(normals @ ref)
            signs[signs == 0] = 1.0

            normals_sum = np.sum(normals * signs[:, None], axis=0)
            axis_norm = float(np.linalg.norm(normals_sum))
            if axis_norm > 1e-12:
                avg_normal = normals_sum / axis_norm
                # Source centroid (area-weighted)
                source_center = np.average(centroids, weights=areas, axis=0)

    # Fall back to PCA principal axis if no usable source normal.
    if avg_normal is None:
        avg_normal = _principal_axis(vertices, source_center)
        source_from_tags = False
    else:
        source_from_tags = True

    # Determine forward axis: should point away from source toward mouth.
    # Project all vertices along avg_normal; mouth is at the extreme.
    projections = vertices @ avg_normal
    source_proj = source_center @ avg_normal
    max_proj = projections.max()
    min_proj = projections.min()

    span = max_proj - min_proj
    if span < 1e-12:
        axis = avg_normal.copy()
    else:
        source_from_min = abs(source_proj - min_proj) / span
        source_from_max = abs(source_proj - max_proj) / span

        if source_from_tags and min(source_from_min, source_from_max) > 0.25:
            # Source is near the midpoint (enclosed geometry where horn
            # throat sits inside a larger enclosure). Trust the source
            # element normal direction rather than the extent heuristic.
            axis = avg_normal.copy()
            logger.info(
                "Enclosed geometry detected (source at %.0f%% of span), "
                "using source normal for axis",
                100 * source_from_min,
            )
        elif source_from_min < source_from_max:
            # Source near min projection: normal already points forward
            axis = avg_normal.copy()
        else:
            # Source near max projection: flip to point forward
            axis = -avg_normal

    # Mouth centre: vertices near the max projection along axis
    proj_along_axis = vertices @ axis
    mouth_threshold = proj_along_axis.max() - 0.02 * (
        proj_along_axis.max() - proj_along_axis.min()
    )
    mouth_verts = vertices[proj_along_axis >= mouth_threshold]
    mouth_center = mouth_verts.mean(axis=0)

    # Transverse vectors via Gram-Schmidt
    ref_x = np.array([1.0, 0.0, 0.0])
    ref_y = np.array([0.0, 1.0, 0.0])
    ref = ref_x if abs(np.dot(axis, ref_x)) < 0.9 else ref_y
    u = ref - np.dot(ref, axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    origin = mouth_center.copy() if origin_at == "mouth" else source_center.copy()

    # Project origin onto symmetry plane for half/quarter models. The
    # half-mesh has vertices at X>=0 (yz symmetry) or Z>=0 (xy), but the
    # effective acoustic centre of the full model is on the plane.
    if symmetry_plane is not None:
        plane = str(symmetry_plane).strip().lower()
        if plane == "yz":
            origin[0] = 0.0
        elif plane == "xz":
            origin[1] = 0.0
        elif plane == "yz+xz":
            origin[0] = 0.0
            origin[1] = 0.0
        elif plane == "xy":
            origin[2] = 0.0

    logger.info(
        "Frame: axis=[%.3f,%.3f,%.3f], origin=%s",
        *axis, origin_at,
    )

    return ObservationFrame(
        axis=axis, origin=origin, u=u, v=v,
        mouth_center=mouth_center, source_center=source_center,
    )


def build_observation_points(
    frame: ObservationFrame,
    config: ObservationConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build observation point arrays on polar arcs.

    When ``config.custom_points`` is set, those arrays are returned directly
    (all planes must have the same number of points). Angles are synthesised
    from ``angle_min/max/count``.

    Returns:
        points: (P, N_angles, 3) array of observation positions
        angles_deg: (N_angles,) array of angles in degrees
    """
    if config.custom_points is not None:
        # Custom observation grids — caller provides exact coordinates.
        angles_deg = np.linspace(
            config.angle_min_deg, config.angle_max_deg, config.angle_count,
        )
        plane_points = []
        for plane in config.planes:
            if plane not in config.custom_points:
                raise ValueError(
                    f"custom_points missing plane {plane!r}; "
                    f"available: {list(config.custom_points.keys())}"
                )
            pts = np.asarray(config.custom_points[plane], dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(
                    f"custom_points[{plane!r}] must be (N, 3), got {pts.shape}"
                )
            plane_points.append(pts)

        # Validate uniform point count across planes
        counts = [p.shape[0] for p in plane_points]
        if len(set(counts)) > 1:
            raise ValueError(
                f"All custom_points planes must have the same number of "
                f"points, got {dict(zip(config.planes, counts))}"
            )
        # Override angle_count to match actual custom point count
        if counts[0] != config.angle_count:
            angles_deg = np.linspace(
                config.angle_min_deg, config.angle_max_deg, counts[0],
            )

        points = np.stack(plane_points, axis=0)  # (P, N_points, 3)
        return points, angles_deg

    angles_deg = np.linspace(
        config.angle_min_deg, config.angle_max_deg, config.angle_count,
    )
    angles_rad = np.deg2rad(angles_deg)
    r = config.distance_m

    plane_points = []

    for plane in config.planes:
        if plane == "horizontal":
            transverse = frame.u
        elif plane == "vertical":
            transverse = frame.v
        elif plane == "diagonal":
            transverse = (frame.u + frame.v) / np.sqrt(2)
        else:
            raise ValueError(f"Unknown plane: {plane!r}")

        # theta=0 is on-axis (along frame.axis), theta=180 is rear
        pts = (
            frame.origin[None, :]
            + r * np.cos(angles_rad)[:, None] * frame.axis[None, :]
            + r * np.sin(angles_rad)[:, None] * transverse[None, :]
        )
        plane_points.append(pts)

    points = np.stack(plane_points, axis=0)  # (P, N_angles, 3)
    return points, angles_deg

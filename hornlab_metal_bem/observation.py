from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import ObservationConfig

logger = logging.getLogger(__name__)

_MIRROR_CLASS_TOLERANCE = 1.0e-7


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
    centered = vertices - center
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


# symmetry_plane here means a mirror-reduced mesh; full image-method
# callers with native symmetry must pass frame_override instead.
def _project_to_symmetry_subspace(
    vector: NDArray[np.float64],
    symmetry_plane: str | None,
) -> NDArray[np.float64]:
    """Zero components normal to requested native symmetry planes."""
    projected = np.array(vector, dtype=np.float64, copy=True)
    if symmetry_plane is None:
        return projected

    plane = str(symmetry_plane).strip().lower()
    if plane == "yz":
        projected[0] = 0.0
    elif plane == "xz":
        projected[1] = 0.0
    elif plane == "yz+xz":
        projected[0] = 0.0
        projected[1] = 0.0
    elif plane == "xy":
        projected[2] = 0.0
    return projected


def infer_frame(
    grid,
    physical_tags: NDArray[np.int32],
    source_tag: int = 2,
    origin_at: str = "mouth",
    symmetry_plane: str | None = None,
) -> ObservationFrame:
    """Infer radiation reference frame from mesh geometry.

    Treats a usable source-element normal as the authoritative forward axis,
    then identifies the mouth as the mesh extreme along that axis. Only the
    PCA fallback, where source winding is unavailable, uses mesh extents to
    resolve the otherwise arbitrary axis sign.

    Args:
        grid: BEM grid object with ``.vertices`` (3, N) and ``.elements``
            (3, M) attributes.
        physical_tags: per-element tag array (shape (M,)).
        source_tag: physical tag of the driver/source disc elements.
            Raises ``ValueError`` if no element carries this tag.
        origin_at: ``"mouth"`` (default, IEC 60268-5) measures from the
            radiating aperture; ``"throat"`` measures from the source
            centroid.
        symmetry_plane: optional plane identifier. When set for a
            mirror-reduced mesh, the inferred axis and observation origin are
            projected onto requested image planes (X=0 for ``yz``, Y=0 for
            ``xz``, Z=0 for legacy ``xy``) for image-source physics in
            half/quarter models.

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
            # not cancel the axis. Anchor the hemisphere to the largest face
            # so a tiny winding outlier cannot reverse an enclosed model's
            # inferred forward direction merely by appearing first.
            ref = normals[int(np.argmax(areas))]
            signs = np.sign(normals @ ref)
            signs[signs == 0] = 1.0

            normals_sum = np.sum(normals * signs[:, None], axis=0)
            axis_norm = float(np.linalg.norm(normals_sum))
            if axis_norm > 1e-12:
                avg_normal = normals_sum / axis_norm
                # Source centroid (area-weighted)
                source_center = np.average(centroids, weights=areas, axis=0)

    # Fall back to PCA principal axis if no usable source normal.
    source_from_tags = avg_normal is not None
    if not source_from_tags:
        avg_normal = _principal_axis(vertices, source_center)

    if symmetry_plane is not None:
        # Full-model axes are mirror-invariant; reduced normals/PCA can be
        # quadrant-biased, so constrain the axis before resolving its sign.
        projected_normal = _project_to_symmetry_subspace(avg_normal, symmetry_plane)
        projected_norm = float(np.linalg.norm(projected_normal))
        if projected_norm > 1e-12:
            avg_normal = projected_normal / projected_norm

    if source_from_tags:
        # Canonical meshes define the source-cap winding semantically: its
        # normal points from the throat toward the mouth. Mesh extents cannot
        # safely second-guess that contract once a cabinet surrounds the horn.
        axis = avg_normal.copy()
    else:
        # PCA eigenvectors have an arbitrary sign. Preserve the legacy extent
        # resolution only for this no-usable-source-normal fallback.
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
            axis = (
                avg_normal.copy()
                if source_from_min < source_from_max
                else -avg_normal
            )

    # Mouth centre: vertices near the max projection along axis
    proj_along_axis = vertices @ axis
    mouth_min = proj_along_axis.min()
    mouth_max = proj_along_axis.max()
    mouth_threshold = mouth_max - 0.02 * (mouth_max - mouth_min)
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
        origin = _project_to_symmetry_subspace(origin, symmetry_plane)

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

        points = np.stack(plane_points, axis=0)  # (P, N_points, 3)
        angles_deg = np.linspace(
            config.angle_min_deg, config.angle_max_deg, counts[0],
        )
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


def build_sphere_grid_points(
    frame: ObservationFrame,
    config: ObservationConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the frame-relative balloon grid requested by ``sphere_grid``.

    theta is the polar angle from ``frame.axis`` (0 = on-axis, matching the
    polar arcs), phi rotates around the axis with phi=0 along ``frame.u``
    (horizontal) and phi=90 along ``frame.v`` (vertical), so the grid's
    phi=0/180 meridian coincides with the horizontal arc and phi=90/270 with
    the vertical arc. Points sit at ``distance_m`` from ``frame.origin``.

    Returns:
        points: (n_theta * n_phi, 3) absolute coordinates, theta-major
        theta_deg: (n_theta * n_phi,) polar angle per point
        phi_deg: (n_theta * n_phi,) azimuth per point in [0, 360)
    """
    if config.sphere_grid is None:
        raise ValueError("build_sphere_grid_points requires sphere_grid to be set")
    n_theta, n_phi = config.sphere_grid
    theta_deg = np.linspace(0.0, float(config.sphere_theta_max_deg), int(n_theta))
    phi_deg = np.arange(int(n_phi), dtype=np.float64) * (360.0 / int(n_phi))

    theta_rad = np.deg2rad(np.repeat(theta_deg, int(n_phi)))
    phi_rad = np.deg2rad(np.tile(phi_deg, int(n_theta)))

    sin_theta = np.sin(theta_rad)
    directions = (
        (sin_theta * np.cos(phi_rad))[:, None] * frame.u[None, :]
        + (sin_theta * np.sin(phi_rad))[:, None] * frame.v[None, :]
        + np.cos(theta_rad)[:, None] * frame.axis[None, :]
    )
    points = frame.origin[None, :] + config.distance_m * directions
    return (
        np.ascontiguousarray(points, dtype=np.float64),
        np.rad2deg(theta_rad),
        np.rad2deg(phi_rad),
    )


def sphere_grid_solid_angle_weights(
    n_theta: int,
    n_phi: int,
    theta_max_deg: float = 180.0,
) -> NDArray[np.float64]:
    """Return exact cell solid angles for a theta/phi balloon grid.

    Theta samples include both endpoints. Each sample owns the band bounded by
    the midpoints to its neighbours, clipped to ``0`` and ``theta_max_deg``;
    integrating ``sin(theta)`` over those bands gives the exact cos-edge
    weights. Phi is periodic, so every column owns exactly ``2*pi/n_phi``.
    The flattened result is theta-major, matching
    :func:`build_sphere_grid_points`.
    """
    n_theta = int(n_theta)
    n_phi = int(n_phi)
    theta_max = float(theta_max_deg)
    if n_theta < 2 or n_phi < 3:
        raise ValueError("sphere grid requires n_theta >= 2 and n_phi >= 3")
    if not np.isfinite(theta_max) or not (0.0 < theta_max <= 180.0):
        raise ValueError("theta_max_deg must be finite and in (0, 180]")

    theta = np.linspace(0.0, np.deg2rad(theta_max), n_theta)
    edges = np.empty(n_theta + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = np.deg2rad(theta_max)
    edges[1:-1] = 0.5 * (theta[:-1] + theta[1:])
    band_weights = np.cos(edges[:-1]) - np.cos(edges[1:])
    phi_width = 2.0 * np.pi / n_phi
    return np.repeat(band_weights * phi_width, n_phi).astype(
        np.float64,
        copy=False,
    )


def integrate_sphere_radiated_power(
    pressure_complex: NDArray[np.complexfloating],
    *,
    distance_m: float,
    solid_angle_weights_sr: NDArray[np.float64],
    air_density: float,
    speed_of_sound: float,
) -> NDArray[np.float64]:
    """Integrate far-field acoustic intensity over a sampled sphere.

    ``pressure_complex`` may have any leading dimensions; its final dimension
    must match the solid-angle weights. The returned array has those leading
    dimensions and units of watts.
    """
    pressure = np.asarray(pressure_complex)
    weights = np.asarray(solid_angle_weights_sr, dtype=np.float64).reshape(-1)
    if pressure.ndim < 1 or pressure.shape[-1] != weights.size:
        raise ValueError(
            "pressure final dimension must match solid_angle_weights_sr"
        )
    radius = float(distance_m)
    rho = float(air_density)
    sound_speed = float(speed_of_sound)
    if not (np.isfinite(radius) and radius > 0.0):
        raise ValueError("distance_m must be finite and positive")
    if not (np.isfinite(rho) and rho > 0.0):
        raise ValueError("air_density must be finite and positive")
    if not (np.isfinite(sound_speed) and sound_speed > 0.0):
        raise ValueError("speed_of_sound must be finite and positive")
    intensity = np.square(np.abs(pressure), dtype=np.float64) / (
        2.0 * rho * sound_speed
    )
    return np.asarray(
        radius * radius * np.sum(intensity * weights, axis=-1),
        dtype=np.float64,
    )


def build_mirror_evaluation_classes(
    points: NDArray[np.float64],
    symmetry_plane: str,
    *,
    tolerance: float = _MIRROR_CLASS_TOLERANCE,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Return mirror-orbit representatives and a full-grid scatter inverse.

    Classes are constructed solely from absolute coordinates. Coordinates are
    quantised at ``tolerance`` before active components are canonicalised with
    ``abs``. A class is merged only when every distinct reflected coordinate in
    its requested orbit is present in the input. Consequently a grid whose
    observation frame is not aligned to the native planes remains exact: any
    missing mirror makes those samples singleton classes instead of assuming a
    theta/phi relationship that the coordinates do not have.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must be finite")
    tol = float(tolerance)
    if not (np.isfinite(tol) and tol > 0.0):
        raise ValueError("tolerance must be finite and positive")

    plane = str(symmetry_plane).strip().lower()
    if plane == "yz":
        components = (0,)
    elif plane == "xz":
        components = (1,)
    elif plane == "yz+xz":
        components = (0, 1)
    else:
        raise ValueError("symmetry_plane must be 'yz', 'xz', or 'yz+xz'")

    quantized = np.rint(pts / tol).astype(np.int64)
    coordinate_keys = [tuple(row.tolist()) for row in quantized]
    available_keys = set(coordinate_keys)

    canonical_groups: dict[tuple[int, int, int], list[int]] = {}
    for index, row in enumerate(quantized):
        canonical = row.copy()
        for component in components:
            canonical[component] = abs(canonical[component])
        canonical_groups.setdefault(tuple(canonical.tolist()), []).append(index)

    representatives: list[int] = []
    inverse = np.empty(pts.shape[0], dtype=np.intp)
    for members in canonical_groups.values():
        seed = quantized[members[0]]
        expected: set[tuple[int, int, int]] = set()
        for mask in range(1 << len(components)):
            reflected = seed.copy()
            for bit, component in enumerate(components):
                if mask & (1 << bit):
                    reflected[component] = -reflected[component]
            expected.add(tuple(reflected.tolist()))

        if expected.issubset(available_keys):
            representative_index = len(representatives)
            representatives.append(members[0])
            inverse[np.asarray(members, dtype=np.intp)] = representative_index
        else:
            for member in members:
                inverse[member] = len(representatives)
                representatives.append(member)

    return (
        np.ascontiguousarray(pts[np.asarray(representatives, dtype=np.intp)]),
        inverse,
    )

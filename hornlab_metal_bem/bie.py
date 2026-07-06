"""Pure acoustic helper routines used by the native Metal solver path."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import SolveConfig, VelocityMode


def _build_axial_face_scale(
    grid,
    physical_tags: NDArray[np.int32],
    source_tags,
    axis: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Per-face ``n_hat . axis`` projection for a rigid axial (piston) source.

    ``axis`` is the forward (throat->mouth) unit axis: pass the observation
    frame's axis, which is already oriented AND, on a symmetry-reduced mesh,
    projected onto the symmetry subspace so a half/quarter cap is not biased off
    the true axis. Each source face is scaled by the cosine between its outward
    normal and that axis -- 1 at the pole, tapering toward the rim. A flat disc
    (every normal along the axis) gives 1.0 on every face, so an axial source
    reduces exactly to the uniform-normal BC there. Faces outside the source tags
    stay 0.0 (never read by the Neumann builder).

    The projection is oriented PER SOURCE TAG so the cap drives outward-positive,
    matching the sign convention of the uniform-normal BC (which drives every
    face at +weight along its outward normal). This only flips a globally
    inverted axis (the frame axis points throat->mouth, which may be opposite the
    cap's outward normals); it is a single scalar sign per tag, so it does NOT
    rectify per-face signs -- a future front/back dipole tag keeps its opposite
    faces opposite.

    Returns ``None`` when the axis is degenerate or no source face is present, so
    the caller keeps the bit-for-bit uniform-normal path.
    """
    axis = np.asarray(axis, dtype=np.float64).reshape(-1)
    if axis.shape[0] != 3:
        return None
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
        return None
    axis = axis / axis_norm

    vertices = np.asarray(grid.vertices.T, dtype=np.float64)
    elements = np.asarray(grid.elements.T, dtype=np.int32)
    n_faces = elements.shape[0]

    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    p2 = vertices[elements[:, 2]]
    # Outward face normals (canonical meshes carry outward winding); the cross
    # magnitude is twice the triangle area (used to normalize per face and to
    # area-weight the per-tag orientation vote).
    raw = np.cross(p1 - p0, p2 - p0)
    mags = np.linalg.norm(raw, axis=1)

    scale = np.zeros(n_faces, dtype=np.float64)
    any_source = False
    for tag in sorted({int(t) for t in source_tags}):
        idx = np.where(physical_tags == tag)[0]
        if idx.size == 0:
            continue
        tag_mags = mags[idx]
        safe_mags = np.where(tag_mags > 1e-15, tag_mags, 1.0)
        unit_normals = raw[idx] / safe_mags[:, None]
        proj = unit_normals @ axis
        # Orient this tag outward-positive (area-weighted), matching the normal
        # BC. Flips the whole tag by one sign; per-face relative signs are kept.
        if float(np.dot(proj, tag_mags)) < 0.0:
            proj = -proj
        scale[idx] = proj
        any_source = True

    return scale if any_source else None


def _build_driver_neumann_coeffs(
    dp0_space,
    physical_tags: NDArray[np.int32],
    omega: float,
    config: SolveConfig,
    dtype: type,
    impedance_tags: set[int] | None = None,
    axial_face_scale: NDArray | None = None,
) -> NDArray:
    """Build DP0 Neumann coefficients for velocity source tags.

    ``impedance_tags`` is the resolved set of tags carrying a Robin (impedance)
    BC at this frequency. The caller (``sweep._build_neumann_rows``) evaluates
    ``impedance_source_callback`` EXACTLY ONCE per frequency upstream and passes
    the resolved tags here, so the skip set used to suppress the velocity BC is
    guaranteed identical to the Robin payload the solver actually applies — a
    stateful or non-repeatable callback can no longer make the two diverge into
    a double BC or an undriven tag. When ``None`` (back-compat: a direct call
    that did not resolve the tags upstream) the set is reconstructed locally from
    the static ``impedance_sources`` plus a single callback evaluation.

    ``axial_face_scale`` is the per-face ``n_hat . axis`` projection from
    ``_build_axial_face_scale`` (config.source_motion == "axial"). When ``None``
    (default / config.source_motion == "normal") every source face gets the same
    normal velocity -- the historical uniform-normal (breathing cap) BC, bit for
    bit unchanged. When supplied, each source face is driven at ``weight`` scaled
    by its projection, i.e. a rigid axial piston (full at the pole, tapering to
    the rim).
    """
    coeffs = np.zeros(dp0_space.global_dof_count, dtype=dtype)
    air_density = config.air_density
    frequency_hz = float(omega) / (2.0 * np.pi) if omega > 0 else 0.0
    velocity_sources = (
        config.velocity_source_callback(frequency_hz)
        if config.velocity_source_callback is not None
        else config.velocity_sources
    )
    # Skip prescribing a velocity BC on any tag carrying a Robin (impedance)
    # BC, otherwise the tag would receive a double boundary condition. The
    # resolved tag set is supplied by the caller (single callback evaluation per
    # frequency). The local-reconstruction fallback unions the STATIC
    # impedance_sources tags with any tags a single callback evaluation returns
    # for this frequency, so a callback-only Robin tag (absent from
    # impedance_sources) is still correctly skipped here.
    if impedance_tags is not None:
        impedance_tag_set = impedance_tags
    else:
        impedance_tag_set = set(config.impedance_sources.keys())
        if config.impedance_source_callback is not None:
            callback_betas = config.impedance_source_callback(frequency_hz)
            impedance_tag_set |= {int(tag) for tag in callback_betas.keys()}
    for tag, weight in velocity_sources.items():
        if tag in impedance_tag_set:
            continue
        mask = physical_tags == tag
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        if axial_face_scale is None:
            # Uniform normal velocity (breathing cap). Unchanged historical path.
            v_n = weight
            if config.velocity_mode == VelocityMode.ACCELERATION:
                v_n = weight / (1j * omega) if omega > 0 else 0.0
            coeffs[idx] = 1j * air_density * omega * v_n
        else:
            # Rigid axial (piston) motion: v_n(face) = weight * (n_hat . axis).
            v_n = weight * axial_face_scale[idx]
            if config.velocity_mode == VelocityMode.ACCELERATION:
                v_n = v_n / (1j * omega) if omega > 0 else np.zeros_like(v_n)
            coeffs[idx] = 1j * air_density * omega * v_n
    return coeffs


def _compute_impedance(
    grid,
    p_surface,
    physical_tags: NDArray[np.int32],
    p1_space,
    source_tag: int = 2,
) -> complex:
    """Area-weighted average complex surface pressure on ``source_tag``.

    ``integral(p dA) / integral(dA)`` in pascals per unit drive. This is not
    a true acoustic impedance: it is not divided by the drive velocity and is
    not normalised to ``rho*c``. The native helper computes the identical
    reduction in ``averageSurfacePressureForTag``; this is the Python fallback
    when the helper returns full surface pressure instead of reductions.
    """
    source_mask = physical_tags == source_tag
    source_elems = np.where(source_mask)[0]

    if len(source_elems) == 0:
        return 0.0 + 0.0j

    areas = np.asarray(grid.volumes)[source_elems]
    coeffs = np.asarray(p_surface.coefficients)

    local2global = np.asarray(p1_space.local2global)
    source_p1_dofs = local2global[source_elems]
    p_at_verts = coeffs[source_p1_dofs]
    p_avg = np.mean(p_at_verts, axis=1)

    total_force = np.sum(p_avg * areas)
    total_area = np.sum(areas)

    if abs(total_area) < 1e-30:
        return 0.0 + 0.0j

    return complex(total_force / total_area)


def compute_surface_pressure_avg(
    grid,
    p_surface,
    physical_tags: NDArray[np.int32],
    p1_space,
    tags: list[int],
) -> dict[int, complex]:
    """Area-weighted average surface pressure on each physical tag."""
    areas = np.asarray(grid.volumes)
    coeffs = np.asarray(p_surface.coefficients)
    local2global = np.asarray(p1_space.local2global)

    result = {}
    for tag in tags:
        mask = physical_tags == tag
        elem_indices = np.where(mask)[0]
        if len(elem_indices) == 0:
            result[tag] = 0.0 + 0.0j
            continue

        elem_areas = areas[elem_indices]
        p1_dofs = local2global[elem_indices]
        p_at_verts = coeffs[p1_dofs]
        p_avg_per_elem = np.mean(p_at_verts, axis=1)

        total_area = np.sum(elem_areas)
        if total_area < 1e-30:
            result[tag] = 0.0 + 0.0j
        else:
            result[tag] = complex(np.sum(p_avg_per_elem * elem_areas) / total_area)

    return result

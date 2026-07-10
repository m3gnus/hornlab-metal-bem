"""Pure acoustic helper routines used by the native Metal solver path."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import (
    AnnularProfile,
    AxialProfile,
    CallableProfile,
    NormalProfile,
    PerFaceProfile,
    SolveConfig,
    SourceMotion,
    TaperProfile,
    VelocityMode,
)


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


def _source_profile_for_tag(config: SolveConfig, tag: int):
    profile_map = {
        int(profile_tag): profile
        for profile_tag, profile in (config.source_velocity_profiles or {}).items()
    }
    if tag in profile_map:
        return profile_map[tag]
    if config.source_motion == SourceMotion.AXIAL:
        return AxialProfile()
    return NormalProfile()


def _normalize_profile_axis(axis: NDArray[np.float64]) -> NDArray[np.float64] | None:
    axis = np.asarray(axis, dtype=np.float64).reshape(-1)
    if axis.shape[0] != 3:
        return None
    axis_norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis_norm) or axis_norm <= 1e-12:
        return None
    return axis / axis_norm


def _tag_axial_projection(
    raw_normals: NDArray[np.float64],
    magnitudes: NDArray[np.float64],
    face_indices: NDArray[np.int64],
    axis: NDArray[np.float64],
) -> NDArray[np.float64]:
    tag_mags = magnitudes[face_indices]
    safe_mags = np.where(tag_mags > 1e-15, tag_mags, 1.0)
    unit_normals = raw_normals[face_indices] / safe_mags[:, None]
    proj = unit_normals @ axis
    if float(np.dot(proj, tag_mags)) < 0.0:
        proj = -proj
    return proj


def _normalized_tag_radius(
    centroids: NDArray[np.float64],
    face_indices: NDArray[np.int64],
    axis: NDArray[np.float64],
    source_center: NDArray[np.float64],
) -> NDArray[np.float64]:
    deltas = centroids[face_indices] - source_center[None, :]
    axial = np.outer(deltas @ axis, axis)
    radial = np.linalg.norm(deltas - axial, axis=1)
    radial_max = float(np.max(radial)) if radial.size else 0.0
    if not np.isfinite(radial_max) or radial_max <= 1e-15:
        return np.zeros_like(radial)
    return np.clip(radial / radial_max, 0.0, 1.0)


def _taper_values(t: NDArray[np.float64], profile: TaperProfile) -> NDArray[np.float64]:
    values = np.ones_like(t, dtype=np.float64)
    transition = t > profile.start
    if np.any(transition):
        x = np.clip((t[transition] - profile.start) / (1.0 - profile.start), 0.0, 1.0)
        if profile.kind == "linear":
            values[transition] = 1.0 - x
        else:
            values[transition] = 0.5 * (1.0 + np.cos(np.pi * x))
    values[t >= 1.0] = 0.0
    return values


def _build_source_face_scale(
    grid,
    physical_tags: NDArray[np.int32],
    config: SolveConfig,
    axis: NDArray[np.float64],
    source_center: NDArray[np.float64],
) -> NDArray | None:
    """Build per-face source velocity multipliers for configured source tags.

    Tags with ``source_velocity_profiles`` override ``config.source_motion``.
    Tags without a profile fall back to the legacy source motion: uniform normal
    (no scale array) or axial piston. Returns ``None`` only when all configured
    source tags are plain normal, preserving the historical scalar Neumann path.
    """
    profile_map = {
        int(profile_tag): profile
        for profile_tag, profile in (config.source_velocity_profiles or {}).items()
    }
    source_tags = sorted({int(tag) for tag in config.velocity_sources} | set(profile_map))
    if not source_tags:
        return None

    effective_profiles = {
        tag: _source_profile_for_tag(config, tag) for tag in source_tags
    }
    if all(isinstance(profile, NormalProfile) for profile in effective_profiles.values()):
        return None

    axis_unit = _normalize_profile_axis(axis)
    if axis_unit is None and all(
        isinstance(profile, (NormalProfile, AxialProfile))
        for profile in effective_profiles.values()
    ):
        return None
    axis_required = any(
        isinstance(profile, (TaperProfile, AnnularProfile, CallableProfile))
        for profile in effective_profiles.values()
    )
    if axis_unit is None and axis_required:
        raise ValueError("source velocity profiles require a non-degenerate axis")

    center = np.asarray(source_center, dtype=np.float64).reshape(-1)
    if center.shape[0] != 3:
        raise ValueError("source_center must have shape (3,)")

    vertices = np.asarray(grid.vertices.T, dtype=np.float64)
    elements = np.asarray(grid.elements.T, dtype=np.int32)
    n_faces = elements.shape[0]

    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    p2 = vertices[elements[:, 2]]
    raw = np.cross(p1 - p0, p2 - p0)
    mags = np.linalg.norm(raw, axis=1)
    safe_mags = np.where(mags > 1e-15, mags, 1.0)
    unit_normals = raw / safe_mags[:, None]
    centroids = (p0 + p1 + p2) / 3.0

    scale = np.zeros(n_faces, dtype=np.complex128)
    any_source = False
    saw_complex = False

    for tag in source_tags:
        idx = np.where(physical_tags == tag)[0]
        if idx.size == 0:
            continue
        profile = effective_profiles[tag]
        any_source = True
        # ``source_center`` belongs to the tag used for frame inference, while
        # configured source tags can be spatially separate. Derive an
        # area-weighted center independently for every tag so identical
        # translated drivers receive identical radial profiles. The supplied
        # center remains the fallback for a degenerate tag.
        tag_center = center
        tag_areas = mags[idx]
        area_sum = float(np.sum(tag_areas))
        if np.isfinite(area_sum) and area_sum > 1.0e-15:
            tag_center = np.average(centroids[idx], weights=tag_areas, axis=0)
        if isinstance(profile, NormalProfile):
            values = np.ones(idx.size, dtype=np.float64)
        elif isinstance(profile, AxialProfile):
            values = (
                np.ones(idx.size, dtype=np.float64)
                if axis_unit is None
                else _tag_axial_projection(raw, mags, idx, axis_unit)
            )
        elif isinstance(profile, TaperProfile):
            assert axis_unit is not None
            axial = _tag_axial_projection(raw, mags, idx, axis_unit)
            values = axial * _taper_values(
                _normalized_tag_radius(centroids, idx, axis_unit, tag_center),
                profile,
            )
        elif isinstance(profile, AnnularProfile):
            assert axis_unit is not None
            axial = _tag_axial_projection(raw, mags, idx, axis_unit)
            t = _normalized_tag_radius(centroids, idx, axis_unit, tag_center)
            annulus = (
                (t >= profile.r_inner) & (t <= profile.r_outer)
            ).astype(np.float64)
            values = axial * annulus
        elif isinstance(profile, PerFaceProfile):
            values = np.asarray(profile.weights, dtype=np.complex128)
            if values.ndim != 1 or values.shape[0] != idx.size:
                raise ValueError(
                    "PerFaceProfile.weights length must equal the number of "
                    f"faces for tag {tag}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError("PerFaceProfile.weights must be finite")
            saw_complex = saw_complex or bool(np.any(values.imag != 0.0))
        elif isinstance(profile, CallableProfile):
            assert axis_unit is not None
            values = np.asarray(
                profile.callback(
                    centroids[idx],
                    unit_normals[idx],
                    axis_unit.copy(),
                    np.asarray(tag_center, dtype=np.float64).copy(),
                ),
                dtype=np.complex128,
            )
            if values.ndim != 1 or values.shape[0] != idx.size:
                raise ValueError(
                    "CallableProfile.callback must return one weight per "
                    f"face for tag {tag}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError("CallableProfile.callback returned non-finite weights")
            saw_complex = saw_complex or bool(np.any(values.imag != 0.0))
        else:  # pragma: no cover - SolveConfig validation rejects this.
            raise ValueError(
                "source_velocity_profiles values must be SourceProfile instances"
            )
        scale[idx] = values

    if not any_source:
        return None
    if saw_complex:
        return scale
    return scale.real.astype(np.float64, copy=False)


def _build_driver_neumann_coeffs(
    dp0_space,
    physical_tags: NDArray[np.int32],
    omega: float,
    config: SolveConfig,
    dtype: type,
    impedance_tags: set[int] | None = None,
    axial_face_scale: NDArray | None = None,
    source_face_scale: NDArray | None = None,
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

    ``source_face_scale`` is the general per-face source multiplier from
    ``_build_source_face_scale``. ``axial_face_scale`` is kept as a back-compat
    alias for the original rigid-piston projection. When no scale is supplied
    (default / config.source_motion == "normal") every source face gets the same
    normal velocity -- the historical uniform-normal (breathing cap) BC, bit for
    bit unchanged.
    """
    if source_face_scale is not None:
        if axial_face_scale is not None:
            raise ValueError("pass only one of source_face_scale or axial_face_scale")
        axial_face_scale = source_face_scale

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
                # Under e^{-i omega t}, a*cos(omega t) integrates to
                # v = a/(-i omega), so q = i rho omega v = -rho a — the
                # momentum equation dp/dn = -rho a_n. Cross-validated against
                # ABEC3 absolute pressure (2026-07-09).
                v_n = weight / (-1j * omega) if omega > 0 else 0.0
            coeffs[idx] = 1j * air_density * omega * v_n
        else:
            # Rigid axial (piston) motion: v_n(face) = weight * (n_hat . axis).
            v_n = weight * axial_face_scale[idx]
            if config.velocity_mode == VelocityMode.ACCELERATION:
                v_n = v_n / (-1j * omega) if omega > 0 else np.zeros_like(v_n)
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

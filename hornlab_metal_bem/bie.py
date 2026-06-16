"""Pure acoustic helper routines used by the native Metal solver path."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import SolveConfig, VelocityMode


def _build_driver_neumann_coeffs(
    dp0_space,
    physical_tags: NDArray[np.int32],
    omega: float,
    config: SolveConfig,
    dtype: type,
    impedance_tags: set[int] | None = None,
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
        v_n = weight
        if config.velocity_mode == VelocityMode.ACCELERATION:
            v_n = weight / (1j * omega) if omega > 0 else 0.0
        coeffs[np.where(mask)[0]] = 1j * air_density * omega * v_n
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

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
) -> NDArray:
    """Build DP0 Neumann coefficients for velocity source tags."""
    coeffs = np.zeros(dp0_space.global_dof_count, dtype=dtype)
    air_density = config.air_density
    impedance_tag_set = set(config.impedance_sources.keys())
    for tag, weight in config.velocity_sources.items():
        if tag in impedance_tag_set:
            continue
        mask = physical_tags == tag
        if not np.any(mask):
            continue
        v_n = weight
        if config.velocity_mode is VelocityMode.ACCELERATION:
            v_n = weight / (1j * omega) if omega > 0 else 0.0
        coeffs[np.where(mask)[0]] = 1j * air_density * omega * v_n
    return coeffs


def _compute_impedance(
    grid,
    p_surface,
    physical_tags: NDArray[np.int32],
    p1_space,
    velocity_weights: NDArray | None = None,
    source_tag: int = 2,
) -> complex:
    """Throat impedance: ``Z = integral(p dA) / Q_eff``."""
    elements = np.asarray(grid.elements.T, dtype=np.int32)
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

    if velocity_weights is not None and len(velocity_weights) == len(source_elems):
        q_eff = np.sum(velocity_weights * areas)
    else:
        q_eff = np.sum(areas)

    if abs(q_eff) < 1e-30:
        return 0.0 + 0.0j

    return complex(total_force / q_eff)


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

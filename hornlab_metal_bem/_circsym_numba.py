"""Portable compiled remainder kernels for the axisymmetric solver.

This module is imported lazily by :mod:`hornlab_metal_bem.circsym`.  Keeping the
Numba import here avoids adding JIT startup work to ordinary package imports and
lets the pure NumPy reference remain the final, always-correct fallback.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def evaluate_far_remainder_onthefly(
    target_rho,
    target_z,
    source_rho,
    source_z,
    measure,
    normal_rho,
    normal_z,
    cos_psi,
    psi_weights,
    has_baffle,
    baffle_z,
    kr,
    ki,
):
    """Evaluate the smooth far remainder without materialising a 4-D tensor."""

    target_count = target_rho.shape[0]
    source_count = source_rho.shape[0]
    line_count = source_rho.shape[1]
    psi_count = cos_psi.shape[0]
    image_count = 2 if has_baffle else 1
    out_s = np.zeros((target_count, source_count), dtype=np.complex128)
    out_h = np.zeros((target_count, source_count), dtype=np.complex128)
    four_pi = 4.0 * math.pi

    for flat_index in prange(target_count * source_count):
        target_index = flat_index // source_count
        source_index = flat_index - target_index * source_count
        rt = target_rho[target_index]
        zt = target_z[target_index]
        nr = normal_rho[source_index]
        source_nz = normal_z[source_index]
        s_re = 0.0
        s_im = 0.0
        h_re = 0.0
        h_im = 0.0

        for image_index in range(image_count):
            nz = -source_nz if image_index else source_nz
            for line_index in range(line_count):
                source_measure = measure[source_index, line_index]
                if source_measure == 0.0:
                    continue
                rs = source_rho[source_index, line_index]
                base_zs = source_z[source_index, line_index]
                zs = 2.0 * baffle_z - base_zs if image_index else base_zs
                dz = zs - zt

                for psi_index in range(psi_count):
                    cp = cos_psi[psi_index]
                    r2 = rt * rt + rs * rs - 2.0 * rt * rs * cp + dz * dz
                    r2 = max(r2, 0.0)
                    distance = math.sqrt(r2)
                    if distance <= 1.0e-13:
                        continue

                    weighted_measure = (
                        source_measure * (2.0 * psi_weights[psi_index]) / four_pi
                    )
                    g_weight = weighted_measure / distance
                    numerator = (rs - rt * cp) * nr + dz * nz
                    h_weight = weighted_measure * numerator / (distance**3)
                    kr_r = kr * distance
                    ki_r = ki * distance
                    decay = math.exp(-ki_r)
                    phase_re = decay * math.cos(kr_r)
                    phase_im = decay * math.sin(kr_r)

                    s_re += (phase_re - 1.0) * g_weight
                    s_im += phase_im * g_weight
                    factor_re = -ki_r - 1.0
                    factor_im = kr_r
                    expr_re = phase_re * factor_re - phase_im * factor_im + 1.0
                    expr_im = phase_re * factor_im + phase_im * factor_re
                    h_re += expr_re * h_weight
                    h_im += expr_im * h_weight

        out_s[target_index, source_index] = complex(s_re, s_im)
        out_h[target_index, source_index] = complex(h_re, h_im)

    return out_s, out_h


@njit(parallel=True, cache=True)
def evaluate_field_onthefly(
    target_rho,
    target_z,
    source_rho,
    source_z,
    measure,
    normal_rho,
    normal_z,
    cos_psi,
    psi_weights,
    has_baffle,
    baffle_z,
    kr,
    ki,
):
    """Evaluate complete ordinary S/H field kernels without NumPy tensors.

    The arithmetic and quadrature match the NumPy reference in ``circsym.py``.
    Keeping the target/source/line/azimuth reduction inside compiled loops avoids
    allocating a target x source x azimuth complex array for every line node.
    Near-boundary pairs are still replaced by the caller's existing specialised
    quadrature, so this routine changes only the ordinary far-pair executor.
    """

    target_count = target_rho.shape[0]
    source_count = source_rho.shape[0]
    line_count = source_rho.shape[1]
    psi_count = cos_psi.shape[0]
    image_count = 2 if has_baffle else 1
    out_s = np.zeros((target_count, source_count), dtype=np.complex128)
    out_h = np.zeros((target_count, source_count), dtype=np.complex128)
    four_pi = 4.0 * math.pi

    for flat_index in prange(target_count * source_count):
        target_index = flat_index // source_count
        source_index = flat_index - target_index * source_count
        rt = target_rho[target_index]
        zt = target_z[target_index]
        nr = normal_rho[source_index]
        source_nz = normal_z[source_index]
        s_re = 0.0
        s_im = 0.0
        h_re = 0.0
        h_im = 0.0

        for image_index in range(image_count):
            nz = -source_nz if image_index else source_nz
            for line_index in range(line_count):
                source_measure = measure[source_index, line_index]
                if source_measure == 0.0:
                    continue
                rs = source_rho[source_index, line_index]
                base_zs = source_z[source_index, line_index]
                zs = 2.0 * baffle_z - base_zs if image_index else base_zs
                dz = zs - zt

                for psi_index in range(psi_count):
                    cp = cos_psi[psi_index]
                    r2 = rt * rt + rs * rs - 2.0 * rt * rs * cp + dz * dz
                    r2 = max(r2, 0.0)
                    distance = math.sqrt(r2)
                    if distance <= 1.0e-13:
                        continue

                    q_re = kr * distance
                    q_im = ki * distance
                    decay = math.exp(-q_im)
                    phase_re = decay * math.cos(q_re)
                    phase_im = decay * math.sin(q_re)
                    weighted_measure = (
                        source_measure * (2.0 * psi_weights[psi_index]) / four_pi
                    )
                    g_scale = weighted_measure / distance
                    s_re += phase_re * g_scale
                    s_im += phase_im * g_scale

                    numerator = (rs - rt * cp) * nr + dz * nz
                    h_scale = weighted_measure * numerator / (distance**3)
                    factor_re = -q_im - 1.0
                    factor_im = q_re
                    h_re += (
                        phase_re * factor_re - phase_im * factor_im
                    ) * h_scale
                    h_im += (
                        phase_re * factor_im + phase_im * factor_re
                    ) * h_scale

        out_s[target_index, source_index] = complex(s_re, s_im)
        out_h[target_index, source_index] = complex(h_re, h_im)

    return out_s, out_h


@njit(parallel=True, cache=True)
def evaluate_near_remainder(R, numerator, weight, kr, ki):
    """Evaluate the near-pair smooth remainder with the analytic small-q limit."""

    pair_count, node_count, psi_count = R.shape
    out_s = np.zeros(pair_count, dtype=np.complex128)
    out_h = np.zeros(pair_count, dtype=np.complex128)

    for pair_index in prange(pair_count):
        s_re = 0.0
        s_im = 0.0
        h_re = 0.0
        h_im = 0.0
        for node_index in range(node_count):
            for psi_index in range(psi_count):
                w = weight[pair_index, node_index, psi_index]
                if w == 0.0:
                    continue
                distance = R[pair_index, node_index, psi_index]
                if distance <= 1.0e-13:
                    s_re += -ki * w
                    s_im += kr * w
                    continue

                q_re = kr * distance
                q_im = ki * distance
                if math.hypot(q_re, q_im) < 1.0e-5:
                    z = complex(-q_im, q_re)
                    z2 = z * z
                    z3 = z2 * z
                    z4 = z3 * z
                    z5 = z4 * z
                    rem_g = (z + 0.5 * z2 + z3 / 6.0 + z4 / 24.0 + z5 / 120.0) / distance

                    q = complex(q_re, q_im)
                    q2 = q * q
                    q3 = q2 * q
                    q4 = q3 * q
                    q5 = q4 * q
                    expr = -0.5 * q2 - (1j / 3.0) * q3 + 0.125 * q4 + (1j / 30.0) * q5
                    remg_re = rem_g.real
                    remg_im = rem_g.imag
                    expr_re = expr.real
                    expr_im = expr.imag
                else:
                    decay = math.exp(-q_im)
                    phase_re = decay * math.cos(q_re)
                    phase_im = decay * math.sin(q_re)
                    remg_re = (phase_re - 1.0) / distance
                    remg_im = phase_im / distance
                    factor_re = -q_im - 1.0
                    factor_im = q_re
                    expr_re = phase_re * factor_re - phase_im * factor_im + 1.0
                    expr_im = phase_re * factor_im + phase_im * factor_re

                h_scale = (
                    w
                    * numerator[pair_index, node_index, psi_index]
                    / (distance**3)
                )
                s_re += remg_re * w
                s_im += remg_im * w
                h_re += expr_re * h_scale
                h_im += expr_im * h_scale

        out_s[pair_index] = complex(s_re, s_im)
        out_h[pair_index] = complex(h_re, h_im)

    return out_s, out_h


@njit(parallel=True, cache=True)
def evaluate_near_remainder_onthefly(
    target_rho,
    target_z,
    source_rho,
    source_z,
    measure,
    normal_rho,
    normal_z,
    cos_psi,
    psi_weights,
    has_baffle,
    baffle_z,
    kr,
    ki,
):
    """Evaluate adaptive near pairs from compact frequency-invariant geometry."""

    pair_count = target_rho.shape[0]
    node_count = source_rho.shape[1]
    psi_count = cos_psi.shape[0]
    image_count = 2 if has_baffle else 1
    out_s = np.zeros(pair_count, dtype=np.complex128)
    out_h = np.zeros(pair_count, dtype=np.complex128)
    four_pi = 4.0 * math.pi

    for pair_index in prange(pair_count):
        rt = target_rho[pair_index]
        zt = target_z[pair_index]
        nr = normal_rho[pair_index]
        source_nz = normal_z[pair_index]
        s_re = 0.0
        s_im = 0.0
        h_re = 0.0
        h_im = 0.0
        for image_index in range(image_count):
            nz = -source_nz if image_index else source_nz
            part_s_re = 0.0
            part_s_im = 0.0
            part_h_re = 0.0
            part_h_im = 0.0
            for node_index in range(node_count):
                source_measure = measure[pair_index, node_index]
                if source_measure == 0.0:
                    continue
                rs = source_rho[pair_index, node_index]
                base_zs = source_z[pair_index, node_index]
                zs = 2.0 * baffle_z - base_zs if image_index else base_zs
                dz = zs - zt
                for psi_index in range(psi_count):
                    cp = cos_psi[psi_index]
                    r2 = rt * rt + rs * rs - 2.0 * rt * rs * cp + dz * dz
                    distance = math.sqrt(max(r2, 0.0))
                    weight = (
                        source_measure * (2.0 * psi_weights[psi_index]) / four_pi
                    )
                    if distance <= 1.0e-13:
                        part_s_re += -ki * weight
                        part_s_im += kr * weight
                        continue

                    q_re = kr * distance
                    q_im = ki * distance
                    if math.hypot(q_re, q_im) < 1.0e-5:
                        z = complex(-q_im, q_re)
                        z2 = z * z
                        z3 = z2 * z
                        z4 = z3 * z
                        z5 = z4 * z
                        rem_g = (
                            z
                            + 0.5 * z2
                            + z3 / 6.0
                            + z4 / 24.0
                            + z5 / 120.0
                        ) / distance

                        q = complex(q_re, q_im)
                        q2 = q * q
                        q3 = q2 * q
                        q4 = q3 * q
                        q5 = q4 * q
                        expr = (
                            -0.5 * q2
                            - (1j / 3.0) * q3
                            + 0.125 * q4
                            + (1j / 30.0) * q5
                        )
                        remg_re = rem_g.real
                        remg_im = rem_g.imag
                        expr_re = expr.real
                        expr_im = expr.imag
                    else:
                        decay = math.exp(-q_im)
                        phase_re = decay * math.cos(q_re)
                        phase_im = decay * math.sin(q_re)
                        remg_re = (phase_re - 1.0) / distance
                        remg_im = phase_im / distance
                        factor_re = -q_im - 1.0
                        factor_im = q_re
                        expr_re = (
                            phase_re * factor_re - phase_im * factor_im + 1.0
                        )
                        expr_im = phase_re * factor_im + phase_im * factor_re

                    numerator = (rs - rt * cp) * nr + dz * nz
                    h_scale = weight * numerator / (distance**3)
                    part_s_re += remg_re * weight
                    part_s_im += remg_im * weight
                    part_h_re += expr_re * h_scale
                    part_h_im += expr_im * h_scale
            s_re += part_s_re
            s_im += part_s_im
            h_re += part_h_re
            h_im += part_h_im

        out_s[pair_index] = complex(s_re, s_im)
        out_h[pair_index] = complex(h_re, h_im)

    return out_s, out_h


__all__ = [
    "evaluate_far_remainder_onthefly",
    "evaluate_field_onthefly",
    "evaluate_near_remainder",
    "evaluate_near_remainder_onthefly",
]

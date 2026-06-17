"""Unit tests for hornlab_metal_bem.losses — pure helpers, no bempp/Metal."""
from __future__ import annotations

import math

import pytest

from hornlab_metal_bem import losses


def test_viscothermal_wall_beta_is_real_and_positive():
    beta = losses.viscothermal_wall_beta(665.0, hydraulic_radius_m=0.01)
    assert beta.imag == 0.0
    assert beta.real > 0.0


def test_viscothermal_wall_beta_scales_as_sqrt_f():
    # beta ~ omega * delta_v ~ omega / sqrt(omega) ~ sqrt(omega) ~ sqrt(f).
    b1 = losses.viscothermal_wall_beta(665.0, hydraulic_radius_m=0.01).real
    b4 = losses.viscothermal_wall_beta(4.0 * 665.0, hydraulic_radius_m=0.01).real
    assert b4 / b1 == pytest.approx(2.0, rel=1e-6)


def test_viscothermal_wall_beta_zero_and_negative_frequency():
    assert losses.viscothermal_wall_beta(0.0, hydraulic_radius_m=0.01) == 0.0 + 0.0j
    assert losses.viscothermal_wall_beta(-5.0, hydraulic_radius_m=0.01) == 0.0 + 0.0j


def test_viscothermal_wall_beta_passive_across_band():
    for f in (50.0, 200.0, 665.0, 2000.0, 10000.0):
        beta = losses.viscothermal_wall_beta(f, hydraulic_radius_m=0.005)
        assert beta.real >= 0.0
        assert math.isfinite(beta.real)


def test_beta_from_surface_impedance_air_matched():
    # Zs == rho0*c0 -> normalized admittance beta == 1.
    beta = losses.beta_from_surface_impedance(losses.RHO0 * losses.C0)
    assert beta == pytest.approx(1.0 + 0.0j)


def test_beta_from_surface_impedance_zero_rejected():
    # Zs -> 0 is pressure-release (infinite admittance), NOT rigid; reject it
    # rather than silently returning beta = 0 (the opposite boundary condition).
    with pytest.raises(ValueError, match="pressure-release"):
        losses.beta_from_surface_impedance(0)


def test_beta_from_surface_impedance_complex():
    zs = complex(2.0 * losses.RHO0 * losses.C0, 0.0)
    assert losses.beta_from_surface_impedance(zs) == pytest.approx(0.5 + 0.0j)


def test_material_beta_priors_are_passive():
    assert losses.MATERIAL_BETA_PRIORS  # non-empty table
    for name, beta in losses.MATERIAL_BETA_PRIORS.items():
        assert isinstance(name, str)
        assert complex(beta).real >= 0.0


def test_constant_beta_callback_is_frequency_independent():
    cb = losses.constant_beta_callback({8: 0.05 + 0.0j})
    assert cb(100.0) == {8: 0.05 + 0.0j}
    assert cb(20000.0) == {8: 0.05 + 0.0j}
    # Returns a copy each call (mutating the result must not leak).
    out = cb(500.0)
    out[8] = 0.99 + 0.0j
    assert cb(500.0) == {8: 0.05 + 0.0j}


def test_viscothermal_port_callback_targets_one_tag():
    cb = losses.viscothermal_port_callback(7, hydraulic_radius_m=0.01)
    result = cb(665.0)
    assert set(result) == {7}
    assert result[7].real > 0.0
    assert result[7].imag == 0.0

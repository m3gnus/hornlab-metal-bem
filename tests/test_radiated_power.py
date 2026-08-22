"""Pure acoustic-power integrator tests (no native helper required)."""
from __future__ import annotations

import numpy as np

from hornlab_metal_bem.bie import (
    integrate_driven_surface_power,
    normal_velocity_from_driver_neumann,
)
from hornlab_metal_bem.config import SolveConfig
from hornlab_metal_bem.sweep import _surface_power_from_tag_averages


def test_surface_power_sign_for_outward_radiation_e_minus_iwt():
    pressure = np.array([4.0 - 3.0j, 2.0 + 5.0j])
    velocity = np.array([0.5 + 0.0j, 0.25 + 0.0j])
    areas = np.array([2.0, 3.0])

    power = integrate_driven_surface_power(pressure, velocity, areas)

    # 0.5*Re(p*conj(v))*A is positive when pressure and outward velocity are
    # in phase under e^{-i*w*t}; the reactive pressure quadrature contributes 0.
    expected = 0.5 * (4.0 * 0.5 * 2.0 + 2.0 * 0.25 * 3.0)
    np.testing.assert_allclose(power, expected)
    assert float(power) > 0.0


def test_surface_power_scales_with_velocity_squared_and_face_area():
    radiation_resistance = 7.0
    velocity = np.array([1.0 + 2.0j, -0.5 + 0.25j])
    areas = np.array([0.2, 0.8])
    pressure = radiation_resistance * velocity

    baseline = integrate_driven_surface_power(pressure, velocity, areas)
    scaled = integrate_driven_surface_power(
        3.0 * pressure,
        3.0 * velocity,
        2.0 * areas,
    )
    expected = 0.5 * radiation_resistance * np.sum(np.abs(velocity) ** 2 * areas)

    np.testing.assert_allclose(baseline, expected)
    np.testing.assert_allclose(scaled, 18.0 * baseline)


def test_neumann_inverse_recovers_exact_velocity_rows():
    rho = 1.2
    omega = 2.0 * np.pi * np.array([100.0, 800.0])
    velocity = np.array(
        [[1.0 + 0.5j, 0.0], [-0.25j, 2.0 - 3.0j]],
        dtype=np.complex128,
    )
    neumann = 1j * rho * omega[:, None] * velocity

    recovered = normal_velocity_from_driver_neumann(neumann, omega, rho)

    np.testing.assert_allclose(recovered, velocity, rtol=2.0e-16, atol=2.0e-16)


def test_uniform_multi_tag_reduction_uses_exact_bc_and_area_averages():
    frequencies = [100.0, 200.0]
    rho = 1.2
    omega = 2.0 * np.pi * np.asarray(frequencies)
    physical_tags = np.array([2, 2, 3], dtype=np.int32)
    areas = np.array([0.25, 0.75, 2.0], dtype=np.float32)
    velocity = np.array(
        [[1.0 + 0.0j, 1.0 + 0.0j, 0.5 + 0.0j]] * 2,
        dtype=np.complex128,
    )
    neumann = (1j * rho * omega[:, None] * velocity).astype(np.complex64)
    pressure_avg = {
        2: np.array([4.0 + 2.0j, 8.0 - 1.0j]),
        3: np.array([3.0 - 5.0j, 6.0 + 2.0j]),
    }

    actual = _surface_power_from_tag_averages(
        surface_pressure_avg=pressure_avg,
        source_tags=[2, 3],
        driver_neumann_rows=neumann,
        completed_frequencies_hz=frequencies,
        physical_tags=physical_tags,
        face_areas_m2=areas,
        config=SolveConfig(air_density=rho),
    )
    expected = 0.5 * np.array(
        [
            4.0 * 1.0 * 1.0 + 3.0 * 0.5 * 2.0,
            8.0 * 1.0 * 1.0 + 6.0 * 0.5 * 2.0,
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-7)

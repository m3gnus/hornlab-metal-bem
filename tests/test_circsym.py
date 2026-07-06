from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j1, spherical_jn, spherical_yn

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem._constants import SPEED_OF_SOUND
from hornlab_metal_bem.circsym import (
    MeridianMesh,
    _integrate_segment_kernel,
    ring_kernel_m0,
)
from hornlab_metal_bem.config import (
    ObservationConfig,
    SolveConfig,
    SourceMotion,
    VelocityMode,
)


def _sphere_meridian(radius: float = 0.1, segments: int = 48) -> MeridianMesh:
    theta = np.linspace(0.0, np.pi, segments + 1)
    points = np.column_stack([radius * np.sin(theta), radius * np.cos(theta)])
    return MeridianMesh.from_polyline(points, tags=2)


def _piston_meridian(radius: float = 0.1, segments: int = 30) -> MeridianMesh:
    points = np.column_stack(
        [np.linspace(0.0, radius, segments + 1), np.zeros(segments + 1)]
    )
    return MeridianMesh.from_polyline(points, tags=2)


def _freq_for_ka(ka: float, radius: float = 0.1) -> float:
    return float(ka) * SPEED_OF_SOUND / (2.0 * np.pi * radius)


def _pulsating_sphere_impedance(ka: np.ndarray) -> np.ndarray:
    # With this package's e^(+ikR), q=+i*rho*omega*v convention, the textbook
    # e^(-iwt) impedance ika/(1+ika) appears conjugated.
    return np.conjugate(1j * ka / (1.0 + 1j * ka))


def _spherical_hankel1(order: int, x: float) -> complex:
    return complex(spherical_jn(order, x) + 1j * spherical_yn(order, x))


def _spherical_hankel1_derivative(order: int, x: float) -> complex:
    return complex(
        spherical_jn(order, x, derivative=True)
        + 1j * spherical_yn(order, x, derivative=True)
    )


def _brute_force_ring_kernel(
    target_rho: float,
    target_z: float,
    source_rho: float,
    source_z: float,
    source_normal: np.ndarray,
    k: complex,
    *,
    samples: int = 65_536,
) -> tuple[complex, complex]:
    psi = (np.arange(samples, dtype=np.float64) + 0.5) * (2.0 * np.pi / samples)
    cos_psi = np.cos(psi)
    r = np.sqrt(
        target_rho**2
        + source_rho**2
        - 2.0 * target_rho * source_rho * cos_psi
        + (target_z - source_z) ** 2
    )
    phase = np.exp(1j * k * r)
    g = phase / (4.0 * np.pi * r)
    drdn_num = (
        (source_rho - target_rho * cos_psi) * source_normal[0]
        + (source_z - target_z) * source_normal[1]
    )
    h = phase * (1j * k * r - 1.0) * drdn_num / (4.0 * np.pi * r**3)
    return complex((2.0 * np.pi / samples) * np.sum(g)), complex(
        (2.0 * np.pi / samples) * np.sum(h)
    )


def test_meridian_from_polyline_derives_outward_normals_and_validates_baffle():
    meridian = _piston_meridian(radius=0.2, segments=2)

    np.testing.assert_allclose(meridian.normals, [[0.0, 1.0], [0.0, 1.0]])
    assert meridian.nodes.shape[1] == 2
    assert "solve_circsym" in metal_bem.__all__
    assert "solve_circsym_frequencies" in metal_bem.__all__
    assert "MeridianMesh" in metal_bem.__all__

    assert SolveConfig(circsym_baffle_z=0.0).circsym_baffle_z == 0.0
    with pytest.raises(ValueError, match="circsym_baffle_z"):
        SolveConfig(circsym_baffle_z=float("nan"))


def test_ring_kernels_match_dense_azimuth_quadrature_off_diagonal_and_near():
    k = 23.0 + 0.15j
    normal = np.array([0.6, -0.8], dtype=np.float64)
    pairs = [
        (0.24, 0.03, 0.11, -0.07),
        (0.30, 0.10, 0.3008, 0.1012),
        (0.0, 0.04, 0.18, -0.02),
    ]

    for target_rho, target_z, source_rho, source_z in pairs:
        subtracted = ring_kernel_m0(
            target_rho,
            target_z,
            source_rho,
            source_z,
            normal,
            k,
            n_psi=192,
        )
        direct = _brute_force_ring_kernel(
            target_rho,
            target_z,
            source_rho,
            source_z,
            normal,
            k,
        )
        np.testing.assert_allclose(subtracted[0], direct[0], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(subtracted[1], direct[1], rtol=1e-6, atol=1e-8)

    meridian = _sphere_meridian(radius=0.1, segments=16)
    geom = meridian.segment_geometry()
    g_self, h_self = _integrate_segment_kernel(
        target_rho=float(geom.rho_mid[5]),
        target_z=float(geom.z_mid[5]),
        meridian=meridian,
        geom=geom,
        source_index=5,
        k=k,
        baffle_z=None,
        n_psi=96,
        target_index=5,
    )
    assert np.isfinite(g_self)
    assert np.isfinite(h_self)


def test_pulsating_sphere_recovers_analytic_impedance_and_uniform_directivity():
    radius = 0.1
    ka = np.array([0.5, 1.5, 3.0], dtype=np.float64)
    config = SolveConfig(
        velocity_sources={2: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        formulation="standard",
        observation=ObservationConfig(
            distance_m=4.0,
            angle_count=37,
            planes=["horizontal", "vertical", "diagonal"],
            origin="throat",
        ),
    )

    result = metal_bem.solve_circsym_frequencies(
        _sphere_meridian(radius=radius, segments=56),
        [_freq_for_ka(value, radius) for value in ka],
        config,
    )

    z_norm = result.impedance / (config.air_density * SPEED_OF_SOUND)
    np.testing.assert_allclose(z_norm, _pulsating_sphere_impedance(ka), rtol=4e-3)
    assert float(np.max(np.abs(result.directivity_db))) < 0.02
    np.testing.assert_allclose(result.pressure_complex[:, 0], result.pressure_complex[:, 1])
    np.testing.assert_allclose(result.pressure_complex[:, 0], result.pressure_complex[:, 2])


def test_rigid_oscillating_sphere_matches_first_order_series():
    radius = 0.1
    ka = np.array([1.0, 3.0], dtype=np.float64)
    meridian = _sphere_meridian(radius=radius, segments=56)
    config = SolveConfig(
        velocity_sources={2: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        source_motion=SourceMotion.AXIAL,
        formulation="standard",
        return_surface_pressure=True,
        observation=ObservationConfig(
            distance_m=5.0,
            angle_count=37,
            planes=["horizontal"],
            origin="throat",
        ),
    )

    result = metal_bem.solve_circsym_frequencies(
        meridian,
        [_freq_for_ka(value, radius) for value in ka],
        config,
    )

    geom = meridian.segment_geometry()
    weights = geom.area_weights
    n_z = meridian.normals[:, 1]
    force = np.sum(result.surface_pressure_complex * n_z[None, :] * weights[None, :], axis=1)
    z_force = force / (config.air_density * SPEED_OF_SOUND * np.pi * radius**2)
    exact = np.array(
        [
            (4.0 / 3.0)
            * 1j
            * _spherical_hankel1(1, value)
            / _spherical_hankel1_derivative(1, value)
            for value in ka
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(z_force, exact, rtol=4e-3)

    amp = np.abs(result.pressure_complex[:, 0])
    amp = amp / amp[:, :1]
    expected = np.abs(np.cos(np.deg2rad(result.observation_angles_deg)))
    sample = np.array([0, 6, 12, 18, 24, 30, 36])
    np.testing.assert_allclose(
        amp[:, sample],
        np.tile(expected[None, sample], (amp.shape[0], 1)),
        atol=2e-3,
    )


def test_baffled_flat_piston_matches_airy_directivity_and_first_null():
    radius = 0.1
    ka_values = np.array([3.0, 8.0], dtype=np.float64)
    config = SolveConfig(
        velocity_sources={2: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        formulation="standard",
        circsym_baffle_z=0.0,
        observation=ObservationConfig(
            distance_m=30.0,
            angle_count=181,
            planes=["horizontal"],
            origin="throat",
        ),
    )

    result = metal_bem.solve_circsym_frequencies(
        _piston_meridian(radius=radius, segments=30),
        [_freq_for_ka(value, radius) for value in ka_values],
        config,
    )

    theta = np.deg2rad(result.observation_angles_deg)
    amp = np.abs(result.pressure_complex[:, 0])
    amp = amp / amp[:, :1]
    for row, ka in enumerate(ka_values):
        x = ka * np.sin(theta)
        theory = np.ones_like(x)
        mask = np.abs(x) > 1e-12
        theory[mask] = np.abs(2.0 * j1(x[mask]) / x[mask])
        if ka < 3.831705970:
            compare = result.observation_angles_deg <= 90.0
        else:
            compare = result.observation_angles_deg <= 24.0
        err_db = 20.0 * np.log10(
            np.maximum(amp[row, compare], 1e-12)
            / np.maximum(theory[compare], 1e-12)
        )
        assert float(np.max(np.abs(err_db))) < 0.03

    first_null = np.rad2deg(np.arcsin(3.831705970 / 8.0))
    search = result.observation_angles_deg <= 45.0
    null_angle = float(result.observation_angles_deg[search][np.argmin(amp[1, search])])
    assert abs(null_angle - first_null) <= 1.0


def test_circsym_default_complex_k_wiring_and_chief_tames_sphere_irregularity():
    radius = 0.1
    meridian = _sphere_meridian(radius=radius, segments=48)
    ka = np.pi
    frequency = _freq_for_ka(ka, radius)
    observation = ObservationConfig(
        distance_m=3.0,
        angle_count=5,
        planes=["horizontal", "vertical"],
        origin="throat",
    )

    default_result = metal_bem.solve_circsym_frequencies(meridian, [frequency])
    assert default_result.config.formulation == "complex_k"
    assert default_result.native_diagnostics[0]["complex_k"] is True

    base = dict(
        velocity_sources={2: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        formulation="standard",
        observation=observation,
    )
    standard = metal_bem.solve_circsym_frequencies(meridian, [frequency], SolveConfig(**base))
    chief = metal_bem.solve_circsym_frequencies(
        meridian,
        [frequency],
        SolveConfig(**base, chief_points=np.array([[0.0, 0.0, 0.0]])),
    )

    exact = _pulsating_sphere_impedance(np.array([ka]))[0]
    standard_error = abs(
        standard.impedance[0] / (standard.config.air_density * SPEED_OF_SOUND) - exact
    )
    chief_error = abs(
        chief.impedance[0] / (chief.config.air_density * SPEED_OF_SOUND) - exact
    )
    assert chief.native_diagnostics[0]["chief_points"] is True
    assert chief.native_diagnostics[0]["chief_points_count"] == 1
    assert chief_error < 0.01
    assert chief_error < 0.1 * standard_error

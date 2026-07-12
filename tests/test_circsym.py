from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j1, spherical_jn, spherical_yn, struve

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem._constants import SPEED_OF_SOUND
from hornlab_metal_bem.circsym import (
    MeridianMesh,
    _BoundaryAssemblyGeometryCache,
    _assemble_boundary_matrices,
    _build_far_remainder_compact_geometry,
    _build_far_remainder_geometry_parts,
    _evaluate_far_remainder_block,
    _evaluate_far_remainder_onthefly_compiled,
    _evaluate_points_pressure,
    _is_flat_baffled_sheet,
    _integrate_segment_kernel,
    _load_circsym_remainder_c_kernel,
    _ring_remainder_kernel_m0,
    _ring_remainder_kernel_m0_targets_batched,
    _validate_closed_or_baffled_meridian,
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


def _baffled_piston_impedance(ka: np.ndarray) -> np.ndarray:
    return 1.0 - j1(2.0 * ka) / ka - 1j * struve(1, 2.0 * ka) / ka


def _spherical_hankel1(order: int, x: float) -> complex:
    return complex(spherical_jn(order, x) + 1j * spherical_yn(order, x))


def _spherical_hankel1_derivative(order: int, x: float) -> complex:
    return complex(
        spherical_jn(order, x, derivative=True)
        + 1j * spherical_yn(order, x, derivative=True)
    )


def _scalar_points_pressure(
    meridian: MeridianMesh,
    pressure: np.ndarray,
    q_total: np.ndarray,
    points: np.ndarray,
    k: complex,
    baffle_z: float | None,
    *,
    n_psi: int,
) -> np.ndarray:
    geom = meridian.segment_geometry()
    out = np.empty(points.shape[0], dtype=np.complex128)
    rayleigh_sheet = _is_flat_baffled_sheet(meridian, baffle_z)
    for i, point in enumerate(points):
        target_rho = float(np.hypot(float(point[0]), float(point[1])))
        target_z = float(point[2])
        s_row = np.empty(meridian.segment_count, dtype=np.complex128)
        h_row = np.empty(meridian.segment_count, dtype=np.complex128)
        for j in range(meridian.segment_count):
            s_row[j], h_row[j] = _integrate_segment_kernel(
                target_rho=target_rho,
                target_z=target_z,
                meridian=meridian,
                geom=geom,
                source_index=j,
                k=k,
                baffle_z=baffle_z,
                n_psi=n_psi,
                target_index=None,
            )
        out[i] = (
            -(s_row @ q_total)
            if rayleigh_sheet
            else h_row @ pressure - s_row @ q_total
        )
    return out


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


@pytest.mark.parametrize(
    "k",
    [1.0e-8 + 0.0j, 1.0e-5 + 1.0e-6j, 0.2 + 0.03j, 30.0 + 1.0j],
)
def test_targets_batched_ring_remainder_matches_scalar_across_q_ranges(k: complex):
    """The target-batched remainder keeps the scalar small-|q| guard."""
    target_rho = np.array([0.02, 0.13, 0.40], dtype=np.float64)
    target_z = np.array([-0.03, 0.08, 0.15], dtype=np.float64)
    source_rho = np.array([[0.01, 0.03, 0.08], [0.12, 0.17, 0.23]])
    source_z = np.array([[-0.07, -0.02, 0.01], [0.05, 0.09, 0.12]])
    normal_rho = np.array([0.6, -0.8], dtype=np.float64)
    normal_z = np.array([0.8, 0.6], dtype=np.float64)

    actual_g, actual_h = _ring_remainder_kernel_m0_targets_batched(
        target_rho,
        target_z,
        source_rho,
        source_z,
        normal_rho,
        normal_z,
        k,
        n_psi=48,
    )
    expected_g = np.empty_like(actual_g)
    expected_h = np.empty_like(actual_h)
    for target_index, (rho, z) in enumerate(zip(target_rho, target_z)):
        for source_index in range(source_rho.shape[0]):
            g_value, h_value = _ring_remainder_kernel_m0(
                float(rho),
                float(z),
                source_rho[source_index],
                source_z[source_index],
                np.array(
                    [normal_rho[source_index], normal_z[source_index]],
                    dtype=np.float64,
                ),
                k,
                n_psi=48,
            )
            expected_g[target_index, source_index] = g_value
            expected_h[target_index, source_index] = h_value

    np.testing.assert_allclose(actual_g, expected_g, rtol=1e-14, atol=1e-30)
    np.testing.assert_allclose(actual_h, expected_h, rtol=1e-14, atol=1e-30)


@pytest.mark.parametrize("baffled_sheet", [False, True])
def test_vectorized_boundary_assembly_matches_scalar_segment_integrals(
    baffled_sheet: bool,
):
    k = 19.0 + 0.03j
    baffle_z = 0.0 if baffled_sheet else None
    meridian = (
        _piston_meridian(radius=0.1, segments=12)
        if baffled_sheet
        else _sphere_meridian(radius=0.1, segments=14)
    )
    geom = meridian.segment_geometry()
    S, H = _assemble_boundary_matrices(meridian, k, baffle_z=baffle_z, n_psi=96)

    S_ref = np.empty_like(S)
    H_ref = np.empty_like(H)
    for i, target in enumerate(geom.midpoints):
        for j in range(meridian.segment_count):
            S_ref[i, j], H_ref[i, j] = _integrate_segment_kernel(
                target_rho=float(target[0]),
                target_z=float(target[1]),
                meridian=meridian,
                geom=geom,
                source_index=j,
                k=k,
                baffle_z=baffle_z,
                n_psi=96,
                target_index=i,
            )

    np.testing.assert_allclose(S, S_ref, rtol=4e-13, atol=4e-14)
    np.testing.assert_allclose(H, H_ref, rtol=4e-13, atol=4e-14)


@pytest.mark.parametrize("baffled_sheet", [False, True])
def test_cached_boundary_assembly_matches_uncached(baffled_sheet: bool):
    k = 41.0 + 0.07j
    baffle_z = 0.0 if baffled_sheet else None
    meridian = (
        _piston_meridian(radius=0.1, segments=11)
        if baffled_sheet
        else _sphere_meridian(radius=0.1, segments=13)
    )
    cache = _BoundaryAssemblyGeometryCache(meridian, baffle_z)

    S_cached, H_cached = _assemble_boundary_matrices(
        meridian,
        k,
        baffle_z=baffle_z,
        n_psi=96,
        geometry_cache=cache,
    )
    S_uncached, H_uncached = _assemble_boundary_matrices(
        meridian,
        k,
        baffle_z=baffle_z,
        n_psi=96,
    )

    np.testing.assert_allclose(S_cached, S_uncached, rtol=6e-13, atol=6e-14)
    np.testing.assert_allclose(H_cached, H_uncached, rtol=6e-13, atol=6e-14)


def test_far_onthefly_compiled_matches_precomputed_reference():
    kernel = _load_circsym_remainder_c_kernel()
    if kernel is None:
        pytest.skip("runtime C compiler is unavailable")

    meridians = [
        (_sphere_meridian(radius=0.1, segments=7), None),
        (_sphere_meridian(radius=0.1, segments=7), -0.137),
    ]
    for meridian, baffle_z in meridians:
        geom = meridian.segment_geometry()
        compact = _build_far_remainder_compact_geometry(
            meridian,
            geom,
            baffle_z,
            n_psi=48,
        )
        reference_parts = _build_far_remainder_geometry_parts(
            meridian,
            geom,
            baffle_z,
            n_psi=48,
        )
        assert compact.source_rho.shape == (meridian.segment_count, 16)
        assert compact.cos_psi.shape == (48,)

        for k in (0.2 + 0.03j, 30.0 + 1.0j):
            actual_s, actual_h = _evaluate_far_remainder_onthefly_compiled(
                kernel,
                compact,
                k,
                workers=2,
            )
            expected_s = np.zeros_like(actual_s)
            expected_h = np.zeros_like(actual_h)
            for reference_part in reference_parts:
                s_part, h_part = _evaluate_far_remainder_block(
                    reference_part,
                    k,
                    0,
                    meridian.segment_count,
                )
                expected_s += s_part
                expected_h += h_part

            np.testing.assert_allclose(
                actual_s,
                expected_s,
                rtol=3e-13,
                atol=3e-14,
            )
            np.testing.assert_allclose(
                actual_h,
                expected_h,
                rtol=3e-13,
                atol=3e-14,
            )


def test_boundary_assembly_cache_release_and_budget_fallback(monkeypatch):
    k = 37.0 + 0.02j
    meridian = _sphere_meridian(radius=0.1, segments=9)
    cache = _BoundaryAssemblyGeometryCache(
        meridian,
        None,
        reusable_n_psi={96},
        n_psi_use_counts={96: 1},
        cache_single_use=False,
    )

    assembled = cache.assemble(
        k,
        n_psi=96,
        meridian=meridian,
        baffle_z=None,
    )
    assert assembled is not None
    S_cached, H_cached = assembled
    assert 96 not in cache._quadrature
    assert cache.assemble(k, n_psi=96, meridian=meridian, baffle_z=None) is None

    S_uncached, H_uncached = _assemble_boundary_matrices(
        meridian,
        k,
        None,
        n_psi=96,
    )
    np.testing.assert_allclose(S_cached, S_uncached, rtol=6e-13, atol=6e-14)
    np.testing.assert_allclose(H_cached, H_uncached, rtol=6e-13, atol=6e-14)

    monkeypatch.setenv("HORNLAB_CIRCSYM_ASSEMBLY_CACHE_MAX_BYTES", "1")
    budget_cache = _BoundaryAssemblyGeometryCache(meridian, None)

    def unexpected_scalar_fallback(*args, **kwargs):
        raise AssertionError("compact far assembly must not use the scalar fallback")

    monkeypatch.setattr(
        "hornlab_metal_bem.circsym._assemble_boundary_matrices_uncached",
        unexpected_scalar_fallback,
    )
    S_budget, H_budget = _assemble_boundary_matrices(
        meridian,
        k,
        None,
        n_psi=96,
        geometry_cache=budget_cache,
    )
    assert budget_cache._quadrature[96].far.source_rho.shape == (9, 16)
    np.testing.assert_allclose(S_budget, S_uncached, rtol=6e-13, atol=6e-14)
    np.testing.assert_allclose(H_budget, H_uncached, rtol=6e-13, atol=6e-14)


def test_vectorized_field_evaluation_matches_scalar_closed_meridian():
    k = 31.0 + 0.0j
    meridian = _sphere_meridian(radius=0.1, segments=18)
    rng = np.random.default_rng(12345)
    pressure = rng.normal(size=meridian.segment_count) + 1j * rng.normal(
        size=meridian.segment_count
    )
    q_total = rng.normal(size=meridian.segment_count) + 1j * rng.normal(
        size=meridian.segment_count
    )
    theta = np.linspace(0.0, np.pi, 19)
    points = np.column_stack(
        [2.0 * np.sin(theta), np.zeros_like(theta), 2.0 * np.cos(theta)]
    )
    points = np.vstack(
        [
            points,
            np.array([[0.052, 0.0, 0.092]], dtype=np.float64),
        ]
    )

    vectorized = _evaluate_points_pressure(
        meridian, pressure, q_total, points, k, None, n_psi=96
    )
    scalar = _scalar_points_pressure(
        meridian, pressure, q_total, points, k, None, n_psi=96
    )

    np.testing.assert_allclose(vectorized, scalar, rtol=1e-10, atol=1e-12)


def test_vectorized_field_evaluation_matches_scalar_baffled_sheet_rayleigh_branch():
    k = 47.0 + 0.0j
    meridian = _piston_meridian(radius=0.1, segments=16)
    rng = np.random.default_rng(6789)
    pressure = rng.normal(size=meridian.segment_count) + 1j * rng.normal(
        size=meridian.segment_count
    )
    q_total = rng.normal(size=meridian.segment_count) + 1j * rng.normal(
        size=meridian.segment_count
    )
    theta = np.linspace(0.0, 0.5 * np.pi, 17)
    points = np.column_stack(
        [3.0 * np.sin(theta), np.zeros_like(theta), 3.0 * np.cos(theta)]
    )

    vectorized = _evaluate_points_pressure(
        meridian, pressure, q_total, points, k, 0.0, n_psi=96
    )
    scalar = _scalar_points_pressure(
        meridian, pressure, q_total, points, k, 0.0, n_psi=96
    )

    np.testing.assert_allclose(vectorized, scalar, rtol=1e-10, atol=1e-12)


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


def test_baffled_flat_piston_absolute_complex_pressure_matches_rayleigh_phase():
    radius = 0.05
    frequency = 1800.0
    distances = np.array([0.6, 1.0, 1.8], dtype=np.float64)
    points = np.column_stack(
        [np.zeros_like(distances), np.zeros_like(distances), distances]
    )
    config = SolveConfig(
        velocity_sources={2: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        formulation="standard",
        circsym_baffle_z=0.0,
        observation=ObservationConfig(
            angle_count=distances.size,
            planes=["horizontal"],
            origin="throat",
            custom_points={"horizontal": points},
        ),
    )
    result = metal_bem.solve_circsym_frequencies(
        _piston_meridian(radius=radius, segments=50),
        [frequency],
        config,
    )

    k = 2.0 * np.pi * frequency / SPEED_OF_SOUND
    omega = 2.0 * np.pi * frequency
    edge_distance = np.sqrt(distances * distances + radius * radius)
    rayleigh_slp = (
        np.exp(1j * k * edge_distance) - np.exp(1j * k * distances)
    ) / (1j * k)
    expected = -1j * config.air_density * omega * rayleigh_slp
    np.testing.assert_allclose(
        result.pressure_complex[0, 0], expected, rtol=2.0e-4, atol=2.0e-4
    )


def test_baffled_flat_piston_surface_impedance_matches_analytic_value():
    radius = 0.1
    ka = np.array([2.0], dtype=np.float64)
    config = SolveConfig(
        velocity_sources={2: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        formulation="standard",
        circsym_baffle_z=0.0,
        observation=ObservationConfig(
            distance_m=10.0,
            angle_count=3,
            planes=["horizontal"],
            origin="throat",
        ),
    )

    result = metal_bem.solve_circsym_frequencies(
        _piston_meridian(radius=radius, segments=60),
        [_freq_for_ka(float(ka[0]), radius)],
        config,
    )

    z_norm = result.impedance / (config.air_density * SPEED_OF_SOUND)
    np.testing.assert_allclose(z_norm, _baffled_piston_impedance(ka), rtol=3e-3, atol=3e-4)


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


def test_open_meridian_without_baffle_is_rejected():
    # A zero-thickness open shell needs a two-trace/open-screen formulation;
    # applying the closed-surface one-trace BIE silently solves the wrong problem.
    meridian = MeridianMesh.from_polyline(
        np.array([[0.0, 0.0], [0.04, 0.03], [0.08, 0.06]], dtype=np.float64),
        tags=2,
    )
    config = SolveConfig(
        velocity_sources={2: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        formulation="standard",
        observation=ObservationConfig(angle_count=3, planes=["horizontal"]),
    )

    with pytest.raises(ValueError, match="bare/open meridians"):
        metal_bem.solve_circsym_frequencies(meridian, [1000.0], config)


def test_closed_meridian_validation_uses_topological_endpoints():
    original = _sphere_meridian(radius=0.1, segments=12)
    permutation = np.array([4, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 11])
    old_to_new = np.empty_like(permutation)
    old_to_new[permutation] = np.arange(permutation.size)
    shuffled = MeridianMesh(
        original.nodes[permutation],
        old_to_new[original.segments],
        original.physical_tags,
        original.normals,
    )
    assert shuffled.nodes[0, 0] > 1.0e-9
    assert shuffled.nodes[-1, 0] > 1.0e-9

    _validate_closed_or_baffled_meridian(shuffled, None)


def test_closed_meridian_validation_rejects_off_axis_topological_endpoints():
    # Array endpoints are on-axis, but the degree-one polyline ends are not.
    meridian = MeridianMesh(
        nodes=np.array(
            [[0.0, -0.01], [0.03, -0.01], [0.08, 0.02], [0.0, 0.01]],
            dtype=np.float64,
        ),
        segments=np.array([[1, 0], [0, 3], [3, 2]], dtype=np.int32),
        physical_tags=np.full(3, 2, dtype=np.int32),
    )
    assert meridian.nodes[0, 0] == 0.0
    assert meridian.nodes[-1, 0] == 0.0

    with pytest.raises(ValueError, match="bare/open meridians"):
        _validate_closed_or_baffled_meridian(meridian, None)


def test_legacy_baffle_image_rejects_recessed_or_nonplanar_meridian():
    meridian = MeridianMesh.from_polyline(
        np.array(
            [[0.0, -0.04], [0.02, -0.04], [0.05, 0.0]],
            dtype=np.float64,
        ),
        tags=2,
    )
    config = SolveConfig(
        velocity_sources={2: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        formulation="standard",
        circsym_baffle_z=0.0,
        observation=ObservationConfig(angle_count=3, planes=["horizontal"]),
    )
    with pytest.raises(ValueError, match="only supported for a coplanar flat"):
        metal_bem.solve_circsym_frequencies(meridian, [1000.0], config)

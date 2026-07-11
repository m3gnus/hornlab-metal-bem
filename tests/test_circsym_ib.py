"""Coupled infinite-baffle (flush-mount horn) CircSym path.

Exact IB = interior BEM on the horn channel + analytic Rayleigh coupling on the
mouth-aperture disc (no image/baffle kernel). Validated against the analytic
Rayleigh baffled-piston directivity in the shallow-stub limit, and against the
physical requirement of a forward beam with zero radiation behind the baffle.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j1

import hornlab_metal_bem as metal_bem
import hornlab_metal_bem.circsym as circsym
from hornlab_metal_bem._constants import SPEED_OF_SOUND
from hornlab_metal_bem.circsym import (
    MeridianMesh,
    _BoundaryAssemblyGeometryCache,
    _assemble_boundary_matrices,
    _evaluate_coupled_ib_points_pressure,
    _integrate_segment_kernel,
)
from hornlab_metal_bem.config import ObservationConfig, SolveConfig, VelocityMode

TAG_THROAT, TAG_WALL, TAG_DISC = 2, 3, 4


def _resample(points: np.ndarray, target: float) -> np.ndarray:
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(float(np.hypot(*(b - a))) / target)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.asarray(out)


def _channel_meridian(r_t, r_m, L, h=0.003):
    """Interior channel, normals into the fluid: throat cap -> wall -> mouth disc at z=0."""
    cap = _resample(np.array([[0.0, -L], [r_t, -L]]), h)
    zs = np.linspace(-L, 0.0, 200)
    rs = r_t + (r_m - r_t) * (zs + L) / L
    wall = _resample(np.column_stack([rs, zs]), h)
    disc = _resample(np.array([[r_m, 0.0], [0.0, 0.0]]), h)
    pts = np.vstack([cap, wall[1:], disc[1:]])
    tags = np.concatenate([
        np.full(len(cap) - 1, TAG_THROAT),
        np.full(len(wall) - 1, TAG_WALL),
        np.full(len(disc) - 1, TAG_DISC),
    ])
    return MeridianMesh.from_polyline(pts, tags)


def _solve(mer, freqs, nang=19, amax=90.0):
    cfg = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        circsym_aperture_tag=TAG_DISC,
        observation=ObservationConfig(distance_m=2.0, angle_min_deg=0.0, angle_max_deg=amax,
                                      angle_count=nang, planes=["horizontal"], origin="mouth"),
    )
    return metal_bem.solve_circsym_frequencies(mer, freqs, cfg)


def test_coupled_ib_dispatch_is_taken_when_aperture_tag_present():
    res = _solve(_channel_meridian(0.05, 0.05, 0.002, h=0.0025), [4000.0])
    assert res.native_diagnostics[0].get("coupled_ib") is True
    assert res.native_diagnostics[0].get("aperture_tag") == TAG_DISC
    assert res.native_diagnostics[0]["aperture_pressure_continuity_rel"] < 1.0e-11
    assert res.surface_pressure_avg is not None
    assert TAG_THROAT in res.surface_pressure_avg
    assert res.surface_pressure_complex is None


def test_vectorized_coupled_ib_field_matches_scalar_aperture_sum():
    k = 52.0 + 0.0j
    meridian = _channel_meridian(0.025, 0.05, 0.04, h=0.0025)
    geom = meridian.segment_geometry()
    aperture_indices = np.where(meridian.physical_tags == TAG_DISC)[0]
    rng = np.random.default_rng(2468)
    q_a = rng.normal(size=aperture_indices.size) + 1j * rng.normal(
        size=aperture_indices.size
    )
    theta = np.linspace(0.0, np.pi, 20)
    points = np.column_stack(
        [2.0 * np.sin(theta), np.zeros_like(theta), 2.0 * np.cos(theta)]
    )

    vectorized = _evaluate_coupled_ib_points_pressure(
        meridian, q_a, aperture_indices, points, k, geom=geom, n_psi=96
    )
    scalar = np.zeros(points.shape[0], dtype=np.complex128)
    for point_index, point in enumerate(points):
        target_z = float(point[2])
        if target_z < 0.0:
            continue
        target_rho = float(np.hypot(float(point[0]), float(point[1])))
        val = 0.0 + 0.0j
        for aperture_local, aperture_index in enumerate(aperture_indices):
            s_val, _ = _integrate_segment_kernel(
                target_rho=target_rho,
                target_z=target_z,
                meridian=meridian,
                geom=geom,
                source_index=int(aperture_index),
                k=k,
                baffle_z=None,
                n_psi=96,
                target_index=None,
            )
            val += 2.0 * s_val * q_a[aperture_local]
        scalar[point_index] = val

    np.testing.assert_allclose(vectorized, scalar, rtol=1e-10, atol=1e-12)
    assert vectorized[-1] == 0.0


def test_coupled_ib_boundary_assembly_cache_matches_uncached():
    k = 67.0 + 0.0j
    meridian = _channel_meridian(0.025, 0.05, 0.04, h=0.004)
    cache = _BoundaryAssemblyGeometryCache(meridian, None)

    S_cached, H_cached = _assemble_boundary_matrices(
        meridian,
        k,
        None,
        n_psi=96,
        geometry_cache=cache,
    )
    S_uncached, H_uncached = _assemble_boundary_matrices(
        meridian,
        k,
        None,
        n_psi=96,
    )

    np.testing.assert_allclose(S_cached, S_uncached, rtol=6e-13, atol=6e-14)
    np.testing.assert_allclose(H_cached, H_uncached, rtol=6e-13, atol=6e-14)


@pytest.mark.parametrize("ka", [1.0, 2.0, 3.0])
def test_coupled_ib_shallow_stub_matches_rayleigh_airy(ka):
    a = 0.05
    mer = _channel_meridian(a, a, 0.002, h=0.0025)  # 2 mm stub ~ flush piston
    freq = ka * SPEED_OF_SOUND / (2.0 * np.pi * a)
    res = _solve(mer, [freq])
    p = np.abs(res.pressure_complex[0, 0])
    degs = res.observation_angles_deg
    d_bem = p / p[0]
    x = ka * np.sin(np.deg2rad(degs))
    airy = np.where(np.abs(x) > 1e-9, 2 * j1(np.where(x == 0, 1, x)) / np.where(x == 0, 1, x), 1.0)
    err_db = np.max(np.abs(20 * np.log10(np.maximum(d_bem, 1e-9) / np.maximum(np.abs(airy), 1e-9))))
    assert err_db < 0.6, f"ka={ka}: {err_db:.2f} dB vs analytic Airy directivity"


def test_coupled_ib_cone_horn_is_a_forward_beam_with_no_rear_radiation():
    mer = _channel_meridian(0.0127, 0.050, 0.080, h=0.003)
    res = _solve(mer, [1000.0, 4000.0, 8000.0], nang=19, amax=180.0)
    degs = res.observation_angles_deg
    on_axis = int(np.argmin(np.abs(degs)))
    for fi in range(3):
        p = np.abs(res.pressure_complex[fi, 0])
        db = 20 * np.log10(np.maximum(p / p[on_axis], 1e-9))
        front = db[degs <= 90.0]
        # on-axis is the max in the front half-space (this is the bug the double-horn broke)
        assert np.max(front) - front[0] < 1.0
        # zero radiation behind the baffle
        assert np.all(db[degs > 91.0] < -40.0)


def test_circsym_missing_aperture_tag_raises_instead_of_free_space():
    """A requested-but-absent circsym_aperture_tag must fail loudly, not silently
    fall back to a free-space (free-standing) sweep with wrong physics."""
    meridian = _channel_meridian(0.025, 0.05, 0.04)
    config = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        circsym_aperture_tag=99,  # not present in the meridian (tags are 2/3/4)
        observation=ObservationConfig(
            distance_m=1.0, angle_min_deg=0.0, angle_max_deg=90.0,
            angle_count=5, planes=["horizontal"], origin="mouth",
        ),
    )
    with pytest.raises(ValueError, match="circsym_aperture_tag 99 is not present"):
        metal_bem.solve_circsym_frequencies(
            meridian, np.array([1000.0]), config
        )


def test_coupled_ib_observation_origin_moves_generated_arc():
    depth = 0.04
    meridian = _channel_meridian(0.025, 0.05, depth, h=0.005)

    def solve(origin: str):
        return metal_bem.solve_circsym_frequencies(
            meridian,
            [1000.0],
            SolveConfig(
                velocity_sources={TAG_THROAT: 1.0},
                velocity_mode=VelocityMode.VELOCITY,
                circsym_aperture_tag=TAG_DISC,
                observation=ObservationConfig(
                    distance_m=1.0,
                    angle_min_deg=0.0,
                    angle_max_deg=60.0,
                    angle_count=3,
                    planes=["horizontal"],
                    origin=origin,
                ),
            ),
        )

    mouth = solve("mouth")
    throat = solve("throat")
    np.testing.assert_allclose(
        throat.observation_points,
        mouth.observation_points + np.array([0.0, 0.0, -depth]),
        atol=1.0e-12,
    )
    assert not np.allclose(throat.pressure_complex, mouth.pressure_complex)


def test_coupled_ib_complex_k_and_robin_are_applied():
    meridian = _channel_meridian(0.025, 0.05, 0.04, h=0.005)
    base = dict(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        circsym_aperture_tag=TAG_DISC,
        observation=ObservationConfig(
            distance_m=1.0,
            angle_count=3,
            planes=["horizontal"],
            origin="mouth",
        ),
    )
    standard = metal_bem.solve_circsym_frequencies(
        meridian, [1500.0], SolveConfig(**base)
    )
    regularized = metal_bem.solve_circsym_frequencies(
        meridian,
        [1500.0],
        SolveConfig(
            **base,
            formulation="complex_k",
            complex_k_shift=0.01,
            impedance_sources={TAG_WALL: 0.08 + 0.02j},
        ),
    )
    diagnostics = regularized.native_diagnostics[0]
    assert diagnostics["complex_k"] is True
    assert diagnostics["robin"] is True
    assert not np.allclose(
        regularized.surface_pressure_avg[TAG_THROAT],
        standard.surface_pressure_avg[TAG_THROAT],
    )


def test_coupled_ib_complex_k_aperture_block_matches_full_assembly(monkeypatch):
    """The reduced real-k Rayleigh block is parity-pinned to the old full path."""
    meridian = _channel_meridian(0.025, 0.05, 0.04, h=0.005)
    config = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        circsym_aperture_tag=TAG_DISC,
        formulation="complex_k",
        complex_k_shift=0.01,
        return_surface_pressure=True,
        observation=ObservationConfig(
            distance_m=1.0,
            angle_count=3,
            planes=["horizontal"],
            origin="mouth",
        ),
    )

    def legacy_full_aperture_matrix(
        meridian,
        aperture_indices,
        k,
        *,
        geom,
        n_psi,
    ):
        del geom
        full_s, _ = circsym._assemble_boundary_matrices(
            meridian,
            k,
            None,
            n_psi=n_psi,
        )
        indices = np.asarray(aperture_indices, dtype=np.int64)
        return full_s[np.ix_(indices, indices)]

    with monkeypatch.context() as context:
        context.setattr(
            circsym,
            "_assemble_coupled_ib_rayleigh_aperture_matrix",
            legacy_full_aperture_matrix,
        )
        legacy = metal_bem.solve_circsym_frequencies(meridian, [1500.0], config)
    optimized = metal_bem.solve_circsym_frequencies(meridian, [1500.0], config)

    np.testing.assert_allclose(
        optimized.pressure_complex,
        legacy.pressure_complex,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimized.directivity_db,
        legacy.directivity_db,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimized.impedance,
        legacy.impedance,
        rtol=1e-12,
        atol=1e-12,
    )
    assert optimized.surface_pressure_complex is not None
    assert legacy.surface_pressure_complex is not None
    np.testing.assert_allclose(
        optimized.surface_pressure_complex,
        legacy.surface_pressure_complex,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimized.surface_pressure_avg[TAG_THROAT],
        legacy.surface_pressure_avg[TAG_THROAT],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "override, match",
    [
        ({"circsym_baffle_z": 0.0}, "does not compose"),
        ({"chief_points": np.array([[0.0, 0.0, -0.02]])}, "does not support chief_points"),
        ({"impedance_sources": {TAG_DISC: 0.1}}, "must not also carry"),
    ],
)
def test_coupled_ib_unsupported_or_conflicting_options_fail_loudly(override, match):
    config = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        circsym_aperture_tag=TAG_DISC,
        observation=ObservationConfig(angle_count=3, planes=["horizontal"]),
        **override,
    )
    with pytest.raises(ValueError, match=match):
        metal_bem.solve_circsym_frequencies(
            _channel_meridian(0.025, 0.05, 0.04, h=0.005),
            [1000.0],
            config,
        )


def test_coupled_ib_rejects_aperture_off_plane_or_wrong_normal():
    original = _channel_meridian(0.025, 0.05, 0.04, h=0.005)
    aperture = original.physical_tags == TAG_DISC

    shifted_nodes = original.nodes.copy()
    shifted_nodes[np.unique(original.segments[aperture]), 1] += 0.001
    shifted = MeridianMesh(
        shifted_nodes,
        original.segments,
        original.physical_tags,
        original.normals,
    )
    with pytest.raises(ValueError, match="global z=0"):
        _solve(shifted, [1000.0])

    wrong_normals = original.normals.copy()
    wrong_normals[aperture] *= -1.0
    wrong = MeridianMesh(
        original.nodes,
        original.segments,
        original.physical_tags,
        wrong_normals,
    )
    with pytest.raises(ValueError, match="normals must point -Z"):
        _solve(wrong, [1000.0])


def test_coupled_ib_rejects_incomplete_or_disconnected_aperture():
    original = _channel_meridian(0.025, 0.05, 0.04, h=0.005)
    aperture_indices = np.where(original.physical_tags == TAG_DISC)[0]
    tags = original.physical_tags.copy()
    tags[aperture_indices[len(aperture_indices) // 2]] = TAG_WALL
    broken = MeridianMesh(
        original.nodes,
        original.segments,
        tags,
        original.normals,
    )
    with pytest.raises(ValueError, match="contiguous mouth-to-axis disc"):
        _solve(broken, [1000.0])

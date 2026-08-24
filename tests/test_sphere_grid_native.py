"""Sphere-grid (balloon) observation solves.

The frame-relative sphere grid must stay aligned with the polar arcs: a grid
point at (theta=90, phi=0) is by construction the same physical location as
the horizontal arc's 90-degree point, so both must return the same pressure
from the same solved system. CircSym gets the same parity check plus a
pulsating-sphere physics check (p ~ e^{ikd}/d about the sphere centre).
"""
from __future__ import annotations

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem._constants import SPEED_OF_SOUND
from hornlab_metal_bem.circsym import MeridianMesh
from hornlab_metal_bem.config import ObservationConfig, SolveConfig, VelocityMode
from hornlab_metal_bem.mesh import LoadedMesh, make_pure_grid
from hornlab_metal_bem.observation import ObservationFrame
from hornlab_metal_bem.result import MeshInfo


def _require_native():
    from hornlab_metal_bem.metal import discover_native_runtime

    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )


def _octasphere(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Unit sphere from a subdivided octahedron (outward-oriented triangles)."""
    vertices = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    triangles = [
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
    ]
    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            cached = midpoint_cache.get(key)
            if cached is not None:
                return cached
            va = vertices[a]
            vb = vertices[b]
            mid = ((va[0] + vb[0]) / 2, (va[1] + vb[1]) / 2, (va[2] + vb[2]) / 2)
            norm = (mid[0] ** 2 + mid[1] ** 2 + mid[2] ** 2) ** 0.5
            vertices.append((mid[0] / norm, mid[1] / norm, mid[2] / norm))
            midpoint_cache[key] = len(vertices) - 1
            return midpoint_cache[key]

        next_triangles = []
        for a, b, c in triangles:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            next_triangles.extend(
                [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
            )
        triangles = next_triangles
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles, dtype=np.int32),
    )


def _capped_sphere_mesh() -> LoadedMesh:
    """Unit sphere with a driven cap at +z (tag 2), rigid elsewhere (tag 1)."""
    vertices, triangles = _octasphere(2)
    centroids = vertices[triangles].mean(axis=1)
    tags = np.ones(triangles.shape[0], dtype=np.int32)
    tags[centroids[:, 2] > 0.55] = 2
    assert np.count_nonzero(tags == 2) > 0
    bbox = (vertices.min(axis=0), vertices.max(axis=0))
    return LoadedMesh(
        grid=make_pure_grid(vertices, triangles),
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles.shape[0],
            physical_groups={1: "rigid", 2: "cap"},
            bounding_box_m=bbox,
        ),
    )


def _pulsating_sphere_mesh(radius: float, *, subdivisions: int = 3) -> LoadedMesh:
    """Closed full-3D sphere with unit normal velocity on every face."""
    vertices, triangles = _octasphere(subdivisions)
    vertices *= radius
    tags = np.full(triangles.shape[0], 2, dtype=np.int32)
    return LoadedMesh(
        grid=make_pure_grid(vertices, triangles),
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles.shape[0],
            physical_groups={2: "pulsating-sphere"},
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
    )


def _quarter_capped_sphere_mesh() -> LoadedMesh:
    """Positive-X/Y quarter sphere with a driven +Z cap."""
    vertices, triangles = _octasphere(2)
    centroids = vertices[triangles].mean(axis=1)
    keep = (centroids[:, 0] >= -1.0e-12) & (centroids[:, 1] >= -1.0e-12)
    triangles = triangles[keep]
    used = np.unique(triangles)
    remap = np.full(vertices.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    vertices = vertices[used]
    triangles = remap[triangles]
    centroids = vertices[triangles].mean(axis=1)
    tags = np.ones(triangles.shape[0], dtype=np.int32)
    tags[centroids[:, 2] > 0.55] = 2
    return LoadedMesh(
        grid=make_pure_grid(vertices, triangles),
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles.shape[0],
            physical_groups={1: "rigid", 2: "cap"},
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
    )


def _pulsating_sphere_meridian(radius: float, *, segments: int = 64) -> MeridianMesh:
    theta = np.linspace(0.0, np.pi, segments + 1)
    return MeridianMesh.from_polyline(
        np.column_stack([radius * np.sin(theta), radius * np.cos(theta)]),
        tags=2,
    )


def _sphere_frame() -> ObservationFrame:
    origin = np.zeros(3, dtype=np.float64)
    return ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=origin,
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=origin,
        source_center=origin,
    )


def _axisymmetric_directivity_index_db(
    pressure: np.ndarray,
    angles_deg: np.ndarray,
) -> np.ndarray:
    """Integrate an axisymmetric polar response over the complete sphere."""
    theta = np.deg2rad(angles_deg)
    normalized_power = np.abs(pressure / pressure[:, :1]) ** 2
    integral = np.trapezoid(
        normalized_power * np.sin(theta)[None, :],
        theta,
        axis=1,
    )
    return 10.0 * np.log10(2.0 / integral)


def _analytic_pulsating_sphere_pressure(
    frequencies_hz: np.ndarray,
    radius: float,
    distance: float,
    air_density: float,
) -> np.ndarray:
    """Outgoing spherical pressure for unit radial surface velocity."""
    k = 2.0 * np.pi * frequencies_hz / SPEED_OF_SOUND
    ka = k * radius
    surface_impedance = air_density * SPEED_OF_SOUND * (
        ka**2 - 1j * ka
    ) / (1.0 + ka**2)
    return surface_impedance * (radius / distance) * np.exp(
        1j * k * (distance - radius)
    )


def _grid_index(n_phi: int, theta_index: int, phi_index: int) -> int:
    return theta_index * n_phi + phi_index


@pytest.mark.slow
def test_native_sphere_grid_matches_coincident_arc_points():
    _require_native()
    mesh = _capped_sphere_mesh()

    # 13 arc angles over 0..180 puts samples at 0 and 90 deg; the (7, 12) grid
    # has theta rows every 30 deg and phi columns every 30 deg, so grid
    # (theta=90, phi=0) coincides with the horizontal arc's 90-degree point
    # and (theta=90, phi=90) with the vertical arc's.
    observation = metal_bem.ObservationConfig(
        planes=["horizontal", "vertical"],
        angle_min_deg=0.0,
        angle_max_deg=180.0,
        angle_count=13,
        distance_m=2.0,
        sphere_grid=(7, 12),
    )
    config = metal_bem.native_config(observation=observation)
    result = metal_bem.solve_frequencies(mesh, [320.0, 640.0], config)

    n_points = 7 * 12
    assert result.sphere_pressure_complex is not None
    assert result.sphere_pressure_complex.shape == (2, n_points)
    assert result.sphere_points.shape == (n_points, 3)
    assert result.sphere_theta_deg.shape == (n_points,)
    assert result.sphere_phi_deg.shape == (n_points,)
    assert np.all(np.isfinite(result.sphere_pressure_complex.view(np.float64)))

    angles = np.asarray(result.observation_angles_deg)
    arc_90 = int(np.argmin(np.abs(angles - 90.0)))
    arc_0 = int(np.argmin(np.abs(angles)))
    h_plane = result.observation_planes.index("horizontal")
    v_plane = result.observation_planes.index("vertical")

    sphere = result.sphere_pressure_complex
    idx_h90 = _grid_index(12, 3, 0)
    idx_v90 = _grid_index(12, 3, 3)

    assert result.sphere_theta_deg[idx_h90] == pytest.approx(90.0)
    assert result.sphere_phi_deg[idx_h90] == pytest.approx(0.0)
    assert result.sphere_phi_deg[idx_v90] == pytest.approx(90.0)

    # Same coordinates, same solved system, same field kernel: near-exact.
    np.testing.assert_allclose(
        sphere[:, idx_h90],
        result.pressure_complex[:, h_plane, arc_90],
        rtol=5e-4,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        sphere[:, idx_v90],
        result.pressure_complex[:, v_plane, arc_90],
        rtol=5e-4,
        atol=1e-9,
    )
    # Every phi column of the theta=0 row is the same pole point.
    pole = sphere[:, :12]
    np.testing.assert_allclose(
        pole,
        np.repeat(
            result.pressure_complex[:, h_plane, arc_0][:, None], 12, axis=1
        ),
        rtol=5e-4,
        atol=1e-9,
    )

    # Grid points actually sit where the arcs sit.
    obs_points = np.asarray(result.observation_points)
    np.testing.assert_allclose(
        result.sphere_points[idx_h90], obs_points[h_plane, arc_90], atol=1e-9
    )


@pytest.mark.slow
def test_native_pulsating_sphere_surface_and_balloon_power_match_analytic():
    _require_native()
    radius = 0.05
    ka = np.array([0.5, 1.0], dtype=np.float64)
    frequencies_hz = ka * SPEED_OF_SOUND / (2.0 * np.pi * radius)
    velocity = 0.75
    observation = ObservationConfig(
        planes=["horizontal"],
        angle_count=3,
        distance_m=2.0,
        sphere_grid=(37, 72),
    )
    config = SolveConfig(
        velocity_sources={2: velocity},
        velocity_mode=VelocityMode.VELOCITY,
        observation=observation,
        frame_override=_sphere_frame(),
        dense_solve_dtype="float64",
    )

    # 2048 faces: the inscribed octasphere's area deficit is 0.3% (1.3% at
    # 512 faces), which keeps the mesh error well inside the 2% analytic gate.
    result = metal_bem.solve_frequencies(
        _pulsating_sphere_mesh(radius, subdivisions=4),
        frequencies_hz,
        config,
    )
    expected = (
        0.5
        * config.air_density
        * SPEED_OF_SOUND
        * velocity**2
        * 4.0
        * np.pi
        * radius**2
        * ka**2
        / (1.0 + ka**2)
    )

    assert result.surface_pressure_complex is None
    assert result.radiated_power_surface_w is not None
    assert result.radiated_power_sphere_w is not None
    assert result.radiated_power_sphere_coverage_sr == pytest.approx(4.0 * np.pi)
    np.testing.assert_allclose(
        result.radiated_power_surface_w,
        expected,
        rtol=0.02,
    )
    agreement_db = 10.0 * np.log10(
        result.radiated_power_sphere_w / result.radiated_power_surface_w
    )
    assert float(np.max(np.abs(agreement_db))) < 0.1
    assert np.all(result.radiated_power_surface_w > 0.0)
    assert np.all(result.radiated_power_sphere_w > 0.0)


@pytest.mark.slow
def test_native_yz_xz_balloon_dedupe_matches_full_evaluation():
    _require_native()
    mesh = _quarter_capped_sphere_mesh()
    common = dict(
        planes=["horizontal"],
        angle_count=3,
        distance_m=2.0,
        sphere_grid=(19, 36),
    )
    deduped = metal_bem.solve_frequencies(
        mesh,
        [320.0, 640.0],
        SolveConfig(
            velocity_sources={2: 1.0},
            velocity_mode=VelocityMode.VELOCITY,
            native_symmetry_plane="yz+xz",
            observation=ObservationConfig(**common),
            frame_override=_sphere_frame(),
            native_check_open_edges=True,
        ),
    )
    full = metal_bem.solve_frequencies(
        mesh,
        [320.0, 640.0],
        SolveConfig(
            velocity_sources={2: 1.0},
            velocity_mode=VelocityMode.VELOCITY,
            native_symmetry_plane="yz+xz",
            observation=ObservationConfig(
                **common,
                sphere_symmetry_dedupe=False,
            ),
            frame_override=_sphere_frame(),
            native_check_open_edges=True,
        ),
    )

    assert deduped.sphere_pressure_complex is not None
    assert full.sphere_pressure_complex is not None
    np.testing.assert_allclose(
        deduped.sphere_pressure_complex,
        full.sphere_pressure_complex,
        rtol=5.0e-4,
        atol=1.0e-7,
    )
    total = 19 * 36
    diagnostics = deduped.native_diagnostics[0]
    assert diagnostics["sphere_targets"] == total
    assert diagnostics["sphere_evaluation_targets"] <= int(0.27 * total)
    assert diagnostics["sphere_symmetry_dedupe"] is True
    assert full.native_diagnostics[0]["sphere_evaluation_targets"] == total
    assert full.native_diagnostics[0]["sphere_symmetry_dedupe"] is False


def test_circsym_sphere_grid_matches_arcs_and_point_source_decay():
    meridian = metal_bem.MeridianMesh.from_polyline(
        np.column_stack(
            [
                0.1 * np.sin(np.linspace(0.0, np.pi, 49)),
                0.1 * np.cos(np.linspace(0.0, np.pi, 49)),
            ]
        ),
        tags=2,
    )
    observation = metal_bem.ObservationConfig(
        planes=["horizontal", "vertical"],
        angle_min_deg=0.0,
        angle_max_deg=180.0,
        angle_count=13,
        distance_m=2.0,
        sphere_grid=(7, 12),
    )
    config = metal_bem.native_config(observation=observation)
    frequency = 1200.0
    result = metal_bem.solve_circsym_frequencies(meridian, [frequency], config)

    n_points = 7 * 12
    assert result.sphere_pressure_complex is not None
    assert result.sphere_pressure_complex.shape == (1, n_points)
    assert result.radiated_power_surface_w is not None
    assert result.radiated_power_sphere_w is not None
    assert result.radiated_power_sphere_coverage_sr == pytest.approx(4.0 * np.pi)
    assert result.radiated_power_surface_w[0] > 0.0
    assert result.radiated_power_sphere_w[0] > 0.0
    assert abs(
        10.0
        * np.log10(
            result.radiated_power_sphere_w[0]
            / result.radiated_power_surface_w[0]
        )
    ) < 0.1
    diagnostics = result.native_diagnostics[0]
    assert diagnostics["sphere_targets"] == n_points
    assert diagnostics["sphere_evaluation_targets"] == 7

    angles = np.asarray(result.observation_angles_deg)
    arc_90 = int(np.argmin(np.abs(angles - 90.0)))
    h_plane = result.observation_planes.index("horizontal")
    sphere = result.sphere_pressure_complex
    idx_h90 = _grid_index(12, 3, 0)
    np.testing.assert_allclose(
        sphere[0, idx_h90],
        result.pressure_complex[0, h_plane, arc_90],
        rtol=1e-6,
    )

    # Pulsating sphere: p ~ e^{+ikd}/d about the sphere centre (origin), so
    # normalizing out the propagation term must leave a constant.
    k = 2.0 * np.pi * frequency / SPEED_OF_SOUND
    d = np.linalg.norm(np.asarray(result.sphere_points), axis=1)
    normalized = sphere[0] * d * np.exp(-1j * k * d)
    magnitudes = np.abs(normalized)
    assert magnitudes.max() / magnitudes.min() == pytest.approx(1.0, abs=0.02)
    np.testing.assert_array_equal(
        sphere[0].reshape(7, 12),
        np.repeat(sphere[0].reshape(7, 12)[:, :1], 12, axis=1),
    )


@pytest.mark.slow
def test_native_full3d_pulsating_sphere_matches_circsym_through_16khz():
    """Qualify CircSym against full-3D Metal on a closed round body.

    This gate intentionally includes an HF case and compares un-normalized
    complex pressure, not only response shape.  The analytic spherical-wave
    reference prevents two implementations from passing through a shared
    amplitude or phase error; the complete 0..180-degree arc also pins the
    integrated directivity index.
    """
    _require_native()
    radius = 0.005
    distance = 1.0
    frequencies_hz = np.array([1000.0, 16000.0], dtype=np.float64)
    observation = ObservationConfig(
        planes=["horizontal", "vertical"],
        angle_min_deg=0.0,
        angle_max_deg=180.0,
        angle_count=73,
        distance_m=distance,
        origin="mouth",
    )
    common = dict(
        velocity_sources={2: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        observation=observation,
        frame_override=_sphere_frame(),
        dense_solve_dtype="float64",
    )

    native = metal_bem.solve_frequencies(
        _pulsating_sphere_mesh(radius),
        frequencies_hz,
        SolveConfig(**common),
    )
    circsym = metal_bem.solve_circsym_frequencies(
        _pulsating_sphere_meridian(radius),
        frequencies_hz,
        SolveConfig(**common),
    )

    native_pressure = native.pressure_complex[:, 0, :]
    circsym_pressure = circsym.pressure_complex[:, 0, :]
    analytic = _analytic_pulsating_sphere_pressure(
        frequencies_hz,
        radius,
        distance,
        SolveConfig().air_density,
    )

    # Both discretizations must independently recover the analytic level and
    # phase.  This is stricter and more diagnostic than normalized-pattern-only
    # parity, especially for a nominally omnidirectional radiator.
    for measured in (native_pressure[:, 0], circsym_pressure[:, 0]):
        level_error_db = 20.0 * np.log10(np.abs(measured / analytic))
        phase_error_deg = np.rad2deg(np.angle(measured / analytic))
        assert float(np.max(np.abs(level_error_db))) < 0.5
        assert float(np.max(np.abs(phase_error_deg))) < 5.0

    parity_ratio = native_pressure / circsym_pressure
    assert float(np.max(np.abs(20.0 * np.log10(np.abs(parity_ratio))))) < 0.5
    assert float(np.max(np.abs(np.rad2deg(np.angle(parity_ratio))))) < 5.0
    assert float(np.max(np.abs(native.directivity_db - circsym.directivity_db))) < 0.5

    native_di = _axisymmetric_directivity_index_db(
        native_pressure,
        native.observation_angles_deg,
    )
    circsym_di = _axisymmetric_directivity_index_db(
        circsym_pressure,
        circsym.observation_angles_deg,
    )
    assert float(np.max(np.abs(native_di - circsym_di))) < 0.1
    assert float(np.max(np.abs(native_di))) < 0.1
    assert float(np.max(np.abs(circsym_di))) < 0.1

    # Rotational invariance is part of the model contract, not merely expected
    # from the analytic solution.
    np.testing.assert_allclose(
        native.pressure_complex[:, 0, :],
        native.pressure_complex[:, 1, :],
        rtol=2.0e-3,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        circsym.pressure_complex[:, 0, :],
        circsym.pressure_complex[:, 1, :],
        rtol=1.0e-10,
        atol=1.0e-10,
    )

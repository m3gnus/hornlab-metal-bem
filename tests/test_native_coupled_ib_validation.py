from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j1, struve

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem.config import ObservationConfig, SolveConfig, VelocityMode
from hornlab_metal_bem.mesh import LoadedMesh, make_pure_grid
from hornlab_metal_bem.metal import discover_native_runtime
from hornlab_metal_bem.observation import ObservationFrame
from hornlab_metal_bem.result import MeshInfo

TAG_THROAT = 2
TAG_WALL = 3
TAG_APERTURE = 4

_TRIANGLE_QX = np.array(
    [
        0.4459484909159651,
        0.0915762135097710,
        0.1081030181680700,
        0.4459484909159651,
        0.8168475729804590,
        0.0915762135097710,
    ],
    dtype=np.float64,
)
_TRIANGLE_QY = np.array(
    [
        0.4459484909159651,
        0.0915762135097700,
        0.4459484909159651,
        0.1081030181680700,
        0.0915762135097700,
        0.8168475729804580,
    ],
    dtype=np.float64,
)
_TRIANGLE_QW = np.array(
    [
        0.5 * 0.2233815896780110,
        0.5 * 0.1099517436553220,
        0.5 * 0.2233815896780110,
        0.5 * 0.2233815896780110,
        0.5 * 0.1099517436553220,
        0.5 * 0.1099517436553220,
    ],
    dtype=np.float64,
)


def _triangulated_disc(
    radius: float,
    *,
    rings: int,
    sectors: int,
    z: float = 0.0,
    normal_sign: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, z)]
    ring_indices: list[list[int]] = []
    for ring in range(1, rings + 1):
        row: list[int] = []
        r = radius * ring / rings
        for sector in range(sectors):
            theta = 2.0 * np.pi * sector / sectors
            row.append(len(vertices))
            vertices.append((r * np.cos(theta), r * np.sin(theta), z))
        ring_indices.append(row)

    triangles: list[list[int]] = []
    first = ring_indices[0]
    for sector in range(sectors):
        nxt = (sector + 1) % sectors
        tri = [0, first[sector], first[nxt]]
        triangles.append(tri if normal_sign > 0 else [tri[0], tri[2], tri[1]])

    for ring in range(1, rings):
        inner = ring_indices[ring - 1]
        outer = ring_indices[ring]
        for sector in range(sectors):
            nxt = (sector + 1) % sectors
            tris = [
                [inner[sector], outer[sector], outer[nxt]],
                [inner[sector], outer[nxt], inner[nxt]],
            ]
            if normal_sign < 0:
                tris = [[a, c, b] for a, b, c in tris]
            triangles.extend(tris)

    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)


def _rayleigh_pressure_uniform_disc(
    vertices: np.ndarray,
    triangles: np.ndarray,
    points: np.ndarray,
    k: float,
) -> np.ndarray:
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    source_points = []
    weights = []
    for xi, eta, weight in zip(_TRIANGLE_QX, _TRIANGLE_QY, _TRIANGLE_QW, strict=True):
        source_points.append((1.0 - xi - eta) * v0 + xi * v1 + eta * v2)
        weights.append(weight * 2.0 * areas)
    sources = np.concatenate(source_points, axis=0)
    source_weights = np.concatenate(weights, axis=0)

    out = np.empty(points.shape[0], dtype=np.complex128)
    for point_index, point in enumerate(points):
        distance = np.linalg.norm(sources - point[None, :], axis=1)
        green = np.exp(1j * k * distance) / (4.0 * np.pi * distance)
        out[point_index] = 2.0 * np.sum(green * source_weights)
    return out


def _airy_directivity(ka: float, angles_deg: np.ndarray) -> np.ndarray:
    x = ka * np.sin(np.deg2rad(angles_deg))
    out = np.ones_like(x, dtype=np.float64)
    mask = np.abs(x) > 1.0e-12
    out[mask] = 2.0 * j1(x[mask]) / x[mask]
    return np.abs(out)


def _first_crossing_deg(
    angles_deg: np.ndarray,
    values_db: np.ndarray,
    target_db: float,
) -> float:
    for i in range(1, angles_deg.size):
        y0 = float(values_db[i - 1] - target_db)
        y1 = float(values_db[i] - target_db)
        if y0 == 0.0:
            return float(angles_deg[i - 1])
        if y0 * y1 <= 0.0:
            frac = y0 / (y0 - y1)
            return float(angles_deg[i - 1] + frac * (angles_deg[i] - angles_deg[i - 1]))
    raise AssertionError(f"no {target_db} dB crossing")


def _straight_channel_mesh(
    radius: float,
    depth: float,
    *,
    rings: int,
    sectors: int,
) -> LoadedMesh:
    top_vertices, top_triangles = _triangulated_disc(
        radius,
        rings=rings,
        sectors=sectors,
        z=0.0,
        normal_sign=1,
    )
    bottom_vertices, bottom_triangles = _triangulated_disc(
        radius,
        rings=rings,
        sectors=sectors,
        z=-depth,
        normal_sign=-1,
    )

    vertices = np.vstack([top_vertices, bottom_vertices])
    bottom_offset = top_vertices.shape[0]
    triangles = [*top_triangles.tolist()]
    tags = [TAG_APERTURE] * top_triangles.shape[0]
    triangles.extend((bottom_triangles + bottom_offset).tolist())
    tags.extend([TAG_THROAT] * bottom_triangles.shape[0])

    top_outer_start = 1 + (rings - 1) * sectors
    bottom_outer_start = bottom_offset + top_outer_start
    for sector in range(sectors):
        nxt = (sector + 1) % sectors
        top0 = top_outer_start + sector
        top1 = top_outer_start + nxt
        bottom0 = bottom_outer_start + sector
        bottom1 = bottom_outer_start + nxt
        triangles.append([bottom0, bottom1, top1])
        triangles.append([bottom0, top1, top0])
        tags.extend([TAG_WALL, TAG_WALL])

    # Coupled-IB 3D meshes use interior-domain orientation:
    # source +Z into the cavity, wall normals inward, and
    # aperture -Z into the cavity. Rayleigh exterior evaluation is selected by
    # aperture_tag rather than aperture triangle winding.
    triangles_arr = np.asarray(triangles, dtype=np.int32)[:, [0, 2, 1]]
    tags_arr = np.asarray(tags, dtype=np.int32)
    grid = make_pure_grid(vertices, triangles_arr)
    return LoadedMesh(
        grid=grid,
        physical_tags=tags_arr,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles_arr.shape[0],
            physical_groups={
                TAG_THROAT: "throat",
                TAG_WALL: "wall",
                TAG_APERTURE: "aperture",
            },
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
    )


def _z_axis_frame(depth: float) -> ObservationFrame:
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        origin=origin,
        u=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        v=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        mouth_center=origin,
        source_center=np.array([0.0, 0.0, -depth], dtype=np.float64),
    )


def test_uniform_full3d_rayleigh_disc_matches_baffled_piston_analytic():
    radius = 0.05
    ka = 3.0
    k = ka / radius
    distance = 5.0
    angles_deg = np.linspace(0.0, 70.0, 29)
    vertices, triangles = _triangulated_disc(radius, rings=20, sectors=128)
    points = np.column_stack(
        [
            distance * np.sin(np.deg2rad(angles_deg)),
            np.zeros_like(angles_deg),
            distance * np.cos(np.deg2rad(angles_deg)),
        ]
    )

    pressure = _rayleigh_pressure_uniform_disc(vertices, triangles, points, k)
    on_axis_expected = (
        np.exp(1j * k * np.hypot(distance, radius)) - np.exp(1j * k * distance)
    ) / (1j * k)
    relative_on_axis_error = abs(pressure[0] - on_axis_expected) / abs(
        on_axis_expected
    )

    directivity_db = 20.0 * np.log10(np.maximum(np.abs(pressure / pressure[0]), 1e-12))
    airy_db = 20.0 * np.log10(np.maximum(_airy_directivity(ka, angles_deg), 1e-12))

    assert relative_on_axis_error < 8.0e-4
    assert np.max(np.abs(directivity_db - airy_db)) < 0.03
    assert abs(
        _first_crossing_deg(angles_deg, directivity_db, -6.0)
        - _first_crossing_deg(angles_deg, airy_db, -6.0)
    ) < 0.3


def test_native_coupled_ib_straight_channel_matches_analytic_piston(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    radius = 0.04
    depth = 0.003
    frequencies_hz = np.array([800.0, 1600.0], dtype=np.float64)
    observation = ObservationConfig(
        distance_m=1.5,
        angle_min_deg=0.0,
        angle_max_deg=90.0,
        angle_count=10,
        planes=["horizontal"],
        origin="mouth",
    )
    native_config = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        aperture_tag=TAG_APERTURE,
        observation=observation,
        metal_native_assembly_mode="corrected",
        dense_solve_dtype="float64",
    )
    native_result = metal_bem.solve_frequencies(
        _straight_channel_mesh(radius, depth, rings=5, sectors=32),
        frequencies_hz,
        native_config,
    )
    native_directivity = native_result.directivity_db[:, 0, :]
    angles = native_result.observation_angles_deg
    for index, frequency in enumerate(frequencies_hz):
        ka = 2.0 * np.pi * frequency * radius / _SPEED_OF_SOUND
        airy_db = 20.0 * np.log10(_airy_directivity(ka, angles))
        np.testing.assert_allclose(native_directivity[index], airy_db, atol=0.15)
    _assert_outward_baffled_piston(
        _coupled_ib_absolute_sign_scale(
            native_result.pressure_complex, frequencies_hz, radius,
            observation.distance_m, angles,
        )
    )

    assert all(entry.get("coupled_ib") is True for entry in native_result.native_diagnostics)
    assert all(
        entry.get("aperture_velocity_basis") == "DP0"
        for entry in native_result.native_diagnostics
    )
    _assert_piston_radiation_impedance(native_result.impedance, frequencies_hz, radius)
    assert native_directivity[:, -1].min() > -20.0


@pytest.mark.slow
def test_native_coupled_channel_surface_and_hemisphere_power_agree():
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    radius = 0.04
    depth = 0.003
    frequencies_hz = np.array([800.0, 1200.0, 1600.0], dtype=np.float64)
    observation = ObservationConfig(
        distance_m=2.0,
        angle_min_deg=0.0,
        angle_max_deg=90.0,
        angle_count=3,
        planes=["horizontal"],
        origin="mouth",
        sphere_grid=(37, 72),
        sphere_theta_max_deg=90.0,
    )
    config = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        aperture_tag=TAG_APERTURE,
        observation=observation,
        frame_override=_z_axis_frame(depth),
        metal_native_assembly_mode="corrected",
        dense_solve_dtype="float64",
    )

    result = metal_bem.solve_frequencies(
        _straight_channel_mesh(radius, depth, rings=5, sectors=32),
        frequencies_hz,
        config,
    )

    assert result.radiated_power_surface_w is not None
    assert result.radiated_power_sphere_w is not None
    assert result.radiated_power_sphere_coverage_sr == pytest.approx(2.0 * np.pi)
    assert np.all(np.isfinite(result.radiated_power_surface_w))
    assert np.all(np.isfinite(result.radiated_power_sphere_w))
    assert np.all(result.radiated_power_surface_w > 0.0)
    assert np.all(result.radiated_power_sphere_w > 0.0)
    agreement_db = 10.0 * np.log10(
        result.radiated_power_sphere_w / result.radiated_power_surface_w
    )
    assert float(np.max(np.abs(agreement_db))) < 0.2


def test_native_coupled_ib_deep_channel_mesh_convergence(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    radius = 0.04
    depth = 0.10
    frequencies_hz = np.array([900.0, 1400.0, 1800.0], dtype=np.float64)
    observation = ObservationConfig(
        distance_m=1.5,
        angle_min_deg=0.0,
        angle_max_deg=90.0,
        angle_count=13,
        planes=["horizontal"],
        origin="mouth",
    )
    native_config = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        aperture_tag=TAG_APERTURE,
        observation=observation,
        metal_native_assembly_mode="corrected",
        dense_solve_dtype="float64",
    )
    native_result = metal_bem.solve_frequencies(
        _straight_channel_mesh(radius, depth, rings=5, sectors=32),
        frequencies_hz,
        native_config,
    )
    refined_result = metal_bem.solve_frequencies(
        _straight_channel_mesh(radius, depth, rings=7, sectors=48),
        frequencies_hz,
        native_config,
    )
    max_error_db = float(np.max(np.abs(
        native_result.directivity_db - refined_result.directivity_db
    )))

    assert all(entry.get("coupled_ib") is True for entry in native_result.native_diagnostics)
    assert max_error_db < 0.8


# ---------------------------------------------------------------------------
# Absolute-pressure (sign/phase) gate against a baffled piston. Normalized
# directivity alone cannot catch global phase or sign errors in IB coupling.

_SPEED_OF_SOUND = 343.0
_AIR_DENSITY = SolveConfig().air_density


def _assert_piston_radiation_impedance(
    measured: np.ndarray, frequencies_hz: np.ndarray, radius: float,
) -> None:
    """Shallow-channel source impedance approaches a rigid baffled piston."""
    ka = 2.0 * np.pi * frequencies_hz * radius / _SPEED_OF_SOUND
    resistance = _AIR_DENSITY * _SPEED_OF_SOUND * (1.0 - j1(2.0 * ka) / ka)
    # This solver uses exp(-i omega t), hence negative mass-like reactance.
    reactance = -_AIR_DENSITY * _SPEED_OF_SOUND * struve(1, 2.0 * ka) / ka
    np.testing.assert_allclose(measured.real, resistance, rtol=0.20)
    np.testing.assert_allclose(measured.imag, reactance, rtol=0.20)


def _analytic_baffled_piston(radius: float, points: np.ndarray, k: float) -> np.ndarray:
    """Absolute complex pressure of a uniform unit-velocity baffled piston.

    This package uses q=+i*rho*omega*v with an exterior field of -S*q, hence
    p=-i*omega*rho times the half-space single-layer. The single-layer helper
    ``_rayleigh_pressure_uniform_disc`` is the analytic reference validated in
    ``test_uniform_full3d_rayleigh_disc_matches_baffled_piston_analytic``.
    """
    omega = k * _SPEED_OF_SOUND
    verts, tris = _triangulated_disc(radius, rings=20, sectors=128)
    return (-1j * omega * _AIR_DENSITY) * _rayleigh_pressure_uniform_disc(
        verts, tris, points, k
    )


def _coupled_ib_absolute_sign_scale(
    pressure_by_freq: np.ndarray,
    frequencies_hz: np.ndarray,
    radius: float,
    distance: float,
    angles_deg: np.ndarray,
) -> list[tuple[complex, float]]:
    """Best-fit complex scale and shape residual vs the analytic baffled piston.

    For a shallow channel the aperture velocity is ~uniform, so the coupled solve
    should equal ``scale * analytic`` with ``scale ~= +1`` (small |scale| drift is
    the real finite-depth-vs-ideal-piston difference). A 180 deg field inversion
    shows up as ``scale.real < 0``; a corrupted system (wrong interior coupling)
    shows up as a large shape residual.
    """
    front = angles_deg <= 90.0
    pts = np.column_stack(
        [
            distance * np.sin(np.deg2rad(angles_deg)),
            np.zeros_like(angles_deg),
            distance * np.cos(np.deg2rad(angles_deg)),
        ]
    )
    out: list[tuple[complex, float]] = []
    for i, f in enumerate(frequencies_hz):
        k = 2.0 * np.pi * float(f) / _SPEED_OF_SOUND
        analytic = _analytic_baffled_piston(radius, pts, k)[front]
        measured = pressure_by_freq[i, 0, :][front]
        scale = complex(np.vdot(analytic, measured) / np.vdot(analytic, analytic))
        resid = float(
            np.linalg.norm(measured - scale * analytic) / np.linalg.norm(measured)
        )
        out.append((scale, resid))
    return out


def _assert_outward_baffled_piston(scales: list[tuple[complex, float]]) -> None:
    for scale, resid in scales:
        # Outward radiation: in-phase with the analytic piston, not inverted.
        assert scale.real > 0.5, f"radiated field inverted (scale={scale:+.3f})"
        assert abs(scale.imag) < 0.30, f"unexpected radiation phase (scale={scale:+.3f})"
        assert 0.8 < abs(scale) < 1.3, f"radiated magnitude off (scale={scale:+.3f})"
        # Directivity shape must track the analytic piston (guards the interior
        # coupling / system, which a naive sign "fix" on the coupling would break).
        assert resid < 0.03, f"directivity shape off analytic piston (resid={resid:.3%})"


@pytest.mark.parametrize("velocity_mode", [VelocityMode.VELOCITY, VelocityMode.ACCELERATION])
def test_native_coupled_ib_radiates_outward_absolute_sign(velocity_mode: str):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    radius, depth, distance = 0.04, 0.003, 1.5
    frequencies_hz = np.array([800.0, 2000.0], dtype=np.float64)
    angles = np.linspace(0.0, 90.0, 10)
    config = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=velocity_mode,
        aperture_tag=TAG_APERTURE,
        observation=ObservationConfig(
            distance_m=distance,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=angles.size,
            planes=["horizontal"],
            origin="mouth",
        ),
        metal_native_assembly_mode="corrected",
        dense_solve_dtype="float64",
    )
    result = metal_bem.solve_frequencies(
        _straight_channel_mesh(radius, depth, rings=5, sectors=32),
        frequencies_hz,
        config,
    )
    if velocity_mode == VelocityMode.ACCELERATION:
        # a = -i*omega*v under exp(-i*omega*t); recover unit velocity.
        result.pressure_complex *= -1j * 2.0 * np.pi * frequencies_hz[:, None, None]
    _assert_outward_baffled_piston(
        _coupled_ib_absolute_sign_scale(
            result.pressure_complex, frequencies_hz, radius, distance, angles
        )
    )

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
from ib_pipe_reference import (
    baffled_piston_on_axis,
    pipe_mouth_velocity,
    pipe_on_axis_pressure,
)

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


def _straight_channel_mesh(
    radius: float,
    depth: float,
    *,
    rings: int,
    sectors: int,
    wall_layers: int = 1,
) -> LoadedMesh:
    """Straight circular channel: throat disc at z=-depth, aperture disc at z=0.

    wall_layers is the number of element bands along the wall between the two
    discs. The historical fixture had one band, which for a 100 mm channel is a
    single element 100 mm long; that under-resolves every axial standing wave.
    """
    if wall_layers < 1:
        raise ValueError("wall_layers must be >= 1")
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

    bottom_offset = top_vertices.shape[0]
    theta = 2.0 * np.pi * np.arange(sectors) / sectors
    middle_rings = [
        np.column_stack(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                np.full(sectors, -depth + depth * layer / wall_layers),
            ]
        )
        for layer in range(1, wall_layers)
    ]
    vertices = np.vstack([top_vertices, bottom_vertices, *middle_rings])
    triangles = [*top_triangles.tolist()]
    tags = [TAG_APERTURE] * top_triangles.shape[0]
    triangles.extend((bottom_triangles + bottom_offset).tolist())
    tags.extend([TAG_THROAT] * bottom_triangles.shape[0])

    top_outer_start = 1 + (rings - 1) * sectors
    bottom_outer_start = bottom_offset + top_outer_start
    middle_start = bottom_offset + bottom_vertices.shape[0]
    # Outer-ring vertex indices from the throat (z=-depth) up to the mouth (z=0).
    ring_indices = [bottom_outer_start + np.arange(sectors)]
    ring_indices.extend(
        middle_start + layer * sectors + np.arange(sectors)
        for layer in range(wall_layers - 1)
    )
    ring_indices.append(top_outer_start + np.arange(sectors))
    for layer in range(wall_layers):
        lower = ring_indices[layer]
        upper = ring_indices[layer + 1]
        for sector in range(sectors):
            nxt = (sector + 1) % sectors
            triangles.append([lower[sector], lower[nxt], upper[nxt]])
            triangles.append([lower[sector], upper[nxt], upper[sector]])
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


def test_ib_reference_model_self_consistency():
    """Guard the analytic reference itself (no engine involved).

    The closed-form on-axis Rayleigh pressure used by the resonant gate must agree
    with a quadrature of the half-space single-layer over a meshed disc, and a
    vanishingly short pipe must deliver the throat velocity to the mouth.
    """
    radius, distance = 0.04, 1.5
    vertices, triangles = _triangulated_disc(radius, rings=20, sectors=128)
    points = np.array([[0.0, 0.0, distance]])
    for frequency in (300.0, 1000.0, 2200.0):
        k = 2.0 * np.pi * frequency / _SPEED_OF_SOUND
        quadrature = (-1j * k * _SPEED_OF_SOUND * _AIR_DENSITY) * (
            _rayleigh_pressure_uniform_disc(vertices, triangles, points, k)[0]
        )
        closed_form = baffled_piston_on_axis(
            radius, distance, k, _AIR_DENSITY, _SPEED_OF_SOUND
        )
        assert abs(quadrature - closed_form) / abs(closed_form) < 8.0e-4
        assert pipe_mouth_velocity(
            k, radius, 1.0e-6, _AIR_DENSITY, _SPEED_OF_SOUND
        ) == pytest.approx(1.0, abs=1.0e-4)


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
            native_result.pressure_complex, frequencies_hz, radius, depth,
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


# ---------------------------------------------------------------------------
# Resonant deep-channel gate against the 1-D pipe + King baffled-piston reference
# (``ib_pipe_reference``; model equations, geometry, source normalization,
# observation convention and valid range are documented there).
#
# Fixture: radius 40 mm, depth 100 mm, unit throat velocity, on-axis point
# 1.5 m from the mouth. Frequency band 300 Hz - 2.2 kHz (ka <= 1.6, below the
# first cross mode at ka = 1.84). The first pipe resonance is near 650 Hz.
#
# Tolerance derivation (Metal, standard formulation, float64 dense solve). Three
# meshes with proportional refinement -- wall layers 10 / 15 / 22 with
# rings x sectors 4x24 / 6x36 / 9x54 (816 / 1872 / 4212 triangles) -- measured
# on the gate grid against the reference:
#
#   mesh   max |dB|  max |phase|: off / in 600-700 Hz   resonance offset   peak level
#   A  10   0.67      4.3 deg / 8.0 deg                  +6.6 Hz            -0.07 dB
#   B  15   0.62      4.4 deg / 7.7 deg                  +6.3 Hz            -0.06 dB
#   C  22   0.59      4.4 deg / 7.5 deg                  +6.2 Hz            -0.05 dB
#
# Mesh-to-mesh differences are small (A-B 0.07 dB / 0.4 deg, B-C 0.03 dB /
# 0.14 deg, A-C 0.10 dB / 0.5 deg), so the residual against the reference is not
# discretisation error. It is the model gap between the exact solver and the
# reference (uniform mouth velocity, plane-wave-only pipe): about +6 Hz (1 %) in
# resonance frequency and 0.6 dB / 4.4 deg elsewhere; the larger in-band phase
# sits at the resonance, where 6 Hz of shift is several degrees of phase. Each
# limit below is that converged gap times 1.5-2 (coarsest mesh included).
#
# A single wall element (the historical fixture, 5 rings x 32 sectors x 1 layer)
# is off by 24 dB and 115 deg and shows no resonance peak, so it fails every
# limit; ``test_..._gate_rejects_single_layer_wall`` keeps that provable.

_GATE_RADIUS_M = 0.04
_GATE_DEPTH_M = 0.10
_GATE_DISTANCE_M = 1.5
# (rings, sectors, wall layers): sectors divisible by 4 so half/quarter cuts land
# on mesh lines.
_MESH_A = dict(rings=4, sectors=24, wall_layers=10)
_MESH_B = dict(rings=6, sectors=36, wall_layers=15)
_MESH_C = dict(rings=9, sectors=54, wall_layers=22)
_MESH_SINGLE_LAYER = dict(rings=5, sectors=32, wall_layers=1)
_GATE_BAND_HZ = np.arange(300.0, 2200.1, 50.0)
_GATE_RESONANCE_SCAN_HZ = np.arange(620.0, 700.1, 4.0)
_GATE_RESONANCE_BAND_HZ = (600.0, 700.0)

_GATE_RESONANCE_TOL_HZ = 12.0   # converged gap 6.2-6.6 Hz
_GATE_PEAK_LEVEL_TOL_DB = 0.25  # converged gap 0.05-0.07 dB
_GATE_LEVEL_TOL_DB = 1.0        # converged gap 0.59-0.67 dB
_GATE_PHASE_TOL_DEG = 8.0       # away from the resonance: converged gap 4.4 deg
_GATE_PHASE_TOL_RESONANCE_DEG = 12.0  # 600-700 Hz: converged gap 7.5-8.0 deg


def _channel_config(
    *,
    depth: float = _GATE_DEPTH_M,
    symmetry: str | None = None,
    angle_count: int = 4,
) -> SolveConfig:
    return SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        aperture_tag=TAG_APERTURE,
        observation=ObservationConfig(
            distance_m=_GATE_DISTANCE_M,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=angle_count,
            planes=["horizontal"],
            origin="mouth",
        ),
        frame_override=_z_axis_frame(depth),
        native_symmetry_plane=symmetry,
        metal_native_assembly_mode="corrected",
        dense_solve_dtype="float64",
    )


def _require_native_helper() -> None:
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )


def _gate_frequencies() -> np.ndarray:
    return np.unique(np.concatenate([_GATE_BAND_HZ, _GATE_RESONANCE_SCAN_HZ]))


def _channel_on_axis(mesh: LoadedMesh, frequencies_hz: np.ndarray) -> np.ndarray:
    result = metal_bem.solve_frequencies(mesh, frequencies_hz, _channel_config())
    assert all(entry.get("coupled_ib") is True for entry in result.native_diagnostics)
    return result.pressure_complex[:, 0, 0]


def _pipe_reference(frequencies_hz: np.ndarray) -> np.ndarray:
    return pipe_on_axis_pressure(
        frequencies_hz,
        radius=_GATE_RADIUS_M,
        depth=_GATE_DEPTH_M,
        distance=_GATE_DISTANCE_M,
        rho=_AIR_DENSITY,
        c=_SPEED_OF_SOUND,
    )


def _resonance_peak(
    frequencies_hz: np.ndarray, pressure: np.ndarray
) -> tuple[float, float]:
    """Peak frequency (parabola through log |p|) and peak level in dB, 600-700 Hz.

    NaN when the maximum sits on the edge of the scan (no resonance found).
    """
    lo, hi = _GATE_RESONANCE_BAND_HZ
    mask = (frequencies_hz >= lo) & (frequencies_hz <= hi)
    f = frequencies_hz[mask]
    y = np.log(np.abs(pressure[mask]))
    i = int(np.argmax(y))
    if not 0 < i < f.size - 1:
        # No interior peak in the scan: report NaN so every comparison fails.
        return float("nan"), float("nan")
    step = f[i + 1] - f[i]
    offset = 0.5 * (y[i - 1] - y[i + 1]) / (y[i - 1] - 2.0 * y[i] + y[i + 1])
    peak_level = y[i] - 0.25 * (y[i - 1] - y[i + 1]) * offset
    return float(f[i] + offset * step), float(20.0 * np.log10(np.e) * peak_level)


def _pipe_gap(frequencies_hz: np.ndarray, pressure: np.ndarray) -> dict[str, float]:
    reference = _pipe_reference(frequencies_hz)
    ratio = pressure / reference
    level_db = 20.0 * np.log10(np.abs(ratio))
    phase_deg = np.degrees(np.angle(ratio))
    in_band = (frequencies_hz >= _GATE_RESONANCE_BAND_HZ[0]) & (
        frequencies_hz <= _GATE_RESONANCE_BAND_HZ[1]
    )
    peak_f, peak_db = _resonance_peak(frequencies_hz, pressure)
    ref_peak_f, ref_peak_db = _resonance_peak(frequencies_hz, reference)
    return {
        "level_db": float(np.max(np.abs(level_db))),
        "phase_off_resonance_deg": float(np.max(np.abs(phase_deg[~in_band]))),
        "phase_resonance_deg": float(np.max(np.abs(phase_deg[in_band]))),
        "resonance_hz": peak_f - ref_peak_f,
        "peak_level_db": peak_db - ref_peak_db,
    }


def _assert_within_pipe_reference(
    frequencies_hz: np.ndarray, pressure: np.ndarray
) -> None:
    gap = _pipe_gap(frequencies_hz, pressure)
    assert abs(gap["resonance_hz"]) < _GATE_RESONANCE_TOL_HZ, gap
    assert abs(gap["peak_level_db"]) < _GATE_PEAK_LEVEL_TOL_DB, gap
    assert gap["level_db"] < _GATE_LEVEL_TOL_DB, gap
    assert gap["phase_off_resonance_deg"] < _GATE_PHASE_TOL_DEG, gap
    assert gap["phase_resonance_deg"] < _GATE_PHASE_TOL_RESONANCE_DEG, gap


@pytest.mark.parametrize("mesh_spec", [_MESH_A, _MESH_B], ids=["10-layers", "15-layers"])
def test_native_coupled_ib_resonant_channel_matches_pipe_reference(mesh_spec):
    """Deep channel: resonance frequency, absolute on-axis level and phase vs the 1-D pipe."""
    _require_native_helper()
    frequencies = _gate_frequencies()
    pressure = _channel_on_axis(
        _straight_channel_mesh(_GATE_RADIUS_M, _GATE_DEPTH_M, **mesh_spec), frequencies
    )
    _assert_within_pipe_reference(frequencies, pressure)


def test_native_coupled_ib_pipe_gate_rejects_single_layer_wall():
    """The historical one-band wall fixture cannot pass the resonant gate."""
    _require_native_helper()
    frequencies = _gate_frequencies()
    pressure = _channel_on_axis(
        _straight_channel_mesh(
            _GATE_RADIUS_M, _GATE_DEPTH_M, **_MESH_SINGLE_LAYER
        ),
        frequencies,
    )
    with pytest.raises(AssertionError):
        _assert_within_pipe_reference(frequencies, pressure)


def test_native_coupled_ib_deep_channel_wall_refinement_converges():
    """Refine the WALL (and the discs in proportion); absolute level and phase converge.

    Compares complex on-axis and off-axis pressure between 10/15/22-layer meshes at
    twenty-one frequencies including the resonance, and checks that every mesh is
    within the pipe-reference gate, so the gate limits are not fitted to one mesh.
    """
    _require_native_helper()
    frequencies = np.unique(
        np.concatenate([np.arange(300.0, 2201.0, 100.0), _GATE_RESONANCE_SCAN_HZ[::4]])
    )
    fields = {}
    for name, spec in (("A", _MESH_A), ("B", _MESH_B), ("C", _MESH_C)):
        result = metal_bem.solve_frequencies(
            _straight_channel_mesh(_GATE_RADIUS_M, _GATE_DEPTH_M, **spec),
            frequencies,
            _channel_config(angle_count=7),
        )
        fields[name] = result.pressure_complex[:, 0, :]

    def difference(a: str, b: str) -> tuple[float, float]:
        ratio = fields[a] / fields[b]
        return (
            float(np.max(np.abs(20.0 * np.log10(np.abs(ratio))))),
            float(np.max(np.abs(np.degrees(np.angle(ratio))))),
        )

    ab_db, ab_deg = difference("A", "B")
    bc_db, bc_deg = difference("B", "C")
    ac_db, ac_deg = difference("A", "C")
    # Measured: A-B 0.07 dB / 0.4 deg, B-C 0.03 dB / 0.13 deg, A-C 0.10 dB / 0.5 deg.
    assert ac_db < 0.25 and ac_deg < 1.5, (ac_db, ac_deg)
    assert bc_db < 0.15 and bc_deg < 1.0, (bc_db, bc_deg)
    assert bc_db < ac_db and bc_deg < ac_deg, "refinement is not converging"
    # The finest mesh also sits inside the gate against the reference. On this
    # sparse grid the resonance peak is not resolved, so only level/phase apply.
    reference = _pipe_reference(frequencies)
    ratio = fields["C"][:, 0] / reference
    assert np.max(np.abs(20.0 * np.log10(np.abs(ratio)))) < _GATE_LEVEL_TOL_DB
    assert np.max(np.abs(np.degrees(np.angle(ratio)))) < _GATE_PHASE_TOL_RESONANCE_DEG


def _cut_channel_mesh(mesh: LoadedMesh, *, quarter: bool) -> LoadedMesh:
    """Keep the x>0 half (or x>0,y>0 quarter) of a channel whose sectors divide by 4."""
    vertices = np.asarray(mesh.grid.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.grid.elements)
    if vertices.shape[0] == 3:
        vertices = vertices.T
    if triangles.shape[0] == 3:
        triangles = triangles.T
    centroids = vertices[triangles].mean(axis=1)
    keep = centroids[:, 0] > 0.0
    if quarter:
        keep &= centroids[:, 1] > 0.0
    used = np.unique(triangles[keep])
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    cut_vertices = vertices[used]
    cut_triangles = remap[triangles[keep]].astype(np.int32)
    return LoadedMesh(
        grid=make_pure_grid(cut_vertices, cut_triangles),
        physical_tags=mesh.physical_tags[keep],
        info=MeshInfo(
            n_vertices=cut_vertices.shape[0],
            n_triangles=cut_triangles.shape[0],
            physical_groups=mesh.info.physical_groups,
            bounding_box_m=(cut_vertices.min(axis=0), cut_vertices.max(axis=0)),
        ),
    )


@pytest.mark.parametrize("plane, quarter", [("yz", False), ("yz+xz", True)])
def test_native_coupled_ib_half_and_quarter_match_full_at_resonance(plane, quarter):
    """Symmetric reduction of the resonant channel reproduces the full model.

    The circular channel is symmetric under x -> -x and y -> -y, so the yz half and
    yz+xz quarter meshes must give the same pressure field and source impedance as
    the full mesh, including at the 650 Hz resonance where the solve is sensitive.
    Measured differences (float32 field path): <= 4.4e-4 relative pressure,
    <= 0.003 dB, <= 0.02 deg; impedance <= 1e-3 relative.
    """
    _require_native_helper()
    frequencies = np.array([300.0, 650.0, 659.0, 1000.0, 2000.0])
    full_mesh = _straight_channel_mesh(_GATE_RADIUS_M, _GATE_DEPTH_M, **_MESH_B)
    full = metal_bem.solve_frequencies(full_mesh, frequencies, _channel_config())
    reduced = metal_bem.solve_frequencies(
        _cut_channel_mesh(full_mesh, quarter=quarter),
        frequencies,
        _channel_config(symmetry=plane),
    )
    assert all(entry.get("coupled_ib") is True for entry in reduced.native_diagnostics)
    relative = np.abs(reduced.pressure_complex - full.pressure_complex) / np.abs(
        full.pressure_complex
    )
    assert float(np.max(relative)) < 2.0e-3
    np.testing.assert_allclose(reduced.impedance, full.impedance, rtol=5.0e-3)


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
    p=-i*omega*rho times the half-space single-layer, evaluated by quadrature with
    ``_rayleigh_pressure_uniform_disc``. On axis it is checked against the closed
    form in ``test_ib_reference_model_self_consistency``.
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
    depth: float,
    distance: float,
    angles_deg: np.ndarray,
) -> list[tuple[complex, float]]:
    """Best-fit complex scale and shape residual vs the pipe-loaded baffled piston.

    The reference field is the unit-velocity baffled piston times the mouth
    velocity of the lossless 1-D channel of the given depth (``pipe_mouth_velocity``:
    the throat velocity is not the mouth velocity, even for 3 mm at 2 kHz, where it
    is 1.08 in magnitude and 5 deg in phase). The coupled solve should therefore be
    ``scale * reference`` with ``scale == 1``. A 180 deg field inversion shows up as
    ``scale.real < 0``; a corrupted system (wrong interior coupling) as a large
    shape residual.
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
        mouth_velocity = pipe_mouth_velocity(
            k, radius, depth, _AIR_DENSITY, _SPEED_OF_SOUND
        )
        analytic = (
            mouth_velocity * _analytic_baffled_piston(radius, pts, k)[front]
        )
        measured = pressure_by_freq[i, 0, :][front]
        scale = complex(np.vdot(analytic, measured) / np.vdot(analytic, analytic))
        resid = float(
            np.linalg.norm(measured - scale * analytic) / np.linalg.norm(measured)
        )
        out.append((scale, resid))
    return out


def _assert_outward_baffled_piston(scales: list[tuple[complex, float]]) -> None:
    # Measured on 5x32 / 8x48 / 10x64-ring shallow (3 mm) meshes at 800 Hz and
    # 2 kHz: |scale| 0.995 / 0.984, phase +0.02 / -0.37 deg, residual 0.1 % / 1 %.
    for scale, resid in scales:
        assert scale.real > 0.0, f"radiated field inverted (scale={scale:+.3f})"
        assert abs(abs(scale) - 1.0) < 0.05, f"radiated magnitude off (scale={scale:+.3f})"
        assert abs(np.degrees(np.angle(scale))) < 3.0, (
            f"radiation phase off (scale={scale:+.3f})"
        )
        # Directivity shape must track the analytic piston (guards the interior
        # coupling / system, which a naive sign "fix" on the coupling would break).
        assert resid < 0.02, f"directivity shape off analytic piston (resid={resid:.3%})"


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
            result.pressure_complex, frequencies_hz, radius, depth, distance, angles
        )
    )

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_metal_bem.boundary_lab import BACKEND_ID, BoundaryLabSolverError, create_backend
from hornlab_metal_bem.boundary_lab import _boundary_lab_channel_drive
from hornlab_metal_bem.boundary_lab import _coerce_symmetry_plane
from hornlab_metal_bem.boundary_lab import _crossover_response
from hornlab_metal_bem.boundary_lab import _frequency_result_from_channel_basis_entry
from hornlab_metal_bem.boundary_lab import _frequency_result_from_log_entry
from hornlab_metal_bem.boundary_lab import _level_polarity_delay_filter_drive
from hornlab_metal_bem.boundary_lab import solve_config_from_boundary_lab


def test_boundary_lab_config_defaults_to_native_metal():
    config, frequencies = solve_config_from_boundary_lab(
        {
            "frequency_min_hz": 800.0,
            "frequency_max_hz": 12_500.0,
            "frequency_count": 9,
            "source_tag": 7,
            "observation_planes": ["horizontal"],
        }
    )

    assert frequencies is None
    assert config.freq_min_hz == 800.0
    assert config.freq_max_hz == 12_500.0
    assert config.freq_count == 9
    assert config.velocity_sources == {7: 1.0}
    assert config.observation.planes == ["horizontal"]


def test_boundary_lab_config_defaults_to_strict_open_edge_check():
    config, _ = solve_config_from_boundary_lab({"symmetry": "yz+xz"})
    assert config.native_check_open_edges is True


@pytest.mark.parametrize("name", ["aperture_tag", "apertureTag"])
def test_boundary_lab_config_forwards_infinite_baffle_aperture_tag(name):
    config, _ = solve_config_from_boundary_lab({name: 12})

    assert config.aperture_tag == 12


def test_boundary_lab_config_rejects_conflicting_aperture_aliases():
    with pytest.raises(BoundaryLabSolverError, match="Conflicting aperture metadata"):
        solve_config_from_boundary_lab({"aperture_tag": 12, "apertureTag": 13})

    with pytest.raises(BoundaryLabSolverError, match="Conflicting aperture_tag"):
        solve_config_from_boundary_lab({}, aperture_tag=12, apertureTag=13)


def test_boundary_lab_config_forwards_open_edge_opt_out_for_open_shells():
    config, _ = solve_config_from_boundary_lab(
        {"symmetry": "yz+xz", "native_check_open_edges": False}
    )
    assert config.native_check_open_edges is False
    # The validator-level alias is also accepted for spec authors.
    alias, _ = solve_config_from_boundary_lab(
        {"symmetry": "yz+xz", "check_open_edges": False}
    )
    assert alias.native_check_open_edges is False


def test_boundary_lab_config_accepts_explicit_frequencies_and_overrides():
    source = SimpleNamespace(
        frequencies_hz=[1000.0, 1600.0],
        velocity_sources={3: 0.5},
        angle_count=5,
    )

    config, frequencies = solve_config_from_boundary_lab(
        source,
        metal_native_threads_per_group=64,
    )

    np.testing.assert_allclose(frequencies, [1000.0, 1600.0])
    assert config.velocity_sources == {3: 0.5}
    assert config.observation.angle_count == 5
    assert config.metal_native_threads_per_group == 64


def test_boundary_lab_backend_exposes_stable_id_and_session():
    backend = create_backend(freq_count=3)
    session = backend.create_session({"freq_min_hz": 900.0})

    assert backend.backend_id == BACKEND_ID
    assert backend.capabilities.supports_channel_resynthesis is True
    assert session.default_overrides == {"freq_count": 3}


def test_boundary_lab_backend_accepts_solve_request_contract():
    config = SimpleNamespace(
        mesh_file="waveguide.msh",
        min_angle=-90.0,
        max_angle=90.0,
        step_size=45.0,
        radiators=(SimpleNamespace(name="Woofer", tag=7, velocity_offset_db=6.0),),
    )
    request = SimpleNamespace(
        config=config,
        frequencies_hz=np.array([1000.0, 1600.0], dtype=np.float32),
    )
    backend = create_backend()
    session = backend.create_session(request)

    assert session.metadata.polar_angle_deg.tolist() == [-90.0, -45.0, 0.0, 45.0, 90.0]
    assert session.metadata.radiator_names.tolist() == ["Woofer"]

    solve_config, frequencies = solve_config_from_boundary_lab(config)
    assert frequencies is None
    assert solve_config.velocity_sources[7] == 1.0
    assert solve_config.velocity_source_callback is not None
    assert solve_config.velocity_source_callback(1000.0)[7] == 10.0 ** (6.0 / 20.0)
    assert solve_config.observation.angle_min_deg == -90.0
    assert solve_config.observation.angle_max_deg == 90.0
    assert solve_config.observation.angle_count == 5


def test_boundary_lab_session_translates_dict_shaped_requests():
    backend = create_backend()
    session = backend.create_session(
        {
            "config": {"frequency_count": 7, "source_tag": 3},
            "frequencies_hz": [100.0, 200.0],
        }
    )

    assert session._simulation_config == {"frequency_count": 7, "source_tag": 3}
    np.testing.assert_allclose(session._frequencies_hz, [100.0, 200.0])


def test_boundary_lab_metadata_angles_match_solved_grid_for_non_divisible_step():
    config = {"min_angle": 0.0, "max_angle": 100.0, "step_size": 30.0}
    backend = create_backend()
    session = backend.create_session({"config": config})

    solve_config, _ = solve_config_from_boundary_lab(config)
    solved = np.linspace(
        solve_config.observation.angle_min_deg,
        solve_config.observation.angle_max_deg,
        solve_config.observation.angle_count,
    )
    np.testing.assert_allclose(session.metadata.polar_angle_deg, solved)


def test_burton_miller_maps_to_complex_k_formulation():
    on, _ = solve_config_from_boundary_lab({"use_burton_miller": True})
    off, _ = solve_config_from_boundary_lab({"use_burton_miller": False})
    absent, _ = solve_config_from_boundary_lab({})
    assert on.formulation == "complex_k"
    assert off.formulation == "standard"
    assert absent.formulation == "standard"


def test_spherical_sampling_builds_sphere_points_and_metadata():
    config = {
        "mesh_file": "m.msh",
        "spherical_sampling_enabled": True,
        "spherical_sampling_points": 32,
        "distance": 3.0,
    }
    solve_config, _ = solve_config_from_boundary_lab(config)
    sphere_points = solve_config.observation.sphere_points
    assert sphere_points is not None
    assert sphere_points.shape == (32, 3)
    # Points lie on the requested-radius sphere (origin-centred), matching
    # Boundary Lab's own Fibonacci-sphere convention.
    np.testing.assert_allclose(np.linalg.norm(sphere_points, axis=1), 3.0, atol=1e-5)

    metadata = create_backend().create_session({"config": config}).metadata
    assert metadata.sphere_metadata is not None
    assert set(metadata.sphere_metadata) == {"r_distance_m", "theta_polar_rad", "phi_azimuth_rad"}
    assert metadata.sphere_metadata["theta_polar_rad"].shape == (32,)


def test_spherical_sampling_disabled_leaves_no_sphere():
    solve_config, _ = solve_config_from_boundary_lab({"mesh_file": "m.msh"})
    assert solve_config.observation.sphere_points is None
    assert create_backend().create_session({"config": {"mesh_file": "m.msh"}}).metadata.sphere_metadata is None


def test_channel_basis_entry_carries_sphere_pressure():
    sphere = np.asarray([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex64)
    entry = {
        "observation_planes": ["horizontal", "vertical"],
        "observation_angles_deg": np.asarray([-45.0, 0.0, 45.0], dtype=np.float32),
        "observation_pressure_complex": np.asarray(
            [[1.0, 2.0, 1.0], [0.5, 1.0, 0.5]], dtype=np.complex64
        ),
        "observation_sphere_pressure_complex": sphere,
        "impedance": 1.0 + 0.0j,
    }
    result = _frequency_result_from_channel_basis_entry(
        1000.0, entry, channel_names=np.asarray(["main"]), channel_configs={}
    )
    assert result.sphere_pressure is not None
    assert result.sphere_pressure.shape == (1, 4)
    # Basis is conjugated to match the channel-synthesis time convention.
    np.testing.assert_allclose(result.sphere_pressure[0], np.conj(sphere))


def test_channel_basis_entry_without_sphere_pressure_stays_none():
    entry = {
        "observation_planes": ["horizontal", "vertical"],
        "observation_angles_deg": np.asarray([-45.0, 0.0, 45.0], dtype=np.float32),
        "observation_pressure_complex": np.asarray(
            [[1.0, 2.0, 1.0], [0.5, 1.0, 0.5]], dtype=np.complex64
        ),
        "impedance": 1.0 + 0.0j,
    }
    result = _frequency_result_from_channel_basis_entry(
        1000.0, entry, channel_names=np.asarray(["main"]), channel_configs={}
    )
    assert result.sphere_pressure is None


def test_boundary_lab_solve_stream_resets_stop_flag_between_streams():
    backend = create_backend()
    session = backend.create_session({"config": {"mesh_file": "m.msh"}})
    session._stop = True

    # The reset happens at solve_stream entry, before any solving starts;
    # pull one step of the generator setup by checking the flag directly.
    stream = session.solve_stream()
    with pytest.raises(BoundaryLabSolverError, match="frequencies_hz"):
        next(stream)
    assert session._stop is False


def test_boundary_lab_session_streams_channel_basis_from_multi_source(monkeypatch):
    import hornlab_metal_bem

    captured = {}

    def fake_solve_multi_source(mesh, sources, solve_config, frequencies_hz):
        captured["mesh"] = mesh
        captured["sources"] = sources
        captured["velocity_source_callback"] = solve_config.velocity_source_callback
        pressure_lf = np.asarray(
            [
                [1.0 + 0.0j, 2.0 + 0.0j, 1.0 + 0.0j],
                [0.5 + 0.0j, 1.0 + 0.0j, 0.5 + 0.0j],
            ],
            dtype=np.complex64,
        )
        pressure_hf = np.asarray(
            [
                [0.25 + 0.0j, 0.5 + 0.0j, 0.25 + 0.0j],
                [0.25 + 0.0j, 0.25 + 0.0j, 0.25 + 0.0j],
            ],
            dtype=np.complex64,
        )
        source_entries = []
        for source in sources:
            if 7 in source:
                source_entries.append({"observation_pressure_complex": pressure_lf, "impedance": 1.0 + 0.0j})
            elif 8 in source:
                source_entries.append({"observation_pressure_complex": pressure_hf, "impedance": 2.0 + 0.0j})
        solve_config.on_frequency_result(
            0,
            float(np.asarray(frequencies_hz)[0]),
            {
                "observation_planes": ["horizontal", "vertical"],
                "observation_angles_deg": np.asarray([-45.0, 0.0, 45.0], dtype=np.float32),
                "source_results": source_entries,
            },
        )
        return []

    monkeypatch.setattr(hornlab_metal_bem, "solve_multi_source", fake_solve_multi_source)

    config = SimpleNamespace(
        mesh_file="waveguide.msh",
        min_angle=-45.0,
        max_angle=45.0,
        step_size=45.0,
        radiators=(
            SimpleNamespace(name="woofer", tag=7, channel="LF", velocity_offset_db=6.0),
            SimpleNamespace(name="tweeter", tag=8, channel="HF", velocity_offset_db=-6.0),
        ),
        channels=(SimpleNamespace(name="LF"), SimpleNamespace(name="HF")),
    )
    request = SimpleNamespace(config=config, frequencies_hz=np.asarray([1000.0], dtype=np.float32))
    session = create_backend().create_session(request)

    (result,) = list(session.solve_stream())

    assert captured["mesh"] == "waveguide.msh"
    assert captured["velocity_source_callback"] is None
    assert captured["sources"][0][8] == pytest.approx(10.0 ** (-6.0 / 20.0))
    assert captured["sources"][1][7] == pytest.approx(10.0 ** (6.0 / 20.0))
    assert result.channel_names.tolist() == ["HF", "LF"]
    assert result.horizontal_pressure.shape == (2, 3)


def test_drive_delay_uses_positive_phase_for_lagging_channel():
    # e^{-i omega t} convention: a delayed channel's phasor rotates by
    # +omega*tau. 1 ms at 250 Hz is a quarter turn: 1 -> +j.
    drive = _level_polarity_delay_filter_drive(
        {"delay_ms": 1.0},
        250.0,
    )
    np.testing.assert_allclose(drive, np.exp(1j * np.pi / 2.0), rtol=1e-12)


def test_drive_rejects_polarity_outside_unit():
    with pytest.raises(BoundaryLabSolverError, match="polarity"):
        _level_polarity_delay_filter_drive({"polarity": 0}, 1000.0)


def test_boundary_lab_channel_delay_uses_negative_phase_on_conjugated_basis():
    # The streaming channel-basis path conjugates the Metal pressures into
    # Boundary Lab's engineering e^{+j omega t} convention before synthesis
    # (see _frequency_result_from_channel_basis_entry), and blab's own
    # channel_drive delays with e^{-j omega tau} on that basis. The adapter's
    # synthesis weight must match: 1 ms at 250 Hz is a quarter turn, 1 -> -j
    # (the conjugate of the solver-side +j).
    drive = _boundary_lab_channel_drive({"delay_ms": 1.0}, 250.0)
    np.testing.assert_allclose(drive, np.exp(-1j * np.pi / 2.0), rtol=1e-12)


@pytest.mark.parametrize("frequency_hz", [250.0, 1000.0, 4000.0, 12_000.0])
def test_boundary_lab_channel_drive_is_conjugate_of_solver_side_drive(frequency_hz):
    # The same channel settings are applied through two disjoint paths: at
    # solve time as velocity drives in the Metal e^{-i omega t} convention
    # (_level_polarity_delay_filter_drive), or post-solve as synthesis
    # weights on the conjugated basis (_boundary_lab_channel_drive). They
    # describe the same physics exactly when the whole drive is a conjugate
    # pair: conj(w_eng * conj(p)) == conj(w_eng) * p == w_solver * p. A sign
    # "fix" to either delay (or a dropped crossover conjugation) breaks this.
    channel = {
        "level_db": 2.0,
        "polarity": -1,
        "delay_ms": 0.37,
        "hpf": {"type": "hpf", "filter": "butterworth", "order": 3, "frequency_hz": 800.0},
        "lpf": {"type": "lpf", "filter": "linkwitz_riley", "order": 4, "frequency_hz": 4000.0},
    }
    np.testing.assert_allclose(
        _boundary_lab_channel_drive(channel, frequency_hz),
        np.conj(_level_polarity_delay_filter_drive(channel, frequency_hz)),
        rtol=1e-12,
    )


def test_crossover_response_is_conjugate_of_scipy_convention():
    # 1st-order Butterworth LPF at cutoff: H(j w_c) = 1/(1+j) in scipy's
    # e^{+j omega t} convention, so the drive must be 1/(1-j).
    response = _crossover_response(
        {"type": "lpf", "filter": "butterworth", "order": 1, "frequency_hz": 1000.0},
        1000.0,
    )
    np.testing.assert_allclose(response, 1.0 / (1.0 - 1.0j), rtol=1e-9)


@pytest.mark.parametrize(
    ("crossover", "match"),
    [
        ({"type": "bandpass", "frequency_hz": 1000.0}, "crossover type"),
        ({"type": "lpf", "filter": "bessel", "frequency_hz": 1000.0}, "crossover filter"),
        ({"type": "lpf"}, "missing frequency_hz"),
        ({"type": "lpf", "frequency_hz": 0.0}, "must be positive"),
        ({"type": "lpf", "order": 0, "frequency_hz": 1000.0}, "order"),
        (
            {"type": "lpf", "filter": "linkwitz_riley", "order": 3, "frequency_hz": 1000.0},
            "even",
        ),
    ],
)
def test_crossover_response_rejects_malformed_configs(crossover, match):
    with pytest.raises(BoundaryLabSolverError, match=match):
        _crossover_response(crossover, 2000.0)


def test_linkwitz_riley_squares_butterworth_sections():
    lr4 = _crossover_response(
        {"type": "lpf", "filter": "linkwitz_riley", "order": 4, "frequency_hz": 1000.0},
        1000.0,
    )
    # LR4 at cutoff is -6 dB.
    np.testing.assert_allclose(abs(lr4), 0.5, rtol=1e-9)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("off", None),
        ("x", "yz"),
        ("y", "xz"),
        ("z", "xy"),
        ("xy", "yz+xz"),
        ("yz", "yz"),
        ("xz", "xz"),
        ("yz+xz", "yz+xz"),
    ],
)
def test_coerce_symmetry_plane_maps_boundary_lab_tokens(token, expected):
    assert _coerce_symmetry_plane(token) == expected


def test_boundary_lab_frequency_result_preserves_streamed_complex_pressure():
    pressure = np.asarray(
        [
            [1.0 + 1.0j, 2.0 - 1.0j],
            [0.5 + 0.25j, 0.25 - 0.5j],
        ],
        dtype=np.complex128,
    )
    result = _frequency_result_from_log_entry(
        1000.0,
        {
            "observation_planes": ["horizontal", "vertical"],
            "observation_directivity_db": np.zeros((2, 2), dtype=np.float64),
            "observation_pressure_complex": pressure,
            "impedance": 1.0 + 0.5j,
            "lapack_info": 0,
            "native_diagnostics": {"assembly_implementation": "test"},
        },
    )

    np.testing.assert_allclose(result.observation_pressure_complex, pressure)
    np.testing.assert_allclose(
        result.horizontal_spl_db,
        20.0 * np.log10(np.abs(pressure[0]) / 20e-6),
    )
    assert result.native_diagnostics["assembly_implementation"] == "test"
    assert result.diagnostics.convergence_info == 0


def test_boundary_lab_channel_basis_result_uses_standard_blab_fields():
    pressure_lf = np.asarray(
        [
            [1.0 + 1.0j, 2.0 + 0.0j, 1.0 - 1.0j],
            [0.5 + 0.0j, 1.0 + 0.0j, 0.5 + 0.0j],
        ],
        dtype=np.complex64,
    )
    pressure_hf = np.asarray(
        [
            [0.25 + 0.0j, 0.5 + 0.5j, 0.25 + 0.0j],
            [0.25 + 0.0j, 0.25 + 0.25j, 0.25 + 0.0j],
        ],
        dtype=np.complex64,
    )

    result = _frequency_result_from_channel_basis_entry(
        1000.0,
        {
            "observation_planes": ["horizontal", "vertical"],
            "observation_angles_deg": np.asarray([-45.0, 0.0, 45.0], dtype=np.float32),
            "source_results": [
                {
                    "observation_pressure_complex": pressure_lf,
                    "impedance": 1.0 + 0.5j,
                    "lapack_info": 0,
                    "backend": "test",
                },
                {
                    "observation_pressure_complex": pressure_hf,
                    "impedance": 2.0 - 0.25j,
                    "lapack_info": 0,
                    "backend": "test",
                },
            ],
        },
        channel_names=np.asarray(["LF", "HF"]),
        channel_configs={},
    )

    assert result.channel_names.tolist() == ["LF", "HF"]
    assert result.horizontal_pressure.shape == (2, 3)
    assert result.vertical_pressure.shape == (2, 3)
    np.testing.assert_allclose(result.horizontal_pressure[0], np.conj(pressure_lf[0]))
    np.testing.assert_allclose(result.impedance, [[1.0, 0.5], [2.0, -0.25]])
    assert result.horizontal_spl_db.shape == (3,)
    assert result.horizontal_spl_norm_db[1] == pytest.approx(0.0)

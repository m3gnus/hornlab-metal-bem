from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_metal_bem.boundary_lab import BACKEND_ID, BoundaryLabSolverError, create_backend
from hornlab_metal_bem.boundary_lab import _coerce_symmetry_plane
from hornlab_metal_bem.boundary_lab import _crossover_response
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
    assert result.native_diagnostics["assembly_implementation"] == "test"
    assert result.diagnostics.convergence_info == 0

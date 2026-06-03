from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hornlab_metal_bem.boundary_lab import BACKEND_ID, create_backend
from hornlab_metal_bem.boundary_lab import _frequency_result_from_log_entry
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

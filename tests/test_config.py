"""Unit tests for hornlab_metal_bem.config — pure dataclass tests, no bempp needed."""
from __future__ import annotations

import pytest

from hornlab_metal_bem.backends import (
    AssemblyBackendUnavailable,
    discover_metal_backend,
    resolve_assembly_backend,
)
import hornlab_metal_bem.backends as backends
from hornlab_metal_bem.config import BIEFormulation, ObservationConfig, SolveConfig
from hornlab_metal_bem.sweep import should_route_native_metal


def test_observation_config_custom_points_defaults_none():
    cfg = ObservationConfig()
    assert cfg.custom_points is None


def test_solve_config_frame_override_defaults_none():
    cfg = SolveConfig()
    assert cfg.frame_override is None


def test_solve_config_air_density_default():
    cfg = SolveConfig()
    assert cfg.air_density == 1.2041


def test_solve_config_air_density_custom():
    cfg = SolveConfig(air_density=1.18)
    assert cfg.air_density == 1.18


def test_solve_config_progress_callback_defaults_none():
    cfg = SolveConfig()
    assert cfg.progress_callback is None


def test_solve_config_on_frequency_result_defaults_none():
    cfg = SolveConfig()
    assert cfg.on_frequency_result is None


def test_solve_config_default_native_metal_controls():
    cfg = SolveConfig()
    assert cfg.formulation == BIEFormulation.STANDARD
    assert cfg.complex_k_shift == 0.005
    assert cfg.impedance_sources == {}
    assert cfg.native_symmetry_plane is None
    assert cfg.native_check_open_edges is True
    assert cfg.metal_native_assembly_mode == "corrected"
    assert cfg.return_surface_pressure is False
    assert cfg.metal_native_threads_per_group is None
    assert cfg.metal_native_matrix_threads_per_group is None
    assert cfg.metal_native_rhs_threads_per_group is None
    assert cfg.metal_native_duffy_threads_per_group is None
    assert cfg.metal_native_field_threads_per_group is None
    assert cfg.dense_solve_rcond_warning_threshold == 1e-6
    assert cfg.mesh_elements_per_wavelength_min == 6.0


def test_solve_config_rejects_unknown_frequency_spacing():
    with pytest.raises(ValueError, match="freq_spacing"):
        SolveConfig(freq_spacing="octave")  # type: ignore[arg-type]


def test_solve_config_rejects_unknown_velocity_mode():
    with pytest.raises(ValueError, match="velocity_mode"):
        SolveConfig(velocity_mode="force")  # type: ignore[arg-type]


def test_solve_config_accepts_experimental_complex_k_and_robin():
    cfg = SolveConfig(
        formulation=BIEFormulation.COMPLEX_K,
        complex_k_shift=0.01,
        impedance_sources={8: 0.05 + 0.01j, 9: 0.02 + 0.0j},
    )

    assert cfg.formulation == "complex_k"
    assert cfg.complex_k_shift == 0.01
    assert cfg.impedance_sources[8] == 0.05 + 0.01j
    assert cfg.impedance_sources[9] == 0.02 + 0.0j


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"formulation": "burton_miller"}, "formulation"),
        ({"complex_k_shift": -0.1}, "complex_k_shift"),
        ({"impedance_sources": {-1: 0.05 + 0.0j}}, "impedance_sources"),
        ({"impedance_sources": {8: complex(float("nan"), 0.0)}}, "impedance_sources"),
    ],
)
def test_solve_config_rejects_invalid_experimental_boundary_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SolveConfig(**kwargs)


def test_solve_config_rejects_unknown_native_symmetry_plane():
    with pytest.raises(ValueError, match="native_symmetry_plane"):
        SolveConfig(native_symmetry_plane="zx")  # type: ignore[arg-type]


def test_solve_config_accepts_native_symmetry_planes():
    assert SolveConfig(native_symmetry_plane="yz").native_symmetry_plane == "yz"
    assert SolveConfig(native_symmetry_plane="xz").native_symmetry_plane == "xz"
    assert SolveConfig(native_symmetry_plane="xy").native_symmetry_plane == "xy"
    assert SolveConfig(native_symmetry_plane="yz+xz").native_symmetry_plane == "yz+xz"


def test_solve_config_accepts_native_check_open_edges_override():
    assert SolveConfig(native_check_open_edges=False).native_check_open_edges is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"freq_count": 0}, "freq_count"),
        ({"freq_min_hz": 0.0}, "freq_min_hz"),
        ({"freq_min_hz": -10.0}, "freq_min_hz"),
        ({"freq_min_hz": 1000.0, "freq_max_hz": 500.0}, "freq_max_hz"),
        ({"mesh_scale": 0.0}, "mesh_scale"),
        ({"air_density": 0.0}, "air_density"),
        (
            {"dense_solve_rcond_warning_threshold": -1.0},
            "dense_solve_rcond_warning_threshold",
        ),
        ({"mesh_elements_per_wavelength_min": 0.0}, "mesh_elements_per_wavelength_min"),
    ],
)
def test_solve_config_rejects_degenerate_sweep_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SolveConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"planes": []}, "planes"),
        ({"distance_m": 0.0}, "distance_m"),
        ({"angle_count": 0}, "angle_count"),
        ({"origin": "Mouth"}, "origin"),
        ({"origin": "centre"}, "origin"),
    ],
)
def test_observation_config_rejects_degenerate_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ObservationConfig(**kwargs)


def test_solve_config_mesh_loading_options_default_off():
    cfg = SolveConfig()
    assert cfg.mesh_validate is True
    assert cfg.mesh_merge_tol == 1e-9
    assert cfg.mesh_repair_normals is False


def test_solve_config_rejects_unknown_metal_native_assembly_mode():
    with pytest.raises(ValueError, match="metal_native_assembly_mode"):
        SolveConfig(metal_native_assembly_mode="unknown")  # type: ignore[arg-type]


def test_solve_config_accepts_metal_native_assembly_modes():
    assert SolveConfig().metal_native_assembly_mode == "corrected"
    assert (
        SolveConfig(metal_native_assembly_mode="optimized")
        .metal_native_assembly_mode
        == "optimized"
    )
    assert (
        SolveConfig(metal_native_assembly_mode="reference")
        .metal_native_assembly_mode
        == "reference"
    )
    assert (
        SolveConfig(metal_native_assembly_mode="parity")
        .metal_native_assembly_mode
        == "parity"
    )


def test_solve_config_accepts_surface_pressure_output_flag():
    assert SolveConfig(return_surface_pressure=True).return_surface_pressure is True


def test_solve_config_accepts_native_threadgroup_override():
    assert (
        SolveConfig(metal_native_threads_per_group=64)
        .metal_native_threads_per_group
        == 64
    )


def test_solve_config_rejects_nonpositive_native_threadgroup_override():
    with pytest.raises(ValueError, match="metal_native_threads_per_group"):
        SolveConfig(metal_native_threads_per_group=0)


def test_solve_config_accepts_per_kernel_native_threadgroup_overrides():
    cfg = SolveConfig(
        metal_native_matrix_threads_per_group=32,
        metal_native_rhs_threads_per_group=64,
        metal_native_duffy_threads_per_group=128,
        metal_native_field_threads_per_group=256,
    )

    assert cfg.metal_native_matrix_threads_per_group == 32
    assert cfg.metal_native_rhs_threads_per_group == 64
    assert cfg.metal_native_duffy_threads_per_group == 128
    assert cfg.metal_native_field_threads_per_group == 256


@pytest.mark.parametrize(
    "field_name",
    [
        "metal_native_matrix_threads_per_group",
        "metal_native_rhs_threads_per_group",
        "metal_native_duffy_threads_per_group",
        "metal_native_field_threads_per_group",
    ],
)
def test_solve_config_rejects_nonpositive_per_kernel_native_threadgroup_override(
    field_name,
):
    with pytest.raises(ValueError, match=field_name):
        SolveConfig(**{field_name: 0})


def test_resolve_backend_raises_when_native_unavailable(monkeypatch):
    class Status:
        available = False
        reason = "native unavailable"

    monkeypatch.setattr(backends, "discover_metal_backend", lambda: Status())

    with pytest.raises(AssemblyBackendUnavailable, match="native unavailable"):
        resolve_assembly_backend()


def test_metal_discovery_reports_native_helper(monkeypatch):
    class NativeStatus:
        available = True
        is_apple_silicon = True
        swift_path = "/usr/bin/swift"
        helper_executable_path = None
        helper_assets_present = True
        unavailable_reasons = ()

    monkeypatch.setattr(backends, "discover_native_runtime", lambda config: NativeStatus())

    status = discover_metal_backend()

    assert status.available is True
    assert status.native_executable == "/usr/bin/swift"
    assert status.native_helper_available is True
    assert status.reason is None


def test_metal_discovery_available_with_compiled_helper_and_no_swift(monkeypatch):
    class NativeStatus:
        available = True
        is_apple_silicon = True
        swift_path = None
        helper_executable_path = "/opt/helper/HornlabMetalBemNative"
        helper_assets_present = True
        unavailable_reasons = ()

    monkeypatch.setattr(backends, "discover_native_runtime", lambda config: NativeStatus())

    status = discover_metal_backend()

    assert status.available is True
    assert status.native_executable == "/opt/helper/HornlabMetalBemNative"


def test_resolve_backend_returns_metal_when_native_discovered(monkeypatch):
    class Status:
        available = True
        reason = None

    monkeypatch.setattr(backends, "discover_metal_backend", lambda: Status())

    resolution = resolve_assembly_backend()

    assert resolution.requested_backend == "metal"
    assert resolution.effective_backend == "metal"
    assert resolution.fallback_used is False


def test_explicit_experimental_metal_request_routes_to_native_sweep():
    cfg = SolveConfig()

    assert should_route_native_metal(cfg) is True


def test_solve_config_callbacks_accept_callables():
    calls = []
    cfg = SolveConfig(
        progress_callback=lambda i, n, f: calls.append(("progress", i)),
        on_frequency_result=lambda i, f, log: True,
    )
    cfg.progress_callback(0, 5, 1000.0)
    assert calls == [("progress", 0)]
    assert cfg.on_frequency_result(0, 1000.0, {}) is True

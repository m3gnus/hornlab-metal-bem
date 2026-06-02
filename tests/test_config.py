"""Unit tests for hornlab_solver.config — pure dataclass tests, no bempp needed."""
from __future__ import annotations

import pytest

from hornlab_solver.backends import (
    AssemblyBackendUnavailable,
    discover_metal_backend,
    resolve_assembly_backend,
)
import hornlab_solver.backends as backends
from hornlab_solver.config import BIEFormulation, ObservationConfig, SolveConfig
from hornlab_solver.sweep import should_route_native_metal


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


def test_solve_config_default_backend_stays_opencl_cpu():
    cfg = SolveConfig()
    assert cfg.assembly_backend == "opencl"
    assert cfg.opencl_device == "cpu"
    assert cfg.experimental_metal_backend is False
    assert cfg.metal_backend_fallback == "opencl"
    assert cfg.native_symmetry_plane is None
    assert cfg.metal_native_assembly_mode == "corrected"
    assert cfg.metal_native_threads_per_group is None
    assert cfg.metal_native_matrix_threads_per_group is None
    assert cfg.metal_native_rhs_threads_per_group is None
    assert cfg.metal_native_duffy_threads_per_group is None
    assert cfg.metal_native_field_threads_per_group is None


def test_solve_config_rejects_unknown_backend():
    with pytest.raises(ValueError, match="assembly_backend"):
        SolveConfig(assembly_backend="cuda")  # type: ignore[arg-type]


def test_solve_config_rejects_unknown_metal_fallback():
    with pytest.raises(ValueError, match="metal_backend_fallback"):
        SolveConfig(metal_backend_fallback="numba")  # type: ignore[arg-type]


def test_solve_config_rejects_unknown_native_symmetry_plane():
    with pytest.raises(ValueError, match="native_symmetry_plane"):
        SolveConfig(native_symmetry_plane="xy")  # type: ignore[arg-type]


def test_solve_config_accepts_native_symmetry_planes():
    assert SolveConfig(native_symmetry_plane="yz").native_symmetry_plane == "yz"
    assert SolveConfig(native_symmetry_plane="xz").native_symmetry_plane == "xz"
    assert SolveConfig(native_symmetry_plane="yz+xz").native_symmetry_plane == "yz+xz"


def test_solve_config_rejects_unknown_metal_native_assembly_mode():
    with pytest.raises(ValueError, match="metal_native_assembly_mode"):
        SolveConfig(metal_native_assembly_mode="parity")  # type: ignore[arg-type]


def test_solve_config_accepts_metal_native_assembly_modes():
    assert SolveConfig().metal_native_assembly_mode == "corrected"
    assert (
        SolveConfig(metal_native_assembly_mode="optimized")
        .metal_native_assembly_mode
        == "optimized"
    )


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


def test_auto_backend_resolves_to_opencl():
    resolution = resolve_assembly_backend(SolveConfig(assembly_backend="auto"))
    assert resolution.effective_backend == "opencl"
    assert resolution.fallback_used is False


def test_metal_backend_falls_back_to_opencl_by_default():
    resolution = resolve_assembly_backend(SolveConfig(assembly_backend="metal"))
    assert resolution.requested_backend == "metal"
    assert resolution.effective_backend == "opencl"
    assert resolution.fallback_used is True
    assert "experimental_metal_backend" in str(resolution.reason)


def test_metal_backend_strict_mode_raises_until_packaged():
    cfg = SolveConfig(
        assembly_backend="metal",
        experimental_metal_backend=True,
        metal_backend_fallback="error",
    )
    with pytest.raises(AssemblyBackendUnavailable):
        resolve_assembly_backend(cfg)


def test_metal_discovery_can_report_native_without_julia(monkeypatch):
    class NativeStatus:
        is_apple_silicon = True
        swift_path = "/usr/bin/swift"
        helper_assets_present = True
        unavailable_reasons = ()

    class JuliaStatus:
        is_apple_silicon = True
        julia_path = None
        backend_assets_present = True
        unavailable_reasons = ("Julia executable not found via PATH.",)

    monkeypatch.setattr(backends, "discover_native_runtime", lambda config: NativeStatus())
    monkeypatch.setattr(backends, "discover_runtime", lambda config: JuliaStatus())

    status = discover_metal_backend()

    assert status.available is True
    assert status.native_executable == "/usr/bin/swift"
    assert status.julia_executable is None
    assert status.native_helper_available is True
    assert status.julia_bridge_available is False
    assert "julia:" in str(status.reason)


def test_metal_strict_mode_still_raises_when_native_discovered(monkeypatch):
    class Status:
        available = True
        reason = None

    monkeypatch.setattr(backends, "discover_metal_backend", lambda: Status())
    cfg = SolveConfig(
        assembly_backend="metal",
        experimental_metal_backend=True,
        metal_backend_fallback="error",
    )

    with pytest.raises(AssemblyBackendUnavailable, match="not wired"):
        resolve_assembly_backend(cfg)


def test_explicit_experimental_metal_request_routes_to_native_sweep():
    cfg = SolveConfig(
        assembly_backend="metal",
        experimental_metal_backend=True,
    )

    assert should_route_native_metal(cfg) is True


def test_native_symmetry_requires_native_metal_route():
    cfg = SolveConfig(native_symmetry_plane="yz")

    with pytest.raises(AssemblyBackendUnavailable, match="native_symmetry_plane"):
        should_route_native_metal(cfg)


def test_native_metal_sweep_rejects_unsupported_formulation_in_strict_mode():
    cfg = SolveConfig(
        assembly_backend="metal",
        experimental_metal_backend=True,
        metal_backend_fallback="error",
        formulation=BIEFormulation.BURTON_MILLER,
    )

    with pytest.raises(AssemblyBackendUnavailable, match="STANDARD"):
        should_route_native_metal(cfg)


def test_solve_config_callbacks_accept_callables():
    calls = []
    cfg = SolveConfig(
        progress_callback=lambda i, n, f: calls.append(("progress", i)),
        on_frequency_result=lambda i, f, log: True,
    )
    cfg.progress_callback(0, 5, 1000.0)
    assert calls == [("progress", 0)]
    assert cfg.on_frequency_result(0, 1000.0, {}) is True

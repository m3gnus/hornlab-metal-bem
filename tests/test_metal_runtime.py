"""Pure tests for the non-executing Metal runtime discovery scaffold."""
from __future__ import annotations

from pathlib import Path

import pytest

from hornlab_solver.metal import (
    MetalRuntimeConfig,
    assert_runtime_available,
    discover_runtime,
)
from hornlab_solver.metal import runtime


def test_discovery_reports_missing_packaged_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("HORNLAB_SOLVER_JULIA", "/opt/julia/bin/julia")

    status = discover_runtime(MetalRuntimeConfig(backend_dir=tmp_path))

    assert status.available is False
    assert status.is_macos is True
    assert status.is_apple_silicon is True
    assert status.julia_path == "/opt/julia/bin/julia"
    assert status.julia_source == "HORNLAB_SOLVER_JULIA"
    assert status.backend_entrypoint == tmp_path / "HornlabSolverMetal.jl"
    assert status.backend_project == tmp_path / "Project.toml"
    assert status.backend_assets_present is False
    assert status.smoke_test_ran is False
    assert status.smoke_test_ok is False
    assert any("Packaged Julia/Metal backend assets" in r for r in status.reasons)


def test_discovery_finds_julia_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("HORNLAB_SOLVER_JULIA", raising=False)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/local/bin/julia")
    monkeypatch.setattr(runtime.Path, "is_file", lambda self: True)

    status = discover_runtime(MetalRuntimeConfig(backend_dir=tmp_path))

    assert status.available is True
    assert status.julia_path == "/usr/local/bin/julia"
    assert status.julia_source == "PATH"
    assert status.backend_assets_present is True
    assert status.smoke_test_ran is False
    assert status.reasons == ()


def test_discovery_accepts_explicit_julia_path(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("HORNLAB_SOLVER_JULIA", "/ignored/julia")
    monkeypatch.setattr(runtime.Path, "is_file", lambda self: True)

    status = discover_runtime(
        MetalRuntimeConfig(
            julia_executable="/explicit/julia",
            backend_dir=tmp_path,
        )
    )

    assert status.available is True
    assert status.julia_path == "/explicit/julia"
    assert status.julia_source == "explicit"


def test_discovery_reports_non_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("HORNLAB_SOLVER_JULIA", raising=False)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)

    status = discover_runtime(MetalRuntimeConfig(backend_dir=tmp_path))

    assert status.available is False
    assert status.is_macos is False
    assert status.is_apple_silicon is False
    assert "Metal backend requires macOS." in status.reasons
    assert any("Julia executable not found" in r for r in status.reasons)


def test_assert_runtime_available_raises_with_reasons(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("HORNLAB_SOLVER_JULIA", raising=False)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="Apple Silicon macOS"):
        assert_runtime_available(MetalRuntimeConfig(backend_dir=tmp_path))


def test_discovery_can_run_packaged_smoke_test(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/local/bin/julia")
    monkeypatch.setattr(runtime.Path, "is_file", lambda self: True)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return runtime.subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    status = discover_runtime(
        MetalRuntimeConfig(backend_dir=tmp_path),
        run_smoke_test=True,
    )

    assert status.available is True
    assert status.smoke_test_ran is True
    assert status.smoke_test_ok is True
    assert status.smoke_test_error is None
    assert calls[0][-1] == "--smoke"


def test_discovery_reports_failed_packaged_smoke_test(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/usr/local/bin/julia")
    monkeypatch.setattr(runtime.Path, "is_file", lambda self: True)

    def fake_run(command, **kwargs):
        return runtime.subprocess.CompletedProcess(
            command,
            1,
            "",
            "Metal device unavailable\n",
        )

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    status = discover_runtime(
        MetalRuntimeConfig(backend_dir=tmp_path),
        run_smoke_test=True,
    )

    assert status.available is False
    assert status.smoke_test_ran is True
    assert status.smoke_test_ok is False
    assert status.smoke_test_error == "Metal device unavailable"
    assert any("smoke test failed" in reason for reason in status.reasons)


def test_config_defaults_to_packaged_metal_directory():
    config = MetalRuntimeConfig()

    assert config.resolved_backend_dir == Path(runtime.__file__).resolve().parent

"""Runtime discovery for the experimental Julia/Metal validation bridge.

Fast discovery does not launch Julia or touch temporary run directories. Callers
that request ``run_smoke_test=True`` execute the packaged Julia entrypoint only
to validate that Metal.jl can initialize and run a tiny kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import subprocess


_DEFAULT_JULIA_ENV_VAR = "HORNLAB_SOLVER_JULIA"
_DEFAULT_BACKEND_ENTRYPOINT = "HornlabSolverMetal.jl"
_DEFAULT_BACKEND_PROJECT = "Project.toml"
_DEFAULT_SMOKE_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class MetalRuntimeConfig:
    """Discovery inputs for the experimental Julia/Metal validation bridge."""

    julia_executable: str | None = None
    julia_env_var: str = _DEFAULT_JULIA_ENV_VAR
    backend_dir: Path | None = None
    backend_entrypoint: str = _DEFAULT_BACKEND_ENTRYPOINT
    backend_project: str = _DEFAULT_BACKEND_PROJECT
    smoke_timeout_s: float = _DEFAULT_SMOKE_TIMEOUT_S

    @property
    def resolved_backend_dir(self) -> Path:
        """Return the package directory expected to contain backend assets."""
        if self.backend_dir is not None:
            return Path(self.backend_dir)
        return Path(__file__).resolve().parent


@dataclass(frozen=True)
class MetalRuntimeStatus:
    """Result of Metal runtime discovery."""

    available: bool
    platform_system: str
    platform_machine: str
    is_macos: bool
    is_apple_silicon: bool
    julia_path: str | None
    julia_source: str | None
    backend_dir: Path
    backend_entrypoint: Path
    backend_project: Path
    backend_assets_present: bool
    smoke_test_ran: bool
    smoke_test_ok: bool
    smoke_test_error: str | None
    reasons: tuple[str, ...]

    @property
    def unavailable_reasons(self) -> tuple[str, ...]:
        """Human-readable blockers when ``available`` is false."""
        return self.reasons


def _find_julia(config: MetalRuntimeConfig) -> tuple[str | None, str | None]:
    if config.julia_executable:
        return config.julia_executable, "explicit"

    env_path = os.environ.get(config.julia_env_var)
    if env_path:
        return env_path, config.julia_env_var

    path_julia = shutil.which("julia")
    if path_julia:
        return path_julia, "PATH"

    return None, None


def discover_runtime(
    config: MetalRuntimeConfig | None = None,
    *,
    run_smoke_test: bool = False,
) -> MetalRuntimeStatus:
    """Inspect host/runtime prerequisites without executing the backend."""
    if config is None:
        config = MetalRuntimeConfig()

    system = platform.system()
    machine = platform.machine()
    normalized_machine = machine.lower()
    is_macos = system == "Darwin"
    is_apple_silicon = is_macos and normalized_machine in {"arm64", "aarch64"}

    julia_path, julia_source = _find_julia(config)
    backend_dir = config.resolved_backend_dir
    backend_entrypoint = backend_dir / config.backend_entrypoint
    backend_project = backend_dir / config.backend_project
    backend_assets_present = backend_entrypoint.is_file() and backend_project.is_file()

    reasons: list[str] = []
    if not is_macos:
        reasons.append("Metal backend requires macOS.")
    elif not is_apple_silicon:
        reasons.append("Metal backend requires Apple Silicon macOS.")

    if julia_path is None:
        reasons.append(
            f"Julia executable not found via {config.julia_env_var} or PATH."
        )

    if not backend_assets_present:
        reasons.append(
            "Packaged Julia/Metal backend assets are not installed under "
            f"{backend_dir}."
        )

    smoke_test_ran = False
    smoke_test_ok = False
    smoke_test_error: str | None = None
    if run_smoke_test and not reasons and julia_path is not None:
        smoke_test_ran = True
        smoke_test_ok, smoke_test_error = _run_backend_smoke_test(
            julia_path,
            backend_dir,
            backend_entrypoint,
            timeout_s=config.smoke_timeout_s,
        )
        if not smoke_test_ok:
            reasons.append(
                "Packaged Julia/Metal backend smoke test failed"
                + (f": {smoke_test_error}" if smoke_test_error else ".")
            )

    return MetalRuntimeStatus(
        available=not reasons,
        platform_system=system,
        platform_machine=machine,
        is_macos=is_macos,
        is_apple_silicon=is_apple_silicon,
        julia_path=julia_path,
        julia_source=julia_source,
        backend_dir=backend_dir,
        backend_entrypoint=backend_entrypoint,
        backend_project=backend_project,
        backend_assets_present=backend_assets_present,
        smoke_test_ran=smoke_test_ran,
        smoke_test_ok=smoke_test_ok,
        smoke_test_error=smoke_test_error,
        reasons=tuple(reasons),
    )


def assert_runtime_available(
    config: MetalRuntimeConfig | None = None,
    *,
    run_smoke_test: bool = False,
) -> MetalRuntimeStatus:
    """Return discovery status or raise ``RuntimeError`` with clear blockers."""
    status = discover_runtime(config, run_smoke_test=run_smoke_test)
    if status.available:
        return status

    raise RuntimeError(
        "Julia/Metal runtime is unavailable: "
        + "; ".join(status.unavailable_reasons)
    )


def _run_backend_smoke_test(
    julia_path: str,
    backend_dir: Path,
    backend_entrypoint: Path,
    *,
    timeout_s: float,
) -> tuple[bool, str | None]:
    command = [
        julia_path,
        f"--project={backend_dir}",
        str(backend_entrypoint),
        "--smoke",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=backend_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)

    if result.returncode == 0:
        return True, None

    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    message = stderr or stdout or f"Julia exited with status {result.returncode}"
    return False, message.splitlines()[-1]

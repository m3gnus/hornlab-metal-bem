"""Assembly backend discovery and production-safe resolution."""
from __future__ import annotations

from dataclasses import dataclass

from .config import SolveConfig
from .metal.native import MetalNativeRuntimeConfig, discover_native_runtime

BEMPP_BACKENDS = frozenset({"opencl", "numba"})


class AssemblyBackendUnavailable(RuntimeError):
    """Raised when a requested experimental backend cannot be used."""


@dataclass(frozen=True)
class MetalBackendStatus:
    """Runtime status for the optional native Metal helper."""

    supported_platform: bool
    native_executable: str | None
    native_helper_available: bool
    reason: str | None

    @property
    def available(self) -> bool:
        return self.supported_platform and self.native_helper_available

    @property
    def packaged_backend_available(self) -> bool:
        """Backward-compatible aggregate asset/runtime availability."""
        return self.native_helper_available


@dataclass(frozen=True)
class AssemblyBackendResolution:
    """Effective backend used by the current production solver path."""

    requested_backend: str
    effective_backend: str
    fallback_used: bool
    reason: str | None = None
    metal_status: MetalBackendStatus | None = None


def discover_metal_backend(
    *,
    native_executable: str | None = None,
) -> MetalBackendStatus:
    """Discover prerequisites for experimental packaged Metal helpers.

    Production routing remains conservative: ``resolve_assembly_backend`` still
    maps every ``metal`` request to the Bempp/OpenCL fallback or raises in
    strict mode.
    """

    native_status = discover_native_runtime(
        MetalNativeRuntimeConfig(swift_executable=native_executable)
    )
    supported_platform = native_status.is_apple_silicon
    native = native_status.swift_path
    native_available = (
        native_status.is_apple_silicon
        and native_status.swift_path is not None
        and native_status.helper_assets_present
    )

    reasons: list[str] = []
    if native_status.unavailable_reasons:
        reasons.append("native: " + "; ".join(native_status.unavailable_reasons))
    reason = "; ".join(reasons) if reasons else None

    return MetalBackendStatus(
        supported_platform=supported_platform,
        native_executable=native,
        native_helper_available=native_available,
        reason=reason,
    )


def resolve_assembly_backend(config: SolveConfig) -> AssemblyBackendResolution:
    """Resolve ``SolveConfig.assembly_backend`` to a current Bempp backend.

    ``metal`` is accepted as an experimental, discoverable request, but it must
    not reach Bempp's ``device_interface`` until a real Metal adapter exists.
    """

    requested = config.assembly_backend
    if requested == "auto":
        return AssemblyBackendResolution(
            requested_backend=requested,
            effective_backend="opencl",
            fallback_used=False,
            reason="auto selects the production OpenCL Bempp backend",
        )

    if requested in BEMPP_BACKENDS:
        return AssemblyBackendResolution(
            requested_backend=requested,
            effective_backend=requested,
            fallback_used=False,
        )

    if requested != "metal":
        raise ValueError(
            "assembly_backend must be one of: auto, opencl, numba, metal"
        )

    status = discover_metal_backend()
    reason = status.reason
    if not config.experimental_metal_backend:
        reason = "Metal backend requested without experimental_metal_backend=True."
    elif status.available:
        # Future promotion point: return effective_backend="metal" once the
        # packaged adapter owns assembly/field evaluation.
        reason = "Metal backend is discovered but not wired into production."

    if config.metal_backend_fallback == "error":
        raise AssemblyBackendUnavailable(reason or "Metal backend is unavailable.")

    return AssemblyBackendResolution(
        requested_backend=requested,
        effective_backend="opencl",
        fallback_used=True,
        reason=reason,
        metal_status=status,
    )

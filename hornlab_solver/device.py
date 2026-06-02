from __future__ import annotations

from functools import lru_cache


class OpenCLError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def configure_opencl(device_type: str = "cpu") -> str:
    """Configure bempp-cl to use a concrete OpenCL device type."""
    normalized = str(device_type or "cpu").strip().lower()
    if normalized not in {"cpu", "gpu"}:
        raise OpenCLError("opencl_device must be 'cpu' or 'gpu'")

    try:
        import bempp_cl.api as bempp_api
        import bempp_cl.core.opencl_kernels as opencl_kernels
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise OpenCLError(f"bempp-cl OpenCL runtime is unavailable: {exc}") from exc

    try:
        setattr(bempp_api, "BOUNDARY_OPERATOR_DEVICE_TYPE", normalized)
        setattr(bempp_api, "POTENTIAL_OPERATOR_DEVICE_TYPE", normalized)

        if normalized == "cpu":
            device = opencl_kernels.default_cpu_device()
            opencl_kernels.default_cpu_context()
        else:
            device = opencl_kernels.default_gpu_device()
            opencl_kernels.default_gpu_context()
            # bempp-cl dense singular assembly still needs a CPU context.
            opencl_kernels.default_cpu_device()
            opencl_kernels.default_cpu_context()
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise OpenCLError(
            f"OpenCL {normalized} device could not be initialized. "
            "Use the HornLab OpenCL CPU Python runtime or install a CPU OpenCL driver."
        ) from exc

    return str(getattr(device, "name", None) or f"OpenCL {normalized.upper()}")

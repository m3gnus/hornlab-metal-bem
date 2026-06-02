from __future__ import annotations

import hornlab_solver as _compat
from hornlab_solver import *  # noqa: F403
from hornlab_solver import SolveConfig as SolveConfig
from hornlab_solver import solve as _compat_solve
from hornlab_solver import solve_frequencies as _compat_solve_frequencies

__all__ = [
    *[name for name in _compat.__all__ if name not in {"solve", "solve_frequencies"}],
    "native_config",
    "solve",
    "solve_frequencies",
]


def native_config(**overrides):
    values = {
        "assembly_backend": "metal",
        "experimental_metal_backend": True,
        "metal_backend_fallback": "error",
        "metal_native_assembly_mode": "corrected",
    }
    values.update(overrides)
    return SolveConfig(**values)


def solve(mesh, config: SolveConfig | None = None):
    return _compat_solve(mesh, native_config() if config is None else config)


def solve_frequencies(mesh, frequencies_hz, config: SolveConfig | None = None):
    return _compat_solve_frequencies(
        mesh,
        frequencies_hz,
        native_config() if config is None else config,
    )

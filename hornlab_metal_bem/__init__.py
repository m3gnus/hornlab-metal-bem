"""Native Apple Metal acoustic BEM solver for HornLab."""
from __future__ import annotations

import numpy as np

from .config import ObservationConfig, SolveConfig, VelocityMode
from .mesh import LoadedMesh, MeshError, load_mesh
from .observation import ObservationFrame, infer_frame
from .result import MeshInfo, SolveResult

__all__ = [
    "native_config",
    "solve",
    "solve_frequencies",
    "load_mesh",
    "SolveConfig",
    "SolveResult",
    "ObservationConfig",
    "VelocityMode",
    "LoadedMesh",
    "MeshInfo",
    "MeshError",
    "ObservationFrame",
]

def native_config(**overrides) -> SolveConfig:
    """Return the supported strict native Metal solve configuration.

    Keyword overrides are passed to ``SolveConfig``. The native Metal path
    supports standard Neumann solves only; unsupported general-solver options
    are intentionally not exported from this namespace.
    """
    values = {
        "metal_native_assembly_mode": "corrected",
    }
    values.update(overrides)
    return SolveConfig(**values)


def _resolve_mesh(mesh, config: SolveConfig) -> LoadedMesh:
    if isinstance(mesh, LoadedMesh):
        return mesh
    return load_mesh(
        mesh,
        scale=config.mesh_scale,
        validate=config.mesh_validate,
        merge_tol=config.mesh_merge_tol,
        repair_normals=config.mesh_repair_normals,
        native_symmetry_plane=config.native_symmetry_plane,
    )


def _resolve_frame(loaded: LoadedMesh, config: SolveConfig) -> ObservationFrame:
    if config.frame_override is not None:
        return config.frame_override

    return infer_frame(
        loaded.grid,
        loaded.physical_tags,
        source_tag=min(config.velocity_sources.keys(), default=2),
        origin_at=config.observation.origin,
        symmetry_plane=config.native_symmetry_plane,
    )


def solve(mesh, config: SolveConfig | None = None) -> SolveResult:
    """Run a native Metal BEM frequency sweep for a mesh."""
    if config is None:
        config = native_config()

    from .sweep import (
        _build_frequency_grid,
        run_sweep_native_metal,
        should_route_native_metal,
    )

    loaded = _resolve_mesh(mesh, config)
    frame = _resolve_frame(loaded, config)
    frequencies = _build_frequency_grid(config)

    should_route_native_metal(config)
    return run_sweep_native_metal(loaded, frequencies, frame, config)


def solve_frequencies(
    mesh,
    frequencies_hz: list[float] | np.ndarray,
    config: SolveConfig | None = None,
) -> SolveResult:
    """Run a native Metal BEM solve at caller-ordered frequencies."""
    if config is None:
        config = native_config()

    from .sweep import run_sweep_native_metal, should_route_native_metal

    loaded = _resolve_mesh(mesh, config)
    frame = _resolve_frame(loaded, config)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)

    should_route_native_metal(config)
    return run_sweep_native_metal(loaded, freqs, frame, config)

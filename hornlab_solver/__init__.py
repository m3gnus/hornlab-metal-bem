"""hornlab-solver — canonical BEM acoustic solver for HornLab.

Public API:
    solve(mesh, config)            → SolveResult (full frequency sweep)
    solve_frequencies(mesh, freqs) → SolveResult (caller-ordered frequencies)
    load_mesh(path, scale)         → LoadedMesh
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from .backends import (
    AssemblyBackendResolution,
    AssemblyBackendUnavailable,
    MetalBackendStatus,
    discover_metal_backend,
    resolve_assembly_backend,
)
from .config import (
    BIEFormulation,
    LinearSolver,
    ObservationConfig,
    SolveConfig,
    VelocityMode,
)
from .device import OpenCLError, configure_opencl
from .metal import DenseBieSystem, MetalBemBackend, MetalBemContext
from .mesh import (
    LoadedMesh,
    MeshError,
    PureFunctionSpace,
    PureGrid,
    load_mesh,
    make_pure_function_spaces,
    to_bempp_loaded_mesh,
)
from .observation import ObservationFrame, infer_frame
from .result import MeshInfo, SolveResult

__all__ = [
    "solve",
    "solve_frequencies",
    "load_mesh",
    "SolveConfig",
    "SolveResult",
    "ObservationConfig",
    "BIEFormulation",
    "LinearSolver",
    "VelocityMode",
    "LoadedMesh",
    "PureGrid",
    "PureFunctionSpace",
    "MeshInfo",
    "MeshError",
    "make_pure_function_spaces",
    "to_bempp_loaded_mesh",
    "ObservationFrame",
    "OpenCLError",
    "configure_opencl",
    "AssemblyBackendResolution",
    "AssemblyBackendUnavailable",
    "MetalBackendStatus",
    "DenseBieSystem",
    "MetalBemBackend",
    "MetalBemContext",
    "discover_metal_backend",
    "resolve_assembly_backend",
]

logger = logging.getLogger(__name__)


def _detect_worker_count() -> int:
    """Auto-detect physical core count (not hyperthreads)."""
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:
        import multiprocessing
        count = multiprocessing.cpu_count() or 1
    return max(1, count)


def _resolve_mesh(mesh, config: SolveConfig) -> LoadedMesh:
    """Accept str, Path, or LoadedMesh."""
    if isinstance(mesh, LoadedMesh):
        return mesh
    grid_backend = "pure" if should_load_pure_grid(config) else "bempp"
    return load_mesh(mesh, scale=config.mesh_scale, grid_backend=grid_backend)


def should_load_pure_grid(config: SolveConfig) -> bool:
    return (
        config.assembly_backend == "metal"
        and config.experimental_metal_backend
    )


def _resolve_frame(loaded: LoadedMesh, config: SolveConfig) -> ObservationFrame:
    """Resolve observation frame: use override, skip for custom_points, or infer."""
    if config.frame_override is not None:
        return config.frame_override

    if config.observation.custom_points is not None:
        # Custom observation points don't need a frame for point construction,
        # but we still build a minimal frame for metadata. Use infer_frame()
        # which works fine for standard geometries. For enclosed geometries
        # where infer_frame might get the axis wrong, callers should set
        # frame_override explicitly.
        pass

    return infer_frame(
        loaded.grid,
        loaded.physical_tags,
        source_tag=min(config.velocity_sources.keys(), default=2),
        origin_at=config.observation.origin,
        symmetry_plane=config.native_symmetry_plane,
    )


def solve(
    mesh,
    config: SolveConfig | None = None,
) -> SolveResult:
    """Run a BEM acoustic solve across a frequency sweep.

    Parameters
    ----------
    mesh : str | Path | LoadedMesh
        Path to a .msh file (with ABEC-convention physical groups),
        or a pre-loaded LoadedMesh from load_mesh().

    config : SolveConfig, optional
        Full solve specification. Defaults to SolveConfig() if None.

    Returns
    -------
    SolveResult with complex pressure, SPL, impedance, and metadata.
    """
    if config is None:
        config = SolveConfig()

    from .sweep import (
        _build_frequency_grid,
        run_sweep_native_metal,
        run_sweep_parallel,
        run_sweep_serial,
        should_route_native_metal,
    )

    loaded = _resolve_mesh(mesh, config)
    frame = _resolve_frame(loaded, config)
    frequencies = _build_frequency_grid(config)

    workers = config.workers
    if workers == 0:
        workers = _detect_worker_count()

    if should_load_pure_grid(config):
        if workers not in (0, 1):
            logger.warning(
                "Native Metal sweeps use one resident validation session; "
                "ignoring workers=%s.",
                workers,
            )
        return run_sweep_native_metal(loaded, frequencies, frame, config)
    if should_route_native_metal(config):
        return run_sweep_native_metal(loaded, frequencies, frame, config)

    if workers <= 1:
        return run_sweep_serial(loaded, frequencies, frame, config)
    else:
        return run_sweep_parallel(loaded, frequencies, frame, config, workers)


def solve_frequencies(
    mesh,
    frequencies_hz: list[float] | np.ndarray,
    config: SolveConfig | None = None,
) -> SolveResult:
    """Solve at specific frequencies in caller-specified order.

    Unlike solve(), this does not generate a frequency grid from config.
    The frequencies are solved in the exact order given -- useful for
    priority-ordered solves with external early-stopping logic.
    """
    if config is None:
        config = SolveConfig()

    from .sweep import run_sweep_native_metal, run_sweep_serial, should_route_native_metal

    loaded = _resolve_mesh(mesh, config)
    frame = _resolve_frame(loaded, config)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)

    if should_load_pure_grid(config) or should_route_native_metal(config):
        return run_sweep_native_metal(loaded, freqs, frame, config)

    # Always serial for caller-ordered frequencies (order matters)
    return run_sweep_serial(loaded, freqs, frame, config)

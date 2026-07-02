"""Native Apple Metal acoustic BEM solver for HornLab."""
from __future__ import annotations

from dataclasses import replace as _replace

import numpy as np

from .config import ObservationConfig, SolveConfig, VelocityMode
from .mesh import LoadedMesh, MeshError, load_mesh
from .observation import ObservationFrame, infer_frame
from .result import MeshInfo, SolveResult

__all__ = [
    "native_config",
    "solve",
    "solve_frequencies",
    "solve_multi_source",
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


def solve_multi_source(
    mesh,
    sources: list[dict[int, complex]],
    config: SolveConfig | None = None,
    frequencies_hz: list[float] | np.ndarray | None = None,
) -> list[SolveResult]:
    """Solve several velocity sources on one mesh with shared factorizations.

    Each entry of ``sources`` is a ``velocity_sources`` dict (tag -> normal
    velocity/acceleration weight; zero-weight tags are legal and record that
    tag's average surface pressure without driving it). The native helper
    assembles and factors each frequency's operator ONCE and back-substitutes
    one right-hand side per source, so N sources cost roughly one solve plus
    N-1 cheap RHS/field passes instead of N full solves.

    Returns one ``SolveResult`` per source, matching a sequential ``solve()``
    with ``config.velocity_sources`` replaced by that source dict to float32
    tolerance. All sources share one observation frame: ``frame_override`` if
    set, else the frame inferred from the FIRST source's lowest tag — pass an
    explicit ``frame_override`` when the per-source frames would differ.
    ``config.velocity_sources`` is ignored. The shared assembly/factorization
    time is attributed to the first result's timings; later results carry
    only their own field-evaluation time.
    """
    if config is None:
        config = native_config()
    if not sources:
        raise ValueError("sources must contain at least one velocity dict")

    from .sweep import (
        _build_frequency_grid,
        run_sweep_native_metal_multi_source,
        should_route_native_metal,
    )

    frame_config = _replace(config, velocity_sources=dict(sources[0]))
    loaded = _resolve_mesh(mesh, config)
    frame = _resolve_frame(loaded, frame_config)
    freqs = (
        _build_frequency_grid(config)
        if frequencies_hz is None
        else np.asarray(frequencies_hz, dtype=np.float64)
    )

    should_route_native_metal(config)
    return run_sweep_native_metal_multi_source(loaded, freqs, frame, config, sources)

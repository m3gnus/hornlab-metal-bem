"""Native Apple Metal acoustic BEM solver for HornLab."""
from __future__ import annotations

from dataclasses import replace as _replace

import numpy as np

from .config import (
    AnnularProfile,
    AxialProfile,
    CallableProfile,
    NormalProfile,
    ObservationConfig,
    PerFaceProfile,
    SolveConfig,
    SourceMotion,
    SourceProfile,
    TaperProfile,
    VelocityMode,
)
from .circsym import MeridianMesh
from .mesh import LoadedMesh, MeshError, load_mesh
from .observation import ObservationFrame, infer_frame
from .result import MeshInfo, SolveResult

__all__ = [
    "native_config",
    "solve",
    "solve_frequencies",
    "solve_circsym",
    "solve_circsym_frequencies",
    "solve_multi_source",
    "load_mesh",
    "MeridianMesh",
    "SolveConfig",
    "SolveResult",
    "ObservationConfig",
    "VelocityMode",
    "SourceMotion",
    "SourceProfile",
    "NormalProfile",
    "AxialProfile",
    "TaperProfile",
    "AnnularProfile",
    "PerFaceProfile",
    "CallableProfile",
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
        aperture_tag=config.aperture_tag,
    )


def _declared_loaded_mesh_aperture_tag(loaded: LoadedMesh) -> int | None:
    """Return coupled-IB metadata carried by a loaded or legacy mesh object."""
    carried = getattr(loaded, "coupled_ib_aperture_tag", None)
    info = getattr(loaded, "info", None)
    physical_groups = getattr(info, "physical_groups", {})
    named = [
        int(tag)
        for tag, name in physical_groups.items()
        if str(name).strip().lower() == "mouth_aperture"
    ]
    if len(named) > 1:
        raise MeshError(
            "LoadedMesh declares more than one mouth_aperture physical group: "
            f"{sorted(named)}"
        )
    named_tag = named[0] if named else None
    if carried is not None and named_tag is not None and int(carried) != named_tag:
        raise MeshError(
            "LoadedMesh coupled-IB metadata conflicts with its mouth_aperture "
            f"physical group ({int(carried)} != {named_tag})"
        )
    detected = int(carried) if carried is not None else named_tag
    if detected is None:
        return None
    present = {int(tag) for tag in np.unique(loaded.physical_tags)}
    if detected not in present:
        raise MeshError(
            f"LoadedMesh declares aperture tag {detected}, but no triangle uses it"
        )
    return detected


def _config_for_loaded_mesh(loaded: LoadedMesh, config: SolveConfig) -> SolveConfig:
    """Resolve mesh-declared coupled-IB semantics into the solve config."""
    detected = _declared_loaded_mesh_aperture_tag(loaded)
    explicit = config.aperture_tag
    if detected is None:
        return config
    if explicit is not None and int(explicit) != detected:
        raise MeshError(
            f"SolveConfig.aperture_tag {int(explicit)} conflicts with the "
            f"LoadedMesh mouth_aperture tag {detected}"
        )
    if explicit is not None:
        return config
    return _replace(config, aperture_tag=detected)


def _project_to_native_symmetry_plane(point: np.ndarray, plane: str | None) -> np.ndarray:
    projected = np.array(point, dtype=np.float64, copy=True)
    if plane == "yz":
        projected[0] = 0.0
    elif plane == "xz":
        projected[1] = 0.0
    elif plane == "yz+xz":
        projected[0] = 0.0
        projected[1] = 0.0
    elif plane == "xy":
        projected[2] = 0.0
    return projected


def _coupled_ib_aperture_origin(loaded: LoadedMesh, aperture_tag: int) -> np.ndarray | None:
    tags = np.asarray(loaded.physical_tags, dtype=np.int32)
    aperture_mask = tags == int(aperture_tag)
    if not np.any(aperture_mask):
        return None

    vertices = np.asarray(loaded.grid.vertices.T, dtype=np.float64)
    elements = np.asarray(loaded.grid.elements.T, dtype=np.int64)
    aperture_elements = elements[aperture_mask]
    if aperture_elements.size == 0:
        return None

    p0 = vertices[aperture_elements[:, 0]]
    p1 = vertices[aperture_elements[:, 1]]
    p2 = vertices[aperture_elements[:, 2]]
    raw_normals = np.cross(p1 - p0, p2 - p0)
    areas = 0.5 * np.linalg.norm(raw_normals, axis=1)
    valid = areas > 1.0e-15
    if not np.any(valid):
        return None

    centroids = (p0[valid] + p1[valid] + p2[valid]) / 3.0
    return np.average(centroids, weights=areas[valid], axis=0)


def _resolve_frame(loaded: LoadedMesh, config: SolveConfig) -> ObservationFrame:
    if config.frame_override is not None:
        return config.frame_override

    frame = infer_frame(
        loaded.grid,
        loaded.physical_tags,
        source_tag=min(config.velocity_sources.keys(), default=2),
        origin_at=config.observation.origin,
        symmetry_plane=config.native_symmetry_plane,
    )
    if config.aperture_tag is not None and config.observation.origin == "mouth":
        aperture_origin = _coupled_ib_aperture_origin(loaded, config.aperture_tag)
        if aperture_origin is not None:
            aperture_origin = _project_to_native_symmetry_plane(
                aperture_origin,
                config.native_symmetry_plane,
            )
            frame = _replace(
                frame,
                origin=aperture_origin,
                mouth_center=aperture_origin,
            )
    return frame


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
    config = _config_for_loaded_mesh(loaded, config)
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
    config = _config_for_loaded_mesh(loaded, config)
    frame = _resolve_frame(loaded, config)
    freqs = np.asarray(frequencies_hz, dtype=np.float64)

    should_route_native_metal(config)
    return run_sweep_native_metal(loaded, freqs, frame, config)


def solve_circsym(
    meridian: MeridianMesh,
    config: SolveConfig | None = None,
) -> SolveResult:
    """Run a pure-Python axisymmetric m=0 BEM frequency sweep."""
    from .circsym import solve_circsym as _solve_circsym

    return _solve_circsym(meridian, config)


def solve_circsym_frequencies(
    meridian: MeridianMesh,
    frequencies_hz: list[float] | np.ndarray,
    config: SolveConfig | None = None,
) -> SolveResult:
    """Run a pure-Python axisymmetric m=0 BEM solve at caller frequencies."""
    from .circsym import solve_circsym_frequencies as _solve_circsym_frequencies

    return _solve_circsym_frequencies(meridian, frequencies_hz, config)


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
    only their own field-evaluation time. If ``config.on_frequency_result`` is
    set, callbacks stream one combined entry per frequency with per-source log
    entries in ``source_results``.
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

    loaded = _resolve_mesh(mesh, config)
    config = _config_for_loaded_mesh(loaded, config)
    frame_config = _replace(config, velocity_sources=dict(sources[0]))
    frame = _resolve_frame(loaded, frame_config)
    freqs = (
        _build_frequency_grid(config)
        if frequencies_hz is None
        else np.asarray(frequencies_hz, dtype=np.float64)
    )

    should_route_native_metal(config)
    return run_sweep_native_metal_multi_source(loaded, freqs, frame, config, sources)

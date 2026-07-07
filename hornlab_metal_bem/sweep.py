"""Native Metal frequency sweep execution."""
from __future__ import annotations

from dataclasses import replace
import logging
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from ._constants import REFERENCE_PRESSURE, SPEED_OF_SOUND
from .backends import AssemblyBackendUnavailable
from .bie import (
    _build_driver_neumann_coeffs,
    _build_source_face_scale,
    _compute_impedance,
    compute_surface_pressure_avg,
)
from .config import (
    BIEFormulation,
    NATIVE_SYMMETRY_PLANES,
    SolveConfig,
)
from .mesh import LoadedMesh, make_pure_function_spaces
from .observation import ObservationFrame, build_observation_points
from .result import SolveResult

logger = logging.getLogger(__name__)

_SMOKE_VALIDATED_HELPERS: dict[tuple[str, float], bool] = {}


def _discover_runtime_smoke_cached():
    """Native runtime discovery that skips the smoke subprocess when this
    process already smoke-validated the same helper binary (path + mtime)."""
    from .metal.native import discover_native_runtime

    runtime = discover_native_runtime(run_smoke_test=False)
    if not runtime.available or runtime.helper_executable_path is None:
        return discover_native_runtime(run_smoke_test=True)

    try:
        key = (
            str(runtime.helper_executable_path),
            runtime.helper_executable_path.stat().st_mtime,
        )
    except OSError:
        return discover_native_runtime(run_smoke_test=True)

    if key in _SMOKE_VALIDATED_HELPERS:
        return runtime

    runtime = discover_native_runtime(run_smoke_test=True)
    if runtime.available:
        _SMOKE_VALIDATED_HELPERS[key] = True
    return runtime


def _build_frequency_grid(config: SolveConfig) -> NDArray[np.float64]:
    if config.freq_spacing == "log":
        return np.geomspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)
    return np.linspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)


def should_route_native_metal(config: SolveConfig) -> bool:
    """Return true when the native Metal path can run this config."""
    if (
        config.native_symmetry_plane is not None
        and config.native_symmetry_plane not in NATIVE_SYMMETRY_PLANES
    ):
        raise AssemblyBackendUnavailable(
            "native_symmetry_plane must be None or one of "
            + ", ".join(repr(p) for p in NATIVE_SYMMETRY_PLANES)
        )
    return True


def _read_f32_exact(path: Path, count: int) -> NDArray[np.float32]:
    values = np.fromfile(path, dtype="<f4")
    if values.size != count:
        raise RuntimeError(
            f"native result file {path} holds {values.size} float32 values, "
            f"expected {count}"
        )
    return values


def _read_complex_f32(
    real_path: Path,
    imag_path: Path,
    shape: tuple[int, ...],
) -> NDArray[np.complex64]:
    count = int(np.prod(shape))
    real = _read_f32_exact(real_path, count).reshape(shape)
    imag = _read_f32_exact(imag_path, count).reshape(shape)
    return np.ascontiguousarray(real + 1j * imag, dtype=np.complex64)


def _native_env_overrides(config: SolveConfig) -> dict[str, str]:
    """Helper-process environment overrides for this solve.

    Passed to the helper subprocess instead of mutating os.environ so
    concurrent solves on different threads cannot race on each other's
    assembly mode or threadgroup overrides.
    """
    overrides: dict[str, str] = {
        "HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE": config.metal_native_assembly_mode,
        "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE": config.dense_solve_dtype,
    }
    if os.environ.get("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE") is None:
        overrides["HORNLAB_METAL_BEM_NATIVE_FIELD_MODE"] = "optimized"
    # complex128 zgesv holds a doublecomplex column-major copy alongside the
    # float32 row-major operator, roughly tripling peak solve memory. CHIEF
    # cases route to a complex128 zgels least-squares solve regardless of
    # dense_solve_dtype and hold a slightly larger (n+m) x n copy per worker.
    # Lower the default solve concurrency for both paths unless the caller
    # pinned it (via the env var on os.environ), keeping peak memory bounded.
    if (
        (config.dense_solve_dtype == "float64" or config.chief_points is not None)
        and os.environ.get("HORNLAB_METAL_BEM_NATIVE_SOLVE_CONCURRENCY") is None
    ):
        overrides["HORNLAB_METAL_BEM_NATIVE_SOLVE_CONCURRENCY"] = "3"
    threadgroup_values = {
        "HORNLAB_METAL_BEM_NATIVE_THREADS_PER_GROUP": (
            config.metal_native_threads_per_group
        ),
        "HORNLAB_METAL_BEM_NATIVE_MATRIX_THREADS_PER_GROUP": (
            config.metal_native_matrix_threads_per_group
        ),
        "HORNLAB_METAL_BEM_NATIVE_RHS_THREADS_PER_GROUP": (
            config.metal_native_rhs_threads_per_group
        ),
        "HORNLAB_METAL_BEM_NATIVE_DUFFY_THREADS_PER_GROUP": (
            config.metal_native_duffy_threads_per_group
        ),
        "HORNLAB_METAL_BEM_NATIVE_FIELD_THREADS_PER_GROUP": (
            config.metal_native_field_threads_per_group
        ),
    }
    for name, value in threadgroup_values.items():
        if value is not None:
            overrides[name] = str(value)
    return overrides


def _k_values_for_native(
    frequencies: NDArray[np.float64],
    config: SolveConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    k_real = (2.0 * np.pi * frequencies / SPEED_OF_SOUND).astype(np.float32)
    if config.formulation == BIEFormulation.COMPLEX_K:
        k_imag = (k_real.astype(np.float64) * config.complex_k_shift).astype(
            np.float32,
        )
    else:
        k_imag = np.zeros_like(k_real, dtype=np.float32)
    return k_real, k_imag


def _active_impedance_sources(
    physical_tags: NDArray[np.int32],
    config: SolveConfig,
) -> dict[int, complex]:
    active: dict[int, complex] = {}
    if not config.impedance_sources:
        return active
    mesh_tags = {int(tag) for tag in np.unique(physical_tags)}
    for tag, beta in config.impedance_sources.items():
        tag_int = int(tag)
        if tag_int not in mesh_tags:
            logger.warning(
                "impedance_sources references tag %d but mesh has no elements "
                "with that tag; skipping",
                tag_int,
            )
            continue
        active[tag_int] = complex(beta)
    return active


def _impedance_sources_for_frequencies(
    physical_tags: NDArray[np.int32],
    frequencies: NDArray[np.float64],
    config: SolveConfig,
) -> list[dict[int, complex]] | dict[int, complex]:
    """Per-frequency beta dicts (callback) or one static dict (no callback).

    When ``config.impedance_source_callback`` is None this returns the single
    static dict from ``_active_impedance_sources`` (back-compat: one dict reused
    for every case). When a callback is set it is evaluated once per frequency;
    the returned ``{tag: beta}`` OVERRIDES the static value for tags present in
    both and EXTENDS it for new tags. Callback tags not present in the mesh are
    skipped with a warning. Passivity is enforced: any non-finite beta or
    ``Re(beta) < 0`` raises ``ValueError`` (the documented anti-pattern of using
    negative admittance to mask the LF blow-up is rejected here).
    """
    static = _active_impedance_sources(physical_tags, config)
    if config.impedance_source_callback is None:
        return static
    mesh_tags = {int(tag) for tag in np.unique(physical_tags)}
    per_case: list[dict[int, complex]] = []
    for freq in frequencies:
        f = float(freq)
        merged = dict(static)
        callback_betas = config.impedance_source_callback(f)
        for tag, beta in callback_betas.items():
            tag_int = int(tag)
            if tag_int not in mesh_tags:
                logger.warning(
                    "impedance_source_callback returned tag %d not in mesh; "
                    "skipping",
                    tag_int,
                )
                continue
            beta_value = complex(beta)
            if not (
                math.isfinite(beta_value.real) and math.isfinite(beta_value.imag)
            ):
                raise ValueError(
                    f"impedance_source_callback({f:.3f}) returned non-finite "
                    f"beta for tag {tag_int}"
                )
            if beta_value.real < 0.0:
                raise ValueError(
                    f"impedance_source_callback({f:.3f}) returned "
                    f"Re(beta)={beta_value.real:.4g} < 0 for tag {tag_int}; "
                    "admittance must be passive (Re(beta) >= 0)"
                )
            merged[tag_int] = beta_value
        per_case.append(merged)
    return per_case


def _build_neumann_rows(
    dp0_space,
    physical_tags: NDArray[np.int32],
    frequencies: NDArray[np.float64],
    config: SolveConfig,
    impedance_sources: list[dict[int, complex]] | dict[int, complex],
    axial_face_scale: NDArray[np.float64] | None = None,
    source_face_scale: NDArray | None = None,
) -> NDArray[np.complex64]:
    """Stack per-frequency Neumann RHS rows.

    ``impedance_sources`` is the per-frequency Robin payload already resolved by
    ``_impedance_sources_for_frequencies`` (a single static dict, or one dict per
    frequency when ``impedance_source_callback`` is set). Its tag set is passed
    into the Neumann builder as the velocity-skip set, so the callback is NEVER
    re-evaluated inside ``bie`` and the skipped tags cannot diverge from the
    Robin tags the solver applies.

    ``source_face_scale`` is the frequency-independent per-face source profile
    multiplier. ``axial_face_scale`` is kept as a back-compat alias for the
    original piston projection. ``None`` keeps the uniform-normal BC. The scale
    is geometry-only, so the caller builds it once and reuses it for every
    frequency row here.
    """
    if source_face_scale is not None:
        if axial_face_scale is not None:
            raise ValueError("pass only one of source_face_scale or axial_face_scale")
        axial_face_scale = source_face_scale

    per_case_list = isinstance(impedance_sources, list)
    rows = []
    for idx, freq in enumerate(frequencies):
        case_sources = (
            impedance_sources[idx] if per_case_list else impedance_sources
        )
        impedance_tags = {int(tag) for tag in case_sources.keys()}
        rows.append(
            _build_driver_neumann_coeffs(
                dp0_space,
                physical_tags,
                2.0 * np.pi * float(freq),
                config,
                np.complex64,
                impedance_tags=impedance_tags,
                source_face_scale=axial_face_scale,
            )
        )
    return np.stack(rows, axis=0)


def _field_points_3xn(obs_points: NDArray[np.float64]) -> NDArray[np.float32]:
    n_planes, n_angles, _ = obs_points.shape
    return np.ascontiguousarray(
        obs_points.reshape(n_planes * n_angles, 3).T,
        dtype=np.float32,
    )


def _append_sphere_field_points(
    field_points: NDArray[np.float32],
    sphere_points: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float32], int]:
    """Append free-standing sphere points after the polar-arc field points.

    The native field kernel evaluates a flat (3, M) point list, so the sphere
    points ride along in the same solve; ``_system_field`` splits them back off
    by count. Returns the combined (3, M) array and the sphere point count
    (0 when disabled, leaving the arc-only behaviour bit-unchanged).
    """
    if sphere_points is None:
        return field_points, 0
    sphere_3xn = np.ascontiguousarray(
        np.asarray(sphere_points, dtype=np.float64).T, dtype=np.float32
    )
    return np.concatenate([field_points, sphere_3xn], axis=1), int(sphere_3xn.shape[1])


def _mesh_vertices_elements(
    mesh: LoadedMesh,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    vertices = np.asarray(mesh.grid.vertices, dtype=np.float64)
    if vertices.ndim == 2 and vertices.shape[0] == 3 and vertices.shape[1] != 3:
        vertices = vertices.T
    elements = np.asarray(mesh.grid.elements, dtype=np.int32)
    if elements.ndim == 2 and elements.shape[0] == 3 and elements.shape[1] != 3:
        elements = elements.T
    return vertices, elements


def _mesh_max_edge_m(mesh: LoadedMesh) -> float:
    vertices, elements = _mesh_vertices_elements(mesh)
    if elements.size == 0:
        return 0.0
    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    p2 = vertices[elements[:, 2]]
    max_edge = max(
        float(np.max(np.linalg.norm(p1 - p0, axis=1))),
        float(np.max(np.linalg.norm(p2 - p1, axis=1))),
        float(np.max(np.linalg.norm(p0 - p2, axis=1))),
    )
    return max_edge


def _warn_near_boundary_chief_points(
    mesh: LoadedMesh,
    chief_points: NDArray[np.float64],
    max_edge_m: float,
) -> None:
    """Warn when a CHIEF point sits on or near the boundary surface.

    CHIEF null-field constraint rows are exact only for strictly interior
    points: a point on (or within roughly a quadrature panel of) the wall
    contributes a wrong constraint row that biases the whole least-squares
    solution with no error signal. Distance is approximated as the minimum
    distance to mesh vertices and triangle centroids, which is within one
    edge length of the true surface distance — cheap and adequate for a
    placement sanity check.
    """
    vertices, elements = _mesh_vertices_elements(mesh)
    if elements.size == 0 or max_edge_m <= 0.0:
        return
    centroids = vertices[elements].mean(axis=1)
    anchors = np.vstack([vertices, centroids])
    pts = np.asarray(chief_points, dtype=np.float64).reshape(-1, 3)
    deltas = pts[:, None, :] - anchors[None, :, :]
    min_d = np.sqrt(np.einsum("pak,pak->pa", deltas, deltas)).min(axis=1)
    threshold = 0.5 * float(max_edge_m)
    near = np.nonzero(min_d < threshold)[0]
    if near.size:
        worst = int(near[np.argmin(min_d[near])])
        logger.warning(
            "[hornlab-metal-bem] %d of %d chief_points sit within 0.5*max_edge "
            "(%.4g m) of the boundary mesh (closest: point %d at %.4g m). CHIEF "
            "points must be strictly interior — near-wall points bias the "
            "least-squares solution without any error signal; move them deeper "
            "into the cavity interior.",
            int(near.size),
            int(pts.shape[0]),
            threshold,
            worst,
            float(min_d[worst]),
        )


def _apply_dense_solve_policy(
    diagnostics: dict,
    *,
    threshold: float,
) -> None:
    diagnostics["dense_solve_rcond_warning_threshold"] = float(threshold)
    diagnostics["dense_solve_suspect"] = False
    if threshold <= 0.0:
        return
    raw_rcond = diagnostics.get("dense_solve_rcond")
    if raw_rcond is None:
        # The CHIEF zgels path returns no rcond estimate, so the conditioning
        # policy cannot run there. Mark it explicitly: a plain
        # dense_solve_suspect=False would read as "checked and fine".
        diagnostics["dense_solve_policy_available"] = False
        return
    diagnostics["dense_solve_policy_available"] = True
    try:
        rcond = float(raw_rcond)
    except (TypeError, ValueError):
        return
    if not math.isfinite(rcond):
        return
    if rcond < threshold:
        diagnostics["dense_solve_suspect"] = True
        diagnostics["dense_solve_recommendation"] = (
            "Treat this frequency as suspect; nudge or densify the frequency "
            "grid around the resonance and compare against nearby points. "
            "This flag detects dense-solve conditioning only."
        )


def _apply_mesh_resolution_policy(
    diagnostics: dict,
    *,
    frequency_hz: float,
    mesh_max_edge_m: float,
    elements_per_wavelength_min: float,
) -> None:
    diagnostics["mesh_max_edge_m"] = float(mesh_max_edge_m)
    diagnostics["mesh_elements_per_wavelength_min"] = float(
        elements_per_wavelength_min
    )
    if mesh_max_edge_m <= 0.0:
        diagnostics["mesh_elements_per_wavelength"] = math.inf
        diagnostics["mesh_max_valid_frequency_hz"] = math.inf
        diagnostics["mesh_resolution_suspect"] = False
        return
    wavelength_m = SPEED_OF_SOUND / float(frequency_hz)
    elements_per_wavelength = wavelength_m / float(mesh_max_edge_m)
    max_valid_frequency_hz = SPEED_OF_SOUND / (
        float(elements_per_wavelength_min) * float(mesh_max_edge_m)
    )
    diagnostics["mesh_elements_per_wavelength"] = float(elements_per_wavelength)
    diagnostics["mesh_max_valid_frequency_hz"] = float(max_valid_frequency_hz)
    diagnostics["mesh_resolution_suspect"] = bool(
        elements_per_wavelength < elements_per_wavelength_min
    )


def _system_reductions(
    system,
    mesh: LoadedMesh,
    p1_space,
    source_tags: list[int],
    impedance_source_tag: int,
) -> tuple[complex, dict[int, complex]]:
    if system.impedance is not None and system.surface_pressure_avg is not None:
        return system.impedance, system.surface_pressure_avg

    if system.pressure_real_f32 is None or system.pressure_imag_f32 is None:
        raise RuntimeError(
            "native solve-field result did not include pressure reductions "
            "or surface pressure files"
        )

    pressure_surface = _read_complex_f32(
        Path(system.pressure_real_f32),
        Path(system.pressure_imag_f32),
        tuple(system.pressure_shape),
    )
    p_surface = SimpleNamespace(coefficients=pressure_surface)
    impedance = _compute_impedance(
        mesh.grid,
        p_surface,
        mesh.physical_tags,
        p1_space,
        source_tag=impedance_source_tag,
    )
    pavg = compute_surface_pressure_avg(
        mesh.grid,
        p_surface,
        mesh.physical_tags,
        p1_space,
        source_tags,
    )
    return impedance, pavg


def _system_field(
    system,
    n_planes: int,
    n_angles: int,
    n_sphere: int = 0,
    field_batch_complex: NDArray[np.complex128] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128] | None]:
    """Return (arc_pressure, sphere_pressure).

    The flat native field result holds the ``n_planes * n_angles`` polar-arc
    points followed by ``n_sphere`` free-standing sphere points. The arc block is
    reshaped to (n_planes, n_angles); the sphere block is returned flat, or None
    when no sphere points were requested.
    """
    if field_batch_complex is not None:
        if system.field_row_index is None:
            raise RuntimeError("mixed batched and per-case native field results")
        field_complex = field_batch_complex[system.field_row_index]
    else:
        field_complex = _read_complex_f32(
            Path(system.field_real_f32),
            Path(system.field_imag_f32),
            tuple(system.field_shape),
        ).astype(np.complex128)
    flat = np.asarray(field_complex).reshape(-1)
    arc_count = n_planes * n_angles
    arc = flat[:arc_count].reshape(n_planes, n_angles)
    sphere = flat[arc_count : arc_count + n_sphere] if n_sphere else None
    return arc, sphere


def _system_surface_pressure(system) -> NDArray[np.complex128]:
    if system.pressure_real_f32 is None or system.pressure_imag_f32 is None:
        raise RuntimeError("native result did not include requested surface pressure")
    return _read_complex_f32(
        Path(system.pressure_real_f32),
        Path(system.pressure_imag_f32),
        tuple(system.pressure_shape),
    ).astype(np.complex128)


def _directivity_from_pressure(
    pressure: NDArray[np.complex128],
    on_axis_idx: int,
) -> NDArray[np.float64]:
    # Floor amplitudes at -120 dB SPL so silent points stay finite, the
    # mapping stays monotonic near zero, and log10 never sees 0.
    floor_amplitude = REFERENCE_PRESSURE * 10.0 ** (-120.0 / 20.0)
    amplitudes = np.maximum(np.abs(pressure), floor_amplitude)
    spl_raw = 20.0 * np.log10(amplitudes / REFERENCE_PRESSURE)
    return spl_raw - spl_raw[:, on_axis_idx][:, None]


def _append_system_result(
    *,
    frequency_hz: float,
    system,
    backend: str,
    timing_s: float,
    mesh: LoadedMesh,
    p1_space,
    source_tags: list[int],
    impedance_source_tag: int,
    n_planes: int,
    n_angles: int,
    on_axis_idx: int,
    field_batch_complex: NDArray[np.complex128] | None,
    n_sphere: int = 0,
    surface_pavg: dict[int, list[complex]],
    pressure_rows: list[NDArray[np.complex128]],
    spl_rows: list[NDArray[np.float64]],
    impedance_rows: list[complex],
    surface_pressure_rows: list[NDArray[np.complex128]] | None,
    native_diagnostics_rows: list[dict],
    solver_log: list[dict],
    completed_freqs: list[float],
    dense_solve_rcond_warning_threshold: float = 0.0,
    mesh_max_edge_m: float = 0.0,
    mesh_elements_per_wavelength_min: float = 6.0,
) -> dict:
    impedance, pavg = _system_reductions(
        system,
        mesh,
        p1_space,
        source_tags,
        impedance_source_tag,
    )
    for tag in source_tags:
        surface_pavg[tag].append(pavg[tag])

    pressure, sphere_pressure = _system_field(
        system, n_planes, n_angles, n_sphere, field_batch_complex
    )
    directivity = _directivity_from_pressure(pressure, on_axis_idx)
    native_diagnostics = dict(getattr(system, "diagnostics", {}) or {})
    _apply_dense_solve_policy(
        native_diagnostics,
        threshold=dense_solve_rcond_warning_threshold,
    )
    _apply_mesh_resolution_policy(
        native_diagnostics,
        frequency_hz=frequency_hz,
        mesh_max_edge_m=mesh_max_edge_m,
        elements_per_wavelength_min=mesh_elements_per_wavelength_min,
    )
    log_entry = {
        "frequency_hz": frequency_hz,
        "iterations": None,
        "timing_s": timing_s,
        "backend": backend,
        "assembly_s": float(system.assembly_s),
        "dense_solve_s": float(system.dense_solve_s),
        "field_s": float(system.field_s),
        "lapack_info": int(system.lapack_info),
        "impedance": impedance,
        "native_diagnostics": native_diagnostics,
        "observation_sphere_pressure_complex": sphere_pressure,
    }

    completed_freqs.append(frequency_hz)
    pressure_rows.append(pressure)
    spl_rows.append(directivity)
    impedance_rows.append(impedance)
    if surface_pressure_rows is not None:
        surface_pressure_rows.append(_system_surface_pressure(system))
    native_diagnostics_rows.append(native_diagnostics)
    solver_log.append(log_entry)
    return log_entry


def run_sweep_native_metal(
    mesh: LoadedMesh,
    frequencies: NDArray[np.float64],
    frame: ObservationFrame,
    config: SolveConfig,
) -> SolveResult:
    """Run an explicit experimental standard-Neumann sweep with native Metal.

    The native helper currently accelerates corrected dense assembly and
    exterior field evaluation. Linear solving remains a general dense CPU solve
    from the assembled non-Hermitian system.
    """
    should_route_native_metal(config)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.size == 0:
        raise ValueError("frequencies must contain at least one value")

    mesh_tags = {int(tag) for tag in np.unique(mesh.physical_tags)}
    missing_tags = sorted(set(config.velocity_sources) - mesh_tags)
    if missing_tags:
        raise ValueError(
            f"velocity_sources tags {missing_tags} are not present in the mesh; "
            f"available physical tags: {sorted(mesh_tags)}"
        )

    try:
        from .metal.geometry import build_metal_geometry_buffers
        from .metal.native import MetalNativeStandardSession
    except Exception as exc:  # pragma: no cover - import/runtime specific.
        raise AssemblyBackendUnavailable(
            f"Native Metal helper could not be imported: {exc}"
        ) from exc

    runtime = _discover_runtime_smoke_cached()
    if not runtime.available:
        reason = "; ".join(runtime.unavailable_reasons)
        raise AssemblyBackendUnavailable(reason)

    t_total = time.time()
    obs_points, angles_deg = build_observation_points(frame, config.observation)
    p1_space, dp0_space = make_pure_function_spaces(mesh.grid)
    geometry_buffers = build_metal_geometry_buffers(
        mesh.grid,
        mesh.physical_tags,
        p1_space,
        dp0_space,
    )
    mesh_max_edge_m = _mesh_max_edge_m(mesh)

    source_tags = list(config.velocity_sources.keys())
    # Per-face source profile multiplier, built once (geometry-only) and reused
    # for every frequency row. The frame axis is symmetry-projected, so reduced
    # (half/quarter) caps stay on-axis. None keeps the uniform-normal BC.
    source_face_scale = _build_source_face_scale(
        mesh.grid,
        mesh.physical_tags,
        config,
        frame.axis,
        frame.source_center,
    )
    # Per-frequency beta dicts when impedance_source_callback is set, else a
    # single static dict. Computed once here (before the streaming/non-streaming
    # branch split) so both branches send the identical per-case payloads.
    impedance_sources_arg = _impedance_sources_for_frequencies(
        mesh.physical_tags, frequencies, config
    )
    # CHIEF interior overdetermination points, (m, 3) metres in the mesh frame.
    # Marshal to the (3, m) float32 layout the helper reader expects (same as the
    # observation points). None leaves the solve a plain square LU/zgesv solve.
    chief_points_3xm = None
    if config.chief_points is not None:
        chief_points_3xm = np.ascontiguousarray(
            np.asarray(config.chief_points, dtype=np.float64).T, dtype=np.float32
        )
        _warn_near_boundary_chief_points(
            mesh,
            np.asarray(config.chief_points, dtype=np.float64),
            mesh_max_edge_m,
        )
    surface_pavg: dict[int, list[complex]] = {tag: [] for tag in source_tags}
    pressure_rows: list[NDArray[np.complex128]] = []
    spl_rows: list[NDArray[np.float64]] = []
    impedance_rows: list[complex] = []
    surface_pressure_rows: list[NDArray[np.complex128]] | None = (
        [] if config.return_surface_pressure else None
    )
    native_diagnostics_rows: list[dict] = []
    solver_log: list[dict] = []
    completed_freqs: list[float] = []
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))
    impedance_source_tag = min(config.velocity_sources.keys(), default=2)
    n_planes, n_angles, _ = obs_points.shape
    field_points, n_sphere = _append_sphere_field_points(
        _field_points_3xn(obs_points), config.observation.sphere_points
    )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=geometry_buffers,
        symmetry_plane=config.native_symmetry_plane,
        aperture_tag=config.aperture_tag,
        velocity_source_tags=source_tags,
        check_open_edges=config.native_check_open_edges,
        runtime_status=runtime,
        extra_env=_native_env_overrides(config),
    ) as session:
        if config.on_frequency_result is None:
            freq_values = np.asarray(frequencies, dtype=np.float64)
            k_values, k_imag_values = _k_values_for_native(freq_values, config)
            neumann_rows = _build_neumann_rows(
                dp0_space,
                mesh.physical_tags,
                freq_values,
                config,
                impedance_sources_arg,
                source_face_scale=source_face_scale,
            )

            logger.info(
                "Running %d-frequency native Metal resident assembly/solve/field batch.",
                len(freq_values),
            )
            systems = session.assemble_solve_evaluate_standard_neumann_batch(
                freq_values,
                k_values,
                neumann_rows,
                field_points,
                k_imag_f32=k_imag_values,
                impedance_sources=impedance_sources_arg,
                batch_id="all_observation_planes",
                operation_id="assembly-solve-field-resident-batch",
                source_tags=source_tags,
                impedance_source_tag=impedance_source_tag,
                write_surface_pressure=config.return_surface_pressure,
                write_batched_field=True,
                dense_solve_dtype=config.dense_solve_dtype,
                chief_points=chief_points_3xm,
                chief_weight=config.chief_weight,
            )
            field_batch_complex: NDArray[np.complex128] | None = None
            if systems and systems[0].field_row_index is not None:
                first = systems[0]
                if first.field_batch_shape is None:
                    raise RuntimeError("native batched field result missing shape")
                field_batch_complex = _read_complex_f32(
                    Path(first.field_real_f32),
                    Path(first.field_imag_f32),
                    tuple(first.field_batch_shape),
                ).astype(np.complex128)

            for i, (freq, system) in enumerate(zip(freq_values, systems)):
                frequency_hz = float(freq)
                timing_s = (
                    float(system.assembly_s)
                    + float(system.dense_solve_s)
                    + float(system.field_s)
                )
                _append_system_result(
                    frequency_hz=frequency_hz,
                    system=system,
                    backend="native_metal_resident_assembly_solve_field_batch",
                    timing_s=timing_s,
                    mesh=mesh,
                    p1_space=p1_space,
                    source_tags=source_tags,
                    impedance_source_tag=impedance_source_tag,
                    n_planes=n_planes,
                    n_angles=n_angles,
                    on_axis_idx=on_axis_idx,
                    field_batch_complex=field_batch_complex,
                    n_sphere=n_sphere,
                    surface_pavg=surface_pavg,
                    pressure_rows=pressure_rows,
                    spl_rows=spl_rows,
                    impedance_rows=impedance_rows,
                    surface_pressure_rows=surface_pressure_rows,
                    native_diagnostics_rows=native_diagnostics_rows,
                    solver_log=solver_log,
                    completed_freqs=completed_freqs,
                    dense_solve_rcond_warning_threshold=(
                        config.dense_solve_rcond_warning_threshold
                    ),
                    mesh_max_edge_m=mesh_max_edge_m,
                    mesh_elements_per_wavelength_min=(
                        config.mesh_elements_per_wavelength_min
                    ),
                )
                if config.progress_callback is not None:
                    config.progress_callback(i, len(freq_values), frequency_hz)
        else:
            freq_values = np.asarray(frequencies, dtype=np.float64)
            k_values, k_imag_values = _k_values_for_native(freq_values, config)
            neumann_rows = _build_neumann_rows(
                dp0_space,
                mesh.physical_tags,
                freq_values,
                config,
                impedance_sources_arg,
                source_face_scale=source_face_scale,
            )
            logger.info(
                "Running %d-frequency native Metal streamed assembly/solve/field batch.",
                len(freq_values),
            )

            def _on_case_result(i: int, system) -> bool | None:
                frequency_hz = float(freq_values[i])
                logger.info(
                    "[%d/%d] %.1f Hz (native Metal %s assembly)",
                    i + 1,
                    len(freq_values),
                    frequency_hz,
                    config.metal_native_assembly_mode,
                )
                timing_s = (
                    float(system.assembly_s)
                    + float(system.dense_solve_s)
                    + float(system.field_s)
                )
                log_entry = _append_system_result(
                    frequency_hz=frequency_hz,
                    system=system,
                    backend="native_metal_resident_assembly_solve_field_streamed",
                    timing_s=timing_s,
                    mesh=mesh,
                    p1_space=p1_space,
                    source_tags=source_tags,
                    impedance_source_tag=impedance_source_tag,
                    n_planes=n_planes,
                    n_angles=n_angles,
                    on_axis_idx=on_axis_idx,
                    field_batch_complex=None,
                    n_sphere=n_sphere,
                    surface_pavg=surface_pavg,
                    pressure_rows=pressure_rows,
                    spl_rows=spl_rows,
                    impedance_rows=impedance_rows,
                    surface_pressure_rows=surface_pressure_rows,
                    native_diagnostics_rows=native_diagnostics_rows,
                    solver_log=solver_log,
                    completed_freqs=completed_freqs,
                    dense_solve_rcond_warning_threshold=(
                        config.dense_solve_rcond_warning_threshold
                    ),
                    mesh_max_edge_m=mesh_max_edge_m,
                    mesh_elements_per_wavelength_min=(
                        config.mesh_elements_per_wavelength_min
                    ),
                )
                if config.progress_callback is not None:
                    config.progress_callback(i, len(freq_values), frequency_hz)
                callback_entry = {
                    **log_entry,
                    "observation_pressure_complex": pressure_rows[-1],
                    "observation_directivity_db": spl_rows[-1],
                    "observation_angles_deg": angles_deg,
                    "observation_planes": config.observation.planes,
                }
                if config.on_frequency_result(i, frequency_hz, callback_entry) is False:
                    logger.info("Early stop requested after %.1f Hz", frequency_hz)
                    return False
                return None

            # One resident helper invocation for the whole sweep; per-case
            # results stream back as each frequency completes, and a False
            # callback return terminates the helper early.
            session.assemble_solve_evaluate_standard_neumann_batch(
                freq_values,
                k_values,
                neumann_rows,
                field_points,
                k_imag_f32=k_imag_values,
                impedance_sources=impedance_sources_arg,
                batch_id="all_observation_planes",
                operation_id="assembly-solve-field-resident-stream",
                source_tags=source_tags,
                impedance_source_tag=impedance_source_tag,
                write_surface_pressure=config.return_surface_pressure,
                on_case_result=_on_case_result,
                dense_solve_dtype=config.dense_solve_dtype,
                chief_points=chief_points_3xm,
                chief_weight=config.chief_weight,
            )

    sp_avg: dict[int, np.ndarray] = {}
    for tag in source_tags:
        sp_avg[tag] = np.array(surface_pavg[tag], dtype=np.complex128)

    timings = {
        "solve_s": sum(float(entry["timing_s"]) for entry in solver_log),
        "assembly_s": sum(float(entry["assembly_s"]) for entry in solver_log),
        "dense_solve_s": sum(float(entry["dense_solve_s"]) for entry in solver_log),
        "directivity_s": sum(float(entry["field_s"]) for entry in solver_log),
        "total_s": time.time() - t_total,
    }

    return SolveResult(
        frequencies_hz=np.array(completed_freqs, dtype=np.float64),
        pressure_complex=np.stack(pressure_rows, axis=0),
        directivity_db=np.stack(spl_rows, axis=0),
        impedance=np.array(impedance_rows, dtype=np.complex128),
        observation_angles_deg=angles_deg,
        observation_points=obs_points,
        observation_planes=config.observation.planes,
        config=config,
        mesh_info=mesh.info,
        timings=timings,
        solver_log=solver_log,
        surface_pressure_avg=sp_avg if sp_avg else None,
        surface_pressure_complex=(
            np.stack(surface_pressure_rows, axis=0)
            if surface_pressure_rows is not None
            else None
        ),
        native_diagnostics=native_diagnostics_rows,
    )


def _extra_source_system_view(system, extra) -> SimpleNamespace:
    """Per-extra-source view over one solved case for the result builders.

    Mirrors the attribute surface `_append_system_result` reads. The shared
    assembly/dense-solve cost is attributed to source 0's result only; extra
    sources report zero there and their own field-evaluation time, so summing
    per-source timings reproduces the true total instead of multiplying the
    shared factorization by the source count.
    """
    return SimpleNamespace(
        impedance=extra.impedance,
        surface_pressure_avg=extra.surface_pressure_avg,
        pressure_real_f32=extra.pressure_real_f32,
        pressure_imag_f32=extra.pressure_imag_f32,
        pressure_shape=extra.pressure_shape,
        field_real_f32=extra.field_real_f32,
        field_imag_f32=extra.field_imag_f32,
        field_shape=extra.field_shape,
        field_row_index=extra.field_row_index,
        field_batch_shape=extra.field_batch_shape,
        assembly_s=0.0,
        dense_solve_s=0.0,
        field_s=extra.field_s,
        lapack_info=system.lapack_info,
        diagnostics=getattr(system, "diagnostics", {}),
    )


def run_sweep_native_metal_multi_source(
    mesh: LoadedMesh,
    frequencies: NDArray[np.float64],
    frame: ObservationFrame,
    config: SolveConfig,
    sources: list[dict[int, complex]],
) -> list[SolveResult]:
    """Solve one sweep with multiple velocity sources sharing each operator.

    Each frequency's system matrix is assembled and factored ONCE in the
    native helper; every entry of ``sources`` contributes one right-hand side
    (multi-RHS) plus its own field evaluation and reductions. Returns one
    ``SolveResult`` per source, each equivalent to a sequential ``solve()``
    with ``config.velocity_sources`` replaced by that source dict (to float32
    tolerance). ``config.velocity_sources`` itself is ignored.

    ``surface_pressure_avg`` is recorded for the sorted union of all source
    tags in EVERY result, so cross-source transfer terms (e.g. radiation-
    impedance matrix columns) come out of one call. Streaming callbacks
    (``on_frequency_result``) are supported and receive a ``source_results``
    list containing one log entry per source. ``velocity_source_callback``
    remains unsupported here because each source vector is fixed for the whole
    shared multi-RHS batch.
    """
    should_route_native_metal(config)
    if not sources:
        raise ValueError("sources must contain at least one velocity dict")
    if config.velocity_source_callback is not None:
        raise ValueError(
            "multi-source solves do not support velocity_source_callback"
        )
    per_source_configs = [
        replace(config, velocity_sources=dict(source)) for source in sources
    ]
    if len(sources) == 1:
        return [
            run_sweep_native_metal(mesh, frequencies, frame, per_source_configs[0])
        ]

    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.size == 0:
        raise ValueError("frequencies must contain at least one value")

    mesh_tags = {int(tag) for tag in np.unique(mesh.physical_tags)}
    for index, source in enumerate(sources):
        missing_tags = sorted(set(source) - mesh_tags)
        if missing_tags:
            raise ValueError(
                f"sources[{index}] tags {missing_tags} are not present in the "
                f"mesh; available physical tags: {sorted(mesh_tags)}"
            )

    try:
        from .metal.geometry import build_metal_geometry_buffers
        from .metal.native import MetalNativeStandardSession
    except Exception as exc:  # pragma: no cover - import/runtime specific.
        raise AssemblyBackendUnavailable(
            f"Native Metal helper could not be imported: {exc}"
        ) from exc

    runtime = _discover_runtime_smoke_cached()
    if not runtime.available:
        reason = "; ".join(runtime.unavailable_reasons)
        raise AssemblyBackendUnavailable(reason)

    t_total = time.time()
    obs_points, angles_deg = build_observation_points(frame, config.observation)
    p1_space, dp0_space = make_pure_function_spaces(mesh.grid)
    geometry_buffers = build_metal_geometry_buffers(
        mesh.grid,
        mesh.physical_tags,
        p1_space,
        dp0_space,
    )
    mesh_max_edge_m = _mesh_max_edge_m(mesh)

    # Union of all source tags: every source's result records the average
    # surface pressure on every tag any source drives (or lists at zero
    # velocity), which is exactly the cross-term data an aperture radiation
    # matrix needs.
    source_tags = sorted({int(tag) for source in sources for tag in source})
    # Per-face source profile multiplier over the union of source tags, built
    # once (geometry-only) on the shared frame. None keeps uniform-normal.
    scale_config = replace(
        config,
        velocity_sources={tag: 1.0 for tag in source_tags},
    )
    source_face_scale = _build_source_face_scale(
        mesh.grid,
        mesh.physical_tags,
        scale_config,
        frame.axis,
        frame.source_center,
    )
    impedance_source_tags = [
        min(source.keys(), default=2) for source in sources
    ]
    impedance_sources_arg = _impedance_sources_for_frequencies(
        mesh.physical_tags, frequencies, config
    )
    chief_points_3xm = None
    if config.chief_points is not None:
        chief_points_3xm = np.ascontiguousarray(
            np.asarray(config.chief_points, dtype=np.float64).T, dtype=np.float32
        )
        _warn_near_boundary_chief_points(
            mesh,
            np.asarray(config.chief_points, dtype=np.float64),
            mesh_max_edge_m,
        )

    n_sources = len(sources)
    surface_pavg = [
        {tag: [] for tag in source_tags} for _ in range(n_sources)
    ]
    pressure_rows: list[list[NDArray[np.complex128]]] = [
        [] for _ in range(n_sources)
    ]
    spl_rows: list[list[NDArray[np.float64]]] = [[] for _ in range(n_sources)]
    impedance_rows: list[list[complex]] = [[] for _ in range(n_sources)]
    surface_pressure_rows: list[list[NDArray[np.complex128]] | None] = [
        [] if config.return_surface_pressure else None for _ in range(n_sources)
    ]
    native_diagnostics_rows: list[list[dict]] = [[] for _ in range(n_sources)]
    solver_logs: list[list[dict]] = [[] for _ in range(n_sources)]
    completed_freqs: list[list[float]] = [[] for _ in range(n_sources)]
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))
    n_planes, n_angles, _ = obs_points.shape
    field_points, n_sphere = _append_sphere_field_points(
        _field_points_3xn(obs_points), config.observation.sphere_points
    )

    freq_values = np.asarray(frequencies, dtype=np.float64)
    k_values, k_imag_values = _k_values_for_native(freq_values, config)
    per_source_neumann = [
        _build_neumann_rows(
            dp0_space,
            mesh.physical_tags,
            freq_values,
            source_config,
            impedance_sources_arg,
            source_face_scale=source_face_scale,
        )
        for source_config in per_source_configs
    ]

    def _append_case_results(
        index: int,
        system,
        *,
        field_batch_complex: NDArray[np.complex128] | None,
    ) -> float:
        frequency_hz = float(freq_values[index])
        if len(system.extra_sources) != n_sources - 1:
            raise RuntimeError(
                "native multi-source case returned "
                f"{len(system.extra_sources)} extra source(s), expected "
                f"{n_sources - 1}"
            )
        timing_s = (
            float(system.assembly_s)
            + float(system.dense_solve_s)
            + float(system.field_s)
        )
        for source_index in range(n_sources):
            source_system = (
                system
                if source_index == 0
                else _extra_source_system_view(
                    system, system.extra_sources[source_index - 1]
                )
            )
            source_timing = (
                timing_s
                if source_index == 0
                else float(source_system.field_s)
            )
            _append_system_result(
                frequency_hz=frequency_hz,
                system=source_system,
                backend="native_metal_resident_multi_source_batch",
                timing_s=source_timing,
                mesh=mesh,
                p1_space=p1_space,
                source_tags=source_tags,
                impedance_source_tag=impedance_source_tags[source_index],
                n_planes=n_planes,
                n_angles=n_angles,
                on_axis_idx=on_axis_idx,
                field_batch_complex=field_batch_complex,
                n_sphere=n_sphere,
                surface_pavg=surface_pavg[source_index],
                pressure_rows=pressure_rows[source_index],
                spl_rows=spl_rows[source_index],
                impedance_rows=impedance_rows[source_index],
                surface_pressure_rows=surface_pressure_rows[source_index],
                native_diagnostics_rows=native_diagnostics_rows[source_index],
                solver_log=solver_logs[source_index],
                completed_freqs=completed_freqs[source_index],
                dense_solve_rcond_warning_threshold=(
                    config.dense_solve_rcond_warning_threshold
                ),
                mesh_max_edge_m=mesh_max_edge_m,
                mesh_elements_per_wavelength_min=(
                    config.mesh_elements_per_wavelength_min
                ),
            )
        return frequency_hz

    def _multi_source_callback_entry() -> dict:
        source_entries: list[dict] = []
        for source_index in range(n_sources):
            log_entry = dict(solver_logs[source_index][-1])
            log_entry.update(
                observation_pressure_complex=pressure_rows[source_index][-1],
                observation_directivity_db=spl_rows[source_index][-1],
                observation_angles_deg=angles_deg,
                observation_planes=config.observation.planes,
            )
            source_entries.append(log_entry)
        return {
            **source_entries[0],
            "source_results": source_entries,
        }

    with MetalNativeStandardSession.create_session(
        geometry_buffers=geometry_buffers,
        symmetry_plane=config.native_symmetry_plane,
        aperture_tag=config.aperture_tag,
        velocity_source_tags=source_tags,
        check_open_edges=config.native_check_open_edges,
        runtime_status=runtime,
        extra_env=_native_env_overrides(config),
    ) as session:
        logger.info(
            "Running %d-frequency x %d-source native Metal multi-RHS batch.",
            len(freq_values),
            n_sources,
        )
        def _on_case_result(i: int, system) -> bool | None:
            frequency_hz = _append_case_results(
                i,
                system,
                field_batch_complex=None,
            )
            if config.progress_callback is not None:
                config.progress_callback(i, len(freq_values), frequency_hz)
            if config.on_frequency_result is not None:
                if config.on_frequency_result(i, frequency_hz, _multi_source_callback_entry()) is False:
                    logger.info("Early stop requested after %.1f Hz", frequency_hz)
                    return False
            return None

        systems = session.assemble_solve_evaluate_standard_neumann_batch(
            freq_values,
            k_values,
            per_source_neumann[0],
            field_points,
            k_imag_f32=k_imag_values,
            impedance_sources=impedance_sources_arg,
            batch_id="all_observation_planes",
            operation_id="assembly-solve-field-resident-batch-multi-source",
            source_tags=source_tags,
            impedance_source_tag=impedance_source_tags[0],
            write_surface_pressure=config.return_surface_pressure,
            write_batched_field=config.on_frequency_result is None,
            on_case_result=_on_case_result if config.on_frequency_result is not None else None,
            dense_solve_dtype=config.dense_solve_dtype,
            chief_points=chief_points_3xm,
            chief_weight=config.chief_weight,
            extra_neumann_dp0=np.stack(per_source_neumann[1:], axis=0),
            extra_impedance_source_tags=impedance_source_tags[1:],
        )
        if config.on_frequency_result is not None:
            systems = []

        field_batch_complex: NDArray[np.complex128] | None = None
        if systems and systems[0].field_row_index is not None:
            first = systems[0]
            if first.field_batch_shape is None:
                raise RuntimeError("native batched field result missing shape")
            field_batch_complex = _read_complex_f32(
                Path(first.field_real_f32),
                Path(first.field_imag_f32),
                tuple(first.field_batch_shape),
            ).astype(np.complex128)

        for i, (freq, system) in enumerate(zip(freq_values, systems)):
            frequency_hz = _append_case_results(
                i,
                system,
                field_batch_complex=field_batch_complex,
            )
            if config.progress_callback is not None:
                config.progress_callback(i, len(freq_values), frequency_hz)

    total_s = time.time() - t_total
    results: list[SolveResult] = []
    for source_index in range(n_sources):
        solver_log = solver_logs[source_index]
        sp_avg: dict[int, np.ndarray] = {
            tag: np.array(values, dtype=np.complex128)
            for tag, values in surface_pavg[source_index].items()
        }
        timings = {
            "solve_s": sum(float(entry["timing_s"]) for entry in solver_log),
            "assembly_s": sum(float(entry["assembly_s"]) for entry in solver_log),
            "dense_solve_s": sum(
                float(entry["dense_solve_s"]) for entry in solver_log
            ),
            "directivity_s": sum(
                float(entry["field_s"]) for entry in solver_log
            ),
            # Shared wall clock: the batch solves every source at once, so the
            # per-source split lives in solve_s/assembly_s/dense_solve_s
            # (source 0 carries the shared factorization cost).
            "total_s": total_s if source_index == 0 else 0.0,
        }
        results.append(
            SolveResult(
                frequencies_hz=np.array(
                    completed_freqs[source_index], dtype=np.float64
                ),
                pressure_complex=np.stack(pressure_rows[source_index], axis=0),
                directivity_db=np.stack(spl_rows[source_index], axis=0),
                impedance=np.array(
                    impedance_rows[source_index], dtype=np.complex128
                ),
                observation_angles_deg=angles_deg,
                observation_points=obs_points,
                observation_planes=config.observation.planes,
                config=per_source_configs[source_index],
                mesh_info=mesh.info,
                timings=timings,
                solver_log=solver_log,
                surface_pressure_avg=sp_avg if sp_avg else None,
                surface_pressure_complex=(
                    np.stack(surface_pressure_rows[source_index], axis=0)
                    if surface_pressure_rows[source_index] is not None
                    else None
                ),
                native_diagnostics=native_diagnostics_rows[source_index],
            )
        )
    return results

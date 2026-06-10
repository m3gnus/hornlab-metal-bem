"""Native Metal frequency sweep execution."""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from ._constants import REFERENCE_PRESSURE, SPEED_OF_SOUND
from .backends import AssemblyBackendUnavailable
from .bie import (
    _build_driver_neumann_coeffs,
    _compute_impedance,
    compute_surface_pressure_avg,
)
from .config import NATIVE_SYMMETRY_PLANES, SolveConfig
from .mesh import LoadedMesh, make_pure_function_spaces
from .observation import ObservationFrame, build_observation_points
from .result import SolveResult

logger = logging.getLogger(__name__)


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


@contextmanager
def _native_environment(config: SolveConfig):
    env_values: dict[str, str | None] = {
        "HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE": config.metal_native_assembly_mode,
        "HORNLAB_METAL_BEM_NATIVE_FIELD_MODE": "optimized",
        "HORNLAB_METAL_BEM_NATIVE_THREADS_PER_GROUP": (
            None
            if config.metal_native_threads_per_group is None
            else str(config.metal_native_threads_per_group)
        ),
        "HORNLAB_METAL_BEM_NATIVE_MATRIX_THREADS_PER_GROUP": (
            None
            if config.metal_native_matrix_threads_per_group is None
            else str(config.metal_native_matrix_threads_per_group)
        ),
        "HORNLAB_METAL_BEM_NATIVE_RHS_THREADS_PER_GROUP": (
            None
            if config.metal_native_rhs_threads_per_group is None
            else str(config.metal_native_rhs_threads_per_group)
        ),
        "HORNLAB_METAL_BEM_NATIVE_DUFFY_THREADS_PER_GROUP": (
            None
            if config.metal_native_duffy_threads_per_group is None
            else str(config.metal_native_duffy_threads_per_group)
        ),
        "HORNLAB_METAL_BEM_NATIVE_FIELD_THREADS_PER_GROUP": (
            None
            if config.metal_native_field_threads_per_group is None
            else str(config.metal_native_field_threads_per_group)
        ),
    }
    previous_values = {name: os.environ.get(name) for name in env_values}

    os.environ["HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE"] = (
        config.metal_native_assembly_mode
    )
    if previous_values["HORNLAB_METAL_BEM_NATIVE_FIELD_MODE"] is None:
        os.environ["HORNLAB_METAL_BEM_NATIVE_FIELD_MODE"] = "optimized"
    for name, value in env_values.items():
        if name.endswith("_ASSEMBLY_MODE") or name.endswith("_FIELD_MODE"):
            continue
        if value is not None:
            os.environ[name] = value

    try:
        yield
    finally:
        for name, previous in previous_values.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def _build_neumann_rows(
    dp0_space,
    physical_tags: NDArray[np.int32],
    frequencies: NDArray[np.float64],
    config: SolveConfig,
) -> NDArray[np.complex64]:
    return np.stack(
        [
            _build_driver_neumann_coeffs(
                dp0_space,
                physical_tags,
                2.0 * np.pi * float(freq),
                config,
                np.complex64,
            )
            for freq in frequencies
        ],
        axis=0,
    )


def _field_points_3xn(obs_points: NDArray[np.float64]) -> NDArray[np.float32]:
    n_planes, n_angles, _ = obs_points.shape
    return np.ascontiguousarray(
        obs_points.reshape(n_planes * n_angles, 3).T,
        dtype=np.float32,
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
    field_batch_complex: NDArray[np.complex128] | None = None,
) -> NDArray[np.complex128]:
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
    return field_complex.reshape(n_planes, n_angles)


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
    surface_pavg: dict[int, list[complex]],
    pressure_rows: list[NDArray[np.complex128]],
    spl_rows: list[NDArray[np.float64]],
    impedance_rows: list[complex],
    surface_pressure_rows: list[NDArray[np.complex128]] | None,
    native_diagnostics_rows: list[dict],
    solver_log: list[dict],
    completed_freqs: list[float],
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

    pressure = _system_field(system, n_planes, n_angles, field_batch_complex)
    directivity = _directivity_from_pressure(pressure, on_axis_idx)
    native_diagnostics = dict(getattr(system, "diagnostics", {}) or {})
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
        from .metal.native import MetalNativeStandardSession, discover_native_runtime
    except Exception as exc:  # pragma: no cover - import/runtime specific.
        raise AssemblyBackendUnavailable(
            f"Native Metal helper could not be imported: {exc}"
        ) from exc

    runtime = discover_native_runtime(run_smoke_test=True)
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

    source_tags = list(config.velocity_sources.keys())
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
    field_points = _field_points_3xn(obs_points)

    with _native_environment(config):
        with MetalNativeStandardSession.create_session(
            geometry_buffers=geometry_buffers,
            symmetry_plane=config.native_symmetry_plane,
        ) as session:
            if config.on_frequency_result is None:
                freq_values = np.asarray(frequencies, dtype=np.float64)
                k_values = (2.0 * np.pi * freq_values / SPEED_OF_SOUND).astype(
                    np.float32,
                )
                neumann_rows = _build_neumann_rows(
                    dp0_space,
                    mesh.physical_tags,
                    freq_values,
                    config,
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
                    batch_id="all_observation_planes",
                    operation_id="assembly-solve-field-resident-batch",
                    source_tags=source_tags,
                    impedance_source_tag=impedance_source_tag,
                    write_surface_pressure=config.return_surface_pressure,
                    write_batched_field=True,
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
                        surface_pavg=surface_pavg,
                        pressure_rows=pressure_rows,
                        spl_rows=spl_rows,
                        impedance_rows=impedance_rows,
                        surface_pressure_rows=surface_pressure_rows,
                        native_diagnostics_rows=native_diagnostics_rows,
                        solver_log=solver_log,
                        completed_freqs=completed_freqs,
                    )
                    if config.progress_callback is not None:
                        config.progress_callback(i, len(freq_values), frequency_hz)
            else:
                for i, freq in enumerate(frequencies):
                    frequency_hz = float(freq)
                    logger.info(
                        "[%d/%d] %.1f Hz (native Metal %s assembly)",
                        i + 1,
                        len(frequencies),
                        frequency_hz,
                        config.metal_native_assembly_mode,
                    )
                    t_freq = time.time()
                    omega = 2.0 * np.pi * frequency_hz
                    k_real = omega / SPEED_OF_SOUND
                    neumann = _build_driver_neumann_coeffs(
                        dp0_space,
                        mesh.physical_tags,
                        omega,
                        config,
                        np.complex64,
                    )
                    system = session.assemble_solve_evaluate_standard_neumann_batch(
                        np.array([frequency_hz], dtype=np.float64),
                        np.array([k_real], dtype=np.float32),
                        neumann.reshape(1, -1),
                        field_points,
                        batch_id="all_observation_planes",
                        operation_id=(
                            f"assembly-solve-field-{i:04d}-"
                            f"{frequency_hz:.6g}hz-all-planes"
                        ),
                        source_tags=source_tags,
                        impedance_source_tag=impedance_source_tag,
                        write_surface_pressure=config.return_surface_pressure,
                    )[0]
                    elapsed = time.time() - t_freq
                    log_entry = _append_system_result(
                        frequency_hz=frequency_hz,
                        system=system,
                        backend="native_metal_resident_assembly_solve_field_single",
                        timing_s=elapsed,
                        mesh=mesh,
                        p1_space=p1_space,
                        source_tags=source_tags,
                        impedance_source_tag=impedance_source_tag,
                        n_planes=n_planes,
                        n_angles=n_angles,
                        on_axis_idx=on_axis_idx,
                        field_batch_complex=None,
                        surface_pavg=surface_pavg,
                        pressure_rows=pressure_rows,
                        spl_rows=spl_rows,
                        impedance_rows=impedance_rows,
                        surface_pressure_rows=surface_pressure_rows,
                        native_diagnostics_rows=native_diagnostics_rows,
                        solver_log=solver_log,
                        completed_freqs=completed_freqs,
                    )

                    if config.progress_callback is not None:
                        config.progress_callback(i, len(frequencies), frequency_hz)
                    callback_entry = {
                        **log_entry,
                        "observation_pressure_complex": pressure_rows[-1],
                        "observation_directivity_db": spl_rows[-1],
                        "observation_angles_deg": angles_deg,
                        "observation_planes": config.observation.planes,
                    }
                    if config.on_frequency_result(i, frequency_hz, callback_entry) is False:
                        logger.info("Early stop requested after %.1f Hz", frequency_hz)
                        break

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

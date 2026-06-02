"""Frequency sweep — serial and parallel execution."""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from ._constants import REFERENCE_PRESSURE, SPEED_OF_SOUND
from .backends import AssemblyBackendUnavailable, resolve_assembly_backend
from .bie import (
    FrequencyResult,
    _build_driver_neumann_coeffs,
    _evaluate_far_field,
    _compute_impedance,
    _operator_kwargs,
    _setup_function_spaces,
    compute_surface_pressure_avg,
    solve_single_frequency,
)
from .config import BIEFormulation, SolveConfig
from .mesh import LoadedMesh, make_pure_function_spaces, to_bempp_loaded_mesh
from .observation import ObservationFrame, build_observation_points
from .result import SolveResult

logger = logging.getLogger(__name__)


def _build_frequency_grid(config: SolveConfig) -> NDArray[np.float64]:
    if config.freq_spacing == "log":
        return np.geomspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)
    return np.linspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)


def should_route_native_metal(config: SolveConfig) -> bool:
    """Return true when an explicit experimental Metal request can own solving.

    The default solver remains Bempp/OpenCL. Native Metal is only selected for
    explicit experimental standard-Neumann sweeps that the package-owned helper
    currently supports.
    """
    if config.native_symmetry_plane is not None and (
        config.assembly_backend != "metal" or not config.experimental_metal_backend
    ):
        raise AssemblyBackendUnavailable(
            "native_symmetry_plane requires assembly_backend='metal' and "
            "experimental_metal_backend=True"
        )
    if config.assembly_backend != "metal" or not config.experimental_metal_backend:
        return False

    unsupported: list[str] = []
    if config.formulation is not BIEFormulation.STANDARD:
        unsupported.append("formulation must be STANDARD")
    if config.impedance_sources:
        unsupported.append("impedance_sources are not supported")
    if unsupported:
        reason = "Native Metal sweep unsupported: " + "; ".join(unsupported)
        if config.metal_backend_fallback == "error":
            raise AssemblyBackendUnavailable(reason)
        logger.warning("%s; falling back to Bempp/OpenCL.", reason)
        return False
    return True


def _read_complex_f32(
    real_path: Path,
    imag_path: Path,
    shape: tuple[int, ...],
) -> NDArray[np.complex64]:
    real = np.fromfile(real_path, dtype="<f4").reshape(shape)
    imag = np.fromfile(imag_path, dtype="<f4").reshape(shape)
    return np.ascontiguousarray(real + 1j * imag, dtype=np.complex64)


def _solve_dense_direct(matrix: NDArray, rhs: NDArray) -> NDArray[np.complex64]:
    try:
        import scipy.linalg

        pressure = scipy.linalg.solve(
            np.asarray(matrix, dtype=np.complex64),
            np.asarray(rhs, dtype=np.complex64),
            assume_a="gen",
            check_finite=False,
        )
    except ImportError:
        pressure = np.linalg.solve(
            np.asarray(matrix, dtype=np.complex64),
            np.asarray(rhs, dtype=np.complex64),
        )
    return np.ascontiguousarray(pressure, dtype=np.complex64)


def _evaluate_directivity(
    freq_results: list[FrequencyResult],
    obs_points: NDArray[np.float64],
    angles_deg: NDArray[np.float64],
    config: SolveConfig,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Evaluate far-field pressure at observation points for all frequencies.

    Returns:
        pressure_complex: (F, P, N_angles)
        spl_db: (F, P, N_angles) — normalised on-axis = 0 dB
    """
    n_freq = len(freq_results)
    n_planes, n_angles, _ = obs_points.shape

    pressure = np.zeros((n_freq, n_planes, n_angles), dtype=np.complex128)
    spl = np.full((n_freq, n_planes, n_angles), -120.0, dtype=np.float64)

    # On-axis index: the angle closest to 0 degrees
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))

    backend = resolve_assembly_backend(config).effective_backend
    op_kwargs = _operator_kwargs(backend, config.precision, config.opencl_device)

    for fi, fr in enumerate(freq_results):
        k_real = 2.0 * np.pi * fr.frequency_hz / SPEED_OF_SOUND

        p1 = fr.pressure_on_surface.space
        dp0 = fr.neumann_data.space

        for pi in range(n_planes):
            pts = obs_points[pi]  # (N_angles, 3)
            p_complex = _evaluate_far_field(
                p1, dp0,
                fr.pressure_on_surface,
                fr.neumann_data,
                k_real, pts, op_kwargs,
            )
            pressure[fi, pi, :] = p_complex

            amplitudes = np.abs(p_complex)
            spl_raw = np.where(
                amplitudes > 1e-15,
                20.0 * np.log10(amplitudes / REFERENCE_PRESSURE),
                -120.0,
            )
            # Normalise: on-axis (0 deg) = 0 dB
            spl[fi, pi, :] = spl_raw - spl_raw[on_axis_idx]

    return pressure, spl


def run_sweep_serial(
    mesh: LoadedMesh,
    frequencies: NDArray[np.float64],
    frame: ObservationFrame,
    config: SolveConfig,
) -> SolveResult:
    """Run frequency sweep in a single process."""
    t_total = time.time()

    obs_points, angles_deg = build_observation_points(frame, config.observation)

    p1_space, dp0_space = _setup_function_spaces(mesh.grid)

    source_tags = list(config.velocity_sources.keys())
    freq_results: list[FrequencyResult] = []
    surface_pavg: dict[int, list[complex]] = {tag: [] for tag in source_tags}
    completed_freqs: list[float] = []

    # Pre-compute op_kwargs for per-frequency far-field evaluation
    # (only used when on_frequency_result is set)
    has_callback = config.on_frequency_result is not None
    callback_pressure_rows: list[NDArray[np.complex128]] = []
    callback_spl_rows: list[NDArray[np.float64]] = []
    if has_callback:
        _backend = resolve_assembly_backend(config).effective_backend
        _ff_op_kwargs = _operator_kwargs(
            _backend, config.precision, config.opencl_device,
        )
        on_axis_idx = int(np.argmin(np.abs(angles_deg)))

    for i, freq in enumerate(frequencies):
        logger.info("[%d/%d] %.1f Hz", i + 1, len(frequencies), freq)
        fr = solve_single_frequency(
            mesh.grid, mesh.physical_tags, freq, config,
            p1_space=p1_space, dp0_space=dp0_space,
        )
        freq_results.append(fr)
        completed_freqs.append(float(freq))

        # Surface pressure average per source tag
        pavg = compute_surface_pressure_avg(
            mesh.grid, fr.pressure_on_surface,
            mesh.physical_tags, p1_space, source_tags,
        )
        for tag in source_tags:
            surface_pavg[tag].append(pavg[tag])

        log_entry = {
            "frequency_hz": fr.frequency_hz,
            "iterations": fr.iterations,
            "timing_s": fr.timing_s,
            "impedance": fr.impedance,
        }

        # When the callback is set, evaluate per-frequency directivity
        # so the caller (e.g. optimizer watcher) can do partial scoring.
        if has_callback:
            k_real = 2.0 * np.pi * fr.frequency_hz / SPEED_OF_SOUND
            n_planes = obs_points.shape[0]
            n_angles = obs_points.shape[1]
            per_freq_pressure = np.zeros(
                (n_planes, n_angles), dtype=np.complex128,
            )
            per_freq_spl = np.full((n_planes, n_angles), -120.0, dtype=np.float64)
            for pi in range(n_planes):
                p_complex = _evaluate_far_field(
                    p1_space, dp0_space,
                    fr.pressure_on_surface, fr.neumann_data,
                    k_real, obs_points[pi], _ff_op_kwargs,
                )
                per_freq_pressure[pi, :] = p_complex
                amplitudes = np.abs(p_complex)
                spl_raw = np.where(
                    amplitudes > 1e-15,
                    20.0 * np.log10(amplitudes / REFERENCE_PRESSURE),
                    -120.0,
                )
                per_freq_spl[pi, :] = spl_raw - spl_raw[on_axis_idx]
            callback_pressure_rows.append(per_freq_pressure)
            callback_spl_rows.append(per_freq_spl)
            log_entry["observation_spl_db"] = per_freq_spl
            log_entry["observation_angles_deg"] = angles_deg
            log_entry["observation_planes"] = config.observation.planes

        # Progress callback
        if config.progress_callback is not None:
            config.progress_callback(i, len(frequencies), float(freq))

        # Early-stopping callback
        if has_callback:
            if not config.on_frequency_result(i, float(freq), log_entry):
                logger.info("Early stop requested after %.1f Hz", freq)
                break

    t_solve = time.time() - t_total

    # Trim frequencies to only those actually completed (for early stopping)
    actual_freqs = np.array(completed_freqs, dtype=np.float64)

    if has_callback and len(callback_pressure_rows) == len(freq_results):
        logger.info("Reusing callback directivity rows for final result.")
        t_dir = 0.0
        pressure = np.stack(callback_pressure_rows, axis=0)
        spl = np.stack(callback_spl_rows, axis=0)
    else:
        logger.info("Evaluating directivity at %d observation points...",
                    obs_points.shape[1] * obs_points.shape[0])
        t_dir = time.time()
        pressure, spl = _evaluate_directivity(
            freq_results, obs_points, angles_deg, config,
        )
        t_dir = time.time() - t_dir

    impedance = np.array(
        [fr.impedance for fr in freq_results], dtype=np.complex128,
    )

    solver_log = [
        {
            "frequency_hz": fr.frequency_hz,
            "iterations": fr.iterations,
            "timing_s": fr.timing_s,
        }
        for fr in freq_results
    ]

    # Build surface_pressure_avg arrays
    sp_avg: dict[int, np.ndarray] = {}
    for tag in source_tags:
        sp_avg[tag] = np.array(surface_pavg[tag], dtype=np.complex128)

    return SolveResult(
        frequencies_hz=actual_freqs,
        pressure_complex=pressure,
        spl_db=spl,
        impedance=impedance,
        observation_angles_deg=angles_deg,
        observation_points=obs_points,
        observation_planes=config.observation.planes,
        config=config,
        mesh_info=mesh.info,
        timings={
            "solve_s": t_solve,
            "directivity_s": t_dir,
            "total_s": time.time() - t_total,
        },
        solver_log=solver_log,
        surface_pressure_avg=sp_avg if sp_avg else None,
    )


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
    if not should_route_native_metal(config):
        return _run_bempp_fallback(mesh, frequencies, frame, config)

    try:
        from .metal.geometry import build_metal_geometry_buffers
        from .metal.native import MetalNativeStandardSession, discover_native_runtime
    except Exception as exc:  # pragma: no cover - import/runtime specific.
        if config.metal_backend_fallback == "error":
            raise AssemblyBackendUnavailable(
                f"Native Metal helper could not be imported: {exc}"
            ) from exc
        logger.warning(
            "Native Metal helper could not be imported: %s; falling back to Bempp/OpenCL.",
            exc,
        )
        return _run_bempp_fallback(mesh, frequencies, frame, config)

    runtime = discover_native_runtime(run_smoke_test=True)
    if not runtime.available:
        reason = "; ".join(runtime.unavailable_reasons)
        if config.metal_backend_fallback == "error":
            raise AssemblyBackendUnavailable(reason)
        logger.warning(
            "Native Metal helper unavailable: %s; falling back to Bempp/OpenCL.",
            reason,
        )
        return _run_bempp_fallback(mesh, frequencies, frame, config)

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
    solver_log: list[dict] = []
    completed_freqs: list[float] = []
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))

    previous_assembly_mode = os.environ.get(
        "HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE"
    )
    previous_field_mode = os.environ.get("HORNLAB_SOLVER_METAL_NATIVE_FIELD_MODE")
    previous_threads_per_group = os.environ.get(
        "HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP"
    )
    per_kernel_threadgroup_env = {
        "HORNLAB_SOLVER_METAL_NATIVE_MATRIX_THREADS_PER_GROUP": (
            config.metal_native_matrix_threads_per_group
        ),
        "HORNLAB_SOLVER_METAL_NATIVE_RHS_THREADS_PER_GROUP": (
            config.metal_native_rhs_threads_per_group
        ),
        "HORNLAB_SOLVER_METAL_NATIVE_DUFFY_THREADS_PER_GROUP": (
            config.metal_native_duffy_threads_per_group
        ),
        "HORNLAB_SOLVER_METAL_NATIVE_FIELD_THREADS_PER_GROUP": (
            config.metal_native_field_threads_per_group
        ),
    }
    previous_per_kernel_threadgroups = {
        name: os.environ.get(name) for name in per_kernel_threadgroup_env
    }
    os.environ["HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE"] = (
        config.metal_native_assembly_mode
    )
    if previous_field_mode is None:
        os.environ["HORNLAB_SOLVER_METAL_NATIVE_FIELD_MODE"] = "optimized"
    if config.metal_native_threads_per_group is not None:
        os.environ["HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP"] = str(
            config.metal_native_threads_per_group
        )
    for name, value in per_kernel_threadgroup_env.items():
        if value is not None:
            os.environ[name] = str(value)
    try:
        with MetalNativeStandardSession.create_session(
            geometry_buffers=geometry_buffers,
            symmetry_plane=config.native_symmetry_plane,
        ) as session:
            if config.on_frequency_result is None:
                freq_values = np.asarray(frequencies, dtype=np.float64)
                k_values = (2.0 * np.pi * freq_values / SPEED_OF_SOUND).astype(
                    np.float32,
                )
                neumann_rows = np.stack(
                    [
                        _build_driver_neumann_coeffs(
                            dp0_space,
                            mesh.physical_tags,
                            2.0 * np.pi * float(freq),
                            config,
                            np.complex64,
                        )
                        for freq in freq_values
                    ],
                    axis=0,
                )

                n_planes, n_angles, _ = obs_points.shape
                field_points = np.ascontiguousarray(
                    obs_points.reshape(n_planes * n_angles, 3).T,
                    dtype=np.float32,
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
                    impedance_source_tag=min(config.velocity_sources.keys(), default=2),
                    write_surface_pressure=False,
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
                    if (
                        system.impedance is not None
                        and system.surface_pressure_avg is not None
                    ):
                        impedance = system.impedance
                        pavg = system.surface_pressure_avg
                    elif (
                        system.pressure_real_f32 is not None
                        and system.pressure_imag_f32 is not None
                    ):
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
                            source_tag=min(config.velocity_sources.keys(), default=2),
                        )
                        pavg = compute_surface_pressure_avg(
                            mesh.grid,
                            p_surface,
                            mesh.physical_tags,
                            p1_space,
                            source_tags,
                        )
                    else:
                        raise RuntimeError(
                            "native solve-field result did not include pressure "
                            "reductions or surface pressure files"
                        )
                    for tag in source_tags:
                        surface_pavg[tag].append(pavg[tag])
                    impedance_rows.append(impedance)
                    completed_freqs.append(frequency_hz)
                    solver_log.append(
                        {
                            "frequency_hz": frequency_hz,
                            "iterations": None,
                            "timing_s": 0.0,
                            "backend": "native_metal_resident_assembly_solve_field_batch",
                            "assembly_s": float(system.assembly_s),
                            "dense_solve_s": float(system.dense_solve_s),
                            "field_s": float(system.field_s),
                            "impedance": impedance,
                        }
                    )
                    if config.progress_callback is not None:
                        config.progress_callback(i, len(freq_values), frequency_hz)

                    per_freq_pressure = np.zeros(
                        (n_planes, n_angles),
                        dtype=np.complex128,
                    )
                    per_freq_spl = np.full(
                        (n_planes, n_angles),
                        -120.0,
                        dtype=np.float64,
                    )
                    if field_batch_complex is not None:
                        if system.field_row_index is None:
                            raise RuntimeError(
                                "mixed batched and per-case native field results"
                            )
                        field_complex = field_batch_complex[system.field_row_index]
                    else:
                        field_complex = _read_complex_f32(
                            Path(system.field_real_f32),
                            Path(system.field_imag_f32),
                            tuple(system.field_shape),
                        ).astype(np.complex128)
                    field_complex = field_complex.reshape(n_planes, n_angles)
                    for pi in range(n_planes):
                        p_complex = field_complex[pi]
                        per_freq_pressure[pi, :] = p_complex
                        amplitudes = np.abs(p_complex)
                        spl_raw = np.where(
                            amplitudes > 1e-15,
                            20.0 * np.log10(amplitudes / REFERENCE_PRESSURE),
                            -120.0,
                        )
                        per_freq_spl[pi, :] = spl_raw - spl_raw[on_axis_idx]
                    pressure_rows.append(per_freq_pressure)
                    spl_rows.append(per_freq_spl)
                    solver_log[i]["timing_s"] = (
                        float(solver_log[i]["assembly_s"])
                        + float(solver_log[i]["dense_solve_s"])
                        + float(solver_log[i]["field_s"])
                    )
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
                    n_planes, n_angles, _ = obs_points.shape
                    field_points = np.ascontiguousarray(
                        obs_points.reshape(n_planes * n_angles, 3).T,
                        dtype=np.float32,
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
                        impedance_source_tag=min(config.velocity_sources.keys(), default=2),
                        write_surface_pressure=False,
                    )[0]
                    if (
                        system.impedance is not None
                        and system.surface_pressure_avg is not None
                    ):
                        impedance = system.impedance
                        pavg = system.surface_pressure_avg
                    elif (
                        system.pressure_real_f32 is not None
                        and system.pressure_imag_f32 is not None
                    ):
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
                            source_tag=min(config.velocity_sources.keys(), default=2),
                        )
                        pavg = compute_surface_pressure_avg(
                            mesh.grid,
                            p_surface,
                            mesh.physical_tags,
                            p1_space,
                            source_tags,
                        )
                    else:
                        raise RuntimeError(
                            "native solve-field result did not include pressure "
                            "reductions or surface pressure files"
                        )
                    for tag in source_tags:
                        surface_pavg[tag].append(pavg[tag])

                    per_freq_pressure = np.zeros(
                        (n_planes, n_angles),
                        dtype=np.complex128,
                    )
                    per_freq_spl = np.full(
                        (n_planes, n_angles),
                        -120.0,
                        dtype=np.float64,
                    )
                    field_complex = _read_complex_f32(
                        Path(system.field_real_f32),
                        Path(system.field_imag_f32),
                        tuple(system.field_shape),
                    ).astype(np.complex128)
                    field_complex = field_complex.reshape(n_planes, n_angles)
                    for pi in range(n_planes):
                        p_complex = field_complex[pi]
                        per_freq_pressure[pi, :] = p_complex
                        amplitudes = np.abs(p_complex)
                        spl_raw = np.where(
                            amplitudes > 1e-15,
                            20.0 * np.log10(amplitudes / REFERENCE_PRESSURE),
                            -120.0,
                        )
                        per_freq_spl[pi, :] = spl_raw - spl_raw[on_axis_idx]

                    elapsed = time.time() - t_freq
                    log_entry = {
                        "frequency_hz": frequency_hz,
                        "iterations": None,
                        "timing_s": elapsed,
                        "backend": "native_metal_resident_assembly_solve_field_single",
                        "assembly_s": float(system.assembly_s),
                        "dense_solve_s": float(system.dense_solve_s),
                        "field_s": float(system.field_s),
                        "impedance": impedance,
                    }
                    solver_log.append(log_entry)
                    completed_freqs.append(frequency_hz)
                    pressure_rows.append(per_freq_pressure)
                    spl_rows.append(per_freq_spl)
                    impedance_rows.append(impedance)

                    if config.progress_callback is not None:
                        config.progress_callback(i, len(frequencies), frequency_hz)
                    callback_entry = {
                        **log_entry,
                        "observation_spl_db": per_freq_spl,
                        "observation_angles_deg": angles_deg,
                        "observation_planes": config.observation.planes,
                    }
                    if not config.on_frequency_result(i, frequency_hz, callback_entry):
                        logger.info("Early stop requested after %.1f Hz", frequency_hz)
                        break
    finally:
        if previous_assembly_mode is None:
            os.environ.pop(
                "HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE",
                None,
            )
        else:
            os.environ["HORNLAB_SOLVER_METAL_NATIVE_ASSEMBLY_MODE"] = (
                previous_assembly_mode
            )
        if previous_field_mode is None:
            os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_FIELD_MODE", None)
        else:
            os.environ["HORNLAB_SOLVER_METAL_NATIVE_FIELD_MODE"] = previous_field_mode
        if previous_threads_per_group is None:
            os.environ.pop("HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP", None)
        else:
            os.environ["HORNLAB_SOLVER_METAL_NATIVE_THREADS_PER_GROUP"] = (
                previous_threads_per_group
            )
        for name, previous_value in previous_per_kernel_threadgroups.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value

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
        spl_db=np.stack(spl_rows, axis=0),
        impedance=np.array(impedance_rows, dtype=np.complex128),
        observation_angles_deg=angles_deg,
        observation_points=obs_points,
        observation_planes=config.observation.planes,
        config=config,
        mesh_info=mesh.info,
        timings=timings,
        solver_log=solver_log,
        surface_pressure_avg=sp_avg if sp_avg else None,
    )


def _run_bempp_fallback(
    mesh: LoadedMesh,
    frequencies: NDArray[np.float64],
    frame: ObservationFrame,
    config: SolveConfig,
) -> SolveResult:
    return run_sweep_serial(to_bempp_loaded_mesh(mesh), frequencies, frame, config)


def run_sweep_parallel(
    mesh: LoadedMesh,
    frequencies: NDArray[np.float64],
    frame: ObservationFrame,
    config: SolveConfig,
    worker_count: int,
) -> SolveResult:
    """Run frequency sweep across multiple processes.

    Each worker solves a chunk of frequencies, evaluates far-field pressure
    at observation points, and returns the results. This avoids shipping
    bempp GridFunction objects across process boundaries.

    Callbacks (progress_callback, on_frequency_result) are not supported
    in parallel mode — they are not picklable across process boundaries.
    """
    if config.progress_callback is not None or config.on_frequency_result is not None:
        raise ValueError(
            "progress_callback and on_frequency_result are not supported in "
            "parallel mode (workers > 1). Use serial mode or set workers=1."
        )

    t_total = time.time()
    obs_points, angles_deg = build_observation_points(frame, config.observation)

    chunks = np.array_split(frequencies, min(worker_count, len(frequencies)))
    chunk_indices = np.array_split(
        np.arange(len(frequencies)), min(worker_count, len(frequencies)),
    )

    n_planes, n_angles, _ = obs_points.shape
    pressure_all = np.zeros(
        (len(frequencies), n_planes, n_angles), dtype=np.complex128,
    )
    spl_all = np.full(
        (len(frequencies), n_planes, n_angles), -120.0, dtype=np.float64,
    )
    impedance_all = np.zeros(len(frequencies), dtype=np.complex128)
    solver_log: list[dict] = [{}] * len(frequencies)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=len(chunks), mp_context=ctx,
    ) as executor:
        futures = {}
        for ci, (chunk_freqs, chunk_idx) in enumerate(
            zip(chunks, chunk_indices),
        ):
            fut = executor.submit(
                _worker_solve_chunk,
                mesh_grid_verts=np.array(mesh.grid.vertices),
                mesh_grid_elems=np.array(mesh.grid.elements),
                physical_tags=mesh.physical_tags,
                frequencies=chunk_freqs,
                obs_points=obs_points,
                angles_deg=angles_deg,
                config=config,
            )
            futures[fut] = chunk_idx

        for fut in as_completed(futures):
            idx = futures[fut]
            chunk_pressure, chunk_spl, chunk_imp, chunk_log = fut.result()
            for local_i, global_i in enumerate(idx):
                pressure_all[global_i] = chunk_pressure[local_i]
                spl_all[global_i] = chunk_spl[local_i]
                impedance_all[global_i] = chunk_imp[local_i]
                solver_log[global_i] = chunk_log[local_i]
            logger.info(
                "Completed chunk: %d frequencies", len(idx),
            )

    return SolveResult(
        frequencies_hz=frequencies,
        pressure_complex=pressure_all,
        spl_db=spl_all,
        impedance=impedance_all,
        observation_angles_deg=angles_deg,
        observation_points=obs_points,
        observation_planes=config.observation.planes,
        config=config,
        mesh_info=mesh.info,
        timings={"total_s": time.time() - t_total},
        solver_log=solver_log,
    )


def _worker_solve_chunk(
    mesh_grid_verts,
    mesh_grid_elems,
    physical_tags,
    frequencies,
    obs_points,
    angles_deg,
    config,
):
    """Worker function: reconstruct grid, solve, evaluate far-field, return arrays."""
    import bempp_cl.api as bempp_api

    from ._constants import REFERENCE_PRESSURE, SPEED_OF_SOUND
    from .bie import (
        _evaluate_far_field,
        _operator_kwargs,
        solve_single_frequency,
    )

    grid = bempp_api.Grid(mesh_grid_verts, mesh_grid_elems)
    from .bie import _setup_function_spaces
    p1_space, dp0_space = _setup_function_spaces(grid)

    n_planes, n_angles, _ = obs_points.shape
    pressure = np.zeros((len(frequencies), n_planes, n_angles), dtype=np.complex128)
    spl = np.full((len(frequencies), n_planes, n_angles), -120.0)
    impedance = np.zeros(len(frequencies), dtype=np.complex128)
    log_entries = []

    # On-axis index: the angle closest to 0 degrees
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))

    backend = resolve_assembly_backend(config).effective_backend
    op_kwargs = _operator_kwargs(backend, config.precision, config.opencl_device)

    for i, freq in enumerate(frequencies):
        fr = solve_single_frequency(
            grid, physical_tags, freq, config,
            p1_space=p1_space, dp0_space=dp0_space,
        )
        impedance[i] = fr.impedance
        log_entries.append({
            "frequency_hz": fr.frequency_hz,
            "iterations": fr.iterations,
            "timing_s": fr.timing_s,
        })

        k_real = 2.0 * np.pi * freq / SPEED_OF_SOUND
        for pi in range(n_planes):
            pts = obs_points[pi]
            p_complex = _evaluate_far_field(
                p1_space, dp0_space,
                fr.pressure_on_surface, fr.neumann_data,
                k_real, pts, op_kwargs,
            )
            pressure[i, pi, :] = p_complex
            amplitudes = np.abs(p_complex)
            spl_raw = np.where(
                amplitudes > 1e-15,
                20.0 * np.log10(amplitudes / REFERENCE_PRESSURE),
                -120.0,
            )
            spl[i, pi, :] = spl_raw - spl_raw[on_axis_idx]

    return pressure, spl, impedance, log_entries

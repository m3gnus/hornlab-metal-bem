"""Native Metal frequency sweep execution."""
from __future__ import annotations

import logging
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
    _compute_impedance,
    compute_surface_pressure_avg,
)
from .config import BIEFormulation, SolveConfig
from .mesh import LoadedMesh, make_pure_function_spaces
from .observation import ObservationFrame, build_observation_points
from .result import SolveResult

logger = logging.getLogger(__name__)


def _build_frequency_grid(config: SolveConfig) -> NDArray[np.float64]:
    if config.freq_spacing == "log":
        return np.geomspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)
    return np.linspace(config.freq_min_hz, config.freq_max_hz, config.freq_count)


def should_route_native_metal(config: SolveConfig) -> bool:
    """Return true when the native Metal path supports the solve config."""
    if config.native_symmetry_plane is not None and (
        config.assembly_backend != "metal" or not config.experimental_metal_backend
    ):
        raise AssemblyBackendUnavailable(
            "native_symmetry_plane requires assembly_backend='metal' and "
            "experimental_metal_backend=True"
        )
    if config.assembly_backend != "metal" or not config.experimental_metal_backend:
        raise AssemblyBackendUnavailable(
            "Native Metal requires assembly_backend='metal' and "
            "experimental_metal_backend=True"
        )

    unsupported: list[str] = []
    if config.formulation is not BIEFormulation.STANDARD:
        unsupported.append("formulation must be STANDARD")
    if config.impedance_sources:
        unsupported.append("impedance_sources are not supported")
    if unsupported:
        reason = "Native Metal sweep unsupported: " + "; ".join(unsupported)
        raise AssemblyBackendUnavailable(reason)
    return True


def _read_complex_f32(
    real_path: Path,
    imag_path: Path,
    shape: tuple[int, ...],
) -> NDArray[np.complex64]:
    real = np.fromfile(real_path, dtype="<f4").reshape(shape)
    imag = np.fromfile(imag_path, dtype="<f4").reshape(shape)
    return np.ascontiguousarray(real + 1j * imag, dtype=np.complex64)


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

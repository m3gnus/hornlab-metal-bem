"""Axisymmetric body-of-revolution acoustic BEM solver.

This module implements the m=0 solver as a NumPy/SciPy sibling to the 3D native
Metal path, with adaptive Metal acceleration for large ordinary field
quadratures. The meridian discretization uses DP0 constants on straight
generating-curve segments with midpoint collocation. Segment integrals carry the
full ring surface measure ``rho ds``; no ``1 / rho`` factors are introduced, so
axis nodes are regular for m=0.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import logging
import math
import os
import platform
import secrets
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.special import ellipe, ellipk

# Re-exported, not used here: harnesses read the shipped default off this
# module. The solve itself takes c from SolveConfig.speed_of_sound.
from ._constants import SPEED_OF_SOUND  # noqa: F401
from .bie import (
    _taper_values,
    integrate_driven_surface_power,
    normal_velocity_from_driver_neumann,
)
from .config import (
    AnnularProfile,
    AxialProfile,
    BIEFormulation,
    CallableProfile,
    NormalProfile,
    PerFaceProfile,
    SolveConfig,
    SourceMotion,
    TaperProfile,
    VelocityMode,
    _resolve_velocity_sources,
)
from .observation import ObservationFrame, build_observation_points
from .result import MeshInfo, SolveResult
from .sweep import (
    _build_frequency_grid,
    _directivity_from_pressure,
    _impedance_sources_for_frequencies,
    _resolve_sphere_observation,
    _sphere_power_from_log,
    _sphere_pressure_from_log,
)

logger = logging.getLogger(__name__)


class CircSymCancelled(RuntimeError):
    """Raised when a CircSym intra-case continuation callback returns False."""


class _CircSymMetalAccelerationError(RuntimeError):
    """Marks a native acceleration failure without masking callback errors."""


def _check_circsym_continue(
    config_or_callback: SolveConfig | Callable[[], bool | None] | None,
) -> None:
    """Run a cancellation checkpoint without masking caller-owned exceptions."""
    callback = (
        config_or_callback.should_continue
        if isinstance(config_or_callback, SolveConfig)
        else config_or_callback
    )
    if callback is not None and callback() is False:
        raise CircSymCancelled("CircSym solve cancelled")


_AZIMUTH_POINTS_MIN = 64
_AZIMUTH_POINTS_PER_KRHO = 4.0
_CIRCSYM_AZIMUTH_POINTS_MIN_ENV = "HORNLAB_CIRCSYM_AZIMUTH_POINTS_MIN"
_LINE_QUAD_ORDER = 16
_SINGULAR_LINE_QUAD_ORDER = 24
_GRADED_POWER = 3.0
_FIELD_KERNEL_BLOCK_ELEMENTS = 8_000_000
_FIELD_KERNEL_MAX_TARGET_BLOCK = 64
_FIELD_KERNEL_PARALLEL_TARGET_BLOCK = 8
_FIELD_KERNEL_MAX_WORKERS = 8
_CIRCSYM_METAL_FIELD_MIN_TERMS = 2_000_000
_CIRCSYM_FIELD_BACKEND_ENV = "HORNLAB_CIRCSYM_FIELD_BACKEND"
_CIRCSYM_METAL_ASSEMBLY_MIN_TERMS = 80_000_000
_CIRCSYM_ASSEMBLY_BACKEND_ENV = "HORNLAB_CIRCSYM_ASSEMBLY_BACKEND"
_CIRCSYM_CPU_REMAINDER_BACKEND_ENV = "HORNLAB_CIRCSYM_CPU_REMAINDER_BACKEND"
_CIRCSYM_CPU_FIELD_BACKEND_ENV = "HORNLAB_CIRCSYM_CPU_FIELD_BACKEND"
_CIRCSYM_C_KERNEL_BUILD_SCHEMA = "circsym-c-kernel-v2"
_CIRCSYM_C_KERNEL_COMPILE_ARGS = ("-O3", "-fPIC", "-pthread")
_CIRCSYM_C_KERNEL_LINK_ARGS = ("-lm",)
_CIRCSYM_C_KERNEL_ORPHAN_GRACE_SECONDS = 300.0
_CIRCSYM_C_KERNEL_GC_MAX_REMOVALS = 8
_CIRCSYM_C_KERNEL_GC_MAX_REMOVAL_BYTES = 64 * 1024 * 1024
_CIRCSYM_C_KERNEL_CACHE_MAX_ARTIFACTS = 64
_CIRCSYM_C_KERNEL_CACHE_MAX_BYTES = 256 * 1024 * 1024
_ASSEMBLY_KERNEL_BLOCK_ELEMENTS = 3_000_000
_ASSEMBLY_KERNEL_MAX_TARGET_BLOCK = 16
_ASSEMBLY_KERNEL_PARALLEL_TARGET_BLOCK = 5
_CANCELLATION_PAIR_BLOCK = 16
_circsym_metal_field_auto_failure: str | None = None
_circsym_metal_assembly_auto_failure: str | None = None
_circsym_c_kernel_failure: str | None = None
_circsym_numba_kernel_failure: str | None = None


@dataclass
class MeridianMesh:
    """Polyline meridian for an axisymmetric body of revolution.

    ``nodes`` is an ``(N, 2)`` float64 array with columns ``(rho, z)`` in metres,
    where ``rho >= 0``. ``segments`` is an ``(M, 2)`` int array of node indices.
    ``physical_tags`` is one integer tag per segment. ``normals`` is an
    ``(M, 2)`` float64 array of outward segment normals in cylindrical
    components ``(n_rho, n_z)``.

    When normals are derived from geometry, the convention is that the physical
    exterior is on the right side of the directed polyline:
    ``n = (-dz, drho) / length``. A sphere meridian therefore runs from the
    +z pole to the -z pole along the outer profile; a baffled piston radius runs
    from the axis out to the rim to get a +z normal.
    """

    nodes: NDArray[np.float64]
    segments: NDArray[np.int32]
    physical_tags: NDArray[np.int32]
    normals: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=np.float64)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("MeridianMesh.nodes must have shape (N, 2) as (rho, z)")
        if nodes.shape[0] < 2:
            raise ValueError("MeridianMesh requires at least two nodes")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("MeridianMesh.nodes must be finite")
        if np.any(nodes[:, 0] < -1e-15):
            raise ValueError("MeridianMesh rho coordinates must be non-negative")
        nodes = nodes.copy()
        nodes[:, 0] = np.maximum(nodes[:, 0], 0.0)

        segments = np.asarray(self.segments, dtype=np.int32)
        if segments.ndim != 2 or segments.shape[1] != 2:
            raise ValueError("MeridianMesh.segments must have shape (M, 2)")
        if segments.shape[0] == 0:
            raise ValueError("MeridianMesh requires at least one segment")
        if np.any(segments < 0) or np.any(segments >= nodes.shape[0]):
            raise ValueError("MeridianMesh.segments contain invalid node indices")

        tags = np.asarray(self.physical_tags, dtype=np.int32).reshape(-1)
        if tags.shape[0] != segments.shape[0]:
            raise ValueError("physical_tags length must equal segment count")

        p0 = nodes[segments[:, 0]]
        p1 = nodes[segments[:, 1]]
        delta = p1 - p0
        lengths = np.linalg.norm(delta, axis=1)
        if np.any(lengths <= 1e-15):
            raise ValueError("MeridianMesh segments must have non-zero length")

        if self.normals is None:
            normals = np.column_stack((-delta[:, 1], delta[:, 0])) / lengths[:, None]
        else:
            normals = np.asarray(self.normals, dtype=np.float64)
            if normals.shape != (segments.shape[0], 2):
                raise ValueError("normals must have shape (M, 2)")
            if not np.all(np.isfinite(normals)):
                raise ValueError("normals must be finite")
            n_norm = np.linalg.norm(normals, axis=1)
            if np.any(n_norm <= 1e-15):
                raise ValueError("normals must be non-zero")
            normals = normals / n_norm[:, None]

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "physical_tags", tags)
        object.__setattr__(self, "normals", normals)

    @classmethod
    def from_polyline(
        cls,
        points_2xN: NDArray[np.float64] | Iterable[Iterable[float]],
        tags: int | Iterable[int] | NDArray[np.int32],
        *,
        close: bool = False,
    ) -> "MeridianMesh":
        """Build a meridian from ordered ``(rho, z)`` polyline points.

        The input may be shaped either ``(N, 2)`` or ``(2, N)``. ``tags`` may be
        a scalar tag applied to every segment, or one tag per generated segment.
        ``close=True`` adds the final segment from the last point to the first;
        it is not needed for the common pole-to-pole sphere meridian.
        """
        pts = np.asarray(points_2xN, dtype=np.float64)
        if pts.ndim != 2:
            raise ValueError("points_2xN must be a 2D array")
        if pts.shape[1] == 2:
            nodes = pts
        elif pts.shape[0] == 2:
            nodes = pts.T
        else:
            raise ValueError("points_2xN must have shape (N, 2) or (2, N)")
        n = nodes.shape[0]
        if n < 2:
            raise ValueError("at least two polyline points are required")
        segs = [[i, i + 1] for i in range(n - 1)]
        if close:
            segs.append([n - 1, 0])
        segments = np.asarray(segs, dtype=np.int32)

        if np.isscalar(tags):
            tag_arr = np.full(segments.shape[0], int(tags), dtype=np.int32)
        else:
            tag_arr = np.asarray(tags, dtype=np.int32).reshape(-1)
            if tag_arr.size != segments.shape[0]:
                raise ValueError("tags must be scalar or have one value per segment")
        return cls(nodes=nodes, segments=segments, physical_tags=tag_arr)

    @property
    def segment_count(self) -> int:
        return int(self.segments.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    def segment_geometry(self) -> SimpleNamespace:
        p0 = self.nodes[self.segments[:, 0]]
        p1 = self.nodes[self.segments[:, 1]]
        delta = p1 - p0
        lengths = np.linalg.norm(delta, axis=1)
        midpoints = 0.5 * (p0 + p1)
        rho_mid = midpoints[:, 0]
        area_weights = 2.0 * np.pi * rho_mid * lengths
        return SimpleNamespace(
            p0=p0,
            p1=p1,
            delta=delta,
            lengths=lengths,
            midpoints=midpoints,
            rho_mid=rho_mid,
            z_mid=midpoints[:, 1],
            area_weights=area_weights,
        )


def run_sweep_circsym(
    meridian: MeridianMesh,
    frequencies: NDArray[np.float64] | Iterable[float],
    config: SolveConfig,
) -> SolveResult:
    """Run the pure-Python axisymmetric m=0 BEM sweep.

    Unknown pressure and prescribed Neumann data are DP0 constants on meridian
    segments. The dense complex128 system is solved by LU for square systems and
    by scaled least squares when CHIEF rows are appended.
    """
    if not isinstance(meridian, MeridianMesh):
        raise TypeError("meridian must be a MeridianMesh")
    if config.return_surface_traces:
        raise ValueError(
            "return_surface_traces is available only for full-3D native Metal solves"
        )
    if config.ground_plane is not None:
        # The axisymmetric path has its own image plane, circsym_baffle_z, and
        # accepts only one normal to the axis of revolution. Silently ignoring
        # ground_plane here would return free-field physics for a half-space
        # request -- the same failure this module already refuses to make for a
        # mis-tagged infinite-baffle solve below.
        raise ValueError(
            f"ground_plane={config.ground_plane!r} is a full-3D native Metal "
            "option; the axisymmetric solver takes a same-sign rigid image "
            "plane through circsym_baffle_z instead, which must be normal to "
            "the axis of revolution"
        )
    if config.circsym_aperture_tag is not None:
        # Dispatch to the coupled infinite-baffle solve whenever an aperture tag
        # is requested. run_sweep_coupled_ib validates that the tag is present and
        # raises a clear error if not; falling through to the free-space sweep
        # here would silently return free-standing physics for a mis-tagged IB
        # request instead of failing loudly.
        return run_sweep_coupled_ib(meridian, frequencies, config)
    frequencies_arr = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if frequencies_arr.size == 0:
        raise ValueError("frequencies must contain at least one value")
    if not np.all(np.isfinite(frequencies_arr)) or np.any(frequencies_arr <= 0.0):
        raise ValueError("frequencies must be finite and positive")

    mesh_tags = {int(tag) for tag in np.unique(meridian.physical_tags)}
    missing_tags = sorted(set(int(tag) for tag in config.velocity_sources) - mesh_tags)
    if missing_tags:
        raise ValueError(
            f"velocity_sources tags {missing_tags} are not present in the meridian; "
            f"available physical tags: {sorted(mesh_tags)}"
        )
    _validate_closed_or_baffled_meridian(meridian, config.circsym_baffle_z)

    t_total = time.time()
    geom = meridian.segment_geometry()
    frame = _infer_circsym_frame(meridian, config)
    obs_points, angles_deg = build_observation_points(frame, config.observation)
    sphere_points_arr, sphere_theta_deg, sphere_phi_deg = _resolve_sphere_observation(
        frame, config.observation
    )
    n_sphere = 0 if sphere_points_arr is None else int(sphere_points_arr.shape[0])
    sphere_evaluation_points, sphere_evaluation_inverse = (
        _axisymmetric_sphere_evaluation_targets(
            sphere_points_arr,
            sphere_theta_deg,
        )
    )
    n_sphere_evaluation = (
        0
        if sphere_evaluation_points is None
        else int(sphere_evaluation_points.shape[0])
    )
    source_tags = list(config.velocity_sources.keys())
    source_scale = _build_source_segment_scale(meridian, config, frame)
    impedance_sources_arg = _impedance_sources_for_frequencies(
        meridian.physical_tags, frequencies_arr, config
    )
    per_case_impedance = isinstance(impedance_sources_arg, list)

    n_planes, n_angles, _ = obs_points.shape
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))
    rho_max = float(np.max(meridian.nodes[:, 0])) if meridian.nodes.size else 0.0
    mesh_max_segment = float(np.max(geom.lengths))
    impedance_source_tag = min(config.velocity_sources.keys(), default=2)
    boundary_free_terms = _boundary_free_terms(meridian, config.circsym_baffle_z)
    n_psi_by_frequency = np.asarray(
        [
            _azimuth_order(_complex_wavenumber(float(frequency_hz), config), rho_max)
            for frequency_hz in frequencies_arr
        ],
        dtype=np.int32,
    )
    k_by_frequency = np.asarray(
        [
            _complex_wavenumber(float(frequency_hz), config)
            for frequency_hz in frequencies_arr
        ],
        dtype=np.complex128,
    )
    # Keep the quadrature order frequency-local. The assembly cache is keyed by
    # n_psi, so repeated orders still reuse geometry without forcing low
    # frequencies to pay the sweep's high-frequency field/assembly cost.
    n_psi_use_counts = _int_value_counts(n_psi_by_frequency)
    assembly_cache = _BoundaryAssemblyGeometryCache(
        meridian,
        config.circsym_baffle_z,
        geom=geom,
        reusable_n_psi=set(n_psi_use_counts),
        n_psi_use_counts=n_psi_use_counts,
        cache_single_use=False,
    )
    assembly_cache.prepare_metal_remainder_batch(
        k_by_frequency,
        n_psi_by_frequency,
        should_continue=config.should_continue,
    )
    field_batch = _prepare_circsym_observation_metal_batch(
        meridian,
        geom,
        obs_points,
        np.asarray(k_by_frequency.real, dtype=np.complex128),
        n_psi_by_frequency,
        config,
        has_sphere_points=sphere_evaluation_points is not None,
    )

    pressure_rows: list[NDArray[np.complex128]] = []
    directivity_rows: list[NDArray[np.float64]] = []
    impedance_rows: list[complex] = []
    solver_log: list[dict] = []
    completed_freqs: list[float] = []
    native_diagnostics: list[dict] = []
    surface_pressure_rows: list[NDArray[np.complex128]] | None = (
        [] if config.return_surface_pressure else None
    )
    surface_pavg: dict[int, list[complex]] = {int(tag): [] for tag in source_tags}
    surface_power_rows: list[float] = []

    for freq_index, frequency_hz in enumerate(frequencies_arr):
        _check_circsym_continue(config)
        frequency = float(frequency_hz)
        t_case = time.time()
        omega = 2.0 * np.pi * frequency
        k = complex(k_by_frequency[freq_index])
        n_psi = int(n_psi_by_frequency[freq_index])
        case_impedance = (
            impedance_sources_arg[freq_index]
            if per_case_impedance
            else impedance_sources_arg
        )
        impedance_tags = {int(tag) for tag in case_impedance.keys()}
        q_driver = _build_driver_neumann_segments(
            meridian,
            omega,
            frequency,
            config,
            impedance_tags=impedance_tags,
            source_scale=source_scale,
        )
        beta = _segment_beta(meridian, case_impedance)

        t_assembly = time.time()
        S, H = _assemble_boundary_matrices(
            meridian,
            k,
            config.circsym_baffle_z,
            n_psi=n_psi,
            geometry_cache=assembly_cache,
            should_continue=config.should_continue,
        )
        _check_circsym_continue(config)
        A = H.copy()
        A[np.diag_indices_from(A)] -= boundary_free_terms
        if np.any(beta != 0.0):
            A -= S * (1j * k * beta)[None, :]
        rhs = S @ q_driver
        assembly_s = time.time() - t_assembly

        t_solve = time.time()
        solve_matrix = A
        solve_rhs = rhs
        chief_residual_rel = None
        chief_rows_count = 0
        dense_solve_rcond: float | None = None
        if config.chief_points is not None:
            chief_S, chief_H = _assemble_chief_matrices(
                meridian,
                np.asarray(config.chief_points, dtype=np.float64),
                k,
                config.circsym_baffle_z,
                n_psi=n_psi,
                should_continue=config.should_continue,
            )
            C = chief_H.copy()
            if np.any(beta != 0.0):
                C -= chief_S * (1j * k * beta)[None, :]
            d = chief_S @ q_driver
            scale = _chief_row_scale(A, C, config.chief_weight)
            solve_matrix = np.vstack([A, scale * C])
            solve_rhs = np.concatenate([rhs, scale * d])
            chief_rows_count = int(C.shape[0])
            pressure, *_ = linalg.lstsq(solve_matrix, solve_rhs)
            chief_residual = C @ pressure - d
            denom = max(float(np.linalg.norm(rhs)), 1e-30)
            chief_residual_rel = float(np.linalg.norm(scale * chief_residual) / denom)
            lapack_info = 0
        else:
            anorm = float(np.linalg.norm(solve_matrix, ord=1))
            lu, piv = linalg.lu_factor(solve_matrix)
            pressure = linalg.lu_solve((lu, piv), solve_rhs)
            dense_solve_rcond = _rcond_from_lu_factor(lu, anorm)
            lapack_info = 0
        dense_solve_s = time.time() - t_solve

        # Evaluate the radiated field at the observation points with the
        # PHYSICAL (real) wavenumber. The complex-k shift regularizes the surface
        # BIE (fictitious-eigenfrequency avoidance) but must NOT attenuate the
        # free-field propagation to the mic: with k complex, the exp(-Im(k)*r)
        # term over-damps the response over the observation distance and grows
        # with frequency (e.g. ~-29 dB at 18 kHz over 2 m with shift 0.005). The
        # 3D solver likewise reconstructs the field at real k.
        k_field = complex(float(k.real), 0.0)
        q_total = q_driver + 1j * k * beta * pressure

        t_field = time.time()
        if field_batch is None:
            field_pressure = _evaluate_observation_pressure(
                meridian,
                pressure,
                q_total,
                obs_points,
                k_field,
                config,
                geom=geom,
                n_psi=n_psi,
                should_continue=config.should_continue,
            )
        else:
            if field_batch.rayleigh_sheet:
                first_unique = -(field_batch.slp[freq_index] @ q_total)
            else:
                first_unique = (
                    field_batch.dlp[freq_index] @ pressure
                    - field_batch.slp[freq_index] @ q_total
                )
            if field_batch.active_targets is None:
                first = first_unique[field_batch.target_inverse]
            else:
                first = np.zeros(obs_points.shape[1], dtype=np.complex128)
                first[field_batch.active_targets] = first_unique[
                    field_batch.target_inverse
                ]
            field_pressure = np.tile(first[None, :], (obs_points.shape[0], 1))
        sphere_pressure_unique = (
            _evaluate_points_pressure(
                meridian,
                pressure,
                q_total,
                sphere_evaluation_points,
                k_field,
                config.circsym_baffle_z,
                geom=geom,
                n_psi=n_psi,
                should_continue=config.should_continue,
            )
            if sphere_evaluation_points is not None
            else None
        )
        sphere_pressure = (
            sphere_pressure_unique[sphere_evaluation_inverse]
            if sphere_pressure_unique is not None
            and sphere_evaluation_inverse is not None
            else sphere_pressure_unique
        )
        field_s = time.time() - t_field

        directivity = _directivity_from_pressure(field_pressure, on_axis_idx)
        normal_velocity = normal_velocity_from_driver_neumann(
            q_driver,
            omega,
            config.air_density,
        )
        surface_power_rows.append(
            float(
                integrate_driven_surface_power(
                    pressure,
                    normal_velocity,
                    geom.area_weights,
                )
            )
        )
        pavg = _surface_pressure_average(meridian, pressure, source_tags)
        for tag in source_tags:
            surface_pavg[int(tag)].append(pavg[int(tag)])
        impedance_rows.append(pavg.get(int(impedance_source_tag), 0.0 + 0.0j))
        pressure_rows.append(field_pressure)
        directivity_rows.append(directivity)
        completed_freqs.append(frequency)
        if surface_pressure_rows is not None:
            surface_pressure_rows.append(np.asarray(pressure, dtype=np.complex128))

        assembly_backend = _select_circsym_assembly_backend(
            meridian.segment_count,
            n_psi,
        )[0]
        cpu_remainder = _circsym_cpu_remainder_status()
        diagnostics = {
            "assembly_implementation": (
                "circsym_cpu_static_metal_remainder_dp0_m0"
                if assembly_backend == "metal"
                else f"circsym_{cpu_remainder['selected']}_dp0_m0"
            ),
            "assembly_backend": assembly_backend,
            "assembly_backend_policy": _requested_circsym_assembly_backend(),
            "cpu_remainder": cpu_remainder,
            "circsym": True,
            "m_mode": 0,
            "azimuth_quadrature_points": int(n_psi),
            "line_quadrature_order": int(_LINE_QUAD_ORDER),
            "singular_line_quadrature_order": int(_SINGULAR_LINE_QUAD_ORDER),
            "complex_k": config.formulation == BIEFormulation.COMPLEX_K,
            "complex_k_shift": float(config.complex_k_shift),
            "circsym_baffle_z": (
                None
                if config.circsym_baffle_z is None
                else float(config.circsym_baffle_z)
            ),
            "dense_solve_rcond": dense_solve_rcond,
            "dense_solve_rcond_estimator": (
                "lapack_gecon_1norm" if dense_solve_rcond is not None else None
            ),
            "mesh_max_edge_m": mesh_max_segment,
            "mesh_elements_per_wavelength": float(config.speed_of_sound)
            / (frequency * mesh_max_segment)
            if mesh_max_segment > 0.0
            else math.inf,
            "sphere_targets": n_sphere,
            "sphere_evaluation_targets": n_sphere_evaluation,
            "field_backend": _circsym_field_backend_summary(
                (n_angles, n_sphere_evaluation),
                meridian.segment_count,
                n_psi,
            ),
            "field_backend_policy": _requested_circsym_field_backend(),
            "cpu_field_kernel": _circsym_cpu_field_status(),
            "chief_points": bool(config.chief_points is not None),
            "chief_points_count": int(chief_rows_count),
            "frequency_batch": {
                "assembly": assembly_cache.metal_batch_diagnostics is not None,
                "field": field_batch is not None,
                "frequency_count": int(frequencies_arr.size),
                "lifecycle": "per_helper_invocation",
                "cross_invocation_persistence": False,
                "near_correction": "cpu_complex128",
                "arithmetic_reduction": False,
            },
        }
        if chief_residual_rel is not None:
            diagnostics["chief_solver"] = "scipy_linalg_lstsq"
            diagnostics["chief_residual_rel"] = chief_residual_rel
        native_diagnostics.append(diagnostics)

        timing_s = time.time() - t_case
        log_entry = {
            "frequency_hz": frequency,
            "iterations": None,
            "timing_s": timing_s,
            "backend": "circsym_python_dp0_m0",
            "assembly_s": assembly_s,
            "dense_solve_s": dense_solve_s,
            "field_s": field_s,
            "lapack_info": lapack_info,
            "impedance": impedance_rows[-1],
            "native_diagnostics": diagnostics,
            "observation_sphere_pressure_complex": sphere_pressure,
        }
        solver_log.append(log_entry)

        if config.progress_callback is not None:
            config.progress_callback(freq_index, len(frequencies_arr), frequency)
        if config.on_frequency_result is not None:
            callback_entry = {
                **log_entry,
                "observation_pressure_complex": field_pressure,
                "observation_directivity_db": directivity,
                "observation_angles_deg": angles_deg,
                "observation_planes": config.observation.planes,
            }
            if config.on_frequency_result(freq_index, frequency, callback_entry) is False:
                logger.info("Early stop requested after %.1f Hz", frequency)
                break

    sp_avg = {
        int(tag): np.asarray(values, dtype=np.complex128)
        for tag, values in surface_pavg.items()
    }
    assembly_batch_s = float(assembly_cache.metal_batch_wall_s)
    field_batch_s = 0.0 if field_batch is None else float(field_batch.wall_s)
    timings = {
        "solve_s": (
            sum(float(entry["timing_s"]) for entry in solver_log)
            + assembly_batch_s
            + field_batch_s
        ),
        "assembly_s": (
            sum(float(entry["assembly_s"]) for entry in solver_log)
            + assembly_batch_s
        ),
        "dense_solve_s": sum(float(entry["dense_solve_s"]) for entry in solver_log),
        "directivity_s": (
            sum(float(entry["field_s"]) for entry in solver_log)
            + field_batch_s
        ),
        "metal_batch_assembly_s": assembly_batch_s,
        "metal_batch_field_s": field_batch_s,
        "total_s": time.time() - t_total,
    }
    for prefix, batch_diagnostics in (
        ("metal_batch_assembly", assembly_cache.metal_batch_diagnostics),
        (
            "metal_batch_field",
            None if field_batch is None else field_batch.diagnostics,
        ),
    ):
        if batch_diagnostics is None:
            continue
        for name in (
            "python_ipc_write_seconds",
            "python_output_read_seconds",
            "python_wall_seconds",
            "input_read_seconds",
            "invocation_setup_seconds",
            "metal_library_seconds",
            "pipeline_seconds",
            "kernel_seconds",
            "kernel_device_seconds",
            "output_write_seconds",
            "helper_wall_seconds",
        ):
            value = batch_diagnostics.get(name)
            if value is not None:
                timings[f"{prefix}_{name}"] = float(value)
    radiated_power_sphere_w, sphere_coverage_sr = _sphere_power_from_log(
        solver_log,
        config,
        n_sphere,
    )

    return SolveResult(
        frequencies_hz=np.asarray(completed_freqs, dtype=np.float64),
        pressure_complex=np.stack(pressure_rows, axis=0),
        directivity_db=np.stack(directivity_rows, axis=0),
        impedance=np.asarray(impedance_rows, dtype=np.complex128),
        observation_angles_deg=angles_deg,
        observation_points=obs_points,
        observation_planes=list(config.observation.planes),
        config=config,
        mesh_info=_mesh_info(meridian),
        timings=timings,
        solver_log=solver_log,
        surface_pressure_avg=sp_avg if sp_avg else None,
        surface_pressure_complex=(
            np.stack(surface_pressure_rows, axis=0)
            if surface_pressure_rows is not None
            else None
        ),
        native_diagnostics=native_diagnostics,
        sphere_pressure_complex=_sphere_pressure_from_log(solver_log, n_sphere),
        sphere_points=sphere_points_arr,
        sphere_theta_deg=sphere_theta_deg,
        sphere_phi_deg=sphere_phi_deg,
        radiated_power_surface_w=np.asarray(
            surface_power_rows,
            dtype=np.float64,
        ),
        radiated_power_sphere_w=radiated_power_sphere_w,
        radiated_power_sphere_coverage_sr=sphere_coverage_sr,
    )


def run_sweep_coupled_ib(
    meridian: MeridianMesh,
    frequencies: NDArray[np.float64] | Iterable[float],
    config: SolveConfig,
) -> SolveResult:
    """Run the exact coupled interior/Rayleigh infinite-baffle CircSym sweep."""
    if not isinstance(meridian, MeridianMesh):
        raise TypeError("meridian must be a MeridianMesh")
    if config.circsym_aperture_tag is None:
        raise ValueError("circsym_aperture_tag must be set for coupled IB solves")
    if config.circsym_baffle_z is not None:
        raise ValueError(
            "circsym_aperture_tag coupled infinite-baffle mode does not compose "
            "with the legacy circsym_baffle_z image kernel"
        )
    if config.chief_points is not None:
        raise ValueError(
            "circsym_aperture_tag coupled infinite-baffle mode does not support "
            "chief_points yet"
        )

    frequencies_arr = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if frequencies_arr.size == 0:
        raise ValueError("frequencies must contain at least one value")
    if not np.all(np.isfinite(frequencies_arr)) or np.any(frequencies_arr <= 0.0):
        raise ValueError("frequencies must be finite and positive")

    tags = meridian.physical_tags
    aperture_tag = int(config.circsym_aperture_tag)
    mesh_tags = {int(tag) for tag in np.unique(tags)}
    if aperture_tag not in mesh_tags:
        raise ValueError(
            f"circsym_aperture_tag {aperture_tag} is not present in the meridian; "
            f"available physical tags: {sorted(mesh_tags)}"
        )

    source_tags = [int(tag) for tag in config.velocity_sources]
    missing_tags = sorted(set(source_tags) - mesh_tags)
    if missing_tags:
        raise ValueError(
            f"velocity_sources tags {missing_tags} are not present in the meridian; "
            f"available physical tags: {sorted(mesh_tags)}"
        )
    if aperture_tag in set(source_tags):
        raise ValueError(
            "circsym_aperture_tag must not also be listed in velocity_sources"
        )

    t_total = time.time()
    geom = meridian.segment_geometry()
    _validate_coupled_ib_meridian(meridian, aperture_tag, geom=geom)
    frame = _infer_circsym_frame(meridian, config)
    source_scale = _build_source_segment_scale(meridian, config, frame)

    idx_a = np.where(tags == aperture_tag)[0]
    n = meridian.segment_count
    m = int(idx_a.size)
    if m == 0:
        raise ValueError("circsym_aperture_tag must select at least one segment")

    idx_t = np.where(np.isin(tags, source_tags))[0]
    if idx_t.size == 0:
        raise ValueError("velocity_sources must select at least one throat segment")
    throat_weights = geom.area_weights[idx_t]
    throat_area = float(np.sum(throat_weights))
    if throat_area <= 1e-30:
        raise ValueError("velocity_sources throat area must be positive")

    obs_points, angles_deg = build_observation_points(frame, config.observation)
    sphere_points_arr, sphere_theta_deg, sphere_phi_deg = _resolve_sphere_observation(
        frame, config.observation
    )
    n_sphere = 0 if sphere_points_arr is None else int(sphere_points_arr.shape[0])
    sphere_evaluation_points, sphere_evaluation_inverse = (
        _axisymmetric_sphere_evaluation_targets(
            sphere_points_arr,
            sphere_theta_deg,
        )
    )
    n_sphere_evaluation = (
        0
        if sphere_evaluation_points is None
        else int(sphere_evaluation_points.shape[0])
    )

    n_planes, n_angles, _ = obs_points.shape
    on_axis_idx = int(np.argmin(np.abs(angles_deg)))
    rho_max = float(np.max(meridian.nodes[:, 0]))
    n_psi_by_frequency = np.asarray(
        [
            _azimuth_order(_complex_wavenumber(float(frequency_hz), config), rho_max)
            for frequency_hz in frequencies_arr
        ],
        dtype=np.int32,
    )
    # Keep the quadrature order frequency-local; cached assembly geometry is
    # stored per n_psi so wide sweeps do not run every case at the HF order.
    n_psi_use_counts = _int_value_counts(n_psi_by_frequency)
    assembly_cache = _BoundaryAssemblyGeometryCache(
        meridian,
        None,
        geom=geom,
        reusable_n_psi=set(n_psi_use_counts),
        n_psi_use_counts=n_psi_use_counts,
        cache_single_use=False,
    )

    pressure_rows: list[NDArray[np.complex128]] = []
    directivity_rows: list[NDArray[np.float64]] = []
    impedance_rows: list[complex] = []
    solver_log: list[dict] = []
    completed_freqs: list[float] = []
    native_diagnostics: list[dict] = []
    surface_pressure_rows: list[NDArray[np.complex128]] | None = (
        [] if config.return_surface_pressure else None
    )
    surface_pavg: dict[int, list[complex]] = {
        int(tag): [] for tag in source_tags
    }
    surface_power_rows: list[float] = []
    impedance_sources_arg = _impedance_sources_for_frequencies(
        meridian.physical_tags, frequencies_arr, config
    )
    per_case_impedance = isinstance(impedance_sources_arg, list)

    for freq_index, frequency_hz in enumerate(frequencies_arr):
        _check_circsym_continue(config)
        frequency = float(frequency_hz)
        t_case = time.time()
        omega = 2.0 * np.pi * frequency
        k = _complex_wavenumber(frequency, config)
        k_field = complex(float(k.real), 0.0)
        n_psi = int(n_psi_by_frequency[freq_index])
        case_impedance = (
            impedance_sources_arg[freq_index]
            if per_case_impedance
            else impedance_sources_arg
        )
        if aperture_tag in {int(tag) for tag in case_impedance}:
            raise ValueError(
                "circsym_aperture_tag must not also carry a Robin/admittance "
                "boundary condition"
            )
        impedance_tags = {int(tag) for tag in case_impedance}

        t_assembly = time.time()
        S, H = _assemble_boundary_matrices(
            meridian,
            k,
            None,
            n_psi=n_psi,
            geometry_cache=assembly_cache,
            should_continue=config.should_continue,
        )
        _check_circsym_continue(config)
        if k.imag == 0.0:
            S_rayleigh_aperture = S[np.ix_(idx_a, idx_a)]
        else:
            # Rayleigh coupling is deliberately evaluated at the physical real
            # wavenumber. Only its aperture-to-aperture single-layer block is
            # consumed, so avoid a discarded full second S/H assembly.
            S_rayleigh_aperture = _assemble_coupled_ib_rayleigh_aperture_matrix(
                meridian,
                idx_a,
                k_field,
                geom=geom,
                n_psi=n_psi,
                should_continue=config.should_continue,
            )
        q_driver = _build_driver_neumann_segments(
            meridian,
            omega,
            frequency,
            config,
            impedance_tags=impedance_tags,
            source_scale=source_scale,
        )
        if np.any(q_driver[idx_a] != 0.0):
            raise ValueError(
                "circsym_aperture_tag must not be driven by velocity_sources"
            )
        free = _boundary_free_terms(meridian, None)
        beta = _segment_beta(meridian, case_impedance)

        A = np.zeros((n + m, n + m), dtype=np.complex128)
        b = np.zeros(n + m, dtype=np.complex128)
        A[:n, :n] = H
        A[np.arange(n), np.arange(n)] -= free
        if np.any(beta != 0.0):
            A[:n, :n] -= S * (1j * k * beta)[None, :]
        A[:n, n:] = -S[:, idx_a]
        b[:n] = S @ q_driver
        A[n + np.arange(m), idx_a] = 1.0
        A[n:, n:] = -2.0 * S_rayleigh_aperture
        assembly_s = time.time() - t_assembly

        t_solve = time.time()
        anorm = float(np.linalg.norm(A, ord=1))
        lu, piv = linalg.lu_factor(A)
        x = linalg.lu_solve((lu, piv), b)
        dense_solve_rcond = _rcond_from_lu_factor(lu, anorm)
        dense_solve_s = time.time() - t_solve

        p_srf = np.asarray(x[:n], dtype=np.complex128)
        q_a = np.asarray(x[n:], dtype=np.complex128)
        aperture_trace = 2.0 * (S_rayleigh_aperture @ q_a)
        aperture_trace_denom = max(float(np.linalg.norm(p_srf[idx_a])), 1.0e-30)
        aperture_pressure_continuity_rel = float(
            np.linalg.norm(aperture_trace - p_srf[idx_a]) / aperture_trace_denom
        )

        t_field = time.time()
        flat_obs = obs_points.reshape(-1, 3)
        flat_pressure = _evaluate_coupled_ib_points_pressure(
            meridian,
            q_a,
            idx_a,
            flat_obs,
            k_field,
            geom=geom,
            n_psi=n_psi,
            should_continue=config.should_continue,
        )
        field_pressure = flat_pressure.reshape(n_planes, n_angles)

        sphere_pressure = None
        if sphere_evaluation_points is not None:
            sphere_pressure_unique = _evaluate_coupled_ib_points_pressure(
                meridian,
                q_a,
                idx_a,
                sphere_evaluation_points,
                k_field,
                geom=geom,
                n_psi=n_psi,
                should_continue=config.should_continue,
            )
            sphere_pressure = (
                sphere_pressure_unique[sphere_evaluation_inverse]
                if sphere_evaluation_inverse is not None
                else sphere_pressure_unique
            )
        field_s = time.time() - t_field

        directivity = _directivity_from_pressure(field_pressure, on_axis_idx)
        normal_velocity = normal_velocity_from_driver_neumann(
            q_driver,
            omega,
            config.air_density,
        )
        surface_power_rows.append(
            float(
                integrate_driven_surface_power(
                    p_srf,
                    normal_velocity,
                    geom.area_weights,
                )
            )
        )
        impedance = complex(np.sum(p_srf[idx_t] * throat_weights) / throat_area)

        pressure_rows.append(field_pressure)
        directivity_rows.append(directivity)
        impedance_rows.append(impedance)
        completed_freqs.append(frequency)
        if surface_pressure_rows is not None:
            surface_pressure_rows.append(p_srf)
        pavg = _surface_pressure_average(meridian, p_srf, source_tags)
        for tag in source_tags:
            surface_pavg[int(tag)].append(pavg[int(tag)])

        assembly_backend = _select_circsym_assembly_backend(
            meridian.segment_count,
            n_psi,
        )[0]
        cpu_remainder = _circsym_cpu_remainder_status()
        diagnostics = {
            "circsym": True,
            "coupled_ib": True,
            "assembly_implementation": (
                "circsym_cpu_static_metal_remainder_dp0_m0_coupled_ib"
                if assembly_backend == "metal"
                else f"circsym_{cpu_remainder['selected']}_dp0_m0_coupled_ib"
            ),
            "assembly_backend": assembly_backend,
            "assembly_backend_policy": _requested_circsym_assembly_backend(),
            "cpu_remainder": cpu_remainder,
            "m_mode": 0,
            "aperture_tag": int(aperture_tag),
            "aperture_segments": int(m),
            "azimuth_quadrature_points": int(n_psi),
            "complex_k": config.formulation == BIEFormulation.COMPLEX_K,
            "complex_k_shift": float(config.complex_k_shift),
            "robin": bool(np.any(beta != 0.0)),
            "aperture_pressure_continuity_rel": aperture_pressure_continuity_rel,
            "dense_solve_rcond": dense_solve_rcond,
            "dense_solve_rcond_estimator": "lapack_gecon_1norm",
            "sphere_targets": n_sphere,
            "sphere_evaluation_targets": n_sphere_evaluation,
            "field_backend": _circsym_field_backend_summary(
                (n_angles, n_sphere_evaluation),
                m,
                n_psi,
            ),
            "field_backend_policy": _requested_circsym_field_backend(),
            "cpu_field_kernel": _circsym_cpu_field_status(),
        }
        native_diagnostics.append(diagnostics)

        timing_s = time.time() - t_case
        log_entry = {
            "frequency_hz": frequency,
            "iterations": None,
            "timing_s": timing_s,
            "backend": "circsym_python_dp0_m0_coupled_ib",
            "assembly_s": assembly_s,
            "dense_solve_s": dense_solve_s,
            "field_s": field_s,
            "lapack_info": 0,
            "impedance": impedance,
            "native_diagnostics": diagnostics,
            "observation_sphere_pressure_complex": sphere_pressure,
        }
        solver_log.append(log_entry)

        if config.progress_callback is not None:
            config.progress_callback(freq_index, len(frequencies_arr), frequency)
        if config.on_frequency_result is not None:
            callback_entry = {
                **log_entry,
                "observation_pressure_complex": field_pressure,
                "observation_directivity_db": directivity,
                "observation_angles_deg": angles_deg,
                "observation_planes": config.observation.planes,
            }
            if (
                config.on_frequency_result(freq_index, frequency, callback_entry)
                is False
            ):
                logger.info("Early stop requested after %.1f Hz", frequency)
                break

    sp_avg = {
        int(tag): np.asarray(values, dtype=np.complex128)
        for tag, values in surface_pavg.items()
    }
    timings = {
        "solve_s": sum(float(entry["timing_s"]) for entry in solver_log),
        "assembly_s": sum(float(entry["assembly_s"]) for entry in solver_log),
        "dense_solve_s": sum(float(entry["dense_solve_s"]) for entry in solver_log),
        "directivity_s": sum(float(entry["field_s"]) for entry in solver_log),
        "total_s": time.time() - t_total,
    }
    radiated_power_sphere_w, sphere_coverage_sr = _sphere_power_from_log(
        solver_log,
        config,
        n_sphere,
    )

    return SolveResult(
        frequencies_hz=np.asarray(completed_freqs, dtype=np.float64),
        pressure_complex=np.stack(pressure_rows, axis=0),
        directivity_db=np.stack(directivity_rows, axis=0),
        impedance=np.asarray(impedance_rows, dtype=np.complex128),
        observation_angles_deg=angles_deg,
        observation_points=obs_points,
        observation_planes=list(config.observation.planes),
        config=config,
        mesh_info=_mesh_info(meridian),
        timings=timings,
        solver_log=solver_log,
        surface_pressure_avg=sp_avg if sp_avg else None,
        surface_pressure_complex=(
            np.stack(surface_pressure_rows, axis=0)
            if surface_pressure_rows is not None
            else None
        ),
        native_diagnostics=native_diagnostics,
        sphere_pressure_complex=_sphere_pressure_from_log(solver_log, n_sphere),
        sphere_points=sphere_points_arr,
        sphere_theta_deg=sphere_theta_deg,
        sphere_phi_deg=sphere_phi_deg,
        radiated_power_surface_w=np.asarray(
            surface_power_rows,
            dtype=np.float64,
        ),
        radiated_power_sphere_w=radiated_power_sphere_w,
        radiated_power_sphere_coverage_sr=sphere_coverage_sr,
    )


def solve_circsym(
    meridian: MeridianMesh,
    config: SolveConfig | None = None,
) -> SolveResult:
    """Run an m=0 CircSym sweep using ``config``'s frequency grid.

    When ``config`` is omitted, CircSym defaults to the complex-k formulation as
    the irregular-frequency cure for closed body-of-revolution surfaces.
    """
    if config is None:
        config = SolveConfig(formulation=BIEFormulation.COMPLEX_K)
    return run_sweep_circsym(meridian, _build_frequency_grid(config), config)


def solve_circsym_frequencies(
    meridian: MeridianMesh,
    frequencies_hz: list[float] | NDArray[np.float64],
    config: SolveConfig | None = None,
) -> SolveResult:
    """Run an m=0 CircSym solve at caller-ordered frequencies."""
    if config is None:
        config = SolveConfig(formulation=BIEFormulation.COMPLEX_K)
    return run_sweep_circsym(
        meridian, np.asarray(frequencies_hz, dtype=np.float64), config
    )


def _complex_wavenumber(frequency_hz: float, config: SolveConfig) -> complex:
    k_real = 2.0 * np.pi * float(frequency_hz) / float(config.speed_of_sound)
    if config.formulation == BIEFormulation.COMPLEX_K:
        return complex(k_real, k_real * float(config.complex_k_shift))
    return complex(k_real, 0.0)


def _azimuth_order(k: complex, rho_max: float) -> int:
    krho = abs(complex(k)) * max(float(rho_max), 0.0)
    raw_minimum = os.environ.get(_CIRCSYM_AZIMUTH_POINTS_MIN_ENV)
    minimum = _AZIMUTH_POINTS_MIN if raw_minimum is None else int(raw_minimum)
    if minimum < 8:
        raise ValueError(f"{_CIRCSYM_AZIMUTH_POINTS_MIN_ENV} must be at least 8")
    return max(minimum, math.ceil(_AZIMUTH_POINTS_PER_KRHO * krho))


@lru_cache(maxsize=64)
def _leggauss01(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x, w = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * (x + 1.0), 0.5 * w


@lru_cache(maxsize=64)
def _leggauss_psi(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x, w = np.polynomial.legendre.leggauss(int(order))
    psi = 0.5 * np.pi * (x + 1.0)
    weights = 0.5 * np.pi * w
    return psi, weights


def _infer_circsym_frame(
    meridian: MeridianMesh,
    config: SolveConfig,
) -> ObservationFrame:
    # Match the full-3D solver contract: a caller-supplied frame is
    # authoritative.  This is required for solver-to-solver parity and for
    # geometries (such as a fully driven closed body) whose source centroid is
    # not the intended acoustic measurement origin.
    if config.frame_override is not None:
        return config.frame_override

    geom = meridian.segment_geometry()
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    source_tags = {int(tag) for tag in config.velocity_sources}
    source_mask = np.isin(meridian.physical_tags, list(source_tags))
    if np.any(source_mask):
        weights = np.maximum(geom.area_weights[source_mask], 0.0)
        if float(np.sum(weights)) > 1e-30:
            source_z = float(np.average(geom.z_mid[source_mask], weights=weights))
        else:
            source_z = float(np.mean(geom.z_mid[source_mask]))
    else:
        source_z = float(np.min(meridian.nodes[:, 1]))
    mouth_z = float(np.max(meridian.nodes[:, 1]))
    origin_z = mouth_z if config.observation.origin == "mouth" else source_z
    return ObservationFrame(
        axis=axis,
        origin=np.array([0.0, 0.0, origin_z], dtype=np.float64),
        u=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        v=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        mouth_center=np.array([0.0, 0.0, mouth_z], dtype=np.float64),
        source_center=np.array([0.0, 0.0, source_z], dtype=np.float64),
    )


def _mesh_info(meridian: MeridianMesh) -> MeshInfo:
    rho_max = float(np.max(meridian.nodes[:, 0]))
    z_min = float(np.min(meridian.nodes[:, 1]))
    z_max = float(np.max(meridian.nodes[:, 1]))
    tags = {int(tag): f"tag_{int(tag)}" for tag in np.unique(meridian.physical_tags)}
    return MeshInfo(
        n_vertices=meridian.node_count,
        n_triangles=meridian.segment_count,
        physical_groups=tags,
        bounding_box_m=(
            np.array([-rho_max, -rho_max, z_min], dtype=np.float64),
            np.array([rho_max, rho_max, z_max], dtype=np.float64),
        ),
    )


def _build_source_segment_scale(
    meridian: MeridianMesh,
    config: SolveConfig,
    frame: ObservationFrame,
) -> NDArray[np.complex128] | NDArray[np.float64] | None:
    profile_map = {
        int(profile_tag): profile
        for profile_tag, profile in (config.source_velocity_profiles or {}).items()
    }
    source_tags = sorted({int(tag) for tag in config.velocity_sources} | set(profile_map))
    fallback_profile = (
        AxialProfile()
        if config.source_motion == SourceMotion.AXIAL
        else NormalProfile()
    )
    if all(
        isinstance(profile_map.get(tag, fallback_profile), NormalProfile)
        for tag in source_tags
    ):
        return None

    geom = meridian.segment_geometry()
    centroids3 = np.column_stack(
        [
            geom.rho_mid,
            np.zeros(meridian.segment_count, dtype=np.float64),
            geom.z_mid,
        ]
    )
    normals3 = np.column_stack(
        [
            meridian.normals[:, 0],
            np.zeros(meridian.segment_count, dtype=np.float64),
            meridian.normals[:, 1],
        ]
    )
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    center = np.asarray(frame.source_center, dtype=np.float64)
    scale = np.zeros(meridian.segment_count, dtype=np.complex128)
    any_source = False

    for tag in source_tags:
        idx = np.where(meridian.physical_tags == tag)[0]
        if idx.size == 0:
            continue
        any_source = True
        profile = profile_map.get(tag, fallback_profile)
        if isinstance(profile, NormalProfile):
            values = np.ones(idx.size, dtype=np.float64)
        elif isinstance(profile, AxialProfile):
            values = _tag_axial_projection_2d(meridian, geom, idx)
        elif isinstance(profile, TaperProfile):
            axial = _tag_axial_projection_2d(meridian, geom, idx)
            values = axial * _taper_values(
                _normalized_tag_radius_2d(meridian, geom, idx),
                profile,
            )
        elif isinstance(profile, AnnularProfile):
            axial = _tag_axial_projection_2d(meridian, geom, idx)
            radial = _normalized_tag_radius_2d(meridian, geom, idx)
            values = axial * (
                (radial >= profile.r_inner) & (radial <= profile.r_outer)
            ).astype(np.float64)
        elif isinstance(profile, PerFaceProfile):
            values = np.asarray(profile.weights, dtype=np.complex128)
            if values.ndim != 1 or values.shape[0] != idx.size:
                raise ValueError(
                    "PerFaceProfile.weights length must equal the number of "
                    f"segments for tag {tag}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError("PerFaceProfile.weights must be finite")
        elif isinstance(profile, CallableProfile):
            values = np.asarray(
                profile.callback(
                    centroids3[idx],
                    normals3[idx],
                    axis.copy(),
                    center.copy(),
                ),
                dtype=np.complex128,
            )
            if values.ndim != 1 or values.shape[0] != idx.size:
                raise ValueError(
                    "CallableProfile.callback must return one weight per "
                    f"segment for tag {tag}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError("CallableProfile.callback returned non-finite weights")
        else:  # pragma: no cover - SolveConfig validation rejects this.
            raise ValueError(
                "source_velocity_profiles values must be SourceProfile instances"
            )
        scale[idx] = values

    if not any_source:
        return None
    if np.any(scale.imag != 0.0):
        return scale
    return scale.real


def _tag_axial_projection_2d(
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    proj = np.asarray(meridian.normals[indices, 1], dtype=np.float64)
    weights = np.asarray(geom.area_weights[indices], dtype=np.float64)
    vote = float(np.dot(proj, weights))
    if vote < -1e-14 * max(float(np.sum(np.abs(weights))), 1.0):
        proj = -proj
    return proj


def _normalized_tag_radius_2d(
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    tag_nodes = np.unique(meridian.segments[indices].reshape(-1))
    rho_max = float(np.max(meridian.nodes[tag_nodes, 0])) if tag_nodes.size else 0.0
    if not np.isfinite(rho_max) or rho_max <= 1e-15:
        return np.zeros(indices.size, dtype=np.float64)
    return np.clip(geom.rho_mid[indices] / rho_max, 0.0, 1.0)


def _build_driver_neumann_segments(
    meridian: MeridianMesh,
    omega: float,
    frequency_hz: float,
    config: SolveConfig,
    *,
    impedance_tags: set[int],
    source_scale: NDArray[np.complex128] | NDArray[np.float64] | None,
) -> NDArray[np.complex128]:
    coeffs = np.zeros(meridian.segment_count, dtype=np.complex128)
    velocity_sources = _resolve_velocity_sources(config, frequency_hz)
    for raw_tag, raw_weight in velocity_sources.items():
        tag = int(raw_tag)
        if tag in impedance_tags:
            continue
        idx = np.where(meridian.physical_tags == tag)[0]
        if idx.size == 0:
            continue
        weight = complex(raw_weight)
        if source_scale is None:
            v_n = np.full(idx.size, weight, dtype=np.complex128)
        else:
            v_n = weight * np.asarray(source_scale[idx], dtype=np.complex128)
        if config.velocity_mode == VelocityMode.ACCELERATION:
            # Under e^{-i omega t}, v = a/(-i omega) for a*cos(omega t), so
            # q = -rho a (momentum: dp/dn = -rho a_n). Matches bie.py and the
            # 2026-07-09 ABEC3 absolute-pressure validation.
            v_n = v_n / (-1j * omega) if omega > 0.0 else np.zeros_like(v_n)
        coeffs[idx] = 1j * config.air_density * omega * v_n
    return coeffs


def _segment_beta(
    meridian: MeridianMesh,
    impedance_sources: dict[int, complex],
) -> NDArray[np.complex128]:
    beta = np.zeros(meridian.segment_count, dtype=np.complex128)
    for tag, value in impedance_sources.items():
        beta[meridian.physical_tags == int(tag)] = complex(value)
    return beta


def _assemble_boundary_matrices(
    meridian: MeridianMesh,
    k: complex,
    baffle_z: float | None,
    *,
    n_psi: int,
    geometry_cache: "_BoundaryAssemblyGeometryCache | None" = None,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    _check_circsym_continue(should_continue)
    if geometry_cache is not None:
        cached = geometry_cache.assemble(
            k,
            n_psi=int(n_psi),
            meridian=meridian,
            baffle_z=baffle_z,
            should_continue=should_continue,
        )
        if cached is not None:
            return cached
    return _assemble_boundary_matrices_uncached(
        meridian,
        k,
        baffle_z,
        n_psi=n_psi,
        should_continue=should_continue,
    )


def _assemble_boundary_matrices_uncached(
    meridian: MeridianMesh,
    k: complex,
    baffle_z: float | None,
    *,
    n_psi: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    geom = meridian.segment_geometry()
    n = meridian.segment_count
    S = np.empty((n, n), dtype=np.complex128)
    H = np.empty((n, n), dtype=np.complex128)
    workers = _assembly_worker_count(n)
    block_size = _assembly_target_block_size(n, int(n_psi))
    if workers > 1:
        block_size = min(block_size, _ASSEMBLY_KERNEL_PARALLEL_TARGET_BLOCK)
    ranges = [
        (start, min(n, start + block_size))
        for start in range(0, n, block_size)
    ]
    if workers <= 1 or len(ranges) <= 1:
        for span in ranges:
            _check_circsym_continue(should_continue)
            start, stop, s_block, h_block = _assemble_boundary_block(
                span[0],
                span[1],
                meridian=meridian,
                geom=geom,
                k=k,
                baffle_z=baffle_z,
                n_psi=n_psi,
            )
            S[start:stop] = s_block
            H[start:stop] = h_block
            _check_circsym_continue(should_continue)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            blocks = executor.map(
                lambda span: _assemble_boundary_block(
                    span[0],
                    span[1],
                    meridian=meridian,
                    geom=geom,
                    k=k,
                    baffle_z=baffle_z,
                    n_psi=n_psi,
                ),
                ranges,
            )
            for start, stop, s_block, h_block in blocks:
                _check_circsym_continue(should_continue)
                S[start:stop] = s_block
                H[start:stop] = h_block
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    return S, H


def _int_value_counts(values: NDArray[np.integer] | Iterable[int]) -> dict[int, int]:
    arr = np.asarray(
        list(values) if not isinstance(values, np.ndarray) else values,
        dtype=np.int64,
    )
    arr = arr.reshape(-1)
    if arr.size == 0:
        return {}
    unique, counts = np.unique(arr, return_counts=True)
    return {int(value): int(count) for value, count in zip(unique, counts)}


@dataclass
class _FarRemainderGeometry:
    R: NDArray[np.float64]
    g_weight: NDArray[np.float64]
    h_weight: NDArray[np.float64]


@dataclass
class _FarRemainderCompactGeometry:
    target_rho: NDArray[np.float64]
    target_z: NDArray[np.float64]
    source_rho: NDArray[np.float64]
    source_z: NDArray[np.float64]
    measure: NDArray[np.float64]
    normal_rho: NDArray[np.float64]
    normal_z: NDArray[np.float64]
    cos_psi: NDArray[np.float64]
    psi_weights: NDArray[np.float64]
    baffle_z: float | None


@dataclass
class _NearRemainderGeometry:
    R: NDArray[np.float64]
    num: NDArray[np.float64]
    weight: NDArray[np.float64]


@dataclass
class _NearPairCompactGeometry:
    target_rho: NDArray[np.float64]
    target_z: NDArray[np.float64]
    source_rho: NDArray[np.float64]
    source_z: NDArray[np.float64]
    measure: NDArray[np.float64]
    normal_rho: NDArray[np.float64]
    normal_z: NDArray[np.float64]
    baffle_z: float | None


@dataclass
class _NearRemainderCompactGeometry:
    pairs: _NearPairCompactGeometry
    cos_psi: NDArray[np.float64]
    psi_weights: NDArray[np.float64]


@dataclass
class _CircSymFieldFrequencyBatch:
    slp: NDArray[np.complex128]
    dlp: NDArray[np.complex128]
    target_inverse: NDArray[np.int64]
    active_targets: NDArray[np.bool_] | None
    rayleigh_sheet: bool
    wall_s: float
    diagnostics: dict[str, Any]


@dataclass
class _BoundaryAssemblyQuadratureGeometry:
    far: _FarRemainderCompactGeometry
    near: _NearRemainderCompactGeometry


class _BoundaryAssemblyGeometryCache:
    def __init__(
        self,
        meridian: MeridianMesh,
        baffle_z: float | None,
        *,
        geom: SimpleNamespace | None = None,
        reusable_n_psi: set[int] | None = None,
        n_psi_use_counts: dict[int, int] | None = None,
        cache_single_use: bool = True,
    ) -> None:
        self.meridian = meridian
        self.baffle_z = baffle_z
        self.geom = geom if geom is not None else meridian.segment_geometry()
        self._reusable_n_psi = (
            None if reusable_n_psi is None else {int(value) for value in reusable_n_psi}
        )
        self._remaining_uses = (
            None
            if n_psi_use_counts is None
            else {int(key): int(value) for key, value in n_psi_use_counts.items()}
        )
        # Retain the legacy keyword for internal callers; compact far geometry
        # is cheap enough that single-use orders are always assembled here.
        del cache_single_use
        self._exhausted_n_psi: set[int] = set()
        self._static_s: NDArray[np.complex128] | None = None
        self._static_h: NDArray[np.complex128] | None = None
        self._near_rows: NDArray[np.int64] | None = None
        self._near_cols: NDArray[np.int64] | None = None
        self._near_pairs: _NearPairCompactGeometry | None = None
        self._quadrature: dict[int, _BoundaryAssemblyQuadratureGeometry] = {}
        self._metal_remainders: dict[
            tuple[float, float, int],
            tuple[NDArray[np.complex128], NDArray[np.complex128]],
        ] = {}
        self.metal_batch_diagnostics: dict[str, Any] | None = None
        self.metal_batch_wall_s = 0.0

    def prepare_metal_remainder_batch(
        self,
        k_values: NDArray[np.complex128],
        n_psi_values: NDArray[np.int32],
        *,
        should_continue: Callable[[], bool | None] | None = None,
    ) -> None:
        """Evaluate all far remainders in one per-invocation frequency batch."""
        kvals = np.asarray(k_values, dtype=np.complex128).reshape(-1)
        orders = np.asarray(n_psi_values, dtype=np.int32).reshape(-1)
        if kvals.shape != orders.shape or kvals.size == 0:
            raise ValueError("k_values and n_psi_values must be non-empty and aligned")
        selections = [
            _select_circsym_assembly_backend(self.meridian.segment_count, int(order))
            for order in orders
        ]
        if any(backend != "metal" for backend, _ in selections):
            return
        runtime_status = selections[0][1]
        if runtime_status is None:
            return
        started = time.perf_counter()
        _, _, near_rows, near_cols = self._static_geometry(
            should_continue=should_continue
        )
        parts = tuple(
            self._quadrature_geometry(
                int(order),
                near_rows,
                near_cols,
                should_continue=should_continue,
            ).far
            for order in orders
        )
        try:
            s_batch, h_batch, diagnostics = (
                _evaluate_far_remainder_onthefly_metal_batch(
                    parts,
                    kvals,
                    runtime_status=runtime_status,
                    should_continue=should_continue,
                )
            )
        except _CircSymMetalAccelerationError as exc:
            if _requested_circsym_assembly_backend() == "metal":
                raise
            _record_circsym_metal_assembly_failure(exc)
            return
        self.metal_batch_wall_s = time.perf_counter() - started
        self.metal_batch_diagnostics = diagnostics
        for index, (k_value, order) in enumerate(zip(kvals, orders, strict=True)):
            self._metal_remainders[_metal_remainder_key(k_value, int(order))] = (
                s_batch[index],
                h_batch[index],
            )

    def assemble(
        self,
        k: complex,
        *,
        n_psi: int,
        meridian: MeridianMesh,
        baffle_z: float | None,
        should_continue: Callable[[], bool | None] | None = None,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]] | None:
        _check_circsym_continue(should_continue)
        if meridian is not self.meridian or not _same_optional_float(
            baffle_z, self.baffle_z
        ):
            return None
        n_psi_int = int(n_psi)
        if self._reusable_n_psi is not None and n_psi_int not in self._reusable_n_psi:
            return None
        if self._remaining_uses is not None and n_psi_int in self._exhausted_n_psi:
            return None
        remainder_kernel = _load_circsym_remainder_kernel()

        try:
            static_s, static_h, near_rows, near_cols = self._static_geometry(
                should_continue=should_continue
            )
            qgeom = self._quadrature_geometry(
                n_psi_int,
                near_rows,
                near_cols,
                should_continue=should_continue,
            )
            result = _assemble_boundary_matrices_from_geometry(
                static_s,
                static_h,
                near_rows,
                near_cols,
                qgeom,
                k,
                n_psi=n_psi_int,
                remainder_kernel=remainder_kernel,
                metal_remainder=self._metal_remainders.pop(
                    _metal_remainder_key(k, n_psi_int),
                    None,
                ),
                should_continue=should_continue,
            )
        except MemoryError:
            self._quadrature.pop(n_psi_int, None)
            return None
        self._release_quadrature_use(n_psi_int)
        return result

    def _static_geometry(
        self,
        *,
        should_continue: Callable[[], bool | None] | None = None,
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.int64],
        NDArray[np.int64],
    ]:
        if (
            self._static_s is None
            or self._static_h is None
            or self._near_rows is None
            or self._near_cols is None
        ):
            static_s, static_h, near_rows, near_cols = _build_boundary_static_geometry(
                self.meridian,
                self.geom,
                self.baffle_z,
                should_continue=should_continue,
            )
            self._static_s = static_s
            self._static_h = static_h
            self._near_rows = near_rows
            self._near_cols = near_cols
        return self._static_s, self._static_h, self._near_rows, self._near_cols

    def _quadrature_geometry(
        self,
        n_psi: int,
        near_rows: NDArray[np.int64],
        near_cols: NDArray[np.int64],
        *,
        should_continue: Callable[[], bool | None] | None = None,
    ) -> _BoundaryAssemblyQuadratureGeometry:
        cached = self._quadrature.get(int(n_psi))
        if cached is None:
            _check_circsym_continue(should_continue)
            if self._near_pairs is None:
                self._near_pairs = _build_near_pair_compact_geometry(
                    self.meridian,
                    self.geom,
                    near_rows,
                    near_cols,
                    self.baffle_z,
                    should_continue=should_continue,
                )
            cached = _BoundaryAssemblyQuadratureGeometry(
                far=_build_far_remainder_compact_geometry(
                    self.meridian,
                    self.geom,
                    self.baffle_z,
                    n_psi=int(n_psi),
                ),
                near=_build_near_remainder_compact_geometry(
                    self._near_pairs,
                    n_psi=int(n_psi),
                ),
            )
            self._quadrature[int(n_psi)] = cached
            _check_circsym_continue(should_continue)
        return cached

    def _release_quadrature_use(self, n_psi: int) -> None:
        if self._remaining_uses is None:
            return
        n_psi_int = int(n_psi)
        remaining = self._remaining_uses.get(n_psi_int)
        if remaining is None:
            return
        remaining -= 1
        if remaining <= 0:
            self._remaining_uses.pop(n_psi_int, None)
            self._quadrature.pop(n_psi_int, None)
            self._exhausted_n_psi.add(n_psi_int)
        else:
            self._remaining_uses[n_psi_int] = remaining


def _same_optional_float(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return float(a) == float(b)


def _metal_remainder_key(k: complex, n_psi: int) -> tuple[float, float, int]:
    value = complex(k)
    return float(value.real), float(value.imag), int(n_psi)


def _assemble_boundary_matrices_from_geometry(
    static_s: NDArray[np.complex128],
    static_h: NDArray[np.complex128],
    near_rows: NDArray[np.int64],
    near_cols: NDArray[np.int64],
    qgeom: _BoundaryAssemblyQuadratureGeometry,
    k: complex,
    *,
    n_psi: int,
    remainder_kernel: _CircsymRemainderKernel | None,
    metal_remainder: tuple[
        NDArray[np.complex128], NDArray[np.complex128]
    ]
    | None = None,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    _check_circsym_continue(should_continue)
    S = static_s.copy()
    H = static_h.copy()
    n = S.shape[0]
    workers = _assembly_worker_count(n)
    assembly_backend, metal_runtime_status = _select_circsym_assembly_backend(
        n,
        int(n_psi),
    )
    far_complete = False
    if assembly_backend == "metal":
        if metal_remainder is not None:
            s_part, h_part = metal_remainder
            S += s_part
            H += h_part
            far_complete = True
        else:
            try:
                s_part, h_part = _evaluate_far_remainder_onthefly_metal(
                    qgeom.far,
                    k,
                    runtime_status=metal_runtime_status,
                    should_continue=should_continue,
                )
                S += s_part
                H += h_part
                far_complete = True
            except _CircSymMetalAccelerationError as exc:
                if _requested_circsym_assembly_backend() == "metal":
                    raise
                _record_circsym_metal_assembly_failure(exc)
    if not far_complete:
        if remainder_kernel is not None:
            s_part, h_part = _evaluate_far_remainder_with_kernel(
                remainder_kernel,
                qgeom.far,
                k,
                workers=workers,
                should_continue=should_continue,
            )
            S += s_part
            H += h_part
        else:
            _accumulate_far_remainder_numpy(
                S,
                H,
                qgeom,
                k,
                n_psi=n_psi,
                workers=workers,
                should_continue=should_continue,
            )

    if near_rows.size:
        _check_circsym_continue(should_continue)
        if remainder_kernel is None:
            s_near, h_near = _evaluate_near_remainder_compact(qgeom.near, k)
        else:
            s_near, h_near = _evaluate_near_remainder_compact_with_kernel(
                remainder_kernel,
                qgeom.near,
                k,
                workers=workers,
            )
        S[near_rows, near_cols] = static_s[near_rows, near_cols] + s_near
        H[near_rows, near_cols] = static_h[near_rows, near_cols] + h_near
    return S, H


def _accumulate_far_remainder_numpy(
    S: NDArray[np.complex128],
    H: NDArray[np.complex128],
    qgeom: _BoundaryAssemblyQuadratureGeometry,
    k: complex,
    *,
    n_psi: int,
    workers: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> None:
    del n_psi
    s_dyn, h_dyn = _evaluate_far_remainder_onthefly_reference(
        qgeom.far,
        k,
        workers=workers,
        should_continue=should_continue,
    )
    S += s_dyn
    H += h_dyn


_CIRCSYM_REMAINDER_C_SOURCE = r"""
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    int64_t nt;
    int64_t ns;
    int64_t nl;
    int64_t np;
    const double *target_rho;
    const double *target_z;
    const double *source_rho;
    const double *source_z;
    const double *measure;
    const double *normal_rho;
    const double *normal_z;
    const double *cos_psi;
    const double *psi_weights;
    int32_t has_baffle;
    double baffle_z;
    double kr;
    double ki;
    double *out_s;
    double *out_h;
} FarOntheflyTask;

typedef struct {
    const FarOntheflyTask *task;
    int64_t start;
    int64_t stop;
} FarOntheflyThreadTask;

typedef struct {
    int64_t pair_count;
    int64_t node_count;
    int64_t psi_count;
    const double *R;
    const double *num;
    const double *weight;
    double kr;
    double ki;
    double *out_s;
    double *out_h;
} NearTask;

typedef struct {
    const NearTask *task;
    int64_t start;
    int64_t stop;
} NearThreadTask;

typedef struct {
    int64_t pair_count;
    int64_t node_count;
    int64_t psi_count;
    const double *target_rho;
    const double *target_z;
    const double *source_rho;
    const double *source_z;
    const double *measure;
    const double *normal_rho;
    const double *normal_z;
    const double *cos_psi;
    const double *psi_weights;
    int32_t has_baffle;
    double baffle_z;
    double kr;
    double ki;
    double *out_s;
    double *out_h;
} NearOntheflyTask;

typedef struct {
    const NearOntheflyTask *task;
    int64_t start;
    int64_t stop;
} NearOntheflyThreadTask;

static inline void cmul(
    double ar,
    double ai,
    double br,
    double bi,
    double *out_re,
    double *out_im
) {
    *out_re = ar * br - ai * bi;
    *out_im = ar * bi + ai * br;
}

static void eval_far_onthefly_range(
    const FarOntheflyTask *task,
    int64_t start,
    int64_t stop
) {
    const int64_t ns = task->ns;
    const int64_t nl = task->nl;
    const int64_t np = task->np;
    const double *target_rho = task->target_rho;
    const double *target_z = task->target_z;
    const double *source_rho = task->source_rho;
    const double *source_z = task->source_z;
    const double *measure = task->measure;
    const double *normal_rho = task->normal_rho;
    const double *normal_z = task->normal_z;
    const double *cos_psi = task->cos_psi;
    const double *psi_weights = task->psi_weights;
    const int32_t image_count = task->has_baffle ? 2 : 1;
    const double baffle_z = task->baffle_z;
    const double kr = task->kr;
    const double ki = task->ki;
    double *out_s = task->out_s;
    double *out_h = task->out_h;
    const double pi = 3.141592653589793238462643383279502884;

    for (int64_t i = start; i < stop; ++i) {
        const double rt = target_rho[i];
        const double zt = target_z[i];
        for (int64_t j = 0; j < ns; ++j) {
            double s_re = 0.0;
            double s_im = 0.0;
            double h_re = 0.0;
            double h_im = 0.0;
            const double nr = normal_rho[j];
            const double source_nz = normal_z[j];

            for (int32_t image = 0; image < image_count; ++image) {
                double part_s_re = 0.0;
                double part_s_im = 0.0;
                double part_h_re = 0.0;
                double part_h_im = 0.0;
                const double nz = image ? -source_nz : source_nz;

                for (int64_t u = 0; u < nl; ++u) {
                    const int64_t source_idx = j * nl + u;
                    const double source_measure = measure[source_idx];
                    if (source_measure == 0.0) {
                        continue;
                    }
                    const double rs = source_rho[source_idx];
                    const double base_zs = source_z[source_idx];
                    const double zs = image ? 2.0 * baffle_z - base_zs : base_zs;
                    const double dz = zs - zt;

                    for (int64_t p = 0; p < np; ++p) {
                        const double cp = cos_psi[p];
                        double r2 =
                            rt * rt + rs * rs - 2.0 * rt * rs * cp + dz * dz;
                        if (r2 < 0.0) {
                            r2 = 0.0;
                        }
                        const double r = sqrt(r2);
                        if (r <= 1.0e-13) {
                            continue;
                        }

                        const double weighted_measure =
                            (source_measure * (2.0 * psi_weights[p])) / (4.0 * pi);
                        const double g_weight = weighted_measure / r;
                        const double num = (rs - rt * cp) * nr + dz * nz;
                        const double h_weight =
                            (weighted_measure * num) / ((r * r) * r);
                        const double kr_r = kr * r;
                        const double ki_r = ki * r;
                        const double decay = exp(-ki_r);
                        const double phase_re = decay * cos(kr_r);
                        const double phase_im = decay * sin(kr_r);

                        part_s_re += (phase_re - 1.0) * g_weight;
                        part_s_im += phase_im * g_weight;

                        const double factor_re = -ki_r - 1.0;
                        const double factor_im = kr_r;
                        const double expr_re =
                            phase_re * factor_re - phase_im * factor_im + 1.0;
                        const double expr_im =
                            phase_re * factor_im + phase_im * factor_re;
                        part_h_re += expr_re * h_weight;
                        part_h_im += expr_im * h_weight;
                    }
                }
                s_re += part_s_re;
                s_im += part_s_im;
                h_re += part_h_re;
                h_im += part_h_im;
            }

            const int64_t out_idx = 2 * (i * ns + j);
            out_s[out_idx] = s_re;
            out_s[out_idx + 1] = s_im;
            out_h[out_idx] = h_re;
            out_h[out_idx + 1] = h_im;
        }
    }
}

static void *eval_far_onthefly_worker(void *raw) {
    const FarOntheflyThreadTask *thread_task =
        (const FarOntheflyThreadTask *)raw;
    eval_far_onthefly_range(thread_task->task, thread_task->start, thread_task->stop);
    return NULL;
}

static void eval_near_range(const NearTask *task, int64_t start, int64_t stop) {
    const int64_t node_count = task->node_count;
    const int64_t psi_count = task->psi_count;
    const double *R = task->R;
    const double *num = task->num;
    const double *weight = task->weight;
    const double kr = task->kr;
    const double ki = task->ki;
    double *out_s = task->out_s;
    double *out_h = task->out_h;

    for (int64_t pair = start; pair < stop; ++pair) {
        double s_re = 0.0;
        double s_im = 0.0;
        double h_re = 0.0;
        double h_im = 0.0;
        int64_t idx = pair * node_count * psi_count;
        for (int64_t node = 0; node < node_count; ++node) {
            for (int64_t psi = 0; psi < psi_count; ++psi, ++idx) {
                const double w = weight[idx];
                if (w == 0.0) {
                    continue;
                }
                const double r = R[idx];
                if (r <= 1.0e-13) {
                    s_re += (-ki) * w;
                    s_im += kr * w;
                    continue;
                }

                const double q_re = kr * r;
                const double q_im = ki * r;
                double remg_re;
                double remg_im;
                double expr_re;
                double expr_im;
                if (hypot(q_re, q_im) < 1.0e-5) {
                    const double z_re = -q_im;
                    const double z_im = q_re;
                    double z2_re, z2_im, z3_re, z3_im, z4_re, z4_im, z5_re, z5_im;
                    cmul(z_re, z_im, z_re, z_im, &z2_re, &z2_im);
                    cmul(z2_re, z2_im, z_re, z_im, &z3_re, &z3_im);
                    cmul(z3_re, z3_im, z_re, z_im, &z4_re, &z4_im);
                    cmul(z4_re, z4_im, z_re, z_im, &z5_re, &z5_im);
                    remg_re = (
                        z_re + 0.5 * z2_re + z3_re / 6.0 +
                        z4_re / 24.0 + z5_re / 120.0
                    ) / r;
                    remg_im = (
                        z_im + 0.5 * z2_im + z3_im / 6.0 +
                        z4_im / 24.0 + z5_im / 120.0
                    ) / r;

                    double q2_re, q2_im, q3_re, q3_im, q4_re, q4_im, q5_re, q5_im;
                    cmul(q_re, q_im, q_re, q_im, &q2_re, &q2_im);
                    cmul(q2_re, q2_im, q_re, q_im, &q3_re, &q3_im);
                    cmul(q3_re, q3_im, q_re, q_im, &q4_re, &q4_im);
                    cmul(q4_re, q4_im, q_re, q_im, &q5_re, &q5_im);
                    expr_re = (
                        -0.5 * q2_re + q3_im / 3.0 +
                        0.125 * q4_re - q5_im / 30.0
                    );
                    expr_im = (
                        -0.5 * q2_im - q3_re / 3.0 +
                        0.125 * q4_im + q5_re / 30.0
                    );
                } else {
                    const double decay = exp(-q_im);
                    const double phase_re = decay * cos(q_re);
                    const double phase_im = decay * sin(q_re);
                    remg_re = (phase_re - 1.0) / r;
                    remg_im = phase_im / r;
                    const double factor_re = -q_im - 1.0;
                    const double factor_im = q_re;
                    expr_re = phase_re * factor_re - phase_im * factor_im + 1.0;
                    expr_im = phase_re * factor_im + phase_im * factor_re;
                }

                const double wh = w * num[idx] / (r * r * r);
                s_re += remg_re * w;
                s_im += remg_im * w;
                h_re += expr_re * wh;
                h_im += expr_im * wh;
            }
        }
        const int64_t out_idx = 2 * pair;
        out_s[out_idx] = s_re;
        out_s[out_idx + 1] = s_im;
        out_h[out_idx] = h_re;
        out_h[out_idx + 1] = h_im;
    }
}

static void *eval_near_worker(void *raw) {
    const NearThreadTask *thread_task = (const NearThreadTask *)raw;
    eval_near_range(thread_task->task, thread_task->start, thread_task->stop);
    return NULL;
}

static void eval_near_onthefly_range(
    const NearOntheflyTask *task,
    int64_t start,
    int64_t stop
) {
    const int64_t node_count = task->node_count;
    const int64_t psi_count = task->psi_count;
    const int32_t image_count = task->has_baffle ? 2 : 1;
    const double four_pi = 4.0 * 3.141592653589793238462643383279502884;

    for (int64_t pair = start; pair < stop; ++pair) {
        const double rt = task->target_rho[pair];
        const double zt = task->target_z[pair];
        const double nr = task->normal_rho[pair];
        const double source_nz = task->normal_z[pair];
        double s_re = 0.0;
        double s_im = 0.0;
        double h_re = 0.0;
        double h_im = 0.0;

        for (int32_t image = 0; image < image_count; ++image) {
            const double nz = image ? -source_nz : source_nz;
            double part_s_re = 0.0;
            double part_s_im = 0.0;
            double part_h_re = 0.0;
            double part_h_im = 0.0;
            for (int64_t node = 0; node < node_count; ++node) {
                const int64_t source_idx = pair * node_count + node;
                const double source_measure = task->measure[source_idx];
                if (source_measure == 0.0) {
                    continue;
                }
                const double rs = task->source_rho[source_idx];
                const double base_zs = task->source_z[source_idx];
                const double zs = image ? 2.0 * task->baffle_z - base_zs : base_zs;
                const double dz = zs - zt;

                for (int64_t psi = 0; psi < psi_count; ++psi) {
                    const double cp = task->cos_psi[psi];
                    double r2 = rt * rt + rs * rs - 2.0 * rt * rs * cp + dz * dz;
                    if (r2 < 0.0) {
                        r2 = 0.0;
                    }
                    const double r = sqrt(r2);
                    const double w = (
                        source_measure * (2.0 * task->psi_weights[psi]) / four_pi
                    );
                    if (r <= 1.0e-13) {
                        part_s_re += (-task->ki) * w;
                        part_s_im += task->kr * w;
                        continue;
                    }

                    const double q_re = task->kr * r;
                    const double q_im = task->ki * r;
                    double remg_re;
                    double remg_im;
                    double expr_re;
                    double expr_im;
                    if (hypot(q_re, q_im) < 1.0e-5) {
                        const double z_re = -q_im;
                        const double z_im = q_re;
                        double z2_re, z2_im, z3_re, z3_im, z4_re, z4_im, z5_re, z5_im;
                        cmul(z_re, z_im, z_re, z_im, &z2_re, &z2_im);
                        cmul(z2_re, z2_im, z_re, z_im, &z3_re, &z3_im);
                        cmul(z3_re, z3_im, z_re, z_im, &z4_re, &z4_im);
                        cmul(z4_re, z4_im, z_re, z_im, &z5_re, &z5_im);
                        remg_re = (
                            z_re + 0.5 * z2_re + z3_re / 6.0 +
                            z4_re / 24.0 + z5_re / 120.0
                        ) / r;
                        remg_im = (
                            z_im + 0.5 * z2_im + z3_im / 6.0 +
                            z4_im / 24.0 + z5_im / 120.0
                        ) / r;

                        double q2_re, q2_im, q3_re, q3_im, q4_re, q4_im, q5_re, q5_im;
                        cmul(q_re, q_im, q_re, q_im, &q2_re, &q2_im);
                        cmul(q2_re, q2_im, q_re, q_im, &q3_re, &q3_im);
                        cmul(q3_re, q3_im, q_re, q_im, &q4_re, &q4_im);
                        cmul(q4_re, q4_im, q_re, q_im, &q5_re, &q5_im);
                        expr_re = (
                            -0.5 * q2_re + q3_im / 3.0 +
                            0.125 * q4_re - q5_im / 30.0
                        );
                        expr_im = (
                            -0.5 * q2_im - q3_re / 3.0 +
                            0.125 * q4_im + q5_re / 30.0
                        );
                    } else {
                        const double decay = exp(-q_im);
                        const double phase_re = decay * cos(q_re);
                        const double phase_im = decay * sin(q_re);
                        remg_re = (phase_re - 1.0) / r;
                        remg_im = phase_im / r;
                        const double factor_re = -q_im - 1.0;
                        const double factor_im = q_re;
                        expr_re = phase_re * factor_re - phase_im * factor_im + 1.0;
                        expr_im = phase_re * factor_im + phase_im * factor_re;
                    }

                    const double numerator = (rs - rt * cp) * nr + dz * nz;
                    const double wh = w * numerator / ((r * r) * r);
                    part_s_re += remg_re * w;
                    part_s_im += remg_im * w;
                    part_h_re += expr_re * wh;
                    part_h_im += expr_im * wh;
                }
            }
            s_re += part_s_re;
            s_im += part_s_im;
            h_re += part_h_re;
            h_im += part_h_im;
        }
        const int64_t out_idx = 2 * pair;
        task->out_s[out_idx] = s_re;
        task->out_s[out_idx + 1] = s_im;
        task->out_h[out_idx] = h_re;
        task->out_h[out_idx + 1] = h_im;
    }
}

static void *eval_near_onthefly_worker(void *raw) {
    const NearOntheflyThreadTask *thread_task =
        (const NearOntheflyThreadTask *)raw;
    eval_near_onthefly_range(
        thread_task->task,
        thread_task->start,
        thread_task->stop
    );
    return NULL;
}

int circsym_eval_far_remainder_onthefly(
    int64_t nt,
    int64_t ns,
    int64_t nl,
    int64_t np,
    const double *target_rho,
    const double *target_z,
    const double *source_rho,
    const double *source_z,
    const double *measure,
    const double *normal_rho,
    const double *normal_z,
    const double *cos_psi,
    const double *psi_weights,
    int32_t has_baffle,
    double baffle_z,
    double kr,
    double ki,
    double *out_s,
    double *out_h,
    int32_t requested_threads
) {
    if (nt < 0 || ns < 0 || nl < 0 || np < 0 ||
        target_rho == NULL || target_z == NULL ||
        source_rho == NULL || source_z == NULL || measure == NULL ||
        normal_rho == NULL || normal_z == NULL ||
        cos_psi == NULL || psi_weights == NULL ||
        out_s == NULL || out_h == NULL) {
        return -1;
    }

    FarOntheflyTask task;
    task.nt = nt;
    task.ns = ns;
    task.nl = nl;
    task.np = np;
    task.target_rho = target_rho;
    task.target_z = target_z;
    task.source_rho = source_rho;
    task.source_z = source_z;
    task.measure = measure;
    task.normal_rho = normal_rho;
    task.normal_z = normal_z;
    task.cos_psi = cos_psi;
    task.psi_weights = psi_weights;
    task.has_baffle = has_baffle;
    task.baffle_z = baffle_z;
    task.kr = kr;
    task.ki = ki;
    task.out_s = out_s;
    task.out_h = out_h;

    int32_t threads = requested_threads;
    if (threads < 1) {
        threads = 1;
    }
    if ((int64_t)threads > nt) {
        threads = (int32_t)nt;
    }
    if (threads <= 1 || nt <= 1) {
        eval_far_onthefly_range(&task, 0, nt);
        return 0;
    }

    pthread_t *handles = (pthread_t *)malloc((size_t)threads * sizeof(pthread_t));
    FarOntheflyThreadTask *thread_tasks = (FarOntheflyThreadTask *)malloc(
        (size_t)threads * sizeof(FarOntheflyThreadTask)
    );
    if (handles == NULL || thread_tasks == NULL) {
        free(handles);
        free(thread_tasks);
        eval_far_onthefly_range(&task, 0, nt);
        return 0;
    }

    int32_t created = 0;
    for (int32_t t = 0; t < threads; ++t) {
        const int64_t start = (nt * (int64_t)t) / (int64_t)threads;
        const int64_t stop = (nt * (int64_t)(t + 1)) / (int64_t)threads;
        thread_tasks[t].task = &task;
        thread_tasks[t].start = start;
        thread_tasks[t].stop = stop;
        if (pthread_create(
                &handles[t], NULL, eval_far_onthefly_worker, &thread_tasks[t]
            ) != 0) {
            break;
        }
        created += 1;
    }

    for (int32_t t = 0; t < created; ++t) {
        pthread_join(handles[t], NULL);
    }
    if (created != threads) {
        eval_far_onthefly_range(&task, 0, nt);
    }
    free(handles);
    free(thread_tasks);
    return 0;
}

int circsym_eval_near_remainder(
    int64_t pair_count,
    int64_t node_count,
    int64_t psi_count,
    const double *R,
    const double *num,
    const double *weight,
    double kr,
    double ki,
    double *out_s,
    double *out_h,
    int32_t requested_threads
) {
    if (pair_count < 0 || node_count < 0 || psi_count < 0 ||
        R == NULL || num == NULL || weight == NULL ||
        out_s == NULL || out_h == NULL) {
        return -1;
    }
    NearTask task;
    task.pair_count = pair_count;
    task.node_count = node_count;
    task.psi_count = psi_count;
    task.R = R;
    task.num = num;
    task.weight = weight;
    task.kr = kr;
    task.ki = ki;
    task.out_s = out_s;
    task.out_h = out_h;

    int32_t threads = requested_threads;
    if (threads < 1) {
        threads = 1;
    }
    if ((int64_t)threads > pair_count) {
        threads = (int32_t)pair_count;
    }
    if (threads <= 1 || pair_count <= 1) {
        eval_near_range(&task, 0, pair_count);
        return 0;
    }

    pthread_t *handles = (pthread_t *)malloc((size_t)threads * sizeof(pthread_t));
    NearThreadTask *thread_tasks =
        (NearThreadTask *)malloc((size_t)threads * sizeof(NearThreadTask));
    if (handles == NULL || thread_tasks == NULL) {
        free(handles);
        free(thread_tasks);
        eval_near_range(&task, 0, pair_count);
        return 0;
    }

    int32_t created = 0;
    for (int32_t t = 0; t < threads; ++t) {
        const int64_t start = (pair_count * (int64_t)t) / (int64_t)threads;
        const int64_t stop = (pair_count * (int64_t)(t + 1)) / (int64_t)threads;
        thread_tasks[t].task = &task;
        thread_tasks[t].start = start;
        thread_tasks[t].stop = stop;
        if (pthread_create(&handles[t], NULL, eval_near_worker, &thread_tasks[t]) != 0) {
            break;
        }
        created += 1;
    }

    for (int32_t t = 0; t < created; ++t) {
        pthread_join(handles[t], NULL);
    }
    if (created != threads) {
        eval_near_range(&task, 0, pair_count);
    }
    free(handles);
    free(thread_tasks);
    return 0;
}

int circsym_eval_near_remainder_onthefly(
    int64_t pair_count,
    int64_t node_count,
    int64_t psi_count,
    const double *target_rho,
    const double *target_z,
    const double *source_rho,
    const double *source_z,
    const double *measure,
    const double *normal_rho,
    const double *normal_z,
    const double *cos_psi,
    const double *psi_weights,
    int32_t has_baffle,
    double baffle_z,
    double kr,
    double ki,
    double *out_s,
    double *out_h,
    int32_t requested_threads
) {
    if (pair_count < 0 || node_count < 0 || psi_count < 0 ||
        target_rho == NULL || target_z == NULL ||
        source_rho == NULL || source_z == NULL || measure == NULL ||
        normal_rho == NULL || normal_z == NULL ||
        cos_psi == NULL || psi_weights == NULL ||
        out_s == NULL || out_h == NULL) {
        return -1;
    }
    NearOntheflyTask task;
    task.pair_count = pair_count;
    task.node_count = node_count;
    task.psi_count = psi_count;
    task.target_rho = target_rho;
    task.target_z = target_z;
    task.source_rho = source_rho;
    task.source_z = source_z;
    task.measure = measure;
    task.normal_rho = normal_rho;
    task.normal_z = normal_z;
    task.cos_psi = cos_psi;
    task.psi_weights = psi_weights;
    task.has_baffle = has_baffle;
    task.baffle_z = baffle_z;
    task.kr = kr;
    task.ki = ki;
    task.out_s = out_s;
    task.out_h = out_h;

    int32_t threads = requested_threads;
    if (threads < 1) {
        threads = 1;
    }
    if ((int64_t)threads > pair_count) {
        threads = (int32_t)pair_count;
    }
    if (threads <= 1 || pair_count <= 1) {
        eval_near_onthefly_range(&task, 0, pair_count);
        return 0;
    }

    pthread_t *handles = (pthread_t *)malloc((size_t)threads * sizeof(pthread_t));
    NearOntheflyThreadTask *thread_tasks = (NearOntheflyThreadTask *)malloc(
        (size_t)threads * sizeof(NearOntheflyThreadTask)
    );
    if (handles == NULL || thread_tasks == NULL) {
        free(handles);
        free(thread_tasks);
        eval_near_onthefly_range(&task, 0, pair_count);
        return 0;
    }

    int32_t created = 0;
    for (int32_t t = 0; t < threads; ++t) {
        const int64_t start = (pair_count * (int64_t)t) / (int64_t)threads;
        const int64_t stop = (pair_count * (int64_t)(t + 1)) / (int64_t)threads;
        thread_tasks[t].task = &task;
        thread_tasks[t].start = start;
        thread_tasks[t].stop = stop;
        if (pthread_create(
                &handles[t], NULL, eval_near_onthefly_worker, &thread_tasks[t]
            ) != 0) {
            break;
        }
        created += 1;
    }

    for (int32_t t = 0; t < created; ++t) {
        pthread_join(handles[t], NULL);
    }
    if (created != threads) {
        const int64_t fallback_start =
            (pair_count * (int64_t)created) / (int64_t)threads;
        eval_near_onthefly_range(&task, fallback_start, pair_count);
    }
    free(handles);
    free(thread_tasks);
    return 0;
}
"""


class _CircsymRemainderCKernel:
    def __init__(self, library_path: str) -> None:
        import fcntl

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        self._lease_fd: int | None = None
        self._lease_fd = os.open(library_path, flags)
        try:
            lease_stat = os.fstat(self._lease_fd)
            path_stat = os.lstat(library_path)
            if (
                _cache_file_identity(lease_stat) != _cache_file_identity(path_stat)
                or not stat.S_ISREG(lease_stat.st_mode)
                or lease_stat.st_uid != os.geteuid()
                or stat.S_IMODE(lease_stat.st_mode) != 0o700
                or lease_stat.st_nlink != 1
                or lease_stat.st_size <= 0
            ):
                raise PermissionError(
                    "CircSym C-kernel generation failed load-time validation"
                )
            fcntl.flock(self._lease_fd, fcntl.LOCK_SH)
            self.library = ctypes.CDLL(library_path)
            self.eval_near = self.library.circsym_eval_near_remainder
            self.eval_near_onthefly = (
                self.library.circsym_eval_near_remainder_onthefly
            )
            double_ptr = ctypes.POINTER(ctypes.c_double)
            self.eval_far_onthefly = self.library.circsym_eval_far_remainder_onthefly
            self.eval_far_onthefly.argtypes = [
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                ctypes.c_int32,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                double_ptr,
                double_ptr,
                ctypes.c_int32,
            ]
            self.eval_far_onthefly.restype = ctypes.c_int
            self.eval_near.argtypes = [
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                double_ptr,
                double_ptr,
                double_ptr,
                ctypes.c_double,
                ctypes.c_double,
                double_ptr,
                double_ptr,
                ctypes.c_int32,
            ]
            self.eval_near.restype = ctypes.c_int
            self.eval_near_onthefly.argtypes = [
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                double_ptr,
                ctypes.c_int32,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                double_ptr,
                double_ptr,
                ctypes.c_int32,
            ]
            self.eval_near_onthefly.restype = ctypes.c_int
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if getattr(self, "_lease_fd", None) is not None:
            os.close(self._lease_fd)
            self._lease_fd = None

    def __del__(self) -> None:
        self.close()

    def validate_build_fingerprint(self, expected: str) -> None:
        fingerprint = self.library.circsym_c_kernel_build_fingerprint
        fingerprint.argtypes = []
        fingerprint.restype = ctypes.c_char_p
        actual = fingerprint()
        if actual is None or actual.decode("ascii", errors="replace") != expected:
            raise RuntimeError(
                "CircSym C-kernel build fingerprint does not match its cache key"
            )


@dataclass(frozen=True)
class _CircsymRemainderKernel:
    backend: str
    implementation: Any


def _circsym_c_kernel_cache_dir() -> str:
    """Return a per-user cache directory for the runtime C kernel.

    The former cache lived directly under ``tempfile.gettempdir()``.  Besides
    being shared by every account on a machine, that let unrelated processes
    compile to the same final library path at the same time.  Follow the host's
    user-cache convention instead; an absolute XDG override is honoured on all
    POSIX hosts so containers and test runners can keep their caches isolated.
    """

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache and os.path.isabs(xdg_cache):
        base = xdg_cache
    elif platform.system() == "Darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.path.expanduser("~/.cache")
    return os.path.join(base, "hornlab-metal-bem", "circsym")


def _circsym_c_kernel_link_mode(platform_name: str) -> str:
    return "-dynamiclib" if platform_name == "darwin" else "-shared"


def _circsym_c_kernel_rendered_source(build_fingerprint: str) -> str:
    if len(build_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in build_fingerprint
    ):
        raise ValueError("invalid CircSym C-kernel build fingerprint")
    return (
        _CIRCSYM_REMAINDER_C_SOURCE
        + "\nconst char *circsym_c_kernel_build_fingerprint(void) {\n"
        + f'    return "{build_fingerprint}";\n'
        + "}\n"
    )


def _circsym_c_kernel_resolved_compiler(compiler: str) -> str:
    resolved = shutil.which(compiler)
    if resolved is None:
        raise FileNotFoundError(f"CircSym C compiler not found: {compiler}")
    return os.path.realpath(resolved)


def _hash_framed_bytes(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _circsym_c_kernel_build_fingerprint(
    *,
    compiler: str,
    platform_name: str,
) -> str:
    """Fingerprint the effective compiler, recipe, macros, and headers."""

    resolved_compiler = _circsym_c_kernel_resolved_compiler(compiler)
    compiler_digest = hashlib.sha256()
    with open(resolved_compiler, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            compiler_digest.update(chunk)

    def probe(arguments: list[str], *, source: bytes | None = None) -> bytes:
        completed = subprocess.run(
            [resolved_compiler, *arguments],
            input=source,
            check=True,
            capture_output=True,
        )
        return completed.stdout + b"\0" + completed.stderr

    link_mode = _circsym_c_kernel_link_mode(platform_name)
    normalized_recipe = (
        *_CIRCSYM_C_KERNEL_COMPILE_ARGS,
        link_mode,
        "<SOURCE>",
        "-o",
        "<OUTPUT>",
        *_CIRCSYM_C_KERNEL_LINK_ARGS,
    )
    # ``-dD`` retains the effective predefined macros while preprocessing the
    # actual headers used by this source. ``-v`` also records the compiler's
    # include search paths and effective sysroot. Every compile argument is
    # deliberately shared with the real compiler invocation below.
    preprocess_source = _circsym_c_kernel_rendered_source("0" * 64).encode()
    preprocessed = probe(
        [
            *_CIRCSYM_C_KERNEL_COMPILE_ARGS,
            "-E",
            "-dD",
            "-P",
            "-v",
            "-x",
            "c",
            "-",
        ],
        source=preprocess_source,
    )

    digest = hashlib.sha256()
    for material in (
        _CIRCSYM_C_KERNEL_BUILD_SCHEMA.encode(),
        os.fsencode(resolved_compiler),
        compiler_digest.digest(),
        probe(["--version"]),
        probe(["-dumpmachine"]),
        "\0".join(normalized_recipe).encode(),
        preprocessed,
    ):
        _hash_framed_bytes(digest, material)
    return digest.hexdigest()


def _circsym_c_kernel_cache_key(
    *,
    build_fingerprint: str | None = None,
    compiler: str | None = None,
    platform_name: str | None = None,
) -> str:
    """Hash the build fingerprint and binary compatibility domain."""

    effective_platform = platform.system().lower() if platform_name is None else platform_name
    if build_fingerprint is None:
        selected_compiler = os.environ.get("CC", "cc") if compiler is None else compiler
        build_fingerprint = _circsym_c_kernel_build_fingerprint(
            compiler=selected_compiler,
            platform_name=effective_platform,
        )

    discriminator = "\0".join(
        (
            sys.platform,
            platform.system(),
            platform.machine(),
            platform.architecture()[0],
            sysconfig.get_platform(),
            ":".join(platform.libc_ver()),
            sys.implementation.name,
            str(sys.implementation.cache_tag or ""),
            str(sysconfig.get_config_var("SOABI") or ""),
            str(ctypes.sizeof(ctypes.c_void_p) * 8),
        )
    )
    material = "\0".join(
        (_CIRCSYM_C_KERNEL_BUILD_SCHEMA, build_fingerprint, discriminator)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _open_private_cache_directory(path: str) -> int:
    """Open a user-owned real directory without following a leaf symlink."""

    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise PermissionError(
            f"CircSym C-kernel cache path is not a real directory: {path}"
        ) from exc
    try:
        path_stat = os.lstat(path)
        fd_stat = os.fstat(fd)
        if not stat.S_ISDIR(path_stat.st_mode) or not stat.S_ISDIR(fd_stat.st_mode):
            raise PermissionError(
                f"CircSym C-kernel cache path is not a directory: {path}"
            )
        if (path_stat.st_dev, path_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino):
            raise PermissionError(
                f"CircSym C-kernel cache path changed while opening: {path}"
            )
        if fd_stat.st_uid != os.geteuid():
            raise PermissionError(
                "CircSym C-kernel cache is not owned by the current user: "
                f"{path}"
            )
        return fd
    except Exception:
        os.close(fd)
        raise


def _ensure_private_cache_directory(path: str) -> None:
    """Create one cache-directory level and make it private without symlinks."""

    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    fd = _open_private_cache_directory(path)
    try:
        os.fchmod(fd, 0o700)
        if stat.S_IMODE(os.fstat(fd).st_mode) != 0o700:
            raise PermissionError(
                f"CircSym C-kernel cache directory is not private: {path}"
            )
    finally:
        os.close(fd)


def _prepare_circsym_c_kernel_cache(cache_dir: str) -> None:
    """Create and permission-check the private runtime-kernel cache."""

    application_dir = os.path.dirname(cache_dir)
    cache_base = os.path.dirname(application_dir)
    os.makedirs(cache_base, exist_ok=True)
    # These are the two application-controlled path components. The platform's
    # cache base may itself be redirected by the user (for example with XDG),
    # but neither component below it may be a symlink or belong to another UID.
    _ensure_private_cache_directory(application_dir)
    _ensure_private_cache_directory(cache_dir)


def _cache_file_identity(file_stat: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
    )


def _validated_private_cache_file(
    path: str,
    *,
    expected_mode: int,
) -> os.stat_result | None:
    """Return a trusted cache-file stat, deleting unsafe non-directories."""

    try:
        file_stat = os.lstat(path)
    except FileNotFoundError:
        return None
    valid = (
        stat.S_ISREG(file_stat.st_mode)
        and file_stat.st_uid == os.geteuid()
        and stat.S_IMODE(file_stat.st_mode) == expected_mode
        and file_stat.st_nlink == 1
        and file_stat.st_size > 0
    )
    if valid:
        return file_stat
    if stat.S_ISDIR(file_stat.st_mode):
        try:
            os.rmdir(path)
        except OSError as exc:
            raise PermissionError(
                f"unsafe directory occupies CircSym cache file path: {path}"
            ) from exc
        return None
    # The containing directory is user-owned and locked at this point. Removing
    # a symlink, FIFO, foreign-owned file, or permissive regular file is safe and
    # ensures no pre-hardening artifact is ever passed to ctypes.
    os.unlink(path)
    return None


def _circsym_c_kernel_artifact_paths(
    cache_dir: str,
    cache_key: str,
    platform_name: str,
) -> tuple[str, str, str, str]:
    """Return source, selection, legacy-library, and extension names."""

    lib_ext = ".dylib" if platform_name == "darwin" else ".so"
    stem = f"circsym_remainder_{cache_key}"
    return (
        os.path.join(cache_dir, f"{stem}.c"),
        os.path.join(cache_dir, f"{stem}.current"),
        os.path.join(cache_dir, f"{stem}{lib_ext}"),
        lib_ext,
    )


def _selected_circsym_c_kernel_library(
    *,
    cache_dir: str,
    cache_key: str,
    platform_name: str,
    selection_path: str,
) -> tuple[str, tuple[int, int, int, int]] | None:
    """Resolve a validated generation name from the private selection file."""

    selection_stat = _validated_private_cache_file(
        selection_path,
        expected_mode=0o600,
    )
    if selection_stat is None:
        return None
    try:
        with open(selection_path, encoding="ascii") as handle:
            generation = handle.read(512).strip()
    except (OSError, UnicodeError):
        os.unlink(selection_path)
        return None
    extension = ".dylib" if platform_name == "darwin" else ".so"
    prefix = f"circsym_remainder_{cache_key}."
    valid_name = (
        generation == os.path.basename(generation)
        and generation.startswith(prefix)
        and generation.endswith(extension)
        and len(generation) > len(prefix) + len(extension)
        and "/" not in generation
        and "\\" not in generation
    )
    if not valid_name:
        os.unlink(selection_path)
        return None
    library_path = os.path.join(cache_dir, generation)
    library_stat = _validated_private_cache_file(
        library_path,
        expected_mode=0o700,
    )
    if library_stat is None:
        os.unlink(selection_path)
        return None
    return library_path, _cache_file_identity(library_stat)


def _unlink_private_cache_file_if_identity(
    path: str,
    identity: tuple[int, int, int, int],
    *,
    expected_mode: int | tuple[int, ...] = 0o700,
    require_unleased: bool = False,
) -> bool:
    """Unlink one regular private file only while its identity still matches."""

    expected_modes = (
        (expected_mode,) if isinstance(expected_mode, int) else expected_mode
    )
    lease_fd: int | None = None
    try:
        if require_unleased:
            import fcntl

            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                lease_fd = os.open(path, flags)
                lease_stat = os.fstat(lease_fd)
                if _cache_file_identity(lease_stat) != identity:
                    return False
                fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, FileNotFoundError, OSError):
                return False
        try:
            file_stat = os.lstat(path)
        except FileNotFoundError:
            return False
        if (
            _cache_file_identity(file_stat) != identity
            or not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.geteuid()
            or stat.S_IMODE(file_stat.st_mode) not in expected_modes
            or file_stat.st_nlink != 1
        ):
            return False
        os.unlink(path)
        return True
    finally:
        if lease_fd is not None:
            os.close(lease_fd)


def _circsym_c_kernel_owned_artifact(
    name: str,
) -> tuple[tuple[int, ...], bool] | None:
    """Return accepted modes and whether an owned cache artifact is a library."""

    hidden = name.startswith(".")
    visible_name = name[1:] if hidden else name
    prefix = "circsym_remainder_"
    if not visible_name.startswith(prefix):
        return None
    keyed_suffix = visible_name[len(prefix) :]
    cache_key, separator, suffix = keyed_suffix.partition(".")
    if (
        not separator
        or len(cache_key) != 24
        or any(character not in "0123456789abcdef" for character in cache_key)
    ):
        return None

    token_characters = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    )

    def valid_token(token: str) -> bool:
        return bool(token) and all(
            character in token_characters for character in token
        )

    if hidden:
        for ending, modes, is_library in (
            (".tmp.c", (0o600,), False),
            (".current.tmp", (0o600,), False),
            (".so.tmp", (0o600, 0o700), True),
            (".dylib.tmp", (0o600, 0o700), True),
        ):
            if suffix.endswith(ending) and valid_token(suffix[: -len(ending)]):
                return modes, is_library
        return None

    if suffix in {"c", "current"}:
        return (0o600,), False
    if suffix in {"so", "dylib"}:
        return (0o700,), True
    for extension in (".so", ".dylib"):
        if suffix.endswith(extension) and valid_token(suffix[: -len(extension)]):
            return (0o700,), True
    return None


def _collect_circsym_c_kernel_orphans(
    *,
    cache_dir: str,
    cache_key: str,
    platform_name: str,
    selected_path: str | None,
    now: float | None = None,
) -> int:
    """Remove a bounded batch of stale artifacts from the whole owned cache."""

    source_path, selection_path, _legacy_path, _extension = (
        _circsym_c_kernel_artifact_paths(cache_dir, cache_key, platform_name)
    )
    protected_paths = {source_path, selection_path}
    if selected_path is not None:
        protected_paths.add(selected_path)
    cutoff = (time.time() if now is None else now) - (
        _CIRCSYM_C_KERNEL_ORPHAN_GRACE_SECONDS
    )
    owned: list[
        tuple[
            int,
            str,
            tuple[int, int, int, int],
            tuple[int, ...],
            bool,
            int,
        ]
    ] = []
    owned_count = 0
    owned_bytes = 0
    with os.scandir(cache_dir) as entries:
        for entry in entries:
            artifact = _circsym_c_kernel_owned_artifact(entry.name)
            if artifact is None:
                continue
            expected_modes, is_library = artifact
            path = os.path.join(cache_dir, entry.name)
            try:
                file_stat = os.lstat(path)
            except FileNotFoundError:
                continue
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_uid != os.geteuid()
                or stat.S_IMODE(file_stat.st_mode) not in expected_modes
                or file_stat.st_nlink != 1
            ):
                continue
            owned_count += 1
            owned_bytes += file_stat.st_size
            if path in protected_paths:
                continue
            owned.append(
                (
                    file_stat.st_mtime_ns,
                    path,
                    _cache_file_identity(file_stat),
                    expected_modes,
                    is_library,
                    file_stat.st_size,
                )
            )

    over_ceiling = (
        owned_count > _CIRCSYM_C_KERNEL_CACHE_MAX_ARTIFACTS
        or owned_bytes > _CIRCSYM_C_KERNEL_CACHE_MAX_BYTES
    )
    candidates = [
        candidate
        for candidate in owned
        if candidate[0] <= int(cutoff * 1_000_000_000) or over_ceiling
    ]
    removed = 0
    removed_bytes = 0
    for _mtime, path, identity, modes, is_library, size in sorted(candidates):
        if removed >= _CIRCSYM_C_KERNEL_GC_MAX_REMOVALS:
            break
        if (
            removed > 0
            and removed_bytes + size > _CIRCSYM_C_KERNEL_GC_MAX_REMOVAL_BYTES
        ):
            break
        unlinked = _unlink_private_cache_file_if_identity(
            path,
            identity,
            expected_mode=modes,
            require_unleased=is_library,
        )
        removed += int(unlinked)
        removed_bytes += size if unlinked else 0
    return removed


def _compile_circsym_remainder_c_kernel(
    *,
    cache_dir: str,
    cache_key: str,
    platform_name: str,
    compiler: str,
    build_fingerprint: str | None = None,
    rejected_library_identity: tuple[int, int, int, int] | None = None,
) -> tuple[str, tuple[int, int, int, int]]:
    """Compile once across processes and publish the library atomically."""

    import fcntl

    resolved_compiler = _circsym_c_kernel_resolved_compiler(compiler)
    if build_fingerprint is None:
        build_fingerprint = _circsym_c_kernel_build_fingerprint(
            compiler=resolved_compiler,
            platform_name=platform_name,
        )
    expected_cache_key = _circsym_c_kernel_cache_key(
        build_fingerprint=build_fingerprint,
        platform_name=platform_name,
    )
    if cache_key != expected_cache_key:
        raise ValueError("CircSym C-kernel cache key does not match its build recipe")
    rendered_source = _circsym_c_kernel_rendered_source(build_fingerprint)

    source_path, selection_path, legacy_library_path, lib_ext = (
        _circsym_c_kernel_artifact_paths(
            cache_dir,
            cache_key,
            platform_name,
        )
    )
    cache_fd = _open_private_cache_directory(cache_dir)
    source_temp: str | None = None
    library_temp: str | None = None
    selection_temp: str | None = None
    generation_path: str | None = None
    try:
        # Lock the directory inode itself. This serializes validation and
        # publication without first having to trust a pre-existing lock file.
        fcntl.flock(cache_fd, fcntl.LOCK_EX)
        directory_stat = os.fstat(cache_fd)
        path_stat = os.lstat(cache_dir)
        if (
            not stat.S_ISDIR(path_stat.st_mode)
            or (path_stat.st_dev, path_stat.st_ino)
            != (directory_stat.st_dev, directory_stat.st_ino)
            or directory_stat.st_uid != os.geteuid()
        ):
            raise PermissionError(
                "CircSym C-kernel cache directory failed locked validation: "
                f"{cache_dir}"
            )
        os.fchmod(cache_fd, 0o700)
        if stat.S_IMODE(os.fstat(cache_fd).st_mode) != 0o700:
            raise PermissionError(
                "CircSym C-kernel cache directory is not private after locking: "
                f"{cache_dir}"
            )

        # Sanitize both canonical artifact names while holding the directory
        # lock. In particular, never trust a hash-named library planted while
        # an older cache directory was group- or world-writable.
        _validated_private_cache_file(source_path, expected_mode=0o600)
        legacy_library_stat = _validated_private_cache_file(
            legacy_library_path,
            expected_mode=0o700,
        )
        if legacy_library_stat is not None:
            # Before generation names, a failed dlopen was healed by replacing
            # this canonical pathname. Dynamic loaders cache handles by name,
            # so the process could keep seeing the rejected image after the
            # bytes changed. Never select that ambiguous legacy pathname.
            os.unlink(legacy_library_path)
        selected = _selected_circsym_c_kernel_library(
            cache_dir=cache_dir,
            cache_key=cache_key,
            platform_name=platform_name,
            selection_path=selection_path,
        )
        if selected is not None:
            library_path, identity = selected
            if rejected_library_identity is None or identity != rejected_library_identity:
                _collect_circsym_c_kernel_orphans(
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                    platform_name=platform_name,
                    selected_path=library_path,
                )
                return library_path, identity
            # The loader rejected this exact generation. Reconfirm both its
            # selection and file identity under the lock before removing it;
            # a concurrent healer may already have published a replacement.
            confirmed = _selected_circsym_c_kernel_library(
                cache_dir=cache_dir,
                cache_key=cache_key,
                platform_name=platform_name,
                selection_path=selection_path,
            )
            if confirmed == selected and _unlink_private_cache_file_if_identity(
                library_path,
                identity,
                require_unleased=True,
            ):
                os.unlink(selection_path)
            selected = None

        _collect_circsym_c_kernel_orphans(
            cache_dir=cache_dir,
            cache_key=cache_key,
            platform_name=platform_name,
            selected_path=None,
        )

        source_fd, source_temp = tempfile.mkstemp(
            prefix=f".circsym_remainder_{cache_key}.",
            # Keep the language-bearing extension last so clang and gcc
            # recognize this as C without relying on a compiler-specific
            # command-line override.
            suffix=".tmp.c",
            dir=cache_dir,
        )
        with os.fdopen(source_fd, "w", encoding="utf-8") as handle:
            handle.write(rendered_source)

        library_fd, library_temp = tempfile.mkstemp(
            prefix=f".circsym_remainder_{cache_key}.",
            suffix=f"{lib_ext}.tmp",
            dir=cache_dir,
        )
        os.close(library_fd)
        command = [
            resolved_compiler,
            *_CIRCSYM_C_KERNEL_COMPILE_ARGS,
            _circsym_c_kernel_link_mode(platform_name),
        ]
        command.extend(
            [source_temp, "-o", library_temp, *_CIRCSYM_C_KERNEL_LINK_ARGS]
        )
        subprocess.run(command, check=True, capture_output=True, text=True)
        if (
            _circsym_c_kernel_build_fingerprint(
                compiler=resolved_compiler,
                platform_name=platform_name,
            )
            != build_fingerprint
        ):
            raise RuntimeError(
                "CircSym C compiler or build inputs changed during compilation"
            )
        os.chmod(library_temp, 0o700)

        generation_path = os.path.join(
            cache_dir,
            f"circsym_remainder_{cache_key}.{secrets.token_hex(12)}{lib_ext}",
        )

        # Both names become visible only after their contents are complete. The
        # generation name is new on every build, so a dynamic loader can never
        # return a cached handle for rejected bytes at an atomically-replaced
        # canonical pathname. The selection file is published last and is the
        # only route by which another process discovers this generation.
        os.replace(source_temp, source_path)
        source_temp = None
        os.replace(library_temp, generation_path)
        library_temp = None
        selection_fd, selection_temp = tempfile.mkstemp(
            prefix=f".circsym_remainder_{cache_key}.",
            suffix=".current.tmp",
            dir=cache_dir,
        )
        with os.fdopen(selection_fd, "w", encoding="ascii") as handle:
            handle.write(os.path.basename(generation_path) + "\n")
        os.chmod(selection_temp, 0o600)
        os.replace(selection_temp, selection_path)
        selection_temp = None
        final_source_stat = _validated_private_cache_file(
            source_path,
            expected_mode=0o600,
        )
        final_selection = _selected_circsym_c_kernel_library(
            cache_dir=cache_dir,
            cache_key=cache_key,
            platform_name=platform_name,
            selection_path=selection_path,
        )
        if final_source_stat is None or final_selection is None:
            raise PermissionError(
                "CircSym C-kernel cache publication failed validation"
            )
        if final_selection[0] != generation_path:
            raise PermissionError(
                "CircSym C-kernel cache selected an unexpected generation"
            )
        generation_path = None
        return final_selection
    finally:
        for temporary in (source_temp, library_temp, selection_temp):
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        if generation_path is not None:
            try:
                current = _selected_circsym_c_kernel_library(
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                    platform_name=platform_name,
                    selection_path=selection_path,
                )
                if current is None or current[0] != generation_path:
                    generation_stat = _validated_private_cache_file(
                        generation_path,
                        expected_mode=0o700,
                    )
                    if generation_stat is not None:
                        _unlink_private_cache_file_if_identity(
                            generation_path,
                            _cache_file_identity(generation_stat),
                        )
            except FileNotFoundError:
                pass
        os.close(cache_fd)


@lru_cache(maxsize=1)
def _load_circsym_remainder_c_kernel() -> _CircsymRemainderCKernel | None:
    global _circsym_c_kernel_failure
    if os.environ.get("HORNLAB_CIRCSYM_DISABLE_C_REMAINDER_KERNEL"):
        _circsym_c_kernel_failure = "disabled by HORNLAB_CIRCSYM_DISABLE_C_REMAINDER_KERNEL"
        return None
    if platform.system() == "Windows":
        _circsym_c_kernel_failure = (
            "the legacy runtime C kernel requires a POSIX compiler and pthreads"
        )
        return None
    try:
        platform_name = platform.system().lower()
        cc = _circsym_c_kernel_resolved_compiler(os.environ.get("CC", "cc"))
        build_fingerprint = _circsym_c_kernel_build_fingerprint(
            compiler=cc,
            platform_name=platform_name,
        )
        cache_key = _circsym_c_kernel_cache_key(
            build_fingerprint=build_fingerprint,
            platform_name=platform_name,
        )
        cache_dir = _circsym_c_kernel_cache_dir()
        _prepare_circsym_c_kernel_cache(cache_dir)
        library_path, library_identity = _compile_circsym_remainder_c_kernel(
            cache_dir=cache_dir,
            cache_key=cache_key,
            platform_name=platform_name,
            compiler=cc,
            build_fingerprint=build_fingerprint,
        )
    except Exception as exc:
        _circsym_c_kernel_failure = str(exc)
        logger.debug("CircSym C remainder kernel unavailable: %s", exc)
        return None
    for attempt in range(2):
        kernel: _CircsymRemainderCKernel | None = None
        try:
            kernel = _CircsymRemainderCKernel(library_path)
            kernel.validate_build_fingerprint(build_fingerprint)
            _circsym_c_kernel_failure = None
            return kernel
        except Exception as exc:
            if kernel is not None:
                kernel.close()
            if attempt == 0:
                logger.debug(
                    "CircSym C remainder kernel load failed; rebuilding once: %s",
                    exc,
                )
                try:
                    library_path, library_identity = (
                        _compile_circsym_remainder_c_kernel(
                            cache_dir=cache_dir,
                            cache_key=cache_key,
                            platform_name=platform_name,
                            compiler=cc,
                            build_fingerprint=build_fingerprint,
                            rejected_library_identity=library_identity,
                        )
                    )
                    continue
                except Exception as rebuild_exc:
                    exc = rebuild_exc
            _circsym_c_kernel_failure = str(exc)
            logger.debug("CircSym C remainder kernel load failed: %s", exc)
            return None
    return None


@lru_cache(maxsize=1)
def _load_circsym_remainder_numba_kernel() -> Any | None:
    global _circsym_numba_kernel_failure
    try:
        from . import _circsym_numba
    except Exception as exc:
        _circsym_numba_kernel_failure = str(exc)
        logger.debug("CircSym Numba remainder kernel unavailable: %s", exc)
        return None
    _circsym_numba_kernel_failure = None
    return _circsym_numba


def _requested_circsym_cpu_remainder_backend() -> str:
    backend = os.environ.get(
        _CIRCSYM_CPU_REMAINDER_BACKEND_ENV,
        "auto",
    ).strip().lower()
    if backend not in {"auto", "c", "numba", "numpy"}:
        raise ValueError(
            f"{_CIRCSYM_CPU_REMAINDER_BACKEND_ENV} must be 'auto', 'c', "
            "'numba', or 'numpy'"
        )
    return backend


@lru_cache(maxsize=1)
def _load_circsym_remainder_kernel() -> _CircsymRemainderKernel | None:
    requested = _requested_circsym_cpu_remainder_backend()
    if requested == "numpy":
        return None

    if requested in {"auto", "c"}:
        c_kernel = _load_circsym_remainder_c_kernel()
        if c_kernel is not None:
            return _CircsymRemainderKernel("c", c_kernel)
        if requested == "c":
            raise RuntimeError(
                "CircSym C remainder backend was requested but is unavailable: "
                + (_circsym_c_kernel_failure or "unknown load failure")
            )

    numba_kernel = _load_circsym_remainder_numba_kernel()
    if numba_kernel is not None:
        return _CircsymRemainderKernel("numba", numba_kernel)
    if requested == "numba":
        raise RuntimeError(
            "CircSym Numba remainder backend was requested but is unavailable: "
            + (_circsym_numba_kernel_failure or "unknown import failure")
        )

    logger.warning(
        "CircSym compiled CPU remainder kernels are unavailable; using the "
        "slower NumPy reference path (C: %s; Numba: %s)",
        _circsym_c_kernel_failure or "not available",
        _circsym_numba_kernel_failure or "not available",
    )
    return None


def _circsym_cpu_remainder_status() -> dict[str, Any]:
    kernel = _load_circsym_remainder_kernel()
    return {
        "policy": _requested_circsym_cpu_remainder_backend(),
        "selected": "numpy" if kernel is None else kernel.backend,
        "c_unavailable_reason": _circsym_c_kernel_failure,
        "numba_unavailable_reason": _circsym_numba_kernel_failure,
    }


def _requested_circsym_cpu_field_backend() -> str:
    backend = os.environ.get(_CIRCSYM_CPU_FIELD_BACKEND_ENV, "numpy").strip().lower()
    if backend not in {"numpy", "numba"}:
        raise ValueError(
            f"{_CIRCSYM_CPU_FIELD_BACKEND_ENV} must be 'numpy' or 'numba'"
        )
    return backend


def _circsym_cpu_field_status() -> dict[str, Any]:
    requested = _requested_circsym_cpu_field_backend()
    implementation = (
        _load_circsym_remainder_numba_kernel() if requested == "numba" else None
    )
    if requested == "numba" and implementation is None:
        raise RuntimeError(
            "CircSym Numba field backend was requested but is unavailable: "
            + (_circsym_numba_kernel_failure or "unknown import failure")
        )
    return {
        "policy": requested,
        "selected": "numba" if implementation is not None else "numpy",
        "numba_unavailable_reason": _circsym_numba_kernel_failure,
    }


def _evaluate_far_remainder_onthefly_compiled(
    kernel: _CircsymRemainderCKernel,
    part: _FarRemainderCompactGeometry,
    k: complex,
    *,
    workers: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    target_rho = np.ascontiguousarray(part.target_rho, dtype=np.float64)
    target_z = np.ascontiguousarray(part.target_z, dtype=np.float64)
    source_rho = np.ascontiguousarray(part.source_rho, dtype=np.float64)
    source_z = np.ascontiguousarray(part.source_z, dtype=np.float64)
    measure = np.ascontiguousarray(part.measure, dtype=np.float64)
    normal_rho = np.ascontiguousarray(part.normal_rho, dtype=np.float64)
    normal_z = np.ascontiguousarray(part.normal_z, dtype=np.float64)
    cos_psi = np.ascontiguousarray(part.cos_psi, dtype=np.float64)
    psi_weights = np.ascontiguousarray(part.psi_weights, dtype=np.float64)
    target_count = int(target_rho.size)
    source_count, line_count = source_rho.shape
    psi_count = int(cos_psi.size)
    s_out = np.empty((target_count, source_count), dtype=np.complex128)
    h_out = np.empty_like(s_out)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    k_value = complex(k)
    block_size = (
        target_count
        if should_continue is None
        else max(1, min(target_count, _ASSEMBLY_KERNEL_MAX_TARGET_BLOCK))
    )
    for start in range(0, target_count, block_size):
        _check_circsym_continue(should_continue)
        stop = min(target_count, start + block_size)
        target_rho_block = target_rho[start:stop]
        target_z_block = target_z[start:stop]
        s_block = s_out[start:stop]
        h_block = h_out[start:stop]
        status = kernel.eval_far_onthefly(
            ctypes.c_int64(stop - start),
            ctypes.c_int64(source_count),
            ctypes.c_int64(line_count),
            ctypes.c_int64(psi_count),
            target_rho_block.ctypes.data_as(double_ptr),
            target_z_block.ctypes.data_as(double_ptr),
            source_rho.ctypes.data_as(double_ptr),
            source_z.ctypes.data_as(double_ptr),
            measure.ctypes.data_as(double_ptr),
            normal_rho.ctypes.data_as(double_ptr),
            normal_z.ctypes.data_as(double_ptr),
            cos_psi.ctypes.data_as(double_ptr),
            psi_weights.ctypes.data_as(double_ptr),
            ctypes.c_int32(part.baffle_z is not None),
            ctypes.c_double(0.0 if part.baffle_z is None else float(part.baffle_z)),
            ctypes.c_double(float(k_value.real)),
            ctypes.c_double(float(k_value.imag)),
            s_block.ctypes.data_as(double_ptr),
            h_block.ctypes.data_as(double_ptr),
            ctypes.c_int32(max(1, min(int(workers), stop - start))),
        )
        if int(status) != 0:
            return _evaluate_far_remainder_onthefly_reference(
                part,
                k,
                workers=workers,
                should_continue=should_continue,
            )
        _check_circsym_continue(should_continue)
    return s_out, h_out


def _evaluate_far_remainder_with_kernel(
    kernel: _CircsymRemainderKernel,
    part: _FarRemainderCompactGeometry,
    k: complex,
    *,
    workers: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if kernel.backend == "c":
        return _evaluate_far_remainder_onthefly_compiled(
            kernel.implementation,
            part,
            k,
            workers=workers,
            should_continue=should_continue,
        )
    if kernel.backend != "numba":
        raise RuntimeError(f"unknown CircSym remainder backend {kernel.backend!r}")

    target_count = int(part.target_rho.size)
    source_count = int(part.source_rho.shape[0])
    s_out = np.empty((target_count, source_count), dtype=np.complex128)
    h_out = np.empty_like(s_out)
    block_size = (
        target_count
        if should_continue is None
        else max(1, min(target_count, _ASSEMBLY_KERNEL_MAX_TARGET_BLOCK))
    )
    k_value = complex(k)
    for start in range(0, target_count, block_size):
        _check_circsym_continue(should_continue)
        stop = min(target_count, start + block_size)
        s_block, h_block = kernel.implementation.evaluate_far_remainder_onthefly(
            np.ascontiguousarray(part.target_rho[start:stop], dtype=np.float64),
            np.ascontiguousarray(part.target_z[start:stop], dtype=np.float64),
            np.ascontiguousarray(part.source_rho, dtype=np.float64),
            np.ascontiguousarray(part.source_z, dtype=np.float64),
            np.ascontiguousarray(part.measure, dtype=np.float64),
            np.ascontiguousarray(part.normal_rho, dtype=np.float64),
            np.ascontiguousarray(part.normal_z, dtype=np.float64),
            np.ascontiguousarray(part.cos_psi, dtype=np.float64),
            np.ascontiguousarray(part.psi_weights, dtype=np.float64),
            part.baffle_z is not None,
            0.0 if part.baffle_z is None else float(part.baffle_z),
            float(k_value.real),
            float(k_value.imag),
        )
        s_out[start:stop] = s_block
        h_out[start:stop] = h_block
        _check_circsym_continue(should_continue)
    return s_out, h_out


def _evaluate_far_remainder_onthefly_metal(
    part: _FarRemainderCompactGeometry,
    k: complex,
    *,
    runtime_status: Any,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    from .metal.native import CircSymMetalCancelled
    from .metal.native import evaluate_circsym_ring_remainder_kernels

    _check_circsym_continue(should_continue)
    try:
        result = evaluate_circsym_ring_remainder_kernels(
            target_rho=part.target_rho,
            target_z=part.target_z,
            source_rho=part.source_rho,
            source_z=part.source_z,
            measure=part.measure,
            normal_rho=part.normal_rho,
            normal_z=part.normal_z,
            cos_psi=part.cos_psi,
            psi_weights=part.psi_weights,
            k=k,
            baffle_z=part.baffle_z,
            runtime_status=runtime_status,
            should_continue=should_continue,
        )
    except CircSymMetalCancelled as exc:
        raise CircSymCancelled("CircSym solve cancelled") from exc
    except Exception as exc:
        raise _CircSymMetalAccelerationError(
            f"native CircSym remainder kernel failed: {exc}"
        ) from exc
    _check_circsym_continue(should_continue)
    return result.slp, result.dlp


def _evaluate_far_remainder_onthefly_metal_batch(
    parts: tuple[_FarRemainderCompactGeometry, ...],
    k_values: NDArray[np.complex128],
    *,
    runtime_status: Any,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    dict[str, Any],
]:
    from .metal.native import CircSymMetalCancelled
    from .metal.native import evaluate_circsym_ring_remainder_kernels_batch

    if not parts:
        raise ValueError("parts must be non-empty")
    _check_circsym_continue(should_continue)
    geometry = parts[0]
    try:
        result = evaluate_circsym_ring_remainder_kernels_batch(
            target_rho=geometry.target_rho,
            target_z=geometry.target_z,
            source_rho=geometry.source_rho,
            source_z=geometry.source_z,
            measure=geometry.measure,
            normal_rho=geometry.normal_rho,
            normal_z=geometry.normal_z,
            cos_psi_by_frequency=tuple(part.cos_psi for part in parts),
            psi_weights_by_frequency=tuple(part.psi_weights for part in parts),
            k_values=np.asarray(k_values, dtype=np.complex128),
            baffle_z=geometry.baffle_z,
            runtime_status=runtime_status,
            should_continue=should_continue,
            operation_id="circsym-assembly-frequency-batch",
        )
    except CircSymMetalCancelled as exc:
        raise CircSymCancelled("CircSym solve cancelled") from exc
    except Exception as exc:
        raise _CircSymMetalAccelerationError(
            f"native CircSym remainder batch failed: {exc}"
        ) from exc
    _check_circsym_continue(should_continue)
    return result.slp, result.dlp, result.diagnostics


def _evaluate_near_remainder_compiled(
    kernel: _CircsymRemainderCKernel,
    part: _NearRemainderGeometry,
    k: complex,
    *,
    workers: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if part.R.shape[0] == 0:
        return (
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.complex128),
        )
    R = part.R if part.R.flags.c_contiguous else np.ascontiguousarray(part.R)
    num = part.num if part.num.flags.c_contiguous else np.ascontiguousarray(part.num)
    weight = (
        part.weight
        if part.weight.flags.c_contiguous
        else np.ascontiguousarray(part.weight)
    )
    pair_count, node_count, psi_count = R.shape
    s_out = np.empty(pair_count, dtype=np.complex128)
    h_out = np.empty_like(s_out)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    status = kernel.eval_near(
        ctypes.c_int64(pair_count),
        ctypes.c_int64(node_count),
        ctypes.c_int64(psi_count),
        R.ctypes.data_as(double_ptr),
        num.ctypes.data_as(double_ptr),
        weight.ctypes.data_as(double_ptr),
        ctypes.c_double(float(complex(k).real)),
        ctypes.c_double(float(complex(k).imag)),
        s_out.ctypes.data_as(double_ptr),
        h_out.ctypes.data_as(double_ptr),
        ctypes.c_int32(max(1, int(workers))),
    )
    if int(status) != 0:
        return _evaluate_near_remainder(part, k)
    return s_out, h_out


def _evaluate_near_remainder_with_kernel(
    kernel: _CircsymRemainderKernel,
    part: _NearRemainderGeometry,
    k: complex,
    *,
    workers: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if kernel.backend == "c":
        return _evaluate_near_remainder_compiled(
            kernel.implementation,
            part,
            k,
            workers=workers,
        )
    if kernel.backend != "numba":
        raise RuntimeError(f"unknown CircSym remainder backend {kernel.backend!r}")
    if part.R.shape[0] == 0:
        return (
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.complex128),
        )
    del workers
    k_value = complex(k)
    return kernel.implementation.evaluate_near_remainder(
        np.ascontiguousarray(part.R, dtype=np.float64),
        np.ascontiguousarray(part.num, dtype=np.float64),
        np.ascontiguousarray(part.weight, dtype=np.float64),
        float(k_value.real),
        float(k_value.imag),
    )


def _evaluate_near_remainder_compact_compiled(
    kernel: _CircsymRemainderCKernel,
    part: _NearRemainderCompactGeometry,
    k: complex,
    *,
    workers: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    pairs = part.pairs
    pair_count = int(pairs.target_rho.size)
    if pair_count == 0:
        return (
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.complex128),
        )
    target_rho = np.ascontiguousarray(pairs.target_rho, dtype=np.float64)
    target_z = np.ascontiguousarray(pairs.target_z, dtype=np.float64)
    source_rho = np.ascontiguousarray(pairs.source_rho, dtype=np.float64)
    source_z = np.ascontiguousarray(pairs.source_z, dtype=np.float64)
    measure = np.ascontiguousarray(pairs.measure, dtype=np.float64)
    normal_rho = np.ascontiguousarray(pairs.normal_rho, dtype=np.float64)
    normal_z = np.ascontiguousarray(pairs.normal_z, dtype=np.float64)
    cos_psi = np.ascontiguousarray(part.cos_psi, dtype=np.float64)
    psi_weights = np.ascontiguousarray(part.psi_weights, dtype=np.float64)
    node_count = int(source_rho.shape[1])
    psi_count = int(cos_psi.size)
    s_out = np.empty(pair_count, dtype=np.complex128)
    h_out = np.empty_like(s_out)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    k_value = complex(k)
    status = kernel.eval_near_onthefly(
        ctypes.c_int64(pair_count),
        ctypes.c_int64(node_count),
        ctypes.c_int64(psi_count),
        target_rho.ctypes.data_as(double_ptr),
        target_z.ctypes.data_as(double_ptr),
        source_rho.ctypes.data_as(double_ptr),
        source_z.ctypes.data_as(double_ptr),
        measure.ctypes.data_as(double_ptr),
        normal_rho.ctypes.data_as(double_ptr),
        normal_z.ctypes.data_as(double_ptr),
        cos_psi.ctypes.data_as(double_ptr),
        psi_weights.ctypes.data_as(double_ptr),
        ctypes.c_int32(pairs.baffle_z is not None),
        ctypes.c_double(0.0 if pairs.baffle_z is None else float(pairs.baffle_z)),
        ctypes.c_double(float(k_value.real)),
        ctypes.c_double(float(k_value.imag)),
        s_out.ctypes.data_as(double_ptr),
        h_out.ctypes.data_as(double_ptr),
        ctypes.c_int32(max(1, min(int(workers), pair_count))),
    )
    if int(status) != 0:
        return _evaluate_near_remainder_compact(part, k)
    return s_out, h_out


def _evaluate_near_remainder_compact_with_kernel(
    kernel: _CircsymRemainderKernel,
    part: _NearRemainderCompactGeometry,
    k: complex,
    *,
    workers: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if kernel.backend == "c":
        return _evaluate_near_remainder_compact_compiled(
            kernel.implementation,
            part,
            k,
            workers=workers,
        )
    if kernel.backend != "numba":
        raise RuntimeError(f"unknown CircSym remainder backend {kernel.backend!r}")
    pairs = part.pairs
    if pairs.target_rho.size == 0:
        return (
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.complex128),
        )
    del workers
    k_value = complex(k)
    return kernel.implementation.evaluate_near_remainder_onthefly(
        np.ascontiguousarray(pairs.target_rho, dtype=np.float64),
        np.ascontiguousarray(pairs.target_z, dtype=np.float64),
        np.ascontiguousarray(pairs.source_rho, dtype=np.float64),
        np.ascontiguousarray(pairs.source_z, dtype=np.float64),
        np.ascontiguousarray(pairs.measure, dtype=np.float64),
        np.ascontiguousarray(pairs.normal_rho, dtype=np.float64),
        np.ascontiguousarray(pairs.normal_z, dtype=np.float64),
        np.ascontiguousarray(part.cos_psi, dtype=np.float64),
        np.ascontiguousarray(part.psi_weights, dtype=np.float64),
        pairs.baffle_z is not None,
        0.0 if pairs.baffle_z is None else float(pairs.baffle_z),
        float(k_value.real),
        float(k_value.imag),
    )


def _evaluate_far_remainder_block(
    part: _FarRemainderGeometry,
    k: complex,
    start: int,
    stop: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    R = part.R[start:stop]
    q = complex(k) * R
    phase = np.exp(1j * q)
    s_rem = np.sum((phase - 1.0) * part.g_weight[start:stop], axis=(2, 3))
    expr = phase * (1j * q - 1.0) + 1.0
    h_rem = np.sum(expr * part.h_weight[start:stop], axis=(2, 3))
    return (
        np.asarray(s_rem, dtype=np.complex128),
        np.asarray(h_rem, dtype=np.complex128),
    )


def _evaluate_far_remainder_onthefly_reference(
    part: _FarRemainderCompactGeometry,
    k: complex,
    *,
    workers: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    target_count = int(part.target_rho.size)
    source_count = int(part.source_rho.shape[0])
    psi_count = int(part.cos_psi.size)
    s_out = np.empty((target_count, source_count), dtype=np.complex128)
    h_out = np.empty_like(s_out)
    block_size = _assembly_target_block_size(source_count, psi_count)
    if workers > 1:
        block_size = min(block_size, _ASSEMBLY_KERNEL_PARALLEL_TARGET_BLOCK)
    ranges = [
        (start, min(target_count, start + block_size))
        for start in range(0, target_count, block_size)
    ]

    def compute_block(
        span: tuple[int, int],
    ) -> tuple[int, int, NDArray[np.complex128], NDArray[np.complex128]]:
        start, stop = span
        s_block = np.zeros((stop - start, source_count), dtype=np.complex128)
        h_block = np.zeros_like(s_block)
        for expanded in _expand_far_remainder_geometry_parts(part, start, stop):
            s_part, h_part = _evaluate_far_remainder_block(
                expanded,
                k,
                0,
                stop - start,
            )
            s_block += s_part
            h_block += h_part
        return start, stop, s_block, h_block

    if workers <= 1 or len(ranges) <= 1:
        for span in ranges:
            _check_circsym_continue(should_continue)
            start, stop, s_block, h_block = compute_block(span)
            s_out[start:stop] = s_block
            h_out[start:stop] = h_block
            _check_circsym_continue(should_continue)
    else:
        executor = ThreadPoolExecutor(max_workers=min(workers, len(ranges)))
        try:
            for start, stop, s_block, h_block in executor.map(compute_block, ranges):
                _check_circsym_continue(should_continue)
                s_out[start:stop] = s_block
                h_out[start:stop] = h_block
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    return s_out, h_out


def _evaluate_near_remainder(
    part: _NearRemainderGeometry,
    k: complex,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if part.R.shape[0] == 0:
        return (
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.complex128),
        )
    R = part.R
    q = complex(k) * R
    with np.errstate(divide="ignore", invalid="ignore"):
        rem_g = np.expm1(1j * q) / R
    rem_g = np.where(R > 1e-13, rem_g, 1j * complex(k))

    expr = np.exp(1j * q) * (1j * q - 1.0) + 1.0
    small = np.abs(q) < 1e-5
    if np.any(small):
        qs = q[small]
        expr = expr.astype(np.complex128, copy=True)
        expr[small] = (
            -0.5 * qs * qs
            - (1j / 3.0) * qs**3
            + 0.125 * qs**4
            + (1j / 30.0) * qs**5
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        rem_h = expr * part.num / (R * R * R)
    rem_h = np.where(R > 1e-13, rem_h, 0.0 + 0.0j)
    return (
        np.asarray(np.sum(rem_g * part.weight, axis=(1, 2)), dtype=np.complex128),
        np.asarray(np.sum(rem_h * part.weight, axis=(1, 2)), dtype=np.complex128),
    )


def _evaluate_near_remainder_compact(
    part: _NearRemainderCompactGeometry,
    k: complex,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Evaluate adaptive near pairs without materialising per-order 3-D tensors."""

    pairs = part.pairs
    pair_count = int(pairs.target_rho.size)
    s_out = np.zeros(pair_count, dtype=np.complex128)
    h_out = np.zeros_like(s_out)
    cos_psi = part.cos_psi[None, :]
    psi_factor = 2.0 * part.psi_weights[None, :] / (4.0 * np.pi)
    k_value = complex(k)
    image_count = 2 if pairs.baffle_z is not None else 1
    for pair_index in range(pair_count):
        node_mask = pairs.measure[pair_index] != 0.0
        if not np.any(node_mask):
            continue
        target_rho = float(pairs.target_rho[pair_index])
        target_z = float(pairs.target_z[pair_index])
        rho_s = pairs.source_rho[pair_index, node_mask][:, None]
        base_z_s = pairs.source_z[pair_index, node_mask][:, None]
        measure = pairs.measure[pair_index, node_mask][:, None]
        normal_rho = float(pairs.normal_rho[pair_index])
        source_normal_z = float(pairs.normal_z[pair_index])
        for image_index in range(image_count):
            if image_index:
                assert pairs.baffle_z is not None
                z_s = 2.0 * float(pairs.baffle_z) - base_z_s
                normal_z = -source_normal_z
            else:
                z_s = base_z_s
                normal_z = source_normal_z
            dz = z_s - target_z
            R2 = (
                target_rho * target_rho
                + rho_s * rho_s
                - 2.0 * target_rho * rho_s * cos_psi
                + dz * dz
            )
            R = np.sqrt(np.maximum(R2, 0.0))
            numerator = (
                (rho_s - target_rho * cos_psi) * normal_rho + dz * normal_z
            )
            weight = measure * psi_factor
            q = k_value * R
            with np.errstate(divide="ignore", invalid="ignore"):
                rem_g = np.expm1(1j * q) / R
            rem_g = np.where(R > 1e-13, rem_g, 1j * k_value)
            expr = np.exp(1j * q) * (1j * q - 1.0) + 1.0
            small = np.abs(q) < 1e-5
            if np.any(small):
                qs = q[small]
                expr = expr.astype(np.complex128, copy=True)
                expr[small] = (
                    -0.5 * qs * qs
                    - (1j / 3.0) * qs**3
                    + 0.125 * qs**4
                    + (1j / 30.0) * qs**5
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                rem_h = expr * numerator / (R * R * R)
            rem_h = np.where(R > 1e-13, rem_h, 0.0 + 0.0j)
            s_out[pair_index] += np.sum(rem_g * weight)
            h_out[pair_index] += np.sum(rem_h * weight)
    return s_out, h_out


def _build_boundary_static_geometry(
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    baffle_z: float | None,
    *,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    n = meridian.segment_count
    source_indices = np.arange(n, dtype=np.int64)
    S = np.empty((n, n), dtype=np.complex128)
    H = np.empty((n, n), dtype=np.complex128)
    block_size = min(_ASSEMBLY_KERNEL_MAX_TARGET_BLOCK, n)
    for start in range(0, n, block_size):
        _check_circsym_continue(should_continue)
        stop = min(n, start + block_size)
        s_block, h_block = _integrate_static_ordinary_segment_kernels_targets_batched(
            target_rho=geom.rho_mid[start:stop],
            target_z=geom.z_mid[start:stop],
            meridian=meridian,
            geom=geom,
            source_indices=source_indices,
            baffle_z=baffle_z,
        )
        S[start:stop] = s_block
        H[start:stop] = h_block

    near_rows, near_cols = _boundary_near_pairs(geom, source_indices)
    for pair_index, (row, col) in enumerate(zip(near_rows, near_cols)):
        if pair_index % _CANCELLATION_PAIR_BLOCK == 0:
            _check_circsym_continue(should_continue)
        S[row, col], H[row, col] = _integrate_static_segment_kernel(
            target_rho=float(geom.rho_mid[row]),
            target_z=float(geom.z_mid[row]),
            meridian=meridian,
            geom=geom,
            source_index=int(col),
            baffle_z=baffle_z,
            target_index=int(row),
        )
    return S, H, near_rows, near_cols


def _boundary_near_pairs(
    geom: SimpleNamespace,
    source_indices: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    far_mask = _ordinary_far_source_mask_targets(
        geom.rho_mid,
        geom.z_mid,
        geom,
        source_indices=source_indices,
    )
    near_rows, near_cols = np.nonzero(~far_mask)
    return near_rows.astype(np.int64, copy=False), near_cols.astype(np.int64, copy=False)


def _integrate_static_ordinary_segment_kernels_targets_batched(
    *,
    target_rho: NDArray[np.float64],
    target_z: NDArray[np.float64],
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    source_indices: NDArray[np.int64] | NDArray[np.int32],
    baffle_z: float | None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    indices = np.asarray(source_indices, dtype=np.int64)
    target_rho_arr = np.asarray(target_rho, dtype=np.float64).reshape(-1)
    target_z_arr = np.asarray(target_z, dtype=np.float64).reshape(-1)
    if indices.size == 0 or target_rho_arr.size == 0:
        shape = (target_rho_arr.size, indices.size)
        return (
            np.empty(shape, dtype=np.complex128),
            np.empty(shape, dtype=np.complex128),
        )

    u, w = _ordinary_interval(0.0, 1.0)
    p0 = geom.p0[indices]
    delta = geom.delta[indices]
    lengths = geom.lengths[indices]
    source = p0[:, None, :] + u[None, :, None] * delta[:, None, :]
    rho_s = source[:, :, 0]
    z_s = source[:, :, 1]
    measure = rho_s * lengths[:, None] * w[None, :]
    normal = meridian.normals[indices]
    normal_rho = normal[:, 0]
    normal_z = normal[:, 1]

    g, h = _ring_static_kernel_m0_targets_batched(
        target_rho_arr,
        target_z_arr,
        rho_s,
        z_s,
        normal_rho,
        normal_z,
    )
    if baffle_z is not None:
        z_img = 2.0 * float(baffle_z) - z_s
        g_i, h_i = _ring_static_kernel_m0_targets_batched(
            target_rho_arr,
            target_z_arr,
            rho_s,
            z_img,
            normal_rho,
            -normal_z,
        )
        g = g + g_i
        h = h + h_i

    return (
        np.asarray(np.sum(g * measure[None, :, :], axis=2), dtype=np.complex128),
        np.asarray(np.sum(h * measure[None, :, :], axis=2), dtype=np.complex128),
    )


def _integrate_static_segment_kernel(
    *,
    target_rho: float,
    target_z: float,
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    source_index: int,
    baffle_z: float | None,
    target_index: int | None,
) -> tuple[complex, complex]:
    p0 = geom.p0[source_index]
    delta = geom.delta[source_index]
    length = float(geom.lengths[source_index])
    normal = meridian.normals[source_index]
    u, w = _segment_quadrature_nodes(
        target_rho=target_rho,
        target_z=target_z,
        source_p0=p0,
        source_delta=delta,
        source_length=length,
        self_pair=target_index == source_index,
    )
    source = p0[None, :] + u[:, None] * delta[None, :]
    rho_s = source[:, 0]
    z_s = source[:, 1]
    measure = rho_s * length * w
    if not np.any(measure != 0.0):
        return 0.0 + 0.0j, 0.0 + 0.0j

    g, h = _ring_static_kernel_m0(target_rho, target_z, rho_s, z_s, normal)
    if baffle_z is not None:
        z_img = 2.0 * float(baffle_z) - z_s
        normal_img = np.array([normal[0], -normal[1]], dtype=np.float64)
        g_i, h_i = _ring_static_kernel_m0(
            target_rho,
            target_z,
            rho_s,
            z_img,
            normal_img,
        )
        g = g + g_i
        h = h + h_i
    return complex(np.sum(g * measure)), complex(np.sum(h * measure))


def _build_far_remainder_compact_geometry(
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    baffle_z: float | None,
    *,
    n_psi: int,
) -> _FarRemainderCompactGeometry:
    u, w = _ordinary_interval(0.0, 1.0)
    psi, psi_weights = _leggauss_psi(int(n_psi))
    p0 = geom.p0
    delta = geom.delta
    lengths = geom.lengths
    source = p0[:, None, :] + u[None, :, None] * delta[:, None, :]
    rho_s = source[:, :, 0]
    z_s = source[:, :, 1]
    measure = rho_s * lengths[:, None] * w[None, :]
    normal_rho = meridian.normals[:, 0]
    normal_z = meridian.normals[:, 1]
    return _FarRemainderCompactGeometry(
        target_rho=np.ascontiguousarray(geom.rho_mid, dtype=np.float64),
        target_z=np.ascontiguousarray(geom.z_mid, dtype=np.float64),
        source_rho=np.ascontiguousarray(rho_s, dtype=np.float64),
        source_z=np.ascontiguousarray(z_s, dtype=np.float64),
        measure=np.ascontiguousarray(measure, dtype=np.float64),
        normal_rho=np.ascontiguousarray(normal_rho, dtype=np.float64),
        normal_z=np.ascontiguousarray(normal_z, dtype=np.float64),
        cos_psi=np.ascontiguousarray(np.cos(psi), dtype=np.float64),
        psi_weights=np.ascontiguousarray(psi_weights, dtype=np.float64),
        baffle_z=None if baffle_z is None else float(baffle_z),
    )


def _build_far_remainder_geometry_parts(
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    baffle_z: float | None,
    *,
    n_psi: int,
) -> tuple[_FarRemainderGeometry, ...]:
    u, w = _ordinary_interval(0.0, 1.0)
    psi, psi_weights = _leggauss_psi(int(n_psi))
    p0 = geom.p0
    delta = geom.delta
    lengths = geom.lengths
    source = p0[:, None, :] + u[None, :, None] * delta[:, None, :]
    rho_s = source[:, :, 0]
    z_s = source[:, :, 1]
    measure = rho_s * lengths[:, None] * w[None, :]
    normal_rho = meridian.normals[:, 0]
    normal_z = meridian.normals[:, 1]
    parts = [
        _build_far_remainder_geometry(
            geom.rho_mid,
            geom.z_mid,
            rho_s,
            z_s,
            measure,
            normal_rho,
            normal_z,
            psi,
            psi_weights,
        )
    ]
    if baffle_z is not None:
        parts.append(
            _build_far_remainder_geometry(
                geom.rho_mid,
                geom.z_mid,
                rho_s,
                2.0 * float(baffle_z) - z_s,
                measure,
                normal_rho,
                -normal_z,
                psi,
                psi_weights,
            )
        )
    return tuple(parts)


def _expand_far_remainder_geometry_parts(
    compact: _FarRemainderCompactGeometry,
    start: int,
    stop: int,
) -> tuple[_FarRemainderGeometry, ...]:
    target_rho = compact.target_rho[int(start) : int(stop)]
    target_z = compact.target_z[int(start) : int(stop)]
    parts = [
        _build_far_remainder_geometry_from_cos(
            target_rho,
            target_z,
            compact.source_rho,
            compact.source_z,
            compact.measure,
            compact.normal_rho,
            compact.normal_z,
            compact.cos_psi,
            compact.psi_weights,
        )
    ]
    if compact.baffle_z is not None:
        parts.append(
            _build_far_remainder_geometry_from_cos(
                target_rho,
                target_z,
                compact.source_rho,
                2.0 * float(compact.baffle_z) - compact.source_z,
                compact.measure,
                compact.normal_rho,
                -compact.normal_z,
                compact.cos_psi,
                compact.psi_weights,
            )
        )
    return tuple(parts)


def _build_far_remainder_geometry(
    target_rho: NDArray[np.float64],
    target_z: NDArray[np.float64],
    rho_s: NDArray[np.float64],
    z_s: NDArray[np.float64],
    measure: NDArray[np.float64],
    normal_rho: NDArray[np.float64],
    normal_z: NDArray[np.float64],
    psi: NDArray[np.float64],
    psi_weights: NDArray[np.float64],
) -> _FarRemainderGeometry:
    return _build_far_remainder_geometry_from_cos(
        target_rho,
        target_z,
        rho_s,
        z_s,
        measure,
        normal_rho,
        normal_z,
        np.cos(psi),
        psi_weights,
    )


def _build_far_remainder_geometry_from_cos(
    target_rho: NDArray[np.float64],
    target_z: NDArray[np.float64],
    rho_s: NDArray[np.float64],
    z_s: NDArray[np.float64],
    measure: NDArray[np.float64],
    normal_rho: NDArray[np.float64],
    normal_z: NDArray[np.float64],
    cos_psi_values: NDArray[np.float64],
    psi_weights: NDArray[np.float64],
) -> _FarRemainderGeometry:
    n_target = int(target_rho.shape[0])
    n_source, n_line = rho_s.shape
    n_psi = int(cos_psi_values.shape[0])
    R_out = np.empty((n_target, n_source, n_line, n_psi), dtype=np.float64)
    g_weight = np.empty_like(R_out)
    h_weight = np.empty_like(R_out)
    cos_psi = np.asarray(cos_psi_values, dtype=np.float64).reshape(1, 1, 1, -1)
    rs = rho_s[None, :, :, None]
    zs = z_s[None, :, :, None]
    nr = np.asarray(normal_rho, dtype=np.float64).reshape(1, -1, 1, 1)
    nz = np.asarray(normal_z, dtype=np.float64).reshape(1, -1, 1, 1)
    weighted_measure = (
        measure[None, :, :, None]
        * (2.0 * psi_weights.reshape(1, 1, 1, -1))
        / (4.0 * np.pi)
    )
    block_size = _assembly_target_block_size(n_target, n_psi)
    for start in range(0, n_target, block_size):
        stop = min(n_target, start + block_size)
        rt = np.asarray(target_rho[start:stop], dtype=np.float64).reshape(-1, 1, 1, 1)
        zt = np.asarray(target_z[start:stop], dtype=np.float64).reshape(-1, 1, 1, 1)
        dz = zs - zt
        R2 = rt * rt + rs * rs - 2.0 * rt * rs * cos_psi + dz * dz
        R = np.sqrt(np.maximum(R2, 0.0))
        num = (rs - rt * cos_psi) * nr + dz * nz
        with np.errstate(divide="ignore", invalid="ignore"):
            g = weighted_measure / R
            h = weighted_measure * num / (R * R * R)
        valid = R > 1e-13
        R_out[start:stop] = R
        g_weight[start:stop] = np.where(valid, g, 0.0)
        h_weight[start:stop] = np.where(valid, h, 0.0)
    return _FarRemainderGeometry(R=R_out, g_weight=g_weight, h_weight=h_weight)


def _build_near_remainder_geometry_parts(
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    near_rows: NDArray[np.int64],
    near_cols: NDArray[np.int64],
    baffle_z: float | None,
    *,
    n_psi: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[_NearRemainderGeometry, ...]:
    _check_circsym_continue(should_continue)
    parts = [
        _build_near_remainder_geometry(
            meridian,
            geom,
            near_rows,
            near_cols,
            baffle_z=None,
            image=False,
            n_psi=int(n_psi),
            should_continue=should_continue,
        )
    ]
    if baffle_z is not None:
        parts.append(
            _build_near_remainder_geometry(
                meridian,
                geom,
                near_rows,
                near_cols,
                baffle_z=baffle_z,
                image=True,
                n_psi=int(n_psi),
                should_continue=should_continue,
            )
        )
    return tuple(parts)


def _build_near_pair_compact_geometry(
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    near_rows: NDArray[np.int64],
    near_cols: NDArray[np.int64],
    baffle_z: float | None,
    *,
    should_continue: Callable[[], bool | None] | None = None,
) -> _NearPairCompactGeometry:
    """Prepare frequency-invariant adaptive line geometry for all near pairs."""

    pair_sources: list[NDArray[np.float64]] = []
    pair_measures: list[NDArray[np.float64]] = []
    max_nodes = 0
    for pair_index, (row, col) in enumerate(zip(near_rows, near_cols)):
        if pair_index % _CANCELLATION_PAIR_BLOCK == 0:
            _check_circsym_continue(should_continue)
        target_rho = float(geom.rho_mid[row])
        target_z = float(geom.z_mid[row])
        source_length = float(geom.lengths[col])
        u, w = _segment_quadrature_nodes(
            target_rho=target_rho,
            target_z=target_z,
            source_p0=geom.p0[col],
            source_delta=geom.delta[col],
            source_length=source_length,
            self_pair=int(row) == int(col),
        )
        source = geom.p0[col][None, :] + u[:, None] * geom.delta[col][None, :]
        pair_sources.append(source)
        pair_measures.append(source[:, 0] * source_length * w)
        max_nodes = max(max_nodes, int(u.size))

    pair_count = int(near_rows.size)
    source_rho = np.zeros((pair_count, max_nodes), dtype=np.float64)
    source_z = np.zeros_like(source_rho)
    measure = np.zeros_like(source_rho)
    for pair_index, (source, source_measure) in enumerate(
        zip(pair_sources, pair_measures)
    ):
        node_count = int(source.shape[0])
        source_rho[pair_index, :node_count] = source[:, 0]
        source_z[pair_index, :node_count] = source[:, 1]
        measure[pair_index, :node_count] = source_measure

    normals = meridian.normals[np.asarray(near_cols, dtype=np.int64)]
    return _NearPairCompactGeometry(
        target_rho=np.ascontiguousarray(geom.rho_mid[near_rows], dtype=np.float64),
        target_z=np.ascontiguousarray(geom.z_mid[near_rows], dtype=np.float64),
        source_rho=source_rho,
        source_z=source_z,
        measure=measure,
        normal_rho=np.ascontiguousarray(normals[:, 0], dtype=np.float64),
        normal_z=np.ascontiguousarray(normals[:, 1], dtype=np.float64),
        baffle_z=None if baffle_z is None else float(baffle_z),
    )


def _build_near_remainder_compact_geometry(
    pairs: _NearPairCompactGeometry,
    *,
    n_psi: int,
) -> _NearRemainderCompactGeometry:
    psi, psi_weights = _leggauss_psi(int(n_psi))
    return _NearRemainderCompactGeometry(
        pairs=pairs,
        cos_psi=np.ascontiguousarray(np.cos(psi), dtype=np.float64),
        psi_weights=np.ascontiguousarray(psi_weights, dtype=np.float64),
    )


def _build_near_remainder_geometry(
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    near_rows: NDArray[np.int64],
    near_cols: NDArray[np.int64],
    *,
    baffle_z: float | None,
    image: bool,
    n_psi: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> _NearRemainderGeometry:
    psi, psi_weights = _leggauss_psi(int(n_psi))
    cos_psi = np.cos(psi)[None, :]
    counts = []
    for pair_index, (row, col) in enumerate(zip(near_rows, near_cols)):
        if pair_index % _CANCELLATION_PAIR_BLOCK == 0:
            _check_circsym_continue(should_continue)
        u, _ = _segment_quadrature_nodes(
            target_rho=float(geom.rho_mid[row]),
            target_z=float(geom.z_mid[row]),
            source_p0=geom.p0[col],
            source_delta=geom.delta[col],
            source_length=float(geom.lengths[col]),
            self_pair=int(row) == int(col),
        )
        counts.append(int(u.size))
    pair_count = int(near_rows.size)
    max_nodes = max(counts, default=0)
    R = np.zeros((pair_count, max_nodes, int(n_psi)), dtype=np.float64)
    num = np.zeros_like(R)
    weight = np.zeros_like(R)
    psi_factor = 2.0 * psi_weights / (4.0 * np.pi)

    for pair_index, (row, col) in enumerate(zip(near_rows, near_cols)):
        if pair_index % _CANCELLATION_PAIR_BLOCK == 0:
            _check_circsym_continue(should_continue)
        target_rho = float(geom.rho_mid[row])
        target_z = float(geom.z_mid[row])
        u, w = _segment_quadrature_nodes(
            target_rho=target_rho,
            target_z=target_z,
            source_p0=geom.p0[col],
            source_delta=geom.delta[col],
            source_length=float(geom.lengths[col]),
            self_pair=int(row) == int(col),
        )
        if u.size == 0:
            continue
        source = geom.p0[col][None, :] + u[:, None] * geom.delta[col][None, :]
        rho_s = source[:, 0]
        z_s = source[:, 1]
        normal = meridian.normals[col]
        normal_rho = float(normal[0])
        normal_z = float(normal[1])
        if image:
            if baffle_z is None:
                raise ValueError("image near geometry requires baffle_z")
            z_s = 2.0 * float(baffle_z) - z_s
            normal_z = -normal_z
        measure = rho_s * float(geom.lengths[col]) * w
        dz = z_s[:, None] - target_z
        rs = rho_s[:, None]
        R2 = (
            target_rho * target_rho
            + rs * rs
            - 2.0 * target_rho * rs * cos_psi
            + dz * dz
        )
        pair_R = np.sqrt(np.maximum(R2, 0.0))
        pair_num = (rs - target_rho * cos_psi) * normal_rho + dz * normal_z
        node_count = int(u.size)
        R[pair_index, :node_count] = pair_R
        num[pair_index, :node_count] = pair_num
        weight[pair_index, :node_count] = measure[:, None] * psi_factor[None, :]
    return _NearRemainderGeometry(R=R, num=num, weight=weight)


def _assembly_target_block_size(segment_count: int, n_psi: int) -> int:
    per_target = max(1, int(segment_count) * _LINE_QUAD_ORDER * max(1, int(n_psi)))
    by_elements = max(1, _ASSEMBLY_KERNEL_BLOCK_ELEMENTS // per_target)
    return max(1, min(_ASSEMBLY_KERNEL_MAX_TARGET_BLOCK, int(by_elements)))


def _assembly_worker_count(segment_count: int) -> int:
    if int(segment_count) < 32:
        return 1
    raw = os.environ.get("HORNLAB_CIRCSYM_ASSEMBLY_THREADS")
    if raw is not None:
        try:
            return max(1, min(int(segment_count), int(raw)))
        except ValueError:
            return 1
    return max(1, min(int(segment_count), 8, os.cpu_count() or 1))


def _assemble_boundary_block(
    start: int,
    stop: int,
    *,
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    k: complex,
    baffle_z: float | None,
    n_psi: int,
) -> tuple[int, int, NDArray[np.complex128], NDArray[np.complex128]]:
    row_indices = np.arange(int(start), int(stop), dtype=np.int64)
    targets = geom.midpoints[row_indices]
    source_indices = np.arange(meridian.segment_count, dtype=np.int64)
    s_block, h_block = _integrate_ordinary_segment_kernels_targets_batched(
        target_rho=targets[:, 0],
        target_z=targets[:, 1],
        meridian=meridian,
        geom=geom,
        source_indices=source_indices,
        k=k,
        baffle_z=baffle_z,
        n_psi=n_psi,
    )
    far_mask = _ordinary_far_source_mask_targets(
        targets[:, 0],
        targets[:, 1],
        geom,
        source_indices=source_indices,
    )
    near_rows, near_cols = np.nonzero(~far_mask)
    for row_local, source_local in zip(near_rows, near_cols):
        row_index = int(row_indices[row_local])
        source_index = int(source_indices[source_local])
        s_block[row_local, source_local], h_block[row_local, source_local] = (
            _integrate_segment_kernel(
                target_rho=float(targets[row_local, 0]),
                target_z=float(targets[row_local, 1]),
                meridian=meridian,
                geom=geom,
                source_index=source_index,
                k=k,
                baffle_z=baffle_z,
                n_psi=n_psi,
                target_index=row_index,
            )
        )
    return int(start), int(stop), s_block, h_block


def _assemble_coupled_ib_rayleigh_aperture_matrix(
    meridian: MeridianMesh,
    aperture_indices: NDArray[np.int64] | NDArray[np.int32],
    k: complex,
    *,
    geom: SimpleNamespace,
    n_psi: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> NDArray[np.complex128]:
    """Assemble the real-k Rayleigh single-layer aperture block only.

    Coupled infinite-baffle assembly consumes this block for the pressure
    trace.  Match the ordinary/near split used by full boundary assembly so
    this reduced work remains numerically identical to slicing a full matrix.
    """
    indices = np.asarray(aperture_indices, dtype=np.int64).reshape(-1)
    targets = geom.midpoints[indices]
    s_block = np.empty((indices.size, indices.size), dtype=np.complex128)
    block_size = _assembly_target_block_size(indices.size, int(n_psi))
    for start in range(0, indices.size, block_size):
        _check_circsym_continue(should_continue)
        stop = min(indices.size, start + block_size)
        s_part, _ = _integrate_ordinary_segment_kernels_targets_batched(
            target_rho=targets[start:stop, 0],
            target_z=targets[start:stop, 1],
            meridian=meridian,
            geom=geom,
            source_indices=indices,
            k=k,
            baffle_z=None,
            n_psi=n_psi,
        )
        s_block[start:stop] = s_part
    far_mask = _ordinary_far_source_mask_targets(
        targets[:, 0],
        targets[:, 1],
        geom,
        source_indices=indices,
    )
    near_rows, near_cols = np.nonzero(~far_mask)
    for pair_index, (row_local, source_local) in enumerate(zip(near_rows, near_cols)):
        if pair_index % _CANCELLATION_PAIR_BLOCK == 0:
            _check_circsym_continue(should_continue)
        s_block[row_local, source_local] = _integrate_segment_kernel(
            target_rho=float(targets[row_local, 0]),
            target_z=float(targets[row_local, 1]),
            meridian=meridian,
            geom=geom,
            source_index=int(indices[source_local]),
            k=k,
            baffle_z=None,
            n_psi=n_psi,
            target_index=int(indices[row_local]),
        )[0]
    return s_block


def _integrate_ordinary_segment_kernels_targets_batched(
    *,
    target_rho: NDArray[np.float64],
    target_z: NDArray[np.float64],
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    source_indices: NDArray[np.int64] | NDArray[np.int32],
    k: complex,
    baffle_z: float | None,
    n_psi: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Ordinary-quadrature block kernel; callers must replace near/self pairs."""
    indices = np.asarray(source_indices, dtype=np.int64)
    target_rho_arr = np.asarray(target_rho, dtype=np.float64).reshape(-1)
    target_z_arr = np.asarray(target_z, dtype=np.float64).reshape(-1)
    if target_rho_arr.shape != target_z_arr.shape:
        raise ValueError("target_rho and target_z must have the same shape")
    if indices.size == 0 or target_rho_arr.size == 0:
        shape = (target_rho_arr.size, indices.size)
        return (
            np.empty(shape, dtype=np.complex128),
            np.empty(shape, dtype=np.complex128),
        )

    u, w = _ordinary_interval(0.0, 1.0)
    p0 = geom.p0[indices]
    delta = geom.delta[indices]
    lengths = geom.lengths[indices]
    source = p0[:, None, :] + u[None, :, None] * delta[:, None, :]
    rho_s = source[:, :, 0]
    z_s = source[:, :, 1]
    measure = rho_s * lengths[:, None] * w[None, :]
    normal = meridian.normals[indices]
    normal_rho = normal[:, 0]
    normal_z = normal[:, 1]

    g_static, h_static = _ring_static_kernel_m0_targets_batched(
        target_rho_arr,
        target_z_arr,
        rho_s,
        z_s,
        normal_rho,
        normal_z,
    )
    g_rem, h_rem = _ring_remainder_kernel_m0_targets_batched(
        target_rho_arr,
        target_z_arr,
        rho_s,
        z_s,
        normal_rho,
        normal_z,
        k,
        n_psi=n_psi,
    )
    g = g_static + g_rem
    h = h_static + h_rem

    if baffle_z is not None:
        z_img = 2.0 * float(baffle_z) - z_s
        g_static_i, h_static_i = _ring_static_kernel_m0_targets_batched(
            target_rho_arr,
            target_z_arr,
            rho_s,
            z_img,
            normal_rho,
            -normal_z,
        )
        g_rem_i, h_rem_i = _ring_remainder_kernel_m0_targets_batched(
            target_rho_arr,
            target_z_arr,
            rho_s,
            z_img,
            normal_rho,
            -normal_z,
            k,
            n_psi=n_psi,
        )
        g = g + g_static_i + g_rem_i
        h = h + h_static_i + h_rem_i

    return (
        np.asarray(np.sum(g * measure[None, :, :], axis=2), dtype=np.complex128),
        np.asarray(np.sum(h * measure[None, :, :], axis=2), dtype=np.complex128),
    )


def _assemble_chief_matrices(
    meridian: MeridianMesh,
    chief_points: NDArray[np.float64],
    k: complex,
    baffle_z: float | None,
    *,
    n_psi: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    pts = np.asarray(chief_points, dtype=np.float64)
    geom = meridian.segment_geometry()
    S = np.empty((pts.shape[0], meridian.segment_count), dtype=np.complex128)
    H = np.empty_like(S)
    for i, point in enumerate(pts):
        _check_circsym_continue(should_continue)
        target_rho = float(math.hypot(float(point[0]), float(point[1])))
        target_z = float(point[2])
        for j in range(meridian.segment_count):
            if j % _CANCELLATION_PAIR_BLOCK == 0:
                _check_circsym_continue(should_continue)
            S[i, j], H[i, j] = _integrate_segment_kernel(
                target_rho=target_rho,
                target_z=target_z,
                meridian=meridian,
                geom=geom,
                source_index=j,
                k=k,
                baffle_z=baffle_z,
                n_psi=n_psi,
                target_index=None,
            )
    return S, H


def _chief_row_scale(
    A: NDArray[np.complex128],
    C: NDArray[np.complex128],
    chief_weight: float,
) -> float:
    c_norm = float(np.linalg.norm(C, ord=np.inf))
    if c_norm <= 1e-30:
        return float(chief_weight)
    a_norm = float(np.linalg.norm(A, ord=np.inf))
    return float(chief_weight) * a_norm / c_norm


def _evaluate_observation_pressure(
    meridian: MeridianMesh,
    pressure: NDArray[np.complex128],
    q_total: NDArray[np.complex128],
    obs_points: NDArray[np.float64],
    k: complex,
    config: SolveConfig,
    *,
    geom: SimpleNamespace | None = None,
    n_psi: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> NDArray[np.complex128]:
    if config.observation.custom_points is None:
        first = _evaluate_points_pressure(
            meridian,
            pressure,
            q_total,
            obs_points[0],
            k,
            config.circsym_baffle_z,
            geom=geom,
            n_psi=n_psi,
            should_continue=should_continue,
        )
        return np.tile(first[None, :], (obs_points.shape[0], 1))

    out = np.empty(obs_points.shape[:2], dtype=np.complex128)
    for plane_index in range(obs_points.shape[0]):
        out[plane_index] = _evaluate_points_pressure(
            meridian,
            pressure,
            q_total,
            obs_points[plane_index],
            k,
            config.circsym_baffle_z,
            geom=geom,
            n_psi=n_psi,
            should_continue=should_continue,
        )
    return out


def _evaluate_points_pressure(
    meridian: MeridianMesh,
    pressure: NDArray[np.complex128],
    q_total: NDArray[np.complex128],
    points: NDArray[np.float64],
    k: complex,
    baffle_z: float | None,
    *,
    geom: SimpleNamespace | None = None,
    n_psi: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> NDArray[np.complex128]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if pts.shape[0] == 0:
        return np.empty(0, dtype=np.complex128)
    if geom is None:
        geom = meridian.segment_geometry()
    target_rho, target_z = _points_target_rho_z(pts)
    rayleigh_sheet = _is_flat_baffled_sheet(meridian, baffle_z, geom=geom)
    if rayleigh_sheet:
        # The image kernel extends the half-space solution symmetrically across
        # the baffle. That continuation is not a physical rear field: retain it
        # only on the side selected by the sheet's outward normal.
        out = np.zeros(pts.shape[0], dtype=np.complex128)
        active = _baffled_sheet_active_targets(
            meridian,
            target_z,
            float(baffle_z),
            geom=geom,
        )
        if not np.any(active):
            return out
        source_indices = np.arange(meridian.segment_count, dtype=np.int64)
        s_mat, _ = _integrate_field_segment_kernels_batched(
            target_rho=target_rho[active],
            target_z=target_z[active],
            meridian=meridian,
            geom=geom,
            source_indices=source_indices,
            k=k,
            baffle_z=baffle_z,
            n_psi=n_psi,
            should_continue=should_continue,
        )
        out[active] = -(s_mat @ q_total)
        return out

    source_indices = np.arange(meridian.segment_count, dtype=np.int64)
    s_mat, h_mat = _integrate_field_segment_kernels_batched(
        target_rho=target_rho,
        target_z=target_z,
        meridian=meridian,
        geom=geom,
        source_indices=source_indices,
        k=k,
        baffle_z=baffle_z,
        n_psi=n_psi,
        should_continue=should_continue,
    )
    return np.asarray(h_mat @ pressure - s_mat @ q_total, dtype=np.complex128)


def _evaluate_coupled_ib_points_pressure(
    meridian: MeridianMesh,
    aperture_neumann: NDArray[np.complex128],
    aperture_indices: NDArray[np.int64] | NDArray[np.int32],
    points: NDArray[np.float64],
    k: complex,
    *,
    geom: SimpleNamespace | None = None,
    n_psi: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> NDArray[np.complex128]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    out = np.zeros(pts.shape[0], dtype=np.complex128)
    if pts.shape[0] == 0:
        return out
    target_rho, target_z = _points_target_rho_z(pts)
    active = target_z >= 0.0
    if not np.any(active):
        return out
    if geom is None:
        geom = meridian.segment_geometry()
    indices = np.asarray(aperture_indices, dtype=np.int64)
    if indices.size == 0:
        return out
    s_mat, _ = _integrate_field_segment_kernels_batched(
        target_rho=target_rho[active],
        target_z=target_z[active],
        meridian=meridian,
        geom=geom,
        source_indices=indices,
        k=k,
        baffle_z=None,
        n_psi=n_psi,
        should_continue=should_continue,
    )
    # The augmented coupling row enforces p_aperture = 2*S_R*q_aperture.
    # Evaluate the exterior Rayleigh field with that same trace convention;
    # changing this sign independently creates a 180-degree pressure jump at
    # the aperture even though normalized directivity remains unchanged.
    out[active] = 2.0 * (s_mat @ np.asarray(aperture_neumann, dtype=np.complex128))
    return out


def _points_target_rho_z(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pts = np.asarray(points, dtype=np.float64)
    return (
        np.hypot(pts[:, 0], pts[:, 1]).astype(np.float64, copy=False),
        pts[:, 2].astype(np.float64, copy=False),
    )


def _axisymmetric_sphere_evaluation_targets(
    points: NDArray[np.float64] | None,
    theta_deg: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.int64] | None]:
    """Collapse a generated sphere grid to one azimuth representative per theta.

    CircSym solves only the rotationally invariant ``m=0`` field, so every phi
    sample at the same generated polar angle has exactly the same pressure.  Use
    the grid's theta metadata instead of tolerance-rounding Cartesian-derived
    ``(rho, z)`` coordinates; the latter can merge genuinely distinct custom
    targets near a boundary.

    Explicit ``sphere_points`` have no theta metadata and pass through unchanged.
    The returned inverse expands evaluated pressures back to the caller's original
    theta-major sphere-grid order.
    """
    if points is None:
        return None, None
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if theta_deg is None:
        return pts, None

    theta = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    if theta.size != pts.shape[0]:
        raise ValueError("sphere theta metadata must match the point count")
    _, first_indices, inverse = np.unique(
        theta,
        return_index=True,
        return_inverse=True,
    )
    if first_indices.size == pts.shape[0]:
        return pts, None
    return (
        np.ascontiguousarray(pts[first_indices], dtype=np.float64),
        np.asarray(inverse, dtype=np.int64),
    )


def _requested_circsym_field_backend() -> str:
    backend = os.environ.get(_CIRCSYM_FIELD_BACKEND_ENV, "auto").strip().lower()
    if backend not in {"auto", "cpu", "metal"}:
        raise ValueError(
            f"{_CIRCSYM_FIELD_BACKEND_ENV} must be 'auto', 'cpu', or 'metal'"
        )
    return backend


@lru_cache(maxsize=1)
def _circsym_metal_runtime_status() -> Any:
    from .metal.native import discover_native_runtime

    return discover_native_runtime(run_smoke_test=True)


def _select_circsym_field_backend(
    target_count: int,
    source_count: int,
    n_psi: int,
) -> tuple[str, Any | None]:
    requested = _requested_circsym_field_backend()
    if requested == "cpu":
        return "cpu", None
    work_terms = (
        int(target_count) * int(source_count) * _LINE_QUAD_ORDER * int(n_psi)
    )
    if requested == "auto" and (
        work_terms < _CIRCSYM_METAL_FIELD_MIN_TERMS
        or _circsym_metal_field_auto_failure is not None
    ):
        return "cpu", None
    status = _circsym_metal_runtime_status()
    if status.available:
        return "metal", status
    if requested == "metal":
        raise RuntimeError(
            "CircSym Metal field backend was requested but is unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    return "cpu", None


def _record_circsym_metal_field_failure(exc: BaseException) -> None:
    global _circsym_metal_field_auto_failure
    if _circsym_metal_field_auto_failure is None:
        _circsym_metal_field_auto_failure = str(exc)
        logger.warning(
            "CircSym Metal field acceleration failed; using CPU for the rest "
            "of this process: %s",
            exc,
        )


def _circsym_field_backend_summary(
    target_counts: Iterable[int],
    source_count: int,
    n_psi: int,
) -> str:
    backends = {
        _select_circsym_field_backend(int(count), source_count, n_psi)[0]
        for count in target_counts
        if int(count) > 0
    }
    return "+".join(sorted(backends)) or "cpu"


def _requested_circsym_assembly_backend() -> str:
    backend = os.environ.get(_CIRCSYM_ASSEMBLY_BACKEND_ENV, "auto").strip().lower()
    if backend not in {"auto", "cpu", "metal"}:
        raise ValueError(
            f"{_CIRCSYM_ASSEMBLY_BACKEND_ENV} must be 'auto', 'cpu', or 'metal'"
        )
    return backend


def _select_circsym_assembly_backend(
    segment_count: int,
    n_psi: int,
) -> tuple[str, Any | None]:
    requested = _requested_circsym_assembly_backend()
    if requested == "cpu":
        return "cpu", None
    work_terms = (
        int(segment_count)
        * int(segment_count)
        * _LINE_QUAD_ORDER
        * int(n_psi)
    )
    if requested == "auto" and (
        work_terms < _CIRCSYM_METAL_ASSEMBLY_MIN_TERMS
        or _circsym_metal_assembly_auto_failure is not None
    ):
        return "cpu", None
    status = _circsym_metal_runtime_status()
    if status.available:
        return "metal", status
    if requested == "metal":
        raise RuntimeError(
            "CircSym Metal assembly backend was requested but is unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    return "cpu", None


def _record_circsym_metal_assembly_failure(exc: BaseException) -> None:
    global _circsym_metal_assembly_auto_failure
    if _circsym_metal_assembly_auto_failure is None:
        _circsym_metal_assembly_auto_failure = str(exc)
        logger.warning(
            "CircSym Metal assembly acceleration failed; using CPU for the rest "
            "of this process: %s",
            exc,
        )


def _prepare_circsym_observation_metal_batch(
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    obs_points: NDArray[np.float64],
    k_values: NDArray[np.complex128],
    n_psi_values: NDArray[np.int32],
    config: SolveConfig,
    *,
    has_sphere_points: bool,
) -> _CircSymFieldFrequencyBatch | None:
    """Prepare the ordinary generated observation field in one Metal batch."""
    if config.observation.custom_points is not None or has_sphere_points:
        return None
    kvals = np.asarray(k_values, dtype=np.complex128).reshape(-1)
    orders = np.asarray(n_psi_values, dtype=np.int32).reshape(-1)
    if kvals.shape != orders.shape or kvals.size == 0:
        raise ValueError("k_values and n_psi_values must be non-empty and aligned")
    selections = [
        _select_circsym_field_backend(
            int(obs_points.shape[1]), meridian.segment_count, int(order)
        )
        for order in orders
    ]
    if any(backend != "metal" for backend, _ in selections):
        return None
    runtime_status = selections[0][1]
    if runtime_status is None:
        return None

    started = time.perf_counter()
    target_rho, target_z = _points_target_rho_z(obs_points[0])
    rayleigh_sheet = _is_flat_baffled_sheet(
        meridian, config.circsym_baffle_z, geom=geom
    )
    active_targets: NDArray[np.bool_] | None = None
    if rayleigh_sheet:
        active_targets = _baffled_sheet_active_targets(
            meridian,
            target_z,
            float(config.circsym_baffle_z),
            geom=geom,
        )
        if not np.any(active_targets):
            return None
    targets = np.column_stack((target_rho, target_z))
    if active_targets is not None:
        targets = targets[active_targets]
    unique_targets, inverse = np.unique(targets, axis=0, return_inverse=True)
    source_indices = np.arange(meridian.segment_count, dtype=np.int64)
    far_mask = _ordinary_far_source_mask_targets(
        unique_targets[:, 0],
        unique_targets[:, 1],
        geom,
        source_indices=source_indices,
    )
    # Retain the adaptive FP64 path for any near observation. The product
    # fixture is a 2 m generated arc and is entirely ordinary/far.
    if not np.all(far_mask):
        return None

    u, w = _ordinary_interval(0.0, 1.0)
    source = (
        geom.p0[:, None, :]
        + u[None, :, None] * geom.delta[:, None, :]
    )
    rho_s = source[:, :, 0]
    z_s = source[:, :, 1]
    line_measure = rho_s * geom.lengths[:, None] * w[None, :]
    quadrature = tuple(_leggauss_psi(int(order)) for order in orders)
    try:
        from .metal.native import CircSymMetalCancelled
        from .metal.native import evaluate_circsym_ring_field_kernels_batch

        result = evaluate_circsym_ring_field_kernels_batch(
            target_rho=unique_targets[:, 0],
            target_z=unique_targets[:, 1],
            source_rho=rho_s,
            source_z=z_s,
            measure=line_measure,
            normal_rho=meridian.normals[:, 0],
            normal_z=meridian.normals[:, 1],
            cos_psi_by_frequency=tuple(np.cos(item[0]) for item in quadrature),
            psi_weights_by_frequency=tuple(item[1] for item in quadrature),
            k_values=kvals,
            baffle_z=config.circsym_baffle_z,
            runtime_status=runtime_status,
            should_continue=config.should_continue,
            operation_id="circsym-observation-frequency-batch",
        )
    except CircSymMetalCancelled as exc:
        raise CircSymCancelled("CircSym solve cancelled") from exc
    except Exception as exc:
        if _requested_circsym_field_backend() == "metal":
            raise
        _record_circsym_metal_field_failure(exc)
        return None
    return _CircSymFieldFrequencyBatch(
        slp=result.slp,
        dlp=result.dlp,
        target_inverse=np.asarray(inverse, dtype=np.int64),
        active_targets=active_targets,
        rayleigh_sheet=rayleigh_sheet,
        wall_s=time.perf_counter() - started,
        diagnostics=result.diagnostics,
    )


def _integrate_field_segment_kernels_batched(
    *,
    target_rho: NDArray[np.float64],
    target_z: NDArray[np.float64],
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    source_indices: NDArray[np.int64] | NDArray[np.int32],
    k: complex,
    baffle_z: float | None,
    n_psi: int,
    should_continue: Callable[[], bool | None] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    _check_circsym_continue(should_continue)
    target_rho_arr = np.asarray(target_rho, dtype=np.float64).reshape(-1)
    target_z_arr = np.asarray(target_z, dtype=np.float64).reshape(-1)
    if target_rho_arr.shape != target_z_arr.shape:
        raise ValueError("target_rho and target_z must have the same shape")
    indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if target_rho_arr.size == 0 or indices.size == 0:
        shape = (target_rho_arr.size, indices.size)
        return (
            np.empty(shape, dtype=np.complex128),
            np.empty(shape, dtype=np.complex128),
        )

    # Arbitrary/custom observation lists can contain exact axisymmetric
    # duplicates (different Cartesian points with identical rho and z).  Evaluate
    # each distinct m=0 target once, then restore the caller's order.  Deliberately
    # use exact float equality here: generated sphere grids receive the stronger,
    # metadata-backed theta collapse above, while near-boundary custom points must
    # never be merged by a geometric rounding tolerance.
    targets = np.column_stack((target_rho_arr, target_z_arr))
    unique_targets, inverse = np.unique(targets, axis=0, return_inverse=True)
    work_rho = unique_targets[:, 0]
    work_z = unique_targets[:, 1]

    s_mat = np.empty((work_rho.size, indices.size), dtype=np.complex128)
    h_mat = np.empty_like(s_mat)
    field_backend, metal_runtime_status = _select_circsym_field_backend(
        work_rho.size,
        indices.size,
        int(n_psi),
    )
    cpu_field_backend = _circsym_cpu_field_status()["selected"]
    workers = (
        1
        if field_backend == "metal" or cpu_field_backend == "numba"
        else _field_kernel_worker_count(work_rho.size, indices.size)
    )
    block_size = (
        int(work_rho.size)
        if field_backend == "metal"
        else _field_kernel_target_block_size(indices.size, int(n_psi))
    )
    if workers > 1:
        block_size = min(block_size, _FIELD_KERNEL_PARALLEL_TARGET_BLOCK)
    ranges = [
        (start, min(work_rho.size, start + block_size))
        for start in range(0, work_rho.size, block_size)
    ]

    def compute_block(
        span: tuple[int, int],
    ) -> tuple[int, int, NDArray[np.complex128], NDArray[np.complex128]]:
        start, stop = span
        block_rho = work_rho[start:stop]
        block_z = work_z[start:stop]
        s_block, h_block = _integrate_ordinary_field_kernels_targets_batched(
            target_rho=block_rho,
            target_z=block_z,
            meridian=meridian,
            geom=geom,
            source_indices=indices,
            k=k,
            baffle_z=baffle_z,
            n_psi=n_psi,
            backend=field_backend,
            metal_runtime_status=metal_runtime_status,
            should_continue=should_continue,
        )
        near_mask = ~_ordinary_far_source_mask_targets(
            block_rho,
            block_z,
            geom,
            source_indices=indices,
        )
        if np.any(near_mask):
            near_points, near_sources = np.nonzero(near_mask)
            for point_local, source_local in zip(near_points, near_sources):
                s_val, h_val = _integrate_segment_kernel(
                    target_rho=float(block_rho[point_local]),
                    target_z=float(block_z[point_local]),
                    meridian=meridian,
                    geom=geom,
                    source_index=int(indices[source_local]),
                    k=k,
                    baffle_z=baffle_z,
                    n_psi=n_psi,
                    target_index=None,
                )
                s_block[point_local, source_local] = s_val
                h_block[point_local, source_local] = h_val
        return start, stop, s_block, h_block

    if workers <= 1 or len(ranges) <= 1:
        for span in ranges:
            _check_circsym_continue(should_continue)
            start, stop, s_block, h_block = compute_block(span)
            s_mat[start:stop] = s_block
            h_mat[start:stop] = h_block
            _check_circsym_continue(should_continue)
    else:
        executor = ThreadPoolExecutor(max_workers=min(workers, len(ranges)))
        try:
            for start, stop, s_block, h_block in executor.map(compute_block, ranges):
                _check_circsym_continue(should_continue)
                s_mat[start:stop] = s_block
                h_mat[start:stop] = h_block
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    return s_mat[inverse], h_mat[inverse]


def _field_kernel_target_block_size(source_count: int, n_psi: int) -> int:
    per_target = max(1, int(source_count) * _LINE_QUAD_ORDER * max(1, int(n_psi)))
    by_elements = max(1, _FIELD_KERNEL_BLOCK_ELEMENTS // per_target)
    return max(1, min(_FIELD_KERNEL_MAX_TARGET_BLOCK, int(by_elements)))


def _field_kernel_worker_count(target_count: int, source_count: int) -> int:
    if int(target_count) < 16 or int(source_count) < 16:
        return 1
    raw = os.environ.get("HORNLAB_CIRCSYM_FIELD_THREADS")
    if raw is not None:
        try:
            return max(1, min(int(target_count), int(raw)))
        except ValueError:
            return 1
    return max(1, min(int(target_count), _FIELD_KERNEL_MAX_WORKERS, os.cpu_count() or 1))


def _ordinary_far_source_mask_targets(
    target_rho: NDArray[np.float64],
    target_z: NDArray[np.float64],
    geom: SimpleNamespace,
    *,
    source_indices: NDArray[np.int64] | NDArray[np.int32],
) -> NDArray[np.bool_]:
    rho = np.asarray(target_rho, dtype=np.float64).reshape(-1)
    z = np.asarray(target_z, dtype=np.float64).reshape(-1)
    indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    target = np.stack([rho, z], axis=1)
    p0 = geom.p0[indices]
    delta = geom.delta[indices]
    lengths = geom.lengths[indices]
    denom = np.maximum(lengths * lengths, 1.0e-30)
    u_star = np.einsum("pnd,nd->pn", target[:, None, :] - p0[None, :, :], delta) / denom
    u_clamped = np.clip(u_star, 0.0, 1.0)
    closest = p0[None, :, :] + u_clamped[:, :, None] * delta[None, :, :]
    distance = np.linalg.norm(target[:, None, :] - closest, axis=2)
    return distance > 1.25 * lengths[None, :]


def _integrate_ordinary_field_kernels_targets_batched(
    *,
    target_rho: NDArray[np.float64],
    target_z: NDArray[np.float64],
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    source_indices: NDArray[np.int64] | NDArray[np.int32],
    k: complex,
    baffle_z: float | None,
    n_psi: int,
    backend: str,
    metal_runtime_status: Any | None,
    should_continue: Callable[[], bool | None] | None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    indices = np.asarray(source_indices, dtype=np.int64)
    target_rho_arr = np.asarray(target_rho, dtype=np.float64).reshape(-1)
    target_z_arr = np.asarray(target_z, dtype=np.float64).reshape(-1)
    if indices.size == 0 or target_rho_arr.size == 0:
        shape = (target_rho_arr.size, indices.size)
        return (
            np.empty(shape, dtype=np.complex128),
            np.empty(shape, dtype=np.complex128),
        )

    u, w = _ordinary_interval(0.0, 1.0)
    psi, psi_weights = _leggauss_psi(int(n_psi))
    cos_psi = np.cos(psi)[None, None, :]
    p0 = geom.p0[indices]
    delta = geom.delta[indices]
    lengths = geom.lengths[indices]
    normal = meridian.normals[indices]
    if backend == "metal":
        source = p0[:, None, :] + u[None, :, None] * delta[:, None, :]
        rho_s = source[:, :, 0]
        z_s = source[:, :, 1]
        line_measure = rho_s * lengths[:, None] * w[None, :]
        try:
            from .metal.native import CircSymMetalCancelled
            from .metal.native import evaluate_circsym_ring_field_kernels

            result = evaluate_circsym_ring_field_kernels(
                target_rho=target_rho_arr,
                target_z=target_z_arr,
                source_rho=rho_s,
                source_z=z_s,
                measure=line_measure,
                normal_rho=normal[:, 0],
                normal_z=normal[:, 1],
                cos_psi=np.cos(psi),
                psi_weights=psi_weights,
                k=k,
                baffle_z=baffle_z,
                runtime_status=metal_runtime_status,
                should_continue=should_continue,
            )
            return result.slp, result.dlp
        except CircSymMetalCancelled as exc:
            raise CircSymCancelled("CircSym solve cancelled") from exc
        except Exception as exc:
            if _requested_circsym_field_backend() == "metal":
                raise
            _record_circsym_metal_field_failure(exc)

    cpu_field = _circsym_cpu_field_status()
    if cpu_field["selected"] == "numba":
        source = p0[:, None, :] + u[None, :, None] * delta[:, None, :]
        rho_s = np.ascontiguousarray(source[:, :, 0], dtype=np.float64)
        z_s = np.ascontiguousarray(source[:, :, 1], dtype=np.float64)
        line_measure = np.ascontiguousarray(
            rho_s * lengths[:, None] * w[None, :], dtype=np.float64
        )
        implementation = _load_circsym_remainder_numba_kernel()
        assert implementation is not None
        k_value = complex(k)
        return implementation.evaluate_field_onthefly(
            np.ascontiguousarray(target_rho_arr, dtype=np.float64),
            np.ascontiguousarray(target_z_arr, dtype=np.float64),
            rho_s,
            z_s,
            line_measure,
            np.ascontiguousarray(normal[:, 0], dtype=np.float64),
            np.ascontiguousarray(normal[:, 1], dtype=np.float64),
            np.ascontiguousarray(np.cos(psi), dtype=np.float64),
            np.ascontiguousarray(psi_weights, dtype=np.float64),
            baffle_z is not None,
            0.0 if baffle_z is None else float(baffle_z),
            float(k_value.real),
            float(k_value.imag),
        )

    normal_rho = normal[:, 0][None, :, None]
    normal_z = normal[:, 1][None, :, None]
    rt = target_rho_arr[:, None, None]
    zt = target_z_arr[:, None, None]

    s_out = np.zeros((target_rho_arr.size, indices.size), dtype=np.complex128)
    h_out = np.zeros_like(s_out)
    for u_node, w_node in zip(u, w):
        source = p0 + float(u_node) * delta
        rho_s = source[:, 0]
        z_s = source[:, 1]
        measure = rho_s * lengths * float(w_node)
        if not np.any(measure != 0.0):
            continue
        rs = rho_s[None, :, None]
        zs = z_s[None, :, None]
        g, h = _ring_dynamic_kernel_m0_targets_direct(
            rt,
            zt,
            rs,
            zs,
            normal_rho,
            normal_z,
            cos_psi,
            psi_weights,
            k,
        )

        if baffle_z is not None:
            z_img = 2.0 * float(baffle_z) - z_s
            g_i, h_i = _ring_dynamic_kernel_m0_targets_direct(
                rt,
                zt,
                rs,
                z_img[None, :, None],
                normal_rho,
                -normal_z,
                cos_psi,
                psi_weights,
                k,
            )
            g = g + g_i
            h = h + h_i

        s_out += g * measure[None, :]
        h_out += h * measure[None, :]

    return s_out, h_out


def _ring_dynamic_kernel_m0_targets_direct(
    target_rho: NDArray[np.float64],
    target_z: NDArray[np.float64],
    source_rho: NDArray[np.float64],
    source_z: NDArray[np.float64],
    normal_rho: NDArray[np.float64],
    normal_z: NDArray[np.float64],
    cos_psi: NDArray[np.float64],
    psi_weights: NDArray[np.float64],
    k: complex,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    dz = source_z - target_z
    R2 = (
        target_rho * target_rho
        + source_rho * source_rho
        - 2.0 * target_rho * source_rho * cos_psi
        + dz * dz
    )
    R = np.sqrt(np.maximum(R2, 0.0))
    phase = np.exp(1j * complex(k) * R)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = phase / (4.0 * np.pi * R)
        num = (source_rho - target_rho * cos_psi) * normal_rho + dz * normal_z
        h = phase * (1j * complex(k) * R - 1.0) * num / (4.0 * np.pi * R ** 3)
    g = np.where(R > 1e-13, g, 0.0 + 0.0j)
    h = np.where(R > 1e-13, h, 0.0 + 0.0j)
    weights = psi_weights[None, None, :]
    return (
        np.asarray(2.0 * np.sum(g * weights, axis=2), dtype=np.complex128),
        np.asarray(2.0 * np.sum(h * weights, axis=2), dtype=np.complex128),
    )


def _is_flat_baffled_sheet(
    meridian: MeridianMesh,
    baffle_z: float | None,
    *,
    geom: SimpleNamespace | None = None,
) -> bool:
    if baffle_z is None:
        return False
    if geom is None:
        geom = meridian.segment_geometry()
    active = geom.area_weights > 1e-30
    if not np.any(active):
        return False
    z_close = np.allclose(geom.midpoints[active, 1], float(baffle_z), atol=1e-10)
    normals = meridian.normals[active]
    normal_close = np.all(np.abs(normals[:, 0]) <= 1e-10) and (
        np.all(np.abs(normals[:, 1] - 1.0) <= 1e-10)
        or np.all(np.abs(normals[:, 1] + 1.0) <= 1e-10)
    )
    return bool(z_close and normal_close)


def _baffled_sheet_active_targets(
    meridian: MeridianMesh,
    target_z: NDArray[np.float64],
    baffle_z: float,
    *,
    geom: SimpleNamespace,
) -> NDArray[np.bool_]:
    active_segments = geom.area_weights > 1e-30
    normal_z = float(meridian.normals[np.flatnonzero(active_segments)[0], 1])
    offset = np.asarray(target_z, dtype=np.float64) - float(baffle_z)
    return normal_z * offset >= -1e-12


def _boundary_free_terms(
    meridian: MeridianMesh,
    baffle_z: float | None,
) -> NDArray[np.float64]:
    terms = np.full(meridian.segment_count, 0.5, dtype=np.float64)
    if _is_flat_baffled_sheet(meridian, baffle_z):
        terms[:] = 1.0
    return terms


def _meridian_node_degrees(segments: NDArray[np.int32]) -> dict[int, int]:
    degrees: dict[int, int] = {}
    for start, end in np.asarray(segments, dtype=np.int64):
        degrees[int(start)] = degrees.get(int(start), 0) + 1
        degrees[int(end)] = degrees.get(int(end), 0) + 1
    return degrees


def _validate_closed_or_baffled_meridian(
    meridian: MeridianMesh,
    baffle_z: float | None,
) -> None:
    if baffle_z is not None:
        if not _is_flat_baffled_sheet(meridian, baffle_z):
            raise ValueError(
                "circsym_baffle_z is only supported for a coplanar flat "
                "Rayleigh sheet; recessed or non-planar waveguides require "
                "circsym_aperture_tag coupled infinite-baffle mode"
            )
        return
    degrees = _meridian_node_degrees(meridian.segments)
    endpoint_nodes = [node for node, degree in degrees.items() if degree == 1]
    if len(endpoint_nodes) != 2:
        raise ValueError(
            "CircSym one-trace BEM requires a closed body-of-revolution meridian "
            "with both endpoints on the symmetry axis; bare/open meridians need "
            "a dedicated open-screen formulation"
        )
    endpoint_rho = meridian.nodes[np.asarray(endpoint_nodes, dtype=np.int64), 0]
    if np.any(endpoint_rho > 1.0e-9):
        raise ValueError(
            "CircSym one-trace BEM requires a closed body-of-revolution meridian "
            "with both endpoints on the symmetry axis; bare/open meridians need "
            "a dedicated open-screen formulation"
        )


def _validate_coupled_ib_meridian(
    meridian: MeridianMesh,
    aperture_tag: int,
    *,
    geom: SimpleNamespace | None = None,
) -> None:
    """Validate the axisymmetric interior-channel/Rayleigh aperture contract."""
    if geom is None:
        geom = meridian.segment_geometry()
    tolerance = 1.0e-9
    tags = meridian.physical_tags
    aperture_indices = np.where(tags == int(aperture_tag))[0]
    if aperture_indices.size == 0:
        raise ValueError("circsym_aperture_tag must select at least one segment")

    aperture_segments = meridian.segments[aperture_indices]
    aperture_nodes = np.unique(aperture_segments.reshape(-1))
    if not np.all(np.abs(meridian.nodes[aperture_nodes, 1]) <= tolerance):
        raise ValueError(
            "circsym_aperture_tag segments must be coplanar on the global z=0 "
            "baffle plane"
        )
    aperture_normals = meridian.normals[aperture_indices]
    if not (
        np.all(np.abs(aperture_normals[:, 0]) <= tolerance)
        and np.all(np.abs(aperture_normals[:, 1] + 1.0) <= tolerance)
    ):
        raise ValueError(
            "circsym_aperture_tag normals must point -Z into the interior cavity"
        )
    if np.any(meridian.nodes[:, 1] > tolerance):
        raise ValueError(
            "coupled infinite-baffle CircSym requires the entire cavity at z <= 0"
        )

    aperture_degrees = _meridian_node_degrees(aperture_segments)
    aperture_endpoints = [node for node, degree in aperture_degrees.items() if degree == 1]
    if (
        len(aperture_endpoints) != 2
        or any(degree > 2 for degree in aperture_degrees.values())
    ):
        raise ValueError(
            "circsym_aperture_tag must form one unbranched contiguous mouth-to-axis disc"
        )
    aperture_adjacency = {node: [] for node in aperture_degrees}
    for start, end in np.asarray(aperture_segments, dtype=np.int64):
        aperture_adjacency[int(start)].append(int(end))
        aperture_adjacency[int(end)].append(int(start))
    visited = {aperture_endpoints[0]}
    stack = [aperture_endpoints[0]]
    while stack:
        node = stack.pop()
        for neighbour in aperture_adjacency[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append(neighbour)
    if visited != set(aperture_degrees):
        raise ValueError(
            "circsym_aperture_tag must form one unbranched contiguous mouth-to-axis disc"
        )
    endpoint_radii = meridian.nodes[np.asarray(aperture_endpoints), 0]
    if int(np.count_nonzero(endpoint_radii <= tolerance)) != 1:
        raise ValueError(
            "circsym_aperture_tag must span exactly from the mouth rim to the symmetry axis"
        )

    full_degrees = _meridian_node_degrees(meridian.segments)
    full_endpoints = [node for node, degree in full_degrees.items() if degree == 1]
    if len(full_endpoints) != 2 or any(degree > 2 for degree in full_degrees.values()):
        raise ValueError(
            "coupled infinite-baffle CircSym meridian must be one closed, unbranched channel"
        )
    if np.any(meridian.nodes[np.asarray(full_endpoints), 0] > tolerance):
        raise ValueError(
            "coupled infinite-baffle CircSym channel must close on the symmetry axis "
            "at the throat and aperture"
        )
    full_adjacency = {node: [] for node in full_degrees}
    for start, end in np.asarray(meridian.segments, dtype=np.int64):
        full_adjacency[int(start)].append(int(end))
        full_adjacency[int(end)].append(int(start))
    visited = {full_endpoints[0]}
    stack = [full_endpoints[0]]
    while stack:
        node = stack.pop()
        for neighbour in full_adjacency[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append(neighbour)
    if visited != set(full_degrees):
        raise ValueError(
            "coupled infinite-baffle CircSym meridian must be one closed, unbranched channel"
        )


def _rcond_from_lu_factor(lu: NDArray[np.complex128], anorm: float) -> float | None:
    if not math.isfinite(float(anorm)) or float(anorm) <= 0.0:
        return None
    gecon = linalg.lapack.get_lapack_funcs("gecon", (lu,))
    rcond, info = gecon(lu, float(anorm), norm="1")
    if int(info) != 0:
        return None
    return float(rcond)


def _surface_pressure_average(
    meridian: MeridianMesh,
    pressure: NDArray[np.complex128],
    tags: list[int],
) -> dict[int, complex]:
    geom = meridian.segment_geometry()
    result: dict[int, complex] = {}
    for tag in tags:
        idx = np.where(meridian.physical_tags == int(tag))[0]
        if idx.size == 0:
            result[int(tag)] = 0.0 + 0.0j
            continue
        weights = geom.area_weights[idx]
        total = float(np.sum(weights))
        if total <= 1e-30:
            result[int(tag)] = 0.0 + 0.0j
        else:
            result[int(tag)] = complex(np.sum(pressure[idx] * weights) / total)
    return result


def _integrate_segment_kernel(
    *,
    target_rho: float,
    target_z: float,
    meridian: MeridianMesh,
    geom: SimpleNamespace | None = None,
    source_index: int,
    k: complex,
    baffle_z: float | None,
    n_psi: int,
    target_index: int | None,
) -> tuple[complex, complex]:
    if geom is None:
        geom = meridian.segment_geometry()
    p0 = geom.p0[source_index]
    delta = geom.delta[source_index]
    length = float(geom.lengths[source_index])
    normal = meridian.normals[source_index]
    u, w = _segment_quadrature_nodes(
        target_rho=target_rho,
        target_z=target_z,
        source_p0=p0,
        source_delta=delta,
        source_length=length,
        self_pair=target_index == source_index,
    )
    source = p0[None, :] + u[:, None] * delta[None, :]
    rho_s = source[:, 0]
    z_s = source[:, 1]
    measure = rho_s * length * w
    if not np.any(measure != 0.0):
        return 0.0 + 0.0j, 0.0 + 0.0j

    g_static, h_static = _ring_static_kernel_m0(
        target_rho, target_z, rho_s, z_s, normal
    )
    g_rem, h_rem = _ring_remainder_kernel_m0(
        target_rho, target_z, rho_s, z_s, normal, k, n_psi=n_psi
    )
    g = g_static + g_rem
    h = h_static + h_rem

    if baffle_z is not None:
        z_img = 2.0 * float(baffle_z) - z_s
        normal_img = np.array([normal[0], -normal[1]], dtype=np.float64)
        g_static_i, h_static_i = _ring_static_kernel_m0(
            target_rho, target_z, rho_s, z_img, normal_img
        )
        g_rem_i, h_rem_i = _ring_remainder_kernel_m0(
            target_rho, target_z, rho_s, z_img, normal_img, k, n_psi=n_psi
        )
        g = g + g_static_i + g_rem_i
        h = h + h_static_i + h_rem_i

    return complex(np.sum(g * measure)), complex(np.sum(h * measure))


def _segment_quadrature_nodes(
    *,
    target_rho: float,
    target_z: float,
    source_p0: NDArray[np.float64],
    source_delta: NDArray[np.float64],
    source_length: float,
    self_pair: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if self_pair:
        left = _graded_interval(0.0, 0.5, cluster_at="right")
        right = _graded_interval(0.5, 1.0, cluster_at="left")
        return (
            np.concatenate([left[0], right[0]]),
            np.concatenate([left[1], right[1]]),
        )

    target = np.array([target_rho, target_z], dtype=np.float64)
    denom = max(float(source_length) ** 2, 1e-30)
    u_star = float(np.dot(target - source_p0, source_delta) / denom)
    u_clamped = min(1.0, max(0.0, u_star))
    closest = source_p0 + u_clamped * source_delta
    distance = float(np.linalg.norm(target - closest))
    if distance > 1.25 * float(source_length):
        return _ordinary_interval(0.0, 1.0)

    if u_clamped <= 1e-6:
        return _graded_interval(0.0, 1.0, cluster_at="left")
    if u_clamped >= 1.0 - 1e-6:
        return _graded_interval(0.0, 1.0, cluster_at="right")

    left = _graded_interval(0.0, u_clamped, cluster_at="right")
    right = _graded_interval(u_clamped, 1.0, cluster_at="left")
    return (
        np.concatenate([left[0], right[0]]),
        np.concatenate([left[1], right[1]]),
    )


def _ordinary_interval(a: float, b: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x, w = _leggauss01(_LINE_QUAD_ORDER)
    width = float(b) - float(a)
    return float(a) + width * x, width * w


def _graded_interval(
    a: float,
    b: float,
    *,
    cluster_at: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x, w = _leggauss01(_SINGULAR_LINE_QUAD_ORDER)
    width = float(b) - float(a)
    if width <= 0.0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if cluster_at == "left":
        xp = x ** _GRADED_POWER
        jac = _GRADED_POWER * x ** (_GRADED_POWER - 1.0)
        return float(a) + width * xp, width * jac * w
    if cluster_at == "right":
        y = 1.0 - x
        xp = 1.0 - y ** _GRADED_POWER
        jac = _GRADED_POWER * y ** (_GRADED_POWER - 1.0)
        return float(a) + width * xp, width * jac * w
    raise ValueError("cluster_at must be 'left' or 'right'")


def _ring_static_kernel_m0(
    target_rho: float,
    target_z: float,
    source_rho: NDArray[np.float64] | float,
    source_z: NDArray[np.float64] | float,
    source_normal: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Static m=0 ring kernels from complete elliptic integrals.

    This is the singular part used for subtraction. The formulas integrate
    ``1 / (4*pi*R)`` and its source-normal derivative over the full source ring.
    """
    rt = float(target_rho)
    zt = float(target_z)
    rs = np.asarray(source_rho, dtype=np.float64)
    zs = np.asarray(source_z, dtype=np.float64)
    n_rho = float(source_normal[0])
    n_z = float(source_normal[1])
    D = (rt + rs) ** 2 + (zt - zs) ** 2
    sqrtD = np.sqrt(D)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(D > 0.0, 4.0 * rt * rs / D, 0.0)
    m = np.clip(m, 0.0, 1.0 - 1e-15)
    K = ellipk(m)
    E = ellipe(m)
    G = K / (np.pi * sqrtD)

    dKdm = _ellipk_derivative(m, K, E)
    dD_dr = 2.0 * (rt + rs)
    dD_dz = 2.0 * (zs - zt)
    with np.errstate(divide="ignore", invalid="ignore"):
        dm_dr = 4.0 * rt / D - (4.0 * rt * rs / (D * D)) * dD_dr
        dm_dz = -(4.0 * rt * rs / (D * D)) * dD_dz
        dF_dr = (
            dKdm * dm_dr / sqrtD - 0.5 * K * dD_dr / (D * sqrtD)
        ) / np.pi
        dF_dz = (
            dKdm * dm_dz / sqrtD - 0.5 * K * dD_dz / (D * sqrtD)
        ) / np.pi
    H = n_rho * dF_dr + n_z * dF_dz
    return np.asarray(G, dtype=np.complex128), np.asarray(H, dtype=np.complex128)


def _ring_static_kernel_m0_targets_batched(
    target_rho: NDArray[np.float64],
    target_z: NDArray[np.float64],
    source_rho: NDArray[np.float64],
    source_z: NDArray[np.float64],
    normal_rho: NDArray[np.float64],
    normal_z: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    rt = np.asarray(target_rho, dtype=np.float64).reshape(-1, 1, 1)
    zt = np.asarray(target_z, dtype=np.float64).reshape(-1, 1, 1)
    rs = np.asarray(source_rho, dtype=np.float64)[None, :, :]
    zs = np.asarray(source_z, dtype=np.float64)[None, :, :]
    n_rho = np.asarray(normal_rho, dtype=np.float64).reshape(1, -1, 1)
    n_z = np.asarray(normal_z, dtype=np.float64).reshape(1, -1, 1)
    D = (rt + rs) ** 2 + (zt - zs) ** 2
    sqrtD = np.sqrt(D)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(D > 0.0, 4.0 * rt * rs / D, 0.0)
    m = np.clip(m, 0.0, 1.0 - 1e-15)
    K = ellipk(m)
    E = ellipe(m)
    G = K / (np.pi * sqrtD)

    dKdm = _ellipk_derivative(m, K, E)
    dD_dr = 2.0 * (rt + rs)
    dD_dz = 2.0 * (zs - zt)
    with np.errstate(divide="ignore", invalid="ignore"):
        dm_dr = 4.0 * rt / D - (4.0 * rt * rs / (D * D)) * dD_dr
        dm_dz = -(4.0 * rt * rs / (D * D)) * dD_dz
        dF_dr = (
            dKdm * dm_dr / sqrtD - 0.5 * K * dD_dr / (D * sqrtD)
        ) / np.pi
        dF_dz = (
            dKdm * dm_dz / sqrtD - 0.5 * K * dD_dz / (D * sqrtD)
        ) / np.pi
    H = n_rho * dF_dr + n_z * dF_dz
    return np.asarray(G, dtype=np.complex128), np.asarray(H, dtype=np.complex128)


def _ellipk_derivative(
    m: NDArray[np.float64],
    K: NDArray[np.float64],
    E: NDArray[np.float64],
) -> NDArray[np.float64]:
    m_arr = np.asarray(m, dtype=np.float64)
    out = np.empty_like(m_arr)
    small = m_arr < 1e-8
    out[small] = (np.pi / 8.0) * (1.0 + 1.125 * m_arr[small])
    regular = ~small
    out[regular] = (
        E[regular] - (1.0 - m_arr[regular]) * K[regular]
    ) / (2.0 * m_arr[regular] * (1.0 - m_arr[regular]))
    return out


def _ring_remainder_kernel_m0(
    target_rho: float,
    target_z: float,
    source_rho: NDArray[np.float64] | float,
    source_z: NDArray[np.float64] | float,
    source_normal: NDArray[np.float64],
    k: complex,
    *,
    n_psi: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Smooth dynamic-static m=0 ring kernel remainder."""
    rs = np.atleast_1d(np.asarray(source_rho, dtype=np.float64))
    zs = np.atleast_1d(np.asarray(source_z, dtype=np.float64))
    psi, weights = _leggauss_psi(int(n_psi))
    cos_psi = np.cos(psi)[None, :]
    rt = float(target_rho)
    zt = float(target_z)
    n_rho = float(source_normal[0])
    n_z = float(source_normal[1])

    dz = zs[:, None] - zt
    R2 = (
        rt * rt
        + rs[:, None] * rs[:, None]
        - 2.0 * rt * rs[:, None] * cos_psi
        + dz * dz
    )
    R = np.sqrt(np.maximum(R2, 0.0))
    q = complex(k) * R
    with np.errstate(divide="ignore", invalid="ignore"):
        rem_g = np.expm1(1j * q) / (4.0 * np.pi * R)
    rem_g = np.where(R > 1e-13, rem_g, 1j * complex(k) / (4.0 * np.pi))

    num = (rs[:, None] - rt * cos_psi) * n_rho + dz * n_z
    expr = np.exp(1j * q) * (1j * q - 1.0) + 1.0
    small = np.abs(q) < 1e-5
    if np.any(small):
        qs = q[small]
        expr = expr.astype(np.complex128, copy=True)
        expr[small] = (
            -0.5 * qs * qs
            - (1j / 3.0) * qs ** 3
            + 0.125 * qs ** 4
            + (1j / 30.0) * qs ** 5
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        rem_h = expr * num / (4.0 * np.pi * R ** 3)
    rem_h = np.where(R > 1e-13, rem_h, 0.0 + 0.0j)

    G = 2.0 * np.sum(rem_g * weights[None, :], axis=1)
    H = 2.0 * np.sum(rem_h * weights[None, :], axis=1)
    if np.ndim(source_rho) == 0:
        return G[0], H[0]
    return G, H


def _ring_remainder_kernel_m0_targets_batched(
    target_rho: NDArray[np.float64],
    target_z: NDArray[np.float64],
    source_rho: NDArray[np.float64],
    source_z: NDArray[np.float64],
    normal_rho: NDArray[np.float64],
    normal_z: NDArray[np.float64],
    k: complex,
    *,
    n_psi: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    rs = np.asarray(source_rho, dtype=np.float64)[None, :, :, None]
    zs = np.asarray(source_z, dtype=np.float64)[None, :, :, None]
    psi, weights = _leggauss_psi(int(n_psi))
    cos_psi = np.cos(psi)[None, None, None, :]
    rt = np.asarray(target_rho, dtype=np.float64).reshape(-1, 1, 1, 1)
    zt = np.asarray(target_z, dtype=np.float64).reshape(-1, 1, 1, 1)
    n_rho = np.asarray(normal_rho, dtype=np.float64).reshape(1, -1, 1, 1)
    n_z = np.asarray(normal_z, dtype=np.float64).reshape(1, -1, 1, 1)

    dz = zs - zt
    R2 = rt * rt + rs * rs - 2.0 * rt * rs * cos_psi + dz * dz
    R = np.sqrt(np.maximum(R2, 0.0))
    q = complex(k) * R
    with np.errstate(divide="ignore", invalid="ignore"):
        rem_g = np.expm1(1j * q) / (4.0 * np.pi * R)
    rem_g = np.where(R > 1e-13, rem_g, 1j * complex(k) / (4.0 * np.pi))

    num = (rs - rt * cos_psi) * n_rho + dz * n_z
    expr = np.exp(1j * q) * (1j * q - 1.0) + 1.0
    small = np.abs(q) < 1e-5
    if np.any(small):
        qs = q[small]
        expr = expr.astype(np.complex128, copy=True)
        expr[small] = (
            -0.5 * qs * qs
            - (1j / 3.0) * qs ** 3
            + 0.125 * qs ** 4
            + (1j / 30.0) * qs ** 5
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        rem_h = expr * num / (4.0 * np.pi * R2 * R)
    rem_h = np.where(R > 1e-13, rem_h, 0.0 + 0.0j)

    G = 2.0 * np.sum(rem_g * weights[None, None, None, :], axis=3)
    H = 2.0 * np.sum(rem_h * weights[None, None, None, :], axis=3)
    return np.asarray(G, dtype=np.complex128), np.asarray(H, dtype=np.complex128)


def ring_kernel_m0(
    target_rho: float,
    target_z: float,
    source_rho: float,
    source_z: float,
    source_normal: NDArray[np.float64],
    k: complex,
    *,
    n_psi: int | None = None,
    baffle_z: float | None = None,
) -> tuple[complex, complex]:
    """Return singular-subtracted m=0 ring kernels ``(G0, H0)``.

    The Green kernel is integrated over the full source ring. ``H0`` is the
    source-normal derivative using normal components ``(n_rho, n_z)``.
    """
    order = int(n_psi or _azimuth_order(k, max(target_rho, source_rho)))
    g_s, h_s = _ring_static_kernel_m0(
        target_rho,
        target_z,
        float(source_rho),
        float(source_z),
        np.asarray(source_normal, dtype=np.float64),
    )
    g_r, h_r = _ring_remainder_kernel_m0(
        target_rho,
        target_z,
        float(source_rho),
        float(source_z),
        np.asarray(source_normal, dtype=np.float64),
        k,
        n_psi=order,
    )
    g = complex(g_s + g_r)
    h = complex(h_s + h_r)
    if baffle_z is not None:
        normal_img = np.array([source_normal[0], -source_normal[1]], dtype=np.float64)
        z_img = 2.0 * float(baffle_z) - float(source_z)
        gi, hi = ring_kernel_m0(
            target_rho,
            target_z,
            source_rho,
            z_img,
            normal_img,
            k,
            n_psi=order,
            baffle_z=None,
        )
        g += gi
        h += hi
    return g, h


def ring_kernel_m0_direct_quadrature(
    target_rho: float,
    target_z: float,
    source_rho: float,
    source_z: float,
    source_normal: NDArray[np.float64],
    k: complex,
    *,
    n_psi: int = 8192,
) -> tuple[complex, complex]:
    """Reference full-ring Gauss quadrature without singular subtraction."""
    psi, weights = _leggauss_psi(int(n_psi))
    g, h = _ring_dynamic_integrand(
        target_rho,
        target_z,
        np.asarray([source_rho], dtype=np.float64),
        np.asarray([source_z], dtype=np.float64),
        np.asarray(source_normal, dtype=np.float64),
        k,
        psi,
    )
    return (
        complex(2.0 * np.sum(g[0] * weights)),
        complex(2.0 * np.sum(h[0] * weights)),
    )


def _ring_dynamic_integrand(
    target_rho: float,
    target_z: float,
    source_rho: NDArray[np.float64],
    source_z: NDArray[np.float64],
    source_normal: NDArray[np.float64],
    k: complex,
    psi: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    cos_psi = np.cos(psi)[None, :]
    rt = float(target_rho)
    zt = float(target_z)
    rs = source_rho[:, None]
    dz = source_z[:, None] - zt
    R = np.sqrt(
        np.maximum(
            rt * rt + rs * rs - 2.0 * rt * rs * cos_psi + dz * dz,
            0.0,
        )
    )
    phase = np.exp(1j * complex(k) * R)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = phase / (4.0 * np.pi * R)
        num = (rs - rt * cos_psi) * float(source_normal[0]) + dz * float(
            source_normal[1]
        )
        H = phase * (1j * complex(k) * R - 1.0) * num / (
            4.0 * np.pi * R ** 3
        )
    return G, H

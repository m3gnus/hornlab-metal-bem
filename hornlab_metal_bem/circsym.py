"""Axisymmetric body-of-revolution acoustic BEM solver.

This module implements the Phase 1 m=0 solver as a pure NumPy/SciPy sibling to
the 3D native Metal path. The meridian discretization uses DP0 constants on
straight generating-curve segments with midpoint collocation. Segment integrals
carry the full ring surface measure ``rho ds``; no ``1 / rho`` factors are
introduced, so axis nodes are regular for m=0.
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
import subprocess
import tempfile
import time
from types import SimpleNamespace
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.special import ellipe, ellipk

from ._constants import SPEED_OF_SOUND
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
)
from .observation import ObservationFrame, build_observation_points
from .result import MeshInfo, SolveResult
from .sweep import (
    _build_frequency_grid,
    _directivity_from_pressure,
    _impedance_sources_for_frequencies,
)

logger = logging.getLogger(__name__)


_AZIMUTH_POINTS_MIN = 64
_AZIMUTH_POINTS_PER_KRHO = 4.0
_LINE_QUAD_ORDER = 16
_SINGULAR_LINE_QUAD_ORDER = 24
_GRADED_POWER = 3.0
_FIELD_KERNEL_BLOCK_ELEMENTS = 8_000_000
_FIELD_KERNEL_MAX_TARGET_BLOCK = 64
_FIELD_KERNEL_PARALLEL_TARGET_BLOCK = 8
_FIELD_KERNEL_MAX_WORKERS = 8
_ASSEMBLY_KERNEL_BLOCK_ELEMENTS = 3_000_000
_ASSEMBLY_KERNEL_MAX_TARGET_BLOCK = 16
_ASSEMBLY_KERNEL_PARALLEL_TARGET_BLOCK = 5
_ASSEMBLY_GEOMETRY_CACHE_MAX_BYTES = 1_000_000_000


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

    for freq_index, frequency_hz in enumerate(frequencies_arr):
        frequency = float(frequency_hz)
        t_case = time.time()
        omega = 2.0 * np.pi * frequency
        k = _complex_wavenumber(frequency, config)
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
        )
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
        field_pressure = _evaluate_observation_pressure(
            meridian,
            pressure,
            q_total,
            obs_points,
            k_field,
            config,
            geom=geom,
            n_psi=n_psi,
        )
        sphere_pressure = (
            _evaluate_points_pressure(
                meridian,
                pressure,
                q_total,
                np.asarray(config.observation.sphere_points, dtype=np.float64),
                k_field,
                config.circsym_baffle_z,
                geom=geom,
                n_psi=n_psi,
            )
            if config.observation.sphere_points is not None
            else None
        )
        field_s = time.time() - t_field

        directivity = _directivity_from_pressure(field_pressure, on_axis_idx)
        pavg = _surface_pressure_average(meridian, pressure, source_tags)
        for tag in source_tags:
            surface_pavg[int(tag)].append(pavg[int(tag)])
        impedance_rows.append(pavg.get(int(impedance_source_tag), 0.0 + 0.0j))
        pressure_rows.append(field_pressure)
        directivity_rows.append(directivity)
        completed_freqs.append(frequency)
        if surface_pressure_rows is not None:
            surface_pressure_rows.append(np.asarray(pressure, dtype=np.complex128))

        diagnostics = {
            "assembly_implementation": "circsym_python_dp0_m0",
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
            "mesh_elements_per_wavelength": SPEED_OF_SOUND
            / (frequency * mesh_max_segment)
            if mesh_max_segment > 0.0
            else math.inf,
            "chief_points": bool(config.chief_points is not None),
            "chief_points_count": int(chief_rows_count),
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
    timings = {
        "solve_s": sum(float(entry["timing_s"]) for entry in solver_log),
        "assembly_s": sum(float(entry["assembly_s"]) for entry in solver_log),
        "dense_solve_s": sum(float(entry["dense_solve_s"]) for entry in solver_log),
        "directivity_s": sum(float(entry["field_s"]) for entry in solver_log),
        "total_s": time.time() - t_total,
    }

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
    impedance_sources_arg = _impedance_sources_for_frequencies(
        meridian.physical_tags, frequencies_arr, config
    )
    per_case_impedance = isinstance(impedance_sources_arg, list)

    for freq_index, frequency_hz in enumerate(frequencies_arr):
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
        )
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
        )
        field_pressure = flat_pressure.reshape(n_planes, n_angles)

        sphere_pressure = None
        if config.observation.sphere_points is not None:
            sphere_points = np.asarray(
                config.observation.sphere_points,
                dtype=np.float64,
            )
            sphere_pressure = _evaluate_coupled_ib_points_pressure(
                meridian,
                q_a,
                idx_a,
                sphere_points,
                k_field,
                geom=geom,
                n_psi=n_psi,
            )
        field_s = time.time() - t_field

        directivity = _directivity_from_pressure(field_pressure, on_axis_idx)
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

        diagnostics = {
            "circsym": True,
            "coupled_ib": True,
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
    k_real = 2.0 * np.pi * float(frequency_hz) / SPEED_OF_SOUND
    if config.formulation == BIEFormulation.COMPLEX_K:
        return complex(k_real, k_real * float(config.complex_k_shift))
    return complex(k_real, 0.0)


def _azimuth_order(k: complex, rho_max: float) -> int:
    krho = abs(complex(k)) * max(float(rho_max), 0.0)
    return max(_AZIMUTH_POINTS_MIN, int(math.ceil(_AZIMUTH_POINTS_PER_KRHO * krho)))


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
) -> NDArray[np.complex128] | None:
    profile_map = {
        int(profile_tag): profile
        for profile_tag, profile in (config.source_velocity_profiles or {}).items()
    }
    source_tags = sorted({int(tag) for tag in config.velocity_sources} | set(profile_map))
    if not source_tags:
        return None
    effective_profiles = {
        tag: profile_map.get(
            tag,
            AxialProfile()
            if config.source_motion == SourceMotion.AXIAL
            else NormalProfile(),
        )
        for tag in source_tags
    }
    if all(isinstance(profile, NormalProfile) for profile in effective_profiles.values()):
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
    saw_complex = False

    for tag in source_tags:
        idx = np.where(meridian.physical_tags == tag)[0]
        if idx.size == 0:
            continue
        any_source = True
        profile = effective_profiles[tag]
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
            saw_complex = saw_complex or bool(np.any(values.imag != 0.0))
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
            saw_complex = saw_complex or bool(np.any(values.imag != 0.0))
        else:  # pragma: no cover - SolveConfig validation rejects this.
            raise ValueError(
                "source_velocity_profiles values must be SourceProfile instances"
            )
        scale[idx] = values

    if not any_source:
        return None
    if saw_complex:
        return scale
    return scale.real.astype(np.float64, copy=False)


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


def _taper_values(t: NDArray[np.float64], profile: TaperProfile) -> NDArray[np.float64]:
    values = np.ones_like(t, dtype=np.float64)
    transition = t > profile.start
    if np.any(transition):
        x = np.clip((t[transition] - profile.start) / (1.0 - profile.start), 0.0, 1.0)
        if profile.kind == "linear":
            values[transition] = 1.0 - x
        else:
            values[transition] = 0.5 * (1.0 + np.cos(np.pi * x))
    values[t >= 1.0] = 0.0
    return values


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
    velocity_sources = (
        config.velocity_source_callback(float(frequency_hz))
        if config.velocity_source_callback is not None
        else config.velocity_sources
    )
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
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if geometry_cache is not None:
        cached = geometry_cache.assemble(
            k,
            n_psi=int(n_psi),
            meridian=meridian,
            baffle_z=baffle_z,
        )
        if cached is not None:
            return cached
    return _assemble_boundary_matrices_uncached(
        meridian,
        k,
        baffle_z,
        n_psi=n_psi,
    )


def _assemble_boundary_matrices_uncached(
    meridian: MeridianMesh,
    k: complex,
    baffle_z: float | None,
    *,
    n_psi: int,
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
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
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
                S[start:stop] = s_block
                H[start:stop] = h_block
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
class _NearRemainderGeometry:
    R: NDArray[np.float64]
    num: NDArray[np.float64]
    weight: NDArray[np.float64]


@dataclass
class _BoundaryAssemblyQuadratureGeometry:
    far_parts: tuple[_FarRemainderGeometry, ...]
    near_parts: tuple[_NearRemainderGeometry, ...]


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
        self._cache_single_use = bool(cache_single_use)
        self._exhausted_n_psi: set[int] = set()
        self._static_s: NDArray[np.complex128] | None = None
        self._static_h: NDArray[np.complex128] | None = None
        self._near_rows: NDArray[np.int64] | None = None
        self._near_cols: NDArray[np.int64] | None = None
        self._quadrature: dict[int, _BoundaryAssemblyQuadratureGeometry] = {}

    def assemble(
        self,
        k: complex,
        *,
        n_psi: int,
        meridian: MeridianMesh,
        baffle_z: float | None,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]] | None:
        if meridian is not self.meridian or not _same_optional_float(
            baffle_z, self.baffle_z
        ):
            return None
        n_psi_int = int(n_psi)
        if self._reusable_n_psi is not None and n_psi_int not in self._reusable_n_psi:
            return None
        if self._remaining_uses is not None and n_psi_int in self._exhausted_n_psi:
            return None
        if (
            n_psi_int not in self._quadrature
            and _estimate_boundary_geometry_cache_bytes(
                self.meridian,
                self.baffle_z,
                n_psi_int,
            )
            > _assembly_geometry_cache_max_bytes()
        ):
            return None
        c_kernel = _load_circsym_remainder_c_kernel()
        remaining_uses = (
            None if self._remaining_uses is None else self._remaining_uses.get(n_psi_int)
        )
        if (
            remaining_uses is not None
            and remaining_uses <= 1
            and n_psi_int not in self._quadrature
            and (c_kernel is None or not self._cache_single_use)
        ):
            return None

        try:
            static_s, static_h, near_rows, near_cols = self._static_geometry()
            qgeom = self._quadrature_geometry(n_psi_int, near_rows, near_cols)
            result = _assemble_boundary_matrices_from_geometry(
                static_s,
                static_h,
                near_rows,
                near_cols,
                qgeom,
                k,
                n_psi=n_psi_int,
                c_kernel=c_kernel,
            )
        except MemoryError:
            self._quadrature.pop(n_psi_int, None)
            return None
        self._release_quadrature_use(n_psi_int)
        return result

    def _static_geometry(
        self,
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
    ) -> _BoundaryAssemblyQuadratureGeometry:
        cached = self._quadrature.get(int(n_psi))
        if cached is None:
            cached = _BoundaryAssemblyQuadratureGeometry(
                far_parts=_build_far_remainder_geometry_parts(
                    self.meridian,
                    self.geom,
                    self.baffle_z,
                    n_psi=int(n_psi),
                ),
                near_parts=_build_near_remainder_geometry_parts(
                    self.meridian,
                    self.geom,
                    near_rows,
                    near_cols,
                    self.baffle_z,
                    n_psi=int(n_psi),
                ),
            )
            self._quadrature[int(n_psi)] = cached
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


def _assembly_geometry_cache_max_bytes() -> int:
    raw = os.environ.get("HORNLAB_CIRCSYM_ASSEMBLY_CACHE_MAX_BYTES")
    if raw is not None:
        try:
            return max(0, int(raw))
        except ValueError:
            return _ASSEMBLY_GEOMETRY_CACHE_MAX_BYTES
    return _ASSEMBLY_GEOMETRY_CACHE_MAX_BYTES


def _estimate_boundary_geometry_cache_bytes(
    meridian: MeridianMesh,
    baffle_z: float | None,
    n_psi: int,
) -> int:
    image_factor = 2 if baffle_z is not None else 1
    far_values = (
        int(meridian.segment_count)
        * int(meridian.segment_count)
        * int(_LINE_QUAD_ORDER)
        * int(n_psi)
    )
    # Far geometry stores R, G weights, and H weights as float64 arrays.
    return int(far_values * 3 * np.dtype(np.float64).itemsize * image_factor)


def _assemble_boundary_matrices_from_geometry(
    static_s: NDArray[np.complex128],
    static_h: NDArray[np.complex128],
    near_rows: NDArray[np.int64],
    near_cols: NDArray[np.int64],
    qgeom: _BoundaryAssemblyQuadratureGeometry,
    k: complex,
    *,
    n_psi: int,
    c_kernel: _CircsymRemainderCKernel | None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    S = static_s.copy()
    H = static_h.copy()
    n = S.shape[0]
    workers = _assembly_worker_count(n)
    if c_kernel is not None:
        for part in qgeom.far_parts:
            s_part, h_part = _evaluate_far_remainder_compiled(
                c_kernel,
                part,
                k,
                workers=workers,
            )
            S += s_part
            H += h_part
    else:
        _accumulate_far_remainder_numpy(S, H, qgeom, k, n_psi=n_psi, workers=workers)

    if near_rows.size:
        s_near = np.zeros(near_rows.size, dtype=np.complex128)
        h_near = np.zeros_like(s_near)
        for part in qgeom.near_parts:
            if c_kernel is None:
                s_part, h_part = _evaluate_near_remainder(part, k)
            else:
                s_part, h_part = _evaluate_near_remainder_compiled(
                    c_kernel,
                    part,
                    k,
                    workers=workers,
                )
            s_near += s_part
            h_near += h_part
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
) -> None:
    n = S.shape[0]
    block_size = _assembly_target_block_size(n, int(n_psi))
    if workers > 1:
        block_size = min(block_size, _ASSEMBLY_KERNEL_PARALLEL_TARGET_BLOCK)
    ranges = [(start, min(n, start + block_size)) for start in range(0, n, block_size)]

    def compute_block(
        span: tuple[int, int],
    ) -> tuple[int, int, NDArray[np.complex128], NDArray[np.complex128]]:
        start, stop = span
        s_dyn = np.zeros((stop - start, n), dtype=np.complex128)
        h_dyn = np.zeros_like(s_dyn)
        for part in qgeom.far_parts:
            s_part, h_part = _evaluate_far_remainder_block(part, k, start, stop)
            s_dyn += s_part
            h_dyn += h_part
        return start, stop, s_dyn, h_dyn

    if workers <= 1 or len(ranges) <= 1:
        for span in ranges:
            start, stop, s_dyn, h_dyn = compute_block(span)
            S[start:stop] += s_dyn
            H[start:stop] += h_dyn
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(ranges))) as executor:
            for start, stop, s_dyn, h_dyn in executor.map(compute_block, ranges):
                S[start:stop] += s_dyn
                H[start:stop] += h_dyn


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
    const double *R;
    const double *gw;
    const double *hw;
    double kr;
    double ki;
    double *out_s;
    double *out_h;
} FarTask;

typedef struct {
    const FarTask *task;
    int64_t start;
    int64_t stop;
} FarThreadTask;

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

static void eval_far_range(const FarTask *task, int64_t start, int64_t stop) {
    const int64_t ns = task->ns;
    const int64_t nl = task->nl;
    const int64_t np = task->np;
    const double *R = task->R;
    const double *gw = task->gw;
    const double *hw = task->hw;
    const double kr = task->kr;
    const double ki = task->ki;
    double *out_s = task->out_s;
    double *out_h = task->out_h;

    for (int64_t i = start; i < stop; ++i) {
        for (int64_t j = 0; j < ns; ++j) {
            double s_re = 0.0;
            double s_im = 0.0;
            double h_re = 0.0;
            double h_im = 0.0;
            int64_t idx = ((i * ns + j) * nl) * np;
            for (int64_t u = 0; u < nl; ++u) {
                for (int64_t p = 0; p < np; ++p, ++idx) {
                    const double r = R[idx];
                    const double g_weight = gw[idx];
                    const double h_weight = hw[idx];
                    if (g_weight == 0.0 && h_weight == 0.0) {
                        continue;
                    }
                    const double kr_r = kr * r;
                    const double ki_r = ki * r;
                    const double decay = exp(-ki_r);
                    const double phase_re = decay * cos(kr_r);
                    const double phase_im = decay * sin(kr_r);

                    s_re += (phase_re - 1.0) * g_weight;
                    s_im += phase_im * g_weight;

                    const double factor_re = -ki_r - 1.0;
                    const double factor_im = kr_r;
                    const double expr_re =
                        phase_re * factor_re - phase_im * factor_im + 1.0;
                    const double expr_im =
                        phase_re * factor_im + phase_im * factor_re;
                    h_re += expr_re * h_weight;
                    h_im += expr_im * h_weight;
                }
            }
            const int64_t out_idx = 2 * (i * ns + j);
            out_s[out_idx] = s_re;
            out_s[out_idx + 1] = s_im;
            out_h[out_idx] = h_re;
            out_h[out_idx + 1] = h_im;
        }
    }
}

static void *eval_far_worker(void *raw) {
    const FarThreadTask *thread_task = (const FarThreadTask *)raw;
    eval_far_range(thread_task->task, thread_task->start, thread_task->stop);
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

int circsym_eval_far_remainder(
    int64_t nt,
    int64_t ns,
    int64_t nl,
    int64_t np,
    const double *R,
    const double *gw,
    const double *hw,
    double kr,
    double ki,
    double *out_s,
    double *out_h,
    int32_t requested_threads
) {
    if (nt < 0 || ns < 0 || nl < 0 || np < 0 ||
        R == NULL || gw == NULL || hw == NULL ||
        out_s == NULL || out_h == NULL) {
        return -1;
    }
    FarTask task;
    task.nt = nt;
    task.ns = ns;
    task.nl = nl;
    task.np = np;
    task.R = R;
    task.gw = gw;
    task.hw = hw;
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
        eval_far_range(&task, 0, nt);
        return 0;
    }

    pthread_t *handles = (pthread_t *)malloc((size_t)threads * sizeof(pthread_t));
    FarThreadTask *thread_tasks =
        (FarThreadTask *)malloc((size_t)threads * sizeof(FarThreadTask));
    if (handles == NULL || thread_tasks == NULL) {
        free(handles);
        free(thread_tasks);
        eval_far_range(&task, 0, nt);
        return 0;
    }

    int32_t created = 0;
    for (int32_t t = 0; t < threads; ++t) {
        const int64_t start = (nt * (int64_t)t) / (int64_t)threads;
        const int64_t stop = (nt * (int64_t)(t + 1)) / (int64_t)threads;
        thread_tasks[t].task = &task;
        thread_tasks[t].start = start;
        thread_tasks[t].stop = stop;
        if (pthread_create(&handles[t], NULL, eval_far_worker, &thread_tasks[t]) != 0) {
            break;
        }
        created += 1;
    }

    for (int32_t t = 0; t < created; ++t) {
        pthread_join(handles[t], NULL);
    }
    if (created != threads) {
        eval_far_range(&task, 0, nt);
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
"""


class _CircsymRemainderCKernel:
    def __init__(self, library_path: str) -> None:
        self.library = ctypes.CDLL(library_path)
        self.eval_far = self.library.circsym_eval_far_remainder
        self.eval_near = self.library.circsym_eval_near_remainder
        double_ptr = ctypes.POINTER(ctypes.c_double)
        self.eval_far.argtypes = [
            ctypes.c_int64,
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
        self.eval_far.restype = ctypes.c_int
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


@lru_cache(maxsize=1)
def _load_circsym_remainder_c_kernel() -> _CircsymRemainderCKernel | None:
    if os.environ.get("HORNLAB_CIRCSYM_DISABLE_C_REMAINDER_KERNEL"):
        return None
    cc = os.environ.get("CC", "cc")
    source_hash = hashlib.sha256(_CIRCSYM_REMAINDER_C_SOURCE.encode("utf-8")).hexdigest()[:16]
    cache_dir = os.path.join(tempfile.gettempdir(), "hornlab-metal-bem-circsym")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        return None
    platform_name = os.uname().sysname.lower() if hasattr(os, "uname") else ""
    lib_ext = ".dylib" if platform_name == "darwin" else ".so"
    source_path = os.path.join(cache_dir, f"circsym_remainder_{source_hash}.c")
    library_path = os.path.join(cache_dir, f"circsym_remainder_{source_hash}{lib_ext}")
    if not os.path.exists(library_path):
        try:
            with open(source_path, "w", encoding="utf-8") as handle:
                handle.write(_CIRCSYM_REMAINDER_C_SOURCE)
            command = [cc, "-O3", "-fPIC"]
            if platform_name == "darwin":
                command.append("-dynamiclib")
            else:
                command.append("-shared")
            command.extend([source_path, "-o", library_path, "-lm", "-pthread"])
            subprocess.run(command, check=True, capture_output=True, text=True)
        except Exception as exc:
            logger.debug("CircSym C remainder kernel unavailable: %s", exc)
            return None
    try:
        return _CircsymRemainderCKernel(library_path)
    except Exception as exc:
        logger.debug("CircSym C remainder kernel load failed: %s", exc)
        return None


def _evaluate_far_remainder_compiled(
    kernel: _CircsymRemainderCKernel,
    part: _FarRemainderGeometry,
    k: complex,
    *,
    workers: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    R = part.R if part.R.flags.c_contiguous else np.ascontiguousarray(part.R)
    g_weight = (
        part.g_weight
        if part.g_weight.flags.c_contiguous
        else np.ascontiguousarray(part.g_weight)
    )
    h_weight = (
        part.h_weight
        if part.h_weight.flags.c_contiguous
        else np.ascontiguousarray(part.h_weight)
    )
    target_count, source_count, line_count, psi_count = R.shape
    s_out = np.empty((target_count, source_count), dtype=np.complex128)
    h_out = np.empty_like(s_out)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    status = kernel.eval_far(
        ctypes.c_int64(target_count),
        ctypes.c_int64(source_count),
        ctypes.c_int64(line_count),
        ctypes.c_int64(psi_count),
        R.ctypes.data_as(double_ptr),
        g_weight.ctypes.data_as(double_ptr),
        h_weight.ctypes.data_as(double_ptr),
        ctypes.c_double(float(complex(k).real)),
        ctypes.c_double(float(complex(k).imag)),
        s_out.ctypes.data_as(double_ptr),
        h_out.ctypes.data_as(double_ptr),
        ctypes.c_int32(max(1, int(workers))),
    )
    if int(status) != 0:
        return _evaluate_far_remainder_block(part, k, 0, target_count)
    return s_out, h_out


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


def _build_boundary_static_geometry(
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    baffle_z: float | None,
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
    for row, col in zip(near_rows, near_cols):
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
    n_target = int(target_rho.shape[0])
    n_source, n_line = rho_s.shape
    n_psi = int(psi.shape[0])
    R_out = np.empty((n_target, n_source, n_line, n_psi), dtype=np.float64)
    g_weight = np.empty_like(R_out)
    h_weight = np.empty_like(R_out)
    cos_psi = np.cos(psi)[None, None, None, :]
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
) -> tuple[_NearRemainderGeometry, ...]:
    parts = [
        _build_near_remainder_geometry(
            meridian,
            geom,
            near_rows,
            near_cols,
            baffle_z=None,
            image=False,
            n_psi=int(n_psi),
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
            )
        )
    return tuple(parts)


def _build_near_remainder_geometry(
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    near_rows: NDArray[np.int64],
    near_cols: NDArray[np.int64],
    *,
    baffle_z: float | None,
    image: bool,
    n_psi: int,
) -> _NearRemainderGeometry:
    psi, psi_weights = _leggauss_psi(int(n_psi))
    cos_psi = np.cos(psi)[None, :]
    counts = []
    for row, col in zip(near_rows, near_cols):
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
) -> NDArray[np.complex128]:
    """Assemble the real-k Rayleigh single-layer aperture block only.

    Coupled infinite-baffle assembly consumes this block for the pressure
    trace.  Match the ordinary/near split used by full boundary assembly so
    this reduced work remains numerically identical to slicing a full matrix.
    """
    indices = np.asarray(aperture_indices, dtype=np.int64).reshape(-1)
    targets = geom.midpoints[indices]
    s_block, _ = _integrate_ordinary_segment_kernels_targets_batched(
        target_rho=targets[:, 0],
        target_z=targets[:, 1],
        meridian=meridian,
        geom=geom,
        source_indices=indices,
        k=k,
        baffle_z=None,
        n_psi=n_psi,
    )
    far_mask = _ordinary_far_source_mask_targets(
        targets[:, 0],
        targets[:, 1],
        geom,
        source_indices=indices,
    )
    near_rows, near_cols = np.nonzero(~far_mask)
    for row_local, source_local in zip(near_rows, near_cols):
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


def _assemble_boundary_row(
    row_index: int,
    *,
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    k: complex,
    baffle_z: float | None,
    n_psi: int,
) -> tuple[int, NDArray[np.complex128], NDArray[np.complex128]]:
    n = meridian.segment_count
    target = geom.midpoints[int(row_index)]
    s_row = np.empty(n, dtype=np.complex128)
    h_row = np.empty(n, dtype=np.complex128)
    far_mask = _ordinary_far_source_mask(target, geom, target_index=int(row_index))
    far_indices = np.nonzero(far_mask)[0]
    if far_indices.size:
        s_row[far_indices], h_row[far_indices] = _integrate_ordinary_segment_kernels_batched(
            target_rho=float(target[0]),
            target_z=float(target[1]),
            meridian=meridian,
            geom=geom,
            source_indices=far_indices,
            k=k,
            baffle_z=baffle_z,
            n_psi=n_psi,
        )
    near_indices = np.nonzero(~far_mask)[0]
    for j in near_indices:
        s_row[j], h_row[j] = _integrate_segment_kernel(
            target_rho=float(target[0]),
            target_z=float(target[1]),
            meridian=meridian,
            geom=geom,
            source_index=int(j),
            k=k,
            baffle_z=baffle_z,
            n_psi=n_psi,
            target_index=int(row_index),
        )
    return int(row_index), s_row, h_row


def _ordinary_far_source_mask(
    target: NDArray[np.float64],
    geom: SimpleNamespace,
    *,
    target_index: int | None,
) -> NDArray[np.bool_]:
    target_arr = np.asarray(target, dtype=np.float64)
    denom = np.maximum(geom.lengths * geom.lengths, 1.0e-30)
    u_star = np.sum((target_arr[None, :] - geom.p0) * geom.delta, axis=1) / denom
    u_clamped = np.clip(u_star, 0.0, 1.0)
    closest = geom.p0 + u_clamped[:, None] * geom.delta
    distance = np.linalg.norm(target_arr[None, :] - closest, axis=1)
    far = distance > 1.25 * geom.lengths
    if target_index is not None:
        far[int(target_index)] = False
    return far


def _integrate_ordinary_segment_kernels_batched(
    *,
    target_rho: float,
    target_z: float,
    meridian: MeridianMesh,
    geom: SimpleNamespace,
    source_indices: NDArray[np.int64] | NDArray[np.int32],
    k: complex,
    baffle_z: float | None,
    n_psi: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    indices = np.asarray(source_indices, dtype=np.int64)
    if indices.size == 0:
        return (
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.complex128),
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
    normal_rho = normal[:, 0:1]
    normal_z = normal[:, 1:2]

    g_static, h_static = _ring_static_kernel_m0_batched(
        target_rho,
        target_z,
        rho_s,
        z_s,
        normal_rho,
        normal_z,
    )
    g_rem, h_rem = _ring_remainder_kernel_m0_batched(
        target_rho,
        target_z,
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
        g_static_i, h_static_i = _ring_static_kernel_m0_batched(
            target_rho,
            target_z,
            rho_s,
            z_img,
            normal_rho,
            -normal_z,
        )
        g_rem_i, h_rem_i = _ring_remainder_kernel_m0_batched(
            target_rho,
            target_z,
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
        np.asarray(np.sum(g * measure, axis=1), dtype=np.complex128),
        np.asarray(np.sum(h * measure, axis=1), dtype=np.complex128),
    )


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
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    pts = np.asarray(chief_points, dtype=np.float64)
    geom = meridian.segment_geometry()
    S = np.empty((pts.shape[0], meridian.segment_count), dtype=np.complex128)
    H = np.empty_like(S)
    for i, point in enumerate(pts):
        target_rho = float(math.hypot(float(point[0]), float(point[1])))
        target_z = float(point[2])
        for j in range(meridian.segment_count):
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
) -> NDArray[np.complex128]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if pts.shape[0] == 0:
        return np.empty(0, dtype=np.complex128)
    if geom is None:
        geom = meridian.segment_geometry()
    target_rho, target_z = _points_target_rho_z(pts)
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
    )
    rayleigh_sheet = _is_flat_baffled_sheet(meridian, baffle_z, geom=geom)
    if rayleigh_sheet:
        # A coplanar baffled disk is an open Rayleigh radiator. The direct
        # closed-surface representation's double-layer pressure term is not
        # part of the textbook piston field, so use the half-space single-layer
        # integral for this narrow geometry.
        return np.asarray(-(s_mat @ q_total), dtype=np.complex128)
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
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
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

    s_mat = np.empty((target_rho_arr.size, indices.size), dtype=np.complex128)
    h_mat = np.empty_like(s_mat)
    workers = _field_kernel_worker_count(target_rho_arr.size, indices.size)
    block_size = _field_kernel_target_block_size(indices.size, int(n_psi))
    if workers > 1:
        block_size = min(block_size, _FIELD_KERNEL_PARALLEL_TARGET_BLOCK)
    ranges = [
        (start, min(target_rho_arr.size, start + block_size))
        for start in range(0, target_rho_arr.size, block_size)
    ]

    def compute_block(
        span: tuple[int, int],
    ) -> tuple[int, int, NDArray[np.complex128], NDArray[np.complex128]]:
        start, stop = span
        block_rho = target_rho_arr[start:stop]
        block_z = target_z_arr[start:stop]
        s_block, h_block = _integrate_ordinary_field_kernels_targets_batched(
            target_rho=block_rho,
            target_z=block_z,
            meridian=meridian,
            geom=geom,
            source_indices=indices,
            k=k,
            baffle_z=baffle_z,
            n_psi=n_psi,
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
            start, stop, s_block, h_block = compute_block(span)
            s_mat[start:stop] = s_block
            h_mat[start:stop] = h_block
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(ranges))) as executor:
            for start, stop, s_block, h_block in executor.map(compute_block, ranges):
                s_mat[start:stop] = s_block
                h_mat[start:stop] = h_block
    return s_mat, h_mat


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
    normal_close = np.all(np.abs(normals[:, 0]) <= 1e-10) and np.all(
        np.abs(np.abs(normals[:, 1]) - 1.0) <= 1e-10
    )
    return bool(z_close and normal_close)


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


def _ring_static_kernel_m0_batched(
    target_rho: float,
    target_z: float,
    source_rho: NDArray[np.float64],
    source_z: NDArray[np.float64],
    normal_rho: NDArray[np.float64],
    normal_z: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    rt = float(target_rho)
    zt = float(target_z)
    rs = np.asarray(source_rho, dtype=np.float64)
    zs = np.asarray(source_z, dtype=np.float64)
    n_rho = np.asarray(normal_rho, dtype=np.float64)
    n_z = np.asarray(normal_z, dtype=np.float64)
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


def _ring_remainder_kernel_m0_batched(
    target_rho: float,
    target_z: float,
    source_rho: NDArray[np.float64],
    source_z: NDArray[np.float64],
    normal_rho: NDArray[np.float64],
    normal_z: NDArray[np.float64],
    k: complex,
    *,
    n_psi: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    rs = np.asarray(source_rho, dtype=np.float64)
    zs = np.asarray(source_z, dtype=np.float64)
    psi, weights = _leggauss_psi(int(n_psi))
    cos_psi = np.cos(psi)[None, None, :]
    rt = float(target_rho)
    zt = float(target_z)
    n_rho = np.asarray(normal_rho, dtype=np.float64)[:, :, None]
    n_z = np.asarray(normal_z, dtype=np.float64)[:, :, None]

    rs3 = rs[:, :, None]
    dz = zs[:, :, None] - zt
    R2 = rt * rt + rs3 * rs3 - 2.0 * rt * rs3 * cos_psi + dz * dz
    R = np.sqrt(np.maximum(R2, 0.0))
    q = complex(k) * R
    with np.errstate(divide="ignore", invalid="ignore"):
        rem_g = np.expm1(1j * q) / (4.0 * np.pi * R)
    rem_g = np.where(R > 1e-13, rem_g, 1j * complex(k) / (4.0 * np.pi))

    num = (rs3 - rt * cos_psi) * n_rho + dz * n_z
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

    G = 2.0 * np.sum(rem_g * weights[None, None, :], axis=2)
    H = 2.0 * np.sum(rem_h * weights[None, None, :], axis=2)
    return np.asarray(G, dtype=np.complex128), np.asarray(H, dtype=np.complex128)


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

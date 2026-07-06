"""Axisymmetric body-of-revolution acoustic BEM solver.

This module implements the Phase 1 m=0 solver as a pure NumPy/SciPy sibling to
the 3D native Metal path. The meridian discretization uses DP0 constants on
straight generating-curve segments with midpoint collocation. Segment integrals
carry the full ring surface measure ``rho ds``; no ``1 / rho`` factors are
introduced, so axis nodes are regular for m=0.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import math
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
        n_psi = _azimuth_order(k, rho_max)
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
            meridian, k, config.circsym_baffle_z, n_psi=n_psi
        )
        A = H.copy()
        A[np.diag_indices_from(A)] -= 0.5
        if np.any(beta != 0.0):
            A -= S * (1j * k * beta)[None, :]
        rhs = S @ q_driver
        assembly_s = time.time() - t_assembly

        t_solve = time.time()
        solve_matrix = A
        solve_rhs = rhs
        chief_residual_rel = None
        chief_rows_count = 0
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
            lu, piv = linalg.lu_factor(solve_matrix)
            pressure = linalg.lu_solve((lu, piv), solve_rhs)
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
        q_total = q_driver + 1j * k_field * beta * pressure

        t_field = time.time()
        field_pressure = _evaluate_observation_pressure(
            meridian,
            pressure,
            q_total,
            obs_points,
            k_field,
            config,
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
            "dense_solve_rcond": float(1.0 / np.linalg.cond(A)),
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
            v_n = v_n / (1j * omega) if omega > 0.0 else np.zeros_like(v_n)
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
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    geom = meridian.segment_geometry()
    n = meridian.segment_count
    S = np.empty((n, n), dtype=np.complex128)
    H = np.empty((n, n), dtype=np.complex128)
    for i in range(n):
        target = geom.midpoints[i]
        for j in range(n):
            S[i, j], H[i, j] = _integrate_segment_kernel(
                target_rho=float(target[0]),
                target_z=float(target[1]),
                meridian=meridian,
                geom=geom,
                source_index=j,
                k=k,
                baffle_z=baffle_z,
                n_psi=n_psi,
                target_index=i,
            )
    return S, H


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
    n_psi: int,
) -> NDArray[np.complex128]:
    pts = np.asarray(points, dtype=np.float64)
    out = np.empty(pts.shape[0], dtype=np.complex128)
    geom = meridian.segment_geometry()
    rayleigh_sheet = _is_flat_baffled_sheet(meridian, baffle_z)
    for i, point in enumerate(pts):
        target_rho = float(math.hypot(float(point[0]), float(point[1])))
        target_z = float(point[2])
        s_row = np.empty(meridian.segment_count, dtype=np.complex128)
        h_row = np.empty(meridian.segment_count, dtype=np.complex128)
        for j in range(meridian.segment_count):
            s_row[j], h_row[j] = _integrate_segment_kernel(
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
        if rayleigh_sheet:
            # A coplanar baffled disk is an open Rayleigh radiator. The direct
            # closed-surface representation's double-layer pressure term is not
            # part of the textbook piston field, so use the half-space
            # single-layer integral for this narrow geometry.
            out[i] = -(s_row @ q_total)
        else:
            out[i] = h_row @ pressure - s_row @ q_total
    return out


def _is_flat_baffled_sheet(meridian: MeridianMesh, baffle_z: float | None) -> bool:
    if baffle_z is None:
        return False
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


def _ellipk_derivative(
    m: NDArray[np.float64],
    K: NDArray[np.float64],
    E: NDArray[np.float64],
) -> NDArray[np.float64]:
    m_arr = np.asarray(m, dtype=np.float64)
    out = np.empty_like(m_arr)
    small = m_arr < 1e-8
    out[small] = (np.pi / 8.0) * (1.0 + 0.75 * m_arr[small])
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

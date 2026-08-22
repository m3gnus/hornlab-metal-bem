from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import SolveConfig


@dataclass
class MeshInfo:
    n_vertices: int
    n_triangles: int
    physical_groups: dict[int, str]
    bounding_box_m: tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass
class SolveResult:
    r"""Native Metal BEM solve output.

    Array dimensions use ``F`` for frequency count, ``P`` for observation
    plane count, and ``N`` for points or angles per plane. Complex values use
    the solver's :math:`e^{-i\omega t}` phase convention.

    ``surface_pressure_complex`` is the optional P1 pressure trace with shape
    ``(F, n_p1_dofs)``. ``surface_neumann_complex`` is the optional *total*
    DP0 outward-normal derivative ``dp/dn`` with shape ``(F, n_dp0_dofs)``;
    its Robin faces include ``q_driver + i*k*beta*avg(p)``. Both traces use the
    same :math:`e^{-i\omega t}` convention as ``pressure_complex``.
    """

    frequencies_hz: NDArray[np.float64]

    # (F, P, N_angles) — complex pressure at every observation point
    pressure_complex: NDArray[np.complex128]

    # (F, P, N_angles) — normalized directivity in dB, on-axis = 0 dB
    directivity_db: NDArray[np.float64]

    # (F,) — area-weighted average complex surface pressure on the impedance
    # source tag (pascals per unit drive). Not divided by drive velocity and
    # not normalised to rho*c.
    impedance: NDArray[np.complex128]

    observation_angles_deg: NDArray[np.float64]
    observation_points: NDArray[np.float64]
    observation_planes: list[str]

    config: SolveConfig
    mesh_info: MeshInfo
    timings: dict[str, float] = field(default_factory=dict)
    solver_log: list[dict] = field(default_factory=list)

    # Area-weighted average surface pressure per velocity-source tag.
    # tag -> (F,) complex array. Populated when velocity_sources has tags.
    surface_pressure_avg: dict[int, NDArray[np.complex128]] | None = None

    # Optional solved P1 surface-pressure trace, shape (F, n_p1_dofs), in the
    # e^{-i omega t} phase convention. Populated when either
    # SolveConfig.return_surface_pressure or return_surface_traces is true.
    surface_pressure_complex: NDArray[np.complex128] | None = None

    # Optional total outward-normal pressure derivative q = dp/dn on DP0,
    # shape (F, n_dp0_dofs), in the e^{-i omega t} phase convention. This is
    # the complete exterior-field trace: on Robin faces it includes
    # q_driver + i*k*beta*avg(p). Populated with surface_pressure_complex when
    # SolveConfig.return_surface_traces is true.
    surface_neumann_complex: NDArray[np.complex128] | None = None

    # Native helper per-frequency diagnostics and resident batch metadata.
    native_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    # Balloon/sphere observation block, populated when ObservationConfig
    # requested sphere sampling. sphere_pressure_complex is (F, M) complex
    # pressure at the M sphere points evaluated from the same solved system
    # as the polar arcs; sphere_points is (M, 3) absolute coordinates.
    # theta/phi are (M,) degrees relative to the observation frame and are
    # set only for frame-relative ``sphere_grid`` requests — explicit
    # ``sphere_points`` callers own their angular metadata.
    sphere_pressure_complex: NDArray[np.complex128] | None = None
    sphere_points: NDArray[np.float64] | None = None
    sphere_theta_deg: NDArray[np.float64] | None = None
    sphere_phi_deg: NDArray[np.float64] | None = None

    # Time-averaged acoustic power delivered through prescribed-velocity
    # boundary faces, shape (F,), in watts. Computed internally even when the
    # optional surface traces are not retained on the result.
    radiated_power_surface_w: NDArray[np.float64] | None = None

    # Far-field power integrated over a frame-relative sphere_grid, shape
    # (F,), in watts. Explicit sphere_points have no quadrature contract and
    # therefore leave these fields unset. Coverage is the integrated solid
    # angle in steradians (4*pi for a full sphere, 2*pi for a hemisphere).
    radiated_power_sphere_w: NDArray[np.float64] | None = None
    radiated_power_sphere_coverage_sr: float | None = None

    @property
    def spl_norm_db(self) -> NDArray[np.float64]:
        """Alias for normalized directivity in dB."""
        return self.directivity_db

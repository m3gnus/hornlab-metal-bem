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
    """Native Metal BEM solve output.

    Array dimensions use ``F`` for frequency count, ``P`` for observation
    plane count, and ``N`` for points or angles per plane.
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

    # Optional solved P1 surface pressure, shape (F, n_p1_dofs), populated
    # when SolveConfig.return_surface_pressure is true.
    surface_pressure_complex: NDArray[np.complex128] | None = None

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

    @property
    def spl_norm_db(self) -> NDArray[np.float64]:
        """Alias for normalized directivity in dB."""
        return self.directivity_db

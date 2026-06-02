from __future__ import annotations

from dataclasses import dataclass, field

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

    frequencies_hz: NDArray[np.float64]

    # (F, P, N_angles) — complex pressure at every observation point
    pressure_complex: NDArray[np.complex128]

    # (F, P, N_angles) — SPL in dB, normalised on-axis = 0 dB
    spl_db: NDArray[np.float64]

    # (F,) — complex throat impedance, normalised to rho*c
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

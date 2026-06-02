from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class BIEFormulation(Enum):
    STANDARD = "standard"
    BURTON_MILLER = "burton_miller"
    COMPLEX_K = "complex_k"


class LinearSolver(Enum):
    AUTO = "auto"
    LU = "lu"
    GMRES = "gmres"


class VelocityMode(Enum):
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"


AssemblyBackend = Literal["opencl", "numba", "auto", "metal"]
MetalBackendFallback = Literal["opencl", "error"]
NativeSymmetryPlane = Literal["yz", "xz", "yz+xz"]
MetalNativeAssemblyMode = Literal["corrected", "optimized"]


@dataclass
class ObservationConfig:
    planes: list[str] = field(default_factory=lambda: ["horizontal", "vertical"])
    distance_m: float = 2.0
    angle_min_deg: float = 0.0
    angle_max_deg: float = 180.0
    angle_count: int = 37
    origin: Literal["mouth", "throat"] = "mouth"

    # Custom observation points: plane_name -> (N, 3) array.
    # When set, build_observation_points() returns these directly
    # instead of constructing polar arcs from the frame.
    custom_points: dict[str, NDArray[np.float64]] | None = None


@dataclass
class SolveConfig:

    # Frequency sweep
    freq_min_hz: float = 500.0
    freq_max_hz: float = 20_000.0
    freq_count: int = 40
    freq_spacing: Literal["log", "linear"] = "log"

    # BIE
    formulation: BIEFormulation = BIEFormulation.STANDARD
    complex_k_shift: float = 0.005
    slp_dlp_quadrature: int = 4
    hyp_adlp_quadrature: int = 4

    # Linear solver
    solver: LinearSolver = LinearSolver.GMRES
    lu_threshold: int = 6000
    gmres_tol: float = 1e-5
    gmres_max_iter: int = 5000

    # Boundary condition
    velocity_mode: VelocityMode = VelocityMode.ACCELERATION
    velocity_sources: dict[int, float] = field(
        default_factory=lambda: {2: 1.0}
    )
    velocity_profile: Literal["piston", "dome", "ring"] = "piston"

    # Robin / impedance boundary condition (wall damping).
    # Maps physical tag -> normalized surface admittance β = ρc / Z_s
    # (dimensionless). β = 0 is rigid wall, β = 1 is air-matched absorber.
    # Light damping (the "clean up unrealistic dips" trick): β ~ 0.02-0.1.
    # When non-empty, the solver substitutes ∂p/∂n = i·k·β·p directly into
    # the BIE and solves once (no iteration); see _assemble_and_solve_impedance.
    impedance_sources: dict[int, complex] = field(default_factory=dict)

    # Observation
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    # Frame override: skip infer_frame() when set.
    # Use this for enclosed geometries where the heuristic may get the
    # axis wrong, or when the caller has a known frame (e.g. WG bridge).
    frame_override: object | None = None  # ObservationFrame, kept as object to avoid circular import

    # Performance
    workers: int = 1
    precision: Literal["single", "double"] = "single"
    assembly_backend: AssemblyBackend = "opencl"
    opencl_device: Literal["cpu", "gpu"] = "cpu"
    experimental_metal_backend: bool = False
    metal_backend_fallback: MetalBackendFallback = "opencl"
    native_symmetry_plane: NativeSymmetryPlane | None = None
    metal_native_assembly_mode: MetalNativeAssemblyMode = "corrected"
    metal_native_threads_per_group: int | None = None
    metal_native_matrix_threads_per_group: int | None = None
    metal_native_rhs_threads_per_group: int | None = None
    metal_native_duffy_threads_per_group: int | None = None
    metal_native_field_threads_per_group: int | None = None

    # Mesh scale (applied on load if mesh isn't already in metres)
    mesh_scale: float = 1.0

    # Air density (kg/m^3). Default 1.2041 matches standard air at 20 C.
    air_density: float = 1.2041

    # Progress callback: called after each frequency solve.
    # Signature: (freq_index: int, total_freqs: int, frequency_hz: float) -> None
    progress_callback: Callable[[int, int, float], None] | None = None

    # Per-frequency result callback for early stopping.
    # Signature: (freq_index: int, frequency_hz: float, log_entry: dict) -> bool
    # Return False to abort the sweep (partial SolveResult is built).
    on_frequency_result: Callable[[int, float, dict], bool] | None = None

    def __post_init__(self) -> None:
        if self.assembly_backend not in {"opencl", "numba", "auto", "metal"}:
            raise ValueError(
                "assembly_backend must be one of: auto, opencl, numba, metal"
            )
        if self.metal_backend_fallback not in {"opencl", "error"}:
            raise ValueError("metal_backend_fallback must be 'opencl' or 'error'")
        if self.native_symmetry_plane not in {None, "yz", "xz", "yz+xz"}:
            raise ValueError(
                "native_symmetry_plane must be None, 'yz', 'xz', or 'yz+xz'"
            )
        if self.metal_native_assembly_mode not in {"corrected", "optimized"}:
            raise ValueError(
                "metal_native_assembly_mode must be 'corrected' or 'optimized'"
            )
        if (
            self.metal_native_threads_per_group is not None
            and self.metal_native_threads_per_group <= 0
        ):
            raise ValueError("metal_native_threads_per_group must be positive")
        for name in (
            "metal_native_matrix_threads_per_group",
            "metal_native_rhs_threads_per_group",
            "metal_native_duffy_threads_per_group",
            "metal_native_field_threads_per_group",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

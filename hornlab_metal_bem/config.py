from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class VelocityMode:
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"


class BIEFormulation:
    STANDARD = "standard"
    COMPLEX_K = "complex_k"


NativeSymmetryPlane = Literal["yz", "xz", "xy", "yz+xz"]
MetalNativeAssemblyMode = Literal["corrected", "optimized", "reference", "parity"]

# Single source of truth for the supported native symmetry planes. Used by
# config validation, native routing, and geometry validation so the lists
# cannot drift apart.
NATIVE_SYMMETRY_PLANES: tuple[str, ...] = ("yz", "xz", "xy", "yz+xz")


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

    def __post_init__(self) -> None:
        if not self.planes:
            raise ValueError("observation planes must not be empty")
        if self.distance_m <= 0:
            raise ValueError("distance_m must be positive")
        if self.angle_count < 1:
            raise ValueError("angle_count must be at least 1")
        if self.origin not in {"mouth", "throat"}:
            raise ValueError("origin must be 'mouth' or 'throat'")


@dataclass
class SolveConfig:

    # Frequency sweep
    freq_min_hz: float = 500.0
    freq_max_hz: float = 20_000.0
    freq_count: int = 40
    freq_spacing: Literal["log", "linear"] = "log"

    # Boundary condition
    formulation: Literal["standard", "complex_k"] = BIEFormulation.STANDARD
    complex_k_shift: float = 0.005
    velocity_mode: Literal["velocity", "acceleration"] = VelocityMode.ACCELERATION
    velocity_sources: dict[int, float] = field(
        default_factory=lambda: {2: 1.0}
    )
    velocity_source_callback: Callable[[float], dict[int, complex]] | None = None
    # Experimental Robin boundary condition. Maps physical tag to normalized
    # surface admittance beta = rho*c/Zs; beta=0 is rigid, beta=1 air-matched.
    impedance_sources: dict[int, complex] = field(default_factory=dict)

    # Observation
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    # Frame override: skip infer_frame() when set.
    # Use this for enclosed geometries where the heuristic may get the
    # axis wrong, or when the caller has a known frame (e.g. WG bridge).
    frame_override: object | None = None  # ObservationFrame, kept as object to avoid circular import

    # Native Metal controls
    native_symmetry_plane: NativeSymmetryPlane | None = None
    metal_native_assembly_mode: MetalNativeAssemblyMode = "corrected"
    return_surface_pressure: bool = False
    metal_native_threads_per_group: int | None = None
    metal_native_matrix_threads_per_group: int | None = None
    metal_native_rhs_threads_per_group: int | None = None
    metal_native_duffy_threads_per_group: int | None = None
    metal_native_field_threads_per_group: int | None = None

    # Diagnostic policy. These mark results suspect; they do not change solver
    # settings and are not an interior-resonance cure.
    dense_solve_rcond_warning_threshold: float = 1e-6
    mesh_elements_per_wavelength_min: float = 6.0

    # Mesh scale (applied on load if mesh isn't already in metres)
    mesh_scale: float = 1.0

    # Mesh loading options forwarded to load_mesh() when solve() is given a
    # path. Ignored for pre-loaded LoadedMesh inputs.
    mesh_validate: bool = True
    mesh_merge_tol: float = 1e-9
    mesh_repair_normals: bool = False

    # Air density (kg/m^3). Default 1.2041 matches standard air at 20 C.
    air_density: float = 1.2041

    # Progress callback: called after each frequency solve.
    # Signature: (freq_index: int, total_freqs: int, frequency_hz: float) -> None
    progress_callback: Callable[[int, int, float], None] | None = None

    # Per-frequency result callback for early stopping.
    # Signature: (freq_index: int, frequency_hz: float, log_entry: dict) -> bool
    # Return exactly False to abort the sweep (partial SolveResult is built);
    # any other return value, including None, continues.
    on_frequency_result: Callable[[int, float, dict], bool] | None = None

    def __post_init__(self) -> None:
        if self.freq_spacing not in {"log", "linear"}:
            raise ValueError("freq_spacing must be 'log' or 'linear'")
        if self.freq_count < 1:
            raise ValueError("freq_count must be at least 1")
        if self.freq_min_hz <= 0:
            raise ValueError("freq_min_hz must be positive")
        if self.freq_max_hz < self.freq_min_hz:
            raise ValueError("freq_max_hz must be >= freq_min_hz")
        if self.mesh_scale <= 0:
            raise ValueError("mesh_scale must be positive")
        if self.air_density <= 0:
            raise ValueError("air_density must be positive")
        if self.dense_solve_rcond_warning_threshold < 0:
            raise ValueError("dense_solve_rcond_warning_threshold must be non-negative")
        if self.mesh_elements_per_wavelength_min <= 0:
            raise ValueError("mesh_elements_per_wavelength_min must be positive")
        if self.formulation not in {BIEFormulation.STANDARD, BIEFormulation.COMPLEX_K}:
            raise ValueError("formulation must be 'standard' or 'complex_k'")
        if self.complex_k_shift < 0:
            raise ValueError("complex_k_shift must be non-negative")
        if self.velocity_mode not in {VelocityMode.VELOCITY, VelocityMode.ACCELERATION}:
            raise ValueError("velocity_mode must be 'velocity' or 'acceleration'")
        for tag, beta in self.impedance_sources.items():
            if int(tag) < 0:
                raise ValueError("impedance_sources tags must be non-negative integers")
            beta_value = complex(beta)
            if not math.isfinite(beta_value.real) or not math.isfinite(beta_value.imag):
                raise ValueError("impedance_sources values must be finite complex numbers")
        if (
            self.native_symmetry_plane is not None
            and self.native_symmetry_plane not in NATIVE_SYMMETRY_PLANES
        ):
            raise ValueError(
                "native_symmetry_plane must be None, 'yz', 'xz', 'xy', or 'yz+xz'"
            )
        if self.metal_native_assembly_mode not in {
            "corrected",
            "optimized",
            "reference",
            "parity",
        }:
            raise ValueError(
                "metal_native_assembly_mode must be 'corrected', 'optimized', "
                "'reference', or 'parity'"
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

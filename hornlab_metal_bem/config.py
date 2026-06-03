from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class VelocityMode:
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"


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

    # Boundary condition
    velocity_mode: Literal["velocity", "acceleration"] = VelocityMode.ACCELERATION
    velocity_sources: dict[int, float] = field(
        default_factory=lambda: {2: 1.0}
    )
    velocity_source_callback: Callable[[float], dict[int, complex]] | None = None

    # Observation
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    # Frame override: skip infer_frame() when set.
    # Use this for enclosed geometries where the heuristic may get the
    # axis wrong, or when the caller has a known frame (e.g. WG bridge).
    frame_override: object | None = None  # ObservationFrame, kept as object to avoid circular import

    # Native Metal controls
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
        if self.freq_spacing not in {"log", "linear"}:
            raise ValueError("freq_spacing must be 'log' or 'linear'")
        if self.velocity_mode not in {VelocityMode.VELOCITY, VelocityMode.ACCELERATION}:
            raise ValueError("velocity_mode must be 'velocity' or 'acceleration'")
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

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import TYPE_CHECKING, Callable, Literal

from ._constants import AIR_DENSITY, SPEED_OF_SOUND

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class VelocityMode:
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"


class SourceMotion:
    """How a driven source tag's prescribed velocity maps onto its faces.

    NORMAL: each source face vibrates along its own outward normal at the same
    speed -- a uniformly "breathing"/pulsating cap. On a curved (dome) cap this
    radiates like a pulsating spherical cap.

    AXIAL: the source moves as a rigid piston along its axis, so the normal
    velocity on each face is ``U * (n_hat . axis)`` -- full at the pole, tapering
    to zero toward the rim. This is the realistic wavefront for a rigid dome /
    diaphragm / cone piston. For a flat disc every normal is the axis, so AXIAL
    reduces exactly to NORMAL.
    """

    NORMAL = "normal"
    AXIAL = "axial"


@dataclass(frozen=True)
class NormalProfile:
    """Uniform normal source velocity on every face of a source tag."""


@dataclass(frozen=True)
class AxialProfile:
    """Rigid piston source velocity: ``v_n = U * (n_hat . axis)``."""


@dataclass(frozen=True)
class TaperProfile:
    """Axial piston velocity with a radial edge taper.

    ``start`` is the normalized radius below which the multiplier is 1.0.
    From ``start`` to the tag's outer radius, the selected taper falls to 0.0.
    """

    kind: Literal["raised_cosine", "linear"] = "raised_cosine"
    start: float = 0.7


@dataclass(frozen=True)
class AnnularProfile:
    """Axial piston velocity limited to a normalized radial ring."""

    r_inner: float
    r_outer: float


@dataclass(frozen=True)
class PerFaceProfile:
    """Explicit per-face source multiplier in physical-tag face order."""

    weights: object


@dataclass(frozen=True)
class CallableProfile:
    """Callable source multiplier hook for measured/modal maps."""

    callback: Callable[[object, object, object, object], object]


SourceProfile = (
    NormalProfile
    | AxialProfile
    | TaperProfile
    | AnnularProfile
    | PerFaceProfile
    | CallableProfile
)


class BIEFormulation:
    STANDARD = "standard"
    COMPLEX_K = "complex_k"


NativeSymmetryPlane = Literal["yz", "xz", "xy", "yz+xz"]
GroundPlane = Literal["xy", "yz", "xz"]
MetalNativeAssemblyMode = Literal["corrected", "optimized", "reference", "parity"]

# Single source of truth for the supported native symmetry planes. Used by
# config validation, native routing, and geometry validation so the lists
# cannot drift apart.
NATIVE_SYMMETRY_PLANES: tuple[str, ...] = ("yz", "xz", "xy", "yz+xz")

# Rigid half-space ground planes. Named for the coordinate plane the rigid
# boundary lies in, matching NATIVE_SYMMETRY_PLANES: "xy" is the Z=0 plane,
# "yz" the X=0 plane, "xz" the Y=0 plane. Only single planes are meaningful --
# two rigid half-spaces would bound a wedge, not a half space.
NATIVE_GROUND_PLANES: tuple[str, ...] = ("xy", "yz", "xz")

# The axis normal to each plane, shared by config validation and the geometry
# validator so the two cannot disagree about which coordinate must stay
# non-negative.
GROUND_PLANE_NORMAL_AXIS: dict[str, int] = {"yz": 0, "xz": 1, "xy": 2}
_MAX_SPHERE_POINTS = 100_000


def _is_integral_value(value: object) -> bool:
    """Return whether ``value`` represents an integer, excluding booleans."""
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value)) and int(value) == value
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_boundary_tag(tag: object, field_name: str) -> int:
    """Validate and normalize one physical-boundary tag."""
    if isinstance(tag, bool) or not isinstance(tag, Integral) or int(tag) < 0:
        raise ValueError(f"{field_name} tags must be non-negative integers")
    return int(tag)


def _validated_velocity_sources(
    sources: object,
    *,
    field_name: str = "velocity_sources",
) -> dict[int, object]:
    """Return a tag-normalized velocity mapping after validating its weights."""
    if not isinstance(sources, dict):
        raise ValueError(f"{field_name} must be a dict mapping tags to weights")
    validated: dict[int, object] = {}
    for tag, weight in sources.items():
        tag_int = _validate_boundary_tag(tag, field_name)
        try:
            weight_value = complex(weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{field_name} weights must be finite complex numbers"
            ) from exc
        if not (
            math.isfinite(weight_value.real) and math.isfinite(weight_value.imag)
        ):
            raise ValueError(f"{field_name} weights must be finite complex numbers")
        validated[tag_int] = weight
    return validated


def _resolve_velocity_sources(
    config: SolveConfig,
    frequency_hz: float,
) -> dict[int, object]:
    """Resolve one frequency's velocity weights within the declared tag set."""
    declared = _validated_velocity_sources(config.velocity_sources)
    if config.velocity_source_callback is None:
        return declared

    frequency_hz = float(frequency_hz)
    field_name = f"velocity_source_callback({frequency_hz:.3f}) result"
    resolved = _validated_velocity_sources(
        config.velocity_source_callback(frequency_hz),
        field_name=field_name,
    )
    extra_tags = sorted(set(resolved) - set(declared))
    if extra_tags:
        raise ValueError(
            f"{field_name} returned tags {extra_tags} that are not declared in "
            "velocity_sources; declare every drivable tag up front (a zero "
            "weight is fine) so frame inference, source velocity profiles, "
            "native session validation, surface pressure averages, and "
            "impedance source selection stay coherent. Returning a subset is "
            "allowed."
        )
    return resolved


def _validated_impedance_sources(
    sources: object,
    *,
    field_name: str = "impedance_sources",
) -> dict[int, complex]:
    """Return a normalized, finite, passive boundary-admittance mapping."""
    if not isinstance(sources, dict):
        raise ValueError(f"{field_name} must be a dict mapping tags to admittances")
    validated: dict[int, complex] = {}
    for tag, beta in sources.items():
        tag_int = _validate_boundary_tag(tag, field_name)
        try:
            beta_value = complex(beta)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{field_name} values must be finite complex numbers"
            ) from exc
        if not (
            math.isfinite(beta_value.real) and math.isfinite(beta_value.imag)
        ):
            raise ValueError(f"{field_name} values must be finite complex numbers")
        if beta_value.real < 0.0:
            raise ValueError(
                f"{field_name} admittance must be passive (Re(beta) >= 0)"
            )
        validated[tag_int] = beta_value
    return validated


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

    # Extra free-standing field points (M, 3) in the mesh frame, evaluated from
    # the SAME solved system as the polar arcs and returned separately as
    # ``observation_sphere_pressure_complex``. Used for full-sphere/balloon
    # sampling: the caller supplies exact coordinates (e.g. a Fibonacci sphere)
    # and pairs them with its own theta/phi metadata. None disables it (no cost,
    # no shape change to the arc outputs).
    sphere_points: NDArray[np.float64] | None = None

    # Frame-relative balloon grid as (n_theta, n_phi). Unlike ``sphere_points``
    # (absolute mesh coordinates), the grid is built inside the solve from the
    # inferred observation frame: theta measured from the frame axis, phi
    # around it with phi=0 along the horizontal (u) direction, radius =
    # ``distance_m``, centred on the observation origin. Theta spans
    # [0, sphere_theta_max_deg] inclusive (90 for half-space balloons); phi
    # spans [0, 360) without the wrap duplicate; ordering is theta-major.
    # Per-frequency pressure lands in ``observation_sphere_pressure_complex``
    # and first-class on SolveResult (sphere_pressure_complex + theta/phi).
    sphere_grid: tuple[int, int] | None = None
    sphere_theta_max_deg: float = 180.0

    # Evaluate only one point per native mirror-symmetry orbit, then scatter
    # back to the complete sphere_grid. This affects frame-relative grids only;
    # explicit sphere_points are always evaluated exactly as supplied. Set
    # False for an in-process A/B. The environment variable
    # HORNLAB_METAL_BEM_SPHERE_SYMMETRY_DEDUPE=0 is an equivalent process-wide
    # escape hatch.
    sphere_symmetry_dedupe: bool = True

    def __post_init__(self) -> None:
        if not self.planes:
            raise ValueError("observation planes must not be empty")
        if not (math.isfinite(self.distance_m) and self.distance_m > 0):
            raise ValueError("distance_m must be finite and positive")
        if not math.isfinite(self.angle_min_deg):
            raise ValueError("angle_min_deg must be finite")
        if not math.isfinite(self.angle_max_deg):
            raise ValueError("angle_max_deg must be finite")
        if not _is_integral_value(self.angle_count) or self.angle_count < 1:
            raise ValueError("angle_count must be at least 1")
        self.angle_count = int(self.angle_count)
        if self.origin not in {"mouth", "throat"}:
            raise ValueError("origin must be 'mouth' or 'throat'")
        if self.sphere_points is not None:
            import numpy as _np

            pts = _np.asarray(self.sphere_points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError("sphere_points must have shape (M, 3)")
            if pts.shape[0] == 0:
                raise ValueError("sphere_points must be non-empty when set")
            if pts.shape[0] > _MAX_SPHERE_POINTS:
                raise ValueError(
                    f"sphere_points is too dense (> {_MAX_SPHERE_POINTS} points)"
                )
            if not _np.all(_np.isfinite(pts)):
                raise ValueError("sphere_points must be finite")
        if self.sphere_grid is not None:
            if self.sphere_points is not None:
                raise ValueError(
                    "sphere_grid and sphere_points are mutually exclusive; "
                    "pass explicit coordinates OR a frame-relative grid"
                )
            grid = tuple(self.sphere_grid)
            if len(grid) != 2:
                raise ValueError("sphere_grid must be (n_theta, n_phi)")
            if not all(_is_integral_value(value) for value in grid):
                raise ValueError("sphere_grid counts must be integers")
            n_theta, n_phi = (int(grid[0]), int(grid[1]))
            if n_theta < 2:
                raise ValueError("sphere_grid n_theta must be at least 2")
            if n_phi < 3:
                raise ValueError("sphere_grid n_phi must be at least 3")
            if n_theta * n_phi > _MAX_SPHERE_POINTS:
                raise ValueError(
                    "sphere_grid is too dense "
                    f"(n_theta*n_phi > {_MAX_SPHERE_POINTS})"
                )
            self.sphere_grid = (n_theta, n_phi)
        if not (0.0 < float(self.sphere_theta_max_deg) <= 180.0):
            raise ValueError("sphere_theta_max_deg must be in (0, 180]")
        if not isinstance(self.sphere_symmetry_dedupe, bool):
            raise ValueError("sphere_symmetry_dedupe must be a bool")


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
    # Direction the prescribed source velocity acts in. "normal" (default) drives
    # each source face along its own outward normal (uniform breathing cap);
    # "axial" drives the source as a rigid piston along its axis, so the per-face
    # normal velocity is U*(n_hat . axis). See SourceMotion. Default "normal"
    # leaves every existing solve bit-for-bit unchanged.
    source_motion: Literal["normal", "axial"] = SourceMotion.NORMAL
    # Optional per-physical-tag source velocity profiles. A configured tag
    # overrides source_motion for that tag; tags with no profile fall back to
    # source_motion. None leaves historical normal/axial behavior unchanged.
    source_velocity_profiles: dict[int, SourceProfile] | None = None
    velocity_sources: dict[int, float] = field(
        default_factory=lambda: {2: 1.0}
    )
    # Frequency-dependent source weights for tags declared in velocity_sources.
    # The callback may omit declared tags (leaving them undriven at that
    # frequency), but may not introduce tags: source geometry, observation
    # frame inference, native session validation, pressure averages, and the
    # impedance reference are established from the static declaration before
    # the frequency loop. Unlike this callback, impedance_source_callback may
    # extend its static mapping because Robin tags do not define that geometry.
    velocity_source_callback: Callable[[float], dict[int, complex]] | None = None
    # Experimental Robin boundary condition. Maps physical tag to normalized
    # surface admittance beta = rho*c/Zs; beta=0 is rigid, beta=1 air-matched.
    impedance_sources: dict[int, complex] = field(default_factory=dict)
    # Frequency-dependent wall admittance. Mirrors velocity_source_callback:
    # called once per solve frequency, returns {tag: beta} that OVERRIDES (for
    # tags also in impedance_sources) and EXTENDS (for new tags) the static
    # impedance_sources for that frequency. beta = rho*c/Zs (normalized
    # admittance). Passivity requires Re(beta) >= 0; the sweep rejects any
    # violation with ValueError.
    #
    # Callback-only Robin tags are fully supported: the sweep resolves the
    # impedance sources exactly once per frequency and threads that resolved
    # tag set into the driver-Neumann builder, so a tag driven by the callback
    # ALONE (absent from the static impedance_sources dict) is correctly
    # skipped for the prescribed-velocity BC — no double boundary condition.
    # Use the static impedance_sources dict only for tags that should always
    # carry a Robin BC at every frequency.
    impedance_source_callback: Callable[[float], dict[int, complex]] | None = None

    # CHIEF (Combined Helmholtz Interior-integral Equation Formulation) points:
    # interior overdetermination points placed *inside* the modeled body's
    # cavities (e.g. an LF front chamber / port volume). Each adds one interior
    # null-field constraint row; the combined system is solved by least squares
    # (zgels, complex128), which removes the exterior-BIE fictitious-eigenvalue
    # non-uniqueness that makes an LF-driven solve blow up at an interior-mode
    # frequency. Shape (m, 3) in metres, in the SAME frame as the mesh vertices.
    # None disables CHIEF (the default; bit-unchanged from the plain solve).
    #
    # Placement guidance: put points in the interior bulk of each enclosed cavity
    # that hosts the spurious mode (4-12 total is standard; start with ~6). Stay
    # strictly inside the watertight surface and ~1 element edge away from the
    # wall (a point too near the boundary makes G(x_c, y) near-singular and
    # pollutes the row). With native_symmetry_plane set, the modeled domain is a
    # reduced wedge and the helper adds analytic images: give points in the
    # reduced frame, inside the reduced cavity, and OFF the symmetry planes (so a
    # point is not self-cancelled by its image). CHIEF composes with both the
    # 'standard' and 'complex_k' formulations. The least-squares path runs in
    # float64 regardless of dense_solve_dtype.
    chief_points: NDArray[np.float64] | None = None
    # Relative weight applied to the CHIEF rows before the least-squares solve.
    # 1.0 (default) auto-scales each case by ||A||_inf/||C||_inf in the helper so
    # the collocation CHIEF rows are numerically comparable to the Galerkin
    # boundary rows; override only to bias the interior constraint harder/softer.
    chief_weight: float = 1.0

    # Axisymmetric pure-Python solver option. None is free field; a finite
    # z-coordinate enables a same-sign rigid/Neumann image source plane for
    # body-of-revolution m=0 solves.
    circsym_baffle_z: float | None = None
    # Axisymmetric exact infinite-baffle coupled solve. Names the meridian
    # segment tag that represents the flush mouth-aperture disc; None keeps the
    # existing CircSym path unchanged.
    circsym_aperture_tag: int | None = None

    # Observation
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    # Frame override: skip infer_frame() when set.
    # Use this when external source winding is not authoritative, several
    # sources need a shared frame, or the caller has a known frame (e.g. WG).
    frame_override: object | None = None  # ObservationFrame, kept as object to avoid circular import

    # Native Metal controls
    # Full-3D native Metal coupled infinite-baffle solve. Names the physical tag
    # for the flush z=0 aperture triangles. Independent of circsym_aperture_tag.
    aperture_tag: int | None = None
    native_symmetry_plane: NativeSymmetryPlane | None = None
    # Rigid (Neumann) infinite half-space boundary, named for the coordinate
    # plane it lies in: "xy" is a rigid floor at Z=0, "yz" a rigid wall at X=0,
    # "xz" a rigid wall at Y=0. The modelled mesh is the COMPLETE radiating
    # body and must lie entirely on the non-negative side of that plane; the
    # solver adds one mirror image of it and returns the pressure in the open
    # half space. Reflection coefficient is +1 (rigid); there is no finite
    # impedance ground.
    #
    # This is NOT native_symmetry_plane. Both use the same image kernel, but
    # they say different things about the mesh you supplied:
    #
    #   native_symmetry_plane -- the mesh is HALF of a mirror-symmetric body,
    #     cut on the plane, and the image completes the physical radiator. The
    #     rim must lie on the plane, and radiated surface power is multiplied
    #     by the number of copies.
    #   ground_plane -- the mesh is the WHOLE body standing next to a rigid
    #     wall it need not touch. The image is fictitious, so radiated surface
    #     power is NOT multiplied, and the observation frame is not projected
    #     onto the plane.
    #
    # A cabinet resting flat on the ground with its contact face removed (rim
    # on the plane) is expressible either way, and native_symmetry_plane has
    # always solved that geometry correctly -- but it reports twice the
    # radiated surface power, because it believes the image is real. Use
    # ground_plane for anything standing on, or flown above, a rigid boundary.
    #
    # None (the default) leaves every existing solve bit-for-bit unchanged.
    # Does not currently compose with native_symmetry_plane or aperture_tag.
    ground_plane: GroundPlane | None = None
    # Minimum clearance, in metres, required between the mesh and the ground
    # plane when the mesh does not touch it. Purely a guard against a body
    # placed so close to its own image that the fixed-order quadrature between
    # the real and image faces loses accuracy; it does not change the solve.
    # Faces that touch the plane exactly are rejected separately.
    ground_plane_min_clearance_m: float = 0.0
    # When True (default), a reduced-domain symmetry mesh must have every open
    # boundary edge on a requested symmetry plane: a closed surface reduced by
    # mirror cuts has its whole rim on the cut planes, so an off-plane open edge
    # signals a mesh cut along an unrequested plane. Set False for open shells
    # whose rim is a real free edge of the full (reduced + mirrored) geometry,
    # e.g. a bare horn radiating from an open mouth. The caller that knows the
    # mesh topology owns this choice; geometry alone cannot distinguish a
    # legitimate mouth rim from a bad cut.
    native_check_open_edges: bool = True
    metal_native_assembly_mode: MetalNativeAssemblyMode = "corrected"
    # Dense LU precision. "float32" (default) matches the historical Complex32
    # LU. "float64" factors/solves the float32-assembled system in complex128
    # (Accelerate zgesv) to recover the 3-4 digits float32 LU loses near a
    # near-singular system, then narrows the solved pressure back to f32;
    # assembly and all downstream buffers/outputs stay float32. Mixed precision.
    # The complex128 buffers roughly triple peak solve memory, so the native
    # routing lowers the default solve concurrency for the float64 path unless
    # the caller pinned HORNLAB_METAL_BEM_NATIVE_SOLVE_CONCURRENCY.
    dense_solve_dtype: Literal["float32", "float64"] = "float32"
    # Dense solve method. "cgesv" is the direct LU that has always shipped.
    # "gmres" is block-Jacobi preconditioned GMRES: measured at 20-35 iterations
    # across kD 5-202 and flat as the mesh refines, so it trades the LU's O(N^3)
    # for O(iterations * N^2) and agrees with the LU to within float32 noise.
    # It is opt-in. `standard` + "gmres" is not refused -- the answer is still
    # correct, just ~6x the iterations on a closed body -- so sweep.py warns on
    # that configuration and again when a measured iteration count crosses the
    # gate the path was accepted against. This field is the only way to select
    # it from Python: the helper env var is overwritten from this field on
    # every solve (see _native_env_overrides in sweep.py).
    dense_solve_implementation: Literal[
        "cgesv", "cgetrf_cgetrs", "gmres"
    ] = "cgesv"
    return_surface_pressure: bool = False
    # Retain the complete P1 pressure and total DP0 Neumann traces needed to
    # re-evaluate the exterior field after the solve. This implies surface
    # pressure output and additionally reconstructs Robin contributions.
    return_surface_traces: bool = False
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
    air_density: float = AIR_DENSITY

    # Speed of sound (m/s). Default 343.0 matches standard air at 20 C and is
    # the value every result published before 2026-09-03 was solved with, so
    # leaving it alone reproduces those bit-for-bit. Set it when matching an
    # external reference that assumes a different value -- ABEC3 defaults to
    # 343.32 m/s, a 0.093% offset that is a fixed bias in every comparison
    # against it. It scales the wavenumber (k = 2*pi*f/c), so it also moves the
    # mesh-resolution diagnostics and the radiated-power reduction.
    speed_of_sound: float = SPEED_OF_SOUND

    # Progress callback: called after each frequency solve.
    # Signature: (freq_index: int, total_freqs: int, frequency_hz: float) -> None
    progress_callback: Callable[[int, int, float], None] | None = None

    # Per-frequency result callback for early stopping.
    # Signature: (freq_index: int, frequency_hz: float, log_entry: dict) -> bool
    # Return exactly False to abort the sweep (partial SolveResult is built);
    # any other return value, including None, continues.
    on_frequency_result: Callable[[int, float, dict], bool] | None = None

    # Fine-grained CircSym cancellation checkpoint. Called within expensive
    # assembly and field-evaluation blocks. Return exactly False to cancel;
    # callers may instead raise their own runtime-specific cancellation exception.
    should_continue: Callable[[], bool | None] | None = None

    def __post_init__(self) -> None:
        if self.freq_spacing not in {"log", "linear"}:
            raise ValueError("freq_spacing must be 'log' or 'linear'")
        if not _is_integral_value(self.freq_count) or self.freq_count < 1:
            raise ValueError("freq_count must be at least 1")
        self.freq_count = int(self.freq_count)
        if not (math.isfinite(self.freq_min_hz) and self.freq_min_hz > 0):
            raise ValueError("freq_min_hz must be finite and positive")
        if not math.isfinite(self.freq_max_hz):
            raise ValueError("freq_max_hz must be finite")
        if self.freq_max_hz < self.freq_min_hz:
            raise ValueError("freq_max_hz must be >= freq_min_hz")
        if not (math.isfinite(self.mesh_scale) and self.mesh_scale > 0):
            raise ValueError("mesh_scale must be finite and positive")
        if not math.isfinite(self.mesh_merge_tol):
            raise ValueError("mesh_merge_tol must be finite")
        if not (math.isfinite(self.air_density) and self.air_density > 0):
            raise ValueError("air_density must be finite and positive")
        if not (math.isfinite(self.speed_of_sound) and self.speed_of_sound > 0):
            raise ValueError("speed_of_sound must be finite and positive")
        if not (
            math.isfinite(self.dense_solve_rcond_warning_threshold)
            and self.dense_solve_rcond_warning_threshold >= 0
        ):
            raise ValueError(
                "dense_solve_rcond_warning_threshold must be finite and non-negative"
            )
        if not (
            math.isfinite(self.mesh_elements_per_wavelength_min)
            and self.mesh_elements_per_wavelength_min > 0
        ):
            raise ValueError(
                "mesh_elements_per_wavelength_min must be finite and positive"
            )
        if self.formulation not in {BIEFormulation.STANDARD, BIEFormulation.COMPLEX_K}:
            raise ValueError("formulation must be 'standard' or 'complex_k'")
        if not (
            math.isfinite(self.complex_k_shift) and self.complex_k_shift >= 0
        ):
            raise ValueError("complex_k_shift must be finite and non-negative")
        if self.velocity_mode not in {VelocityMode.VELOCITY, VelocityMode.ACCELERATION}:
            raise ValueError("velocity_mode must be 'velocity' or 'acceleration'")
        if self.source_motion not in {SourceMotion.NORMAL, SourceMotion.AXIAL}:
            raise ValueError("source_motion must be 'normal' or 'axial'")
        _validated_velocity_sources(self.velocity_sources)
        if (
            self.velocity_source_callback is not None
            and not callable(self.velocity_source_callback)
        ):
            raise ValueError("velocity_source_callback must be callable or None")
        if self.source_velocity_profiles is not None:
            if not isinstance(self.source_velocity_profiles, dict):
                raise ValueError("source_velocity_profiles must be a dict or None")
            for tag, profile in self.source_velocity_profiles.items():
                _validate_boundary_tag(tag, "source_velocity_profiles")
                _validate_source_profile(profile)
        _validated_impedance_sources(self.impedance_sources)
        if (
            self.impedance_source_callback is not None
            and not callable(self.impedance_source_callback)
        ):
            raise ValueError("impedance_source_callback must be callable or None")
        if self.chief_points is not None:
            import numpy as _np

            pts = _np.asarray(self.chief_points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError("chief_points must have shape (m, 3)")
            if pts.shape[0] == 0:
                raise ValueError("chief_points must be non-empty when set")
            if not _np.all(_np.isfinite(pts)):
                raise ValueError("chief_points must be finite")
        if not (math.isfinite(self.chief_weight) and self.chief_weight > 0):
            raise ValueError("chief_weight must be finite and positive")
        if self.circsym_baffle_z is not None and not math.isfinite(
            float(self.circsym_baffle_z)
        ):
            raise ValueError("circsym_baffle_z must be finite or None")
        if self.circsym_aperture_tag is not None:
            if (
                isinstance(self.circsym_aperture_tag, bool)
                or not isinstance(self.circsym_aperture_tag, Integral)
                or self.circsym_aperture_tag <= 0
            ):
                raise ValueError(
                    "circsym_aperture_tag must be a positive int or None"
                )
        if self.aperture_tag is not None:
            if (
                isinstance(self.aperture_tag, bool)
                or not isinstance(self.aperture_tag, Integral)
                or self.aperture_tag <= 0
            ):
                raise ValueError("aperture_tag must be a positive int or None")
        if (
            self.native_symmetry_plane is not None
            and self.native_symmetry_plane not in NATIVE_SYMMETRY_PLANES
        ):
            raise ValueError(
                "native_symmetry_plane must be None, 'yz', 'xz', 'xy', or 'yz+xz'"
            )
        if self.ground_plane is not None:
            if self.ground_plane not in NATIVE_GROUND_PLANES:
                raise ValueError(
                    "ground_plane must be None, 'xy', 'yz', or 'xz'"
                )
            if self.native_symmetry_plane is not None:
                raise ValueError(
                    "ground_plane does not yet compose with "
                    "native_symmetry_plane; solve the full body against the "
                    "ground, or drop the ground plane"
                )
            if self.aperture_tag is not None:
                raise ValueError(
                    "ground_plane does not compose with the coupled "
                    "infinite-baffle aperture_tag mode"
                )
        if not (
            math.isfinite(self.ground_plane_min_clearance_m)
            and self.ground_plane_min_clearance_m >= 0.0
        ):
            raise ValueError(
                "ground_plane_min_clearance_m must be finite and non-negative"
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
        if self.dense_solve_dtype not in {"float32", "float64"}:
            raise ValueError("dense_solve_dtype must be 'float32' or 'float64'")
        if self.dense_solve_implementation not in {
            "cgesv",
            "cgetrf_cgetrs",
            "gmres",
        }:
            raise ValueError(
                "dense_solve_implementation must be 'cgesv', 'cgetrf_cgetrs' "
                "or 'gmres'"
            )
        if (
            self.dense_solve_implementation == "gmres"
            and self.dense_solve_dtype == "float64"
        ):
            raise ValueError(
                "dense_solve_implementation='gmres' has no float64 path; use "
                "dense_solve_dtype='float32' or a direct solver"
            )
        for name in (
            "metal_native_threads_per_group",
            "metal_native_matrix_threads_per_group",
            "metal_native_rhs_threads_per_group",
            "metal_native_duffy_threads_per_group",
            "metal_native_field_threads_per_group",
        ):
            value = getattr(self, name)
            if value is not None:
                if not _is_integral_value(value) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
                setattr(self, name, int(value))


def _validate_source_profile(profile: SourceProfile) -> None:
    if isinstance(profile, (NormalProfile, AxialProfile)):
        return
    if isinstance(profile, TaperProfile):
        if profile.kind not in {"raised_cosine", "linear"}:
            raise ValueError(
                "TaperProfile.kind must be 'raised_cosine' or 'linear'"
            )
        if not (math.isfinite(profile.start) and 0.0 <= profile.start < 1.0):
            raise ValueError("TaperProfile.start must be finite and in [0, 1)")
        return
    if isinstance(profile, AnnularProfile):
        if not (
            math.isfinite(profile.r_inner)
            and math.isfinite(profile.r_outer)
            and 0.0 <= profile.r_inner <= profile.r_outer <= 1.0
        ):
            raise ValueError(
                "AnnularProfile radii must be finite with "
                "0 <= r_inner <= r_outer <= 1"
            )
        return
    if isinstance(profile, PerFaceProfile):
        import numpy as _np

        try:
            values = _np.asarray(profile.weights, dtype=_np.complex128)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "PerFaceProfile.weights must be convertible to a complex array"
            ) from exc
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                "PerFaceProfile.weights must be a non-empty 1D array"
            )
        if not _np.all(_np.isfinite(values)):
            raise ValueError("PerFaceProfile.weights must be finite")
        return
    if isinstance(profile, CallableProfile):
        if not callable(profile.callback):
            raise ValueError("CallableProfile.callback must be callable")
        return
    raise ValueError(
        "source_velocity_profiles values must be SourceProfile instances"
    )

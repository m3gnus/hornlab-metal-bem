from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
import queue
import platform
import threading
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from scipy import signal

from .config import BIEFormulation, ObservationConfig, SolveConfig, VelocityMode

REFERENCE_PRESSURE_PA = 20e-6

try:  # Boundary Lab is optional when this package is used standalone.
    from blab.solvers.base import (
        FrequencyResult,
        FrequencySolveTimings,
        SolveMetadata,
        SolverCapabilities,
        SolverDiagnostics,
    )
except Exception:  # pragma: no cover - exercised only outside Boundary Lab.
    @dataclass(frozen=True)
    class FrequencySolveTimings:
        assembly_s: float = 0.0
        solve_s: float = 0.0
        field_s: float = 0.0

    @dataclass(frozen=True)
    class SolverDiagnostics:
        convergence_info: int | None = None
        message: str | None = None

    @dataclass(frozen=True)
    class FrequencyResult:
        freq_hz: float
        horizontal_spl_norm_db: np.ndarray
        vertical_spl_norm_db: np.ndarray
        impedance: np.ndarray
        horizontal_spl_db: np.ndarray | None = None
        vertical_spl_db: np.ndarray | None = None
        sphere_spl_norm_db: np.ndarray | None = None
        channel_names: np.ndarray | None = None
        horizontal_pressure: np.ndarray | None = None
        vertical_pressure: np.ndarray | None = None
        sphere_pressure: np.ndarray | None = None
        timings: FrequencySolveTimings = field(default_factory=FrequencySolveTimings)
        diagnostics: SolverDiagnostics | None = None

        @property
        def has_channel_basis(self) -> bool:
            return (
                self.channel_names is not None
                and self.horizontal_pressure is not None
                and self.vertical_pressure is not None
            )

    @dataclass(frozen=True)
    class SolveMetadata:
        polar_angle_deg: np.ndarray
        radiator_names: np.ndarray
        sphere_metadata: dict[str, np.ndarray] | None = None

    @dataclass(frozen=True)
    class SolverCapabilities:
        supports_spherical_sampling: bool = False
        supports_impedance: bool = True
        supports_burton_miller: bool = False
        supports_flat_target_normalization: bool = True
        supports_channel_resynthesis: bool = True
        supports_cancellation: bool = True
        supports_streaming: bool = True
        supports_remote_assets: bool = False
        supports_parallel_workers: bool = False
        supports_symmetry: bool = True
        is_remote: bool = False


BACKEND_ID = "hornlab_metal"
__all__ = [
    "BACKEND_ID",
    "BoundaryLabBackend",
    "BoundaryLabSession",
    "BoundaryLabSolverError",
    "create_backend",
    "is_apple_silicon",
    "solve_config_from_boundary_lab",
]


class BoundaryLabSolverError(ValueError):
    pass


@dataclass(frozen=True)
class BoundaryLabBackend:
    """Solver backend implementing Boundary Lab's local backend protocol."""

    backend_id: str = BACKEND_ID
    label: str = "HornLab Metal BEM"
    capabilities: SolverCapabilities = field(
        default_factory=lambda: SolverCapabilities(
            supports_spherical_sampling=True,
            supports_burton_miller=False,
            supports_flat_target_normalization=True,
            supports_channel_resynthesis=True,
            supports_cancellation=True,
            supports_streaming=True,
            supports_remote_assets=False,
            supports_parallel_workers=False,
            supports_symmetry=True,
            is_remote=False,
        )
    )
    default_overrides: dict[str, Any] = field(default_factory=dict)

    def create_session(
        self,
        request_or_config: Any | None = None,
    ) -> "BoundaryLabSession":
        return BoundaryLabSession(
            request_or_config=request_or_config,
            default_overrides=dict(self.default_overrides),
        )

    def supports(self, simulation_config: Any | None = None) -> bool:
        return is_apple_silicon()


@dataclass
class BoundaryLabSession:
    """Boundary Lab solver session backed by the native Metal sweep."""

    request_or_config: Any | None = None
    default_overrides: dict[str, Any] = field(default_factory=dict)
    _stop: bool = False

    def solve(
        self,
        mesh: Any,
        simulation_config: Any | None = None,
        **overrides: Any,
    ):
        from . import solve as _solve
        from . import solve_frequencies as _solve_frequencies

        config_source = (
            self._simulation_config if simulation_config is None else simulation_config
        )
        merged_overrides = {**self.default_overrides, **overrides}
        solve_config, frequencies_hz = solve_config_from_boundary_lab(
            config_source,
            **merged_overrides,
        )
        if frequencies_hz is None and simulation_config is None:
            # Request-shaped inputs carry frequencies next to the config.
            frequencies_hz = self._frequencies_hz
        if frequencies_hz is not None:
            return _solve_frequencies(mesh, frequencies_hz, solve_config)
        return _solve(mesh, solve_config)

    @property
    def _simulation_config(self) -> Any | None:
        if self.request_or_config is None:
            return None
        nested = _first(self.request_or_config, "config", default=None)
        return nested if nested is not None else self.request_or_config

    @property
    def _frequencies_hz(self) -> np.ndarray | None:
        frequencies = _first(self.request_or_config, "frequencies_hz", default=None)
        return _coerce_frequencies(frequencies)

    @property
    def metadata(self) -> SolveMetadata:
        cfg = self._simulation_config
        sphere = _boundary_lab_sphere(cfg)
        sphere_metadata = None
        if sphere is not None:
            _points, theta_polar, phi_azimuth, r_distance = sphere
            sphere_metadata = {
                "r_distance_m": r_distance,
                "theta_polar_rad": theta_polar,
                "phi_azimuth_rad": phi_azimuth,
            }
        return SolveMetadata(
            polar_angle_deg=_boundary_lab_angles(cfg).astype(np.float32, copy=False),
            radiator_names=np.asarray(_radiator_names(cfg)),
            sphere_metadata=sphere_metadata,
        )

    def solve_stream(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[FrequencyResult]:
        from . import solve_multi_source as _solve_multi_source

        # A previous stream (even one that ran to completion) leaves _stop
        # set; reset so session reuse does not silently truncate to one
        # frequency.
        self._stop = False

        cfg = self._simulation_config
        if cfg is None:
            raise BoundaryLabSolverError("Boundary Lab solve request is missing a SimulationConfig.")

        frequencies_hz = self._frequencies_hz
        if frequencies_hz is None:
            raise BoundaryLabSolverError("Boundary Lab solve request is missing frequencies_hz.")

        mesh_path = _first(cfg, "mesh_file", default=None)
        if mesh_path is None:
            raise BoundaryLabSolverError("Boundary Lab SimulationConfig is missing mesh_file.")

        result_queue: queue.Queue[FrequencyResult | BaseException | None] = queue.Queue()
        channel_names, channel_sources = _channel_basis_sources(cfg)
        channel_configs = _boundary_lab_channel_configs_by_name(cfg)

        def should_stop() -> bool:
            return self._stop or (stop_requested is not None and stop_requested())

        def on_frequency_result(index: int, frequency_hz: float, entry: dict[str, Any]) -> bool:
            result_queue.put(
                _frequency_result_from_channel_basis_entry(
                    frequency_hz,
                    entry,
                    channel_names=channel_names,
                    channel_configs=channel_configs,
                )
            )
            return not should_stop()

        overrides = {**self.default_overrides, "on_frequency_result": on_frequency_result}
        overrides.update(
            velocity_sources=dict(channel_sources[0]),
            velocity_source_callback=None,
        )
        solve_config, translated_frequencies = solve_config_from_boundary_lab(cfg, **overrides)
        frequencies = frequencies_hz if frequencies_hz is not None else translated_frequencies
        if frequencies is None:
            raise BoundaryLabSolverError("No frequencies available for Metal solve.")

        def run() -> None:
            try:
                _solve_multi_source(mesh_path, channel_sources, solve_config, frequencies_hz=frequencies)
            except BaseException as exc:
                result_queue.put(exc)
            finally:
                result_queue.put(None)

        worker = threading.Thread(target=run, name="hornlab-metal-boundary-lab-solve", daemon=True)
        worker.start()

        try:
            while True:
                item = result_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
                if should_stop():
                    self._stop = True
        finally:
            self._stop = True
            worker.join(timeout=1.0)

    def stop(self) -> None:
        self._stop = True

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "BoundaryLabSession":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def create_backend(**default_overrides: Any) -> BoundaryLabBackend:
    return BoundaryLabBackend(default_overrides=dict(default_overrides))


def is_apple_silicon() -> bool:
    return (
        platform.system() == "Darwin"
        and platform.machine() in {"arm64", "aarch64"}
    )


def solve_config_from_boundary_lab(
    simulation_config: Any | None = None,
    **overrides: Any,
) -> tuple[SolveConfig, np.ndarray | None]:
    """Translate a Boundary Lab-like config object into ``SolveConfig``.

    The translator accepts dictionaries and ordinary config objects. Unknown
    fields are ignored so Boundary Lab can evolve independently.
    """
    frequencies = _first(
        simulation_config,
        "frequencies_hz",
        "frequency_hz",
        "frequencies",
        default=None,
    )
    frequencies_hz = _coerce_frequencies(frequencies)

    # Derive the observation arc directly from the Boundary Lab angle grid so
    # the solved angles always match the metadata.polar_angle_deg the adapter
    # publishes. With a step-based grid whose step does not divide the span,
    # the grid's last angle (not the configured max) is the true endpoint.
    boundary_angles = _boundary_lab_angles(simulation_config)
    sphere = _boundary_lab_sphere(simulation_config)
    observation = ObservationConfig(
        planes=list(
            _first(
                simulation_config,
                "planes",
                "observation_planes",
                default=["horizontal", "vertical"],
            )
        ),
        distance_m=float(
            _first(
                simulation_config,
                "distance",
                "distance_m",
                "observation_distance_m",
                default=2.0,
            )
        ),
        angle_min_deg=float(boundary_angles[0]),
        angle_max_deg=float(boundary_angles[-1]),
        angle_count=len(boundary_angles),
        origin=_first(
            simulation_config,
            "origin",
            "observation_origin",
            default="mouth",
        ),
        sphere_points=None if sphere is None else sphere[0],
    )

    config_values: dict[str, Any] = {
        "freq_min_hz": float(
            _first(
                simulation_config,
                "freq_min",
                "freq_min_hz",
                "frequency_min_hz",
                "min_frequency_hz",
                default=500.0,
            )
        ),
        "freq_max_hz": float(
            _first(
                simulation_config,
                "freq_max",
                "freq_max_hz",
                "frequency_max_hz",
                "max_frequency_hz",
                default=20_000.0,
            )
        ),
        "freq_count": int(
            _first(
                simulation_config,
                "freq_count",
                "frequency_count",
                default=40,
            )
        ),
        "freq_spacing": _first(
            simulation_config,
            "freq_spacing",
            "frequency_spacing",
            default="log",
        ),
        "velocity_sources": _coerce_velocity_sources(simulation_config),
        "velocity_source_callback": _coerce_velocity_source_callback(simulation_config),
        "velocity_mode": VelocityMode.VELOCITY,
        # Boundary Lab's Burton-Miller toggle requests fictitious-eigenvalue
        # robustness. The Metal core has no Burton-Miller operator, but its
        # complex-wavenumber formulation solves the same non-uniqueness, so map
        # the intent onto complex_k rather than silently ignoring it.
        "formulation": (
            BIEFormulation.COMPLEX_K
            if bool(_first(simulation_config, "use_burton_miller", default=False))
            else BIEFormulation.STANDARD
        ),
        "mesh_scale": float(
            _first(simulation_config, "mesh_scale", "scale_factor", default=1.0)
        ),
        "air_density": float(_first(simulation_config, "rho", "air_density", default=1.2041)),
        "native_symmetry_plane": _coerce_symmetry_plane(
            _first(simulation_config, "symmetry", default="off")
        ),
        # Open-shell meshes (a bare horn radiating from an open mouth) have a
        # real free rim off the symmetry planes; the spec opts those out of the
        # cut-plane open-edge guard. Defaults to the strict check for closed
        # mirror-reduced meshes.
        "native_check_open_edges": bool(
            _first(
                simulation_config,
                "native_check_open_edges",
                "check_open_edges",
                default=True,
            )
        ),
        # Boundary Lab and mesher metadata use both snake_case and camelCase.
        # Preserve the coupled infinite-baffle contract through translation so
        # a declared aperture cannot silently become an ordinary free-space
        # solve.
        "aperture_tag": _coerce_boundary_aperture_tag(simulation_config),
        "observation": observation,
        "metal_native_assembly_mode": "corrected",
    }
    if "apertureTag" in overrides:
        camel_value = overrides.pop("apertureTag")
        if "aperture_tag" in overrides and overrides["aperture_tag"] != camel_value:
            raise BoundaryLabSolverError(
                "Conflicting aperture_tag and apertureTag overrides"
            )
        overrides["aperture_tag"] = camel_value
    config_values.update(overrides)
    return SolveConfig(**config_values), frequencies_hz


def _first(source: Any | None, *names: str, default: Any) -> Any:
    if source is None:
        return default
    for name in names:
        if isinstance(source, dict) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _coerce_frequencies(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return np.asarray([float(value)], dtype=np.float64)
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return np.asarray(list(value), dtype=np.float64)
    raise BoundaryLabSolverError("frequencies_hz must be a number or sequence")


def _coerce_boundary_aperture_tag(source: Any | None) -> int | None:
    values: list[tuple[str, Any]] = []
    for name in ("aperture_tag", "apertureTag"):
        if isinstance(source, dict):
            if name in source and source[name] is not None:
                values.append((name, source[name]))
        elif source is not None and hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                values.append((name, value))
    if not values:
        return None

    normalized: list[tuple[str, int]] = []
    for name, value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise BoundaryLabSolverError(f"{name} must be a positive integer or null")
        normalized.append((name, int(value)))

    unique = {tag for _, tag in normalized}
    if len(unique) != 1:
        detail = ", ".join(f"{name}={tag}" for name, tag in normalized)
        raise BoundaryLabSolverError(f"Conflicting aperture metadata: {detail}")
    return normalized[0][1]


def _coerce_velocity_sources(source: Any | None) -> dict[int, float]:
    value = _first(source, "velocity_sources", default=None)
    if value is not None:
        return {int(k): float(v) for k, v in dict(value).items()}
    radiators = tuple(_first(source, "radiators", default=()) or ())
    if radiators:
        return {
            int(_first(radiator, "tag", default=2)): 1.0
            for radiator in radiators
        }
    source_tag = int(_first(source, "source_tag", "driver_tag", default=2))
    source_weight = float(
        _first(source, "source_weight", "velocity_weight", default=1.0)
    )
    return {source_tag: source_weight}


def _channel_basis_sources(source: Any | None) -> tuple[np.ndarray, list[dict[int, complex]]]:
    """Return Boundary Lab channel-basis source vectors.

    Boundary Lab solves one unit source per active channel and applies channel
    gain, polarity, delay, crossover, and flat-target correction after solving.
    Radiator ``velocity_offset_db`` is part of the source basis because it is a
    per-radiator scale, not a post-solve channel control.
    """
    radiators = tuple(_first(source, "radiators", default=()) or ())
    if not radiators:
        return np.asarray(["main"]), [
            {int(tag): complex(value) for tag, value in _coerce_velocity_sources(source).items()}
        ]

    names = tuple(sorted({str(_first(radiator, "channel", default="main")) for radiator in radiators}))
    sources: list[dict[int, complex]] = []
    for name in names:
        values: dict[int, complex] = {}
        for radiator in radiators:
            if str(_first(radiator, "channel", default="main")) != name:
                continue
            tag = int(_first(radiator, "tag", default=2))
            velocity_offset = 10.0 ** (float(_first(radiator, "velocity_offset_db", default=0.0)) / 20.0)
            values[tag] = values.get(tag, 0.0 + 0.0j) + complex(velocity_offset)
        if not values:
            raise BoundaryLabSolverError(f"Channel {name!r} has no driven radiators.")
        sources.append(values)
    return np.asarray(names), sources


def _coerce_velocity_source_callback(source: Any | None) -> Callable[[float], dict[int, complex]] | None:
    if _first(source, "velocity_sources", default=None) is not None:
        return None
    radiators = tuple(_first(source, "radiators", default=()) or ())
    if not radiators:
        return None
    channels = _channel_configs_by_name(source)

    def callback(frequency_hz: float) -> dict[int, complex]:
        drives: dict[int, complex] = {}
        for radiator in radiators:
            tag = int(_first(radiator, "tag", default=2))
            channel = channels.get(str(_first(radiator, "channel", default="main")))
            drive = (
                _channel_drive(channel, frequency_hz)
                if channel is not None
                else _radiator_drive(radiator, frequency_hz)
            )
            drive *= 10.0 ** (
                float(_first(radiator, "velocity_offset_db", default=0.0)) / 20.0
            )
            drives[tag] = drives.get(tag, 0.0 + 0.0j) + drive
        return drives

    return callback


def _channel_configs_by_name(source: Any | None) -> dict[str, Any]:
    channels = tuple(_first(source, "channels", default=()) or ())
    return {str(_first(channel, "name", default="main")): channel for channel in channels}


def _boundary_lab_channel_configs_by_name(source: Any | None) -> dict[str, Any]:
    channels = _channel_configs_by_name(source)
    if channels:
        return channels

    resolved: dict[str, Any] = {}
    for radiator in tuple(_first(source, "radiators", default=()) or ()):
        name = str(_first(radiator, "channel", default="main"))
        resolved[name] = radiator
    return resolved


def _radiator_drive(radiator: Any, frequency_hz: float) -> complex:
    return _level_polarity_delay_filter_drive(radiator, frequency_hz)


def _channel_drive(channel: Any, frequency_hz: float) -> complex:
    return _level_polarity_delay_filter_drive(channel, frequency_hz)


def _boundary_lab_channel_drive(channel: Any | None, frequency_hz: float) -> complex:
    if channel is None:
        return 1.0 + 0.0j
    omega = 2.0 * np.pi * float(frequency_hz)
    level = 10.0 ** (float(_first(channel, "level_db", default=0.0)) / 20.0)
    polarity = int(_first(channel, "polarity", default=1))
    if polarity not in {-1, 1}:
        raise BoundaryLabSolverError(
            f"polarity must be -1 or 1, got {polarity!r}"
        )
    delay = np.exp(-1j * omega * (float(_first(channel, "delay_ms", default=0.0)) / 1000.0))
    crossover = 1.0 + 0.0j
    for name in ("hpf", "lpf"):
        crossover_config = _first(channel, name, default=None)
        if crossover_config is not None and str(_first(crossover_config, "type", default="none")).lower() != "none":
            # _crossover_response returns the Metal/e^{-i wt} convention.
            # Boundary Lab's channel synthesis uses the conjugate convention.
            crossover *= np.conj(_crossover_response(crossover_config, frequency_hz))
    return complex(level * polarity * delay * crossover)


def _level_polarity_delay_filter_drive(source: Any, frequency_hz: float) -> complex:
    omega = 2.0 * np.pi * float(frequency_hz)
    level = 10.0 ** (float(_first(source, "level_db", default=0.0)) / 20.0)
    polarity = int(_first(source, "polarity", default=1))
    if polarity not in {-1, 1}:
        raise BoundaryLabSolverError(
            f"polarity must be -1 or 1, got {polarity!r}"
        )
    # The BEM core uses the e^{-i omega t} convention (Green's kernel
    # e^{+ikr}/4*pi*r, Neumann coefficient +i*rho*omega*v_n). A time delay of
    # tau therefore multiplies the phasor by e^{+i omega tau}.
    delay = np.exp(1j * omega * (float(_first(source, "delay_ms", default=0.0)) / 1000.0))
    crossover = 1.0 + 0.0j
    for name in ("hpf", "lpf"):
        crossover_config = _first(source, name, default=None)
        if crossover_config is not None and str(_first(crossover_config, "type", default="none")).lower() != "none":
            crossover *= _crossover_response(crossover_config, frequency_hz)
    return complex(level * polarity * delay * crossover)


_CROSSOVER_TYPES = {"highpass", "hpf", "lowpass", "lpf"}
_CROSSOVER_FILTERS = {"butterworth", "linkwitz_riley"}


def _crossover_response(crossover: Any, frequency_hz: float) -> complex:
    crossover_type = str(_first(crossover, "type", default="none")).lower()
    if crossover_type == "none":
        return 1.0 + 0.0j
    if crossover_type not in _CROSSOVER_TYPES:
        raise BoundaryLabSolverError(
            f"Unsupported crossover type {crossover_type!r}; "
            "expected one of 'highpass', 'hpf', 'lowpass', 'lpf', or 'none'"
        )

    filter_name = str(_first(crossover, "filter", default="butterworth")).lower()
    if filter_name not in _CROSSOVER_FILTERS:
        raise BoundaryLabSolverError(
            f"Unsupported crossover filter {filter_name!r}; "
            "expected 'butterworth' or 'linkwitz_riley'"
        )
    order = int(_first(crossover, "order", default=1))
    if order < 1:
        raise BoundaryLabSolverError(f"crossover order must be >= 1, got {order}")
    cutoff = _first(crossover, "frequency_hz", default=None)
    if cutoff is None:
        raise BoundaryLabSolverError(
            f"crossover of type {crossover_type!r} is missing frequency_hz"
        )
    cutoff_hz = float(cutoff)
    if cutoff_hz <= 0:
        raise BoundaryLabSolverError("crossover frequency_hz must be positive")
    if filter_name == "linkwitz_riley":
        if order % 2 != 0:
            raise BoundaryLabSolverError(
                f"linkwitz_riley order must be even, got {order}"
            )
        section_order = order // 2
        section = _butterworth_response(crossover_type, section_order, cutoff_hz, frequency_hz)
        return section * section
    return _butterworth_response(crossover_type, order, cutoff_hz, frequency_hz)


def _butterworth_response(crossover_type: str, order: int, cutoff_hz: float, frequency_hz: float) -> complex:
    btype = "highpass" if crossover_type in {"highpass", "hpf"} else "lowpass"
    b, a = signal.butter(order, 2.0 * np.pi * cutoff_hz, btype=btype, analog=True)
    _, response = signal.freqs(b, a, worN=[2.0 * np.pi * frequency_hz])
    # scipy's analog H(j omega) assumes the e^{+j omega t} convention; the BEM
    # core uses e^{-i omega t}, so the drive phasor is the conjugate.
    return complex(np.conj(response[0]))


def _boundary_lab_angles(source: Any | None) -> np.ndarray:
    explicit_count = _first(source, "angle_count", "n_angles", default=None)
    angle_min = float(_first(source, "angle_min_deg", "min_angle_deg", "min_angle", default=0.0))
    angle_max = float(_first(source, "angle_max_deg", "max_angle_deg", "max_angle", default=180.0))
    if explicit_count is not None:
        return np.linspace(angle_min, angle_max, int(explicit_count), dtype=np.float64)

    step = float(_first(source, "step_size", "polar_angle_step_deg", default=5.0))
    if step <= 0:
        raise BoundaryLabSolverError("step_size must be positive.")
    return np.clip(
        np.arange(angle_min, angle_max + 0.5 * step, step, dtype=np.float64),
        angle_min,
        angle_max,
    )


def _boundary_lab_sphere(
    source: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Fibonacci-sphere observation points + metadata for balloon sampling.

    Mirrors Boundary Lab's own ``build_fibonacci_sphere_observation_points`` so
    the balloon is identical across solver backends: an origin-centred sphere of
    radius = observation distance, theta measured from +Z, phi in the X-Y plane.
    Returns ``(points_xyz (N,3), theta_polar_rad (N,), phi_azimuth_rad (N,),
    r_distance_m (N,))`` or ``None`` when spherical sampling is disabled.
    """
    if not bool(_first(source, "spherical_sampling_enabled", default=False)):
        return None
    count = int(
        _first(source, "spherical_sampling_points", "balloon_sampling_points", default=6000)
    )
    if count <= 0:
        return None
    distance = float(
        _first(source, "distance", "distance_m", "observation_distance_m", default=2.0)
    )
    axial_offset = float(_first(source, "axial_offset", "axial_offset_m", default=0.0))

    indices = np.arange(count, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - (2.0 * indices + 1.0) / count
    xy_radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    phi = indices * golden_angle
    x = xy_radius * np.cos(phi)
    y = xy_radius * np.sin(phi)
    points = distance * np.vstack([x, y, z])
    points[2, :] += axial_offset
    points_xyz = np.ascontiguousarray(points.T, dtype=np.float64)
    theta_polar = np.arccos(np.clip(z, -1.0, 1.0)).astype(np.float32)
    phi_azimuth = np.mod(np.arctan2(y, x), 2.0 * np.pi).astype(np.float32)
    r_distance = np.full(count, distance, dtype=np.float32)
    return points_xyz, theta_polar, phi_azimuth, r_distance


def _radiator_names(source: Any | None) -> tuple[str, ...]:
    radiators = tuple(_first(source, "radiators", default=()) or ())
    if radiators:
        return tuple(str(_first(radiator, "name", default=f"tag_{_first(radiator, 'tag', default=2)}")) for radiator in radiators)
    return ("throat",)


def _coerce_symmetry_plane(symmetry: Any) -> str | None:
    """Map Boundary Lab symmetry tokens onto native plane names.

    Boundary Lab names the mirrored *axes* ("x" mirrors across X=0, "xy"
    mirrors across both X=0 and Y=0), while the native solver names the
    *plane* ("yz" is the X=0 plane, "xy" is the Z=0 plane). The same token
    "xy" therefore means quarter symmetry here but a z-mirror in SolveConfig
    — do not pass native plane intentions through this adapter.
    """
    mode = str(symmetry or "off").strip().lower()
    if mode in {"", "off", "none"}:
        return None
    if mode == "x":
        return "yz"
    if mode == "y":
        return "xz"
    if mode == "z":
        return "xy"
    if mode == "xy":
        return "yz+xz"
    if mode in {"yz", "xz", "yz+xz"}:
        return mode
    raise BoundaryLabSolverError(f"Unsupported Boundary Lab symmetry mode for Metal: {symmetry!r}")


def _frequency_result_from_log_entry(frequency_hz: float, entry: dict[str, Any]) -> FrequencyResult:
    planes = list(entry.get("observation_planes") or ["horizontal", "vertical"])
    directivity = np.asarray(entry["observation_directivity_db"], dtype=np.float32)
    horizontal = _plane_spl(directivity, planes, "horizontal")
    vertical = _plane_spl(directivity, planes, "vertical")
    pressure = entry.get("observation_pressure_complex")
    pressure_complex = (
        np.asarray(pressure, dtype=np.complex64) if pressure is not None else None
    )
    horizontal_raw = None
    vertical_raw = None
    if pressure_complex is not None:
        horizontal_raw = _pressure_to_spl(_plane_pressure(pressure_complex, planes, "horizontal"))
        vertical_raw = _pressure_to_spl(_plane_pressure(pressure_complex, planes, "vertical"))
    impedance = _impedance_array(entry.get("impedance"))
    native_diagnostics = dict(entry.get("native_diagnostics") or {})
    timings = FrequencySolveTimings(
        assembly_s=float(entry.get("assembly_s", 0.0) or 0.0),
        solve_s=float(entry.get("dense_solve_s", entry.get("solve_s", 0.0)) or 0.0),
        field_s=float(entry.get("field_s", 0.0) or 0.0),
    )
    diagnostics = SolverDiagnostics(
        convergence_info=entry.get("lapack_info"),
        message=str(entry.get("backend", "native_metal")),
    )
    kwargs = {
        "freq_hz": float(frequency_hz),
        "horizontal_spl_norm_db": horizontal,
        "vertical_spl_norm_db": vertical,
        "impedance": impedance,
        "horizontal_spl_db": horizontal_raw,
        "vertical_spl_db": vertical_raw,
        "sphere_spl_norm_db": None,
        "timings": timings,
        "diagnostics": diagnostics,
    }
    result = FrequencyResult(**kwargs)
    _attach_result_extras(
        result,
        observation_pressure_complex=pressure_complex,
        native_diagnostics=native_diagnostics,
    )
    return result


def _frequency_result_from_channel_basis_entry(
    frequency_hz: float,
    entry: dict[str, Any],
    *,
    channel_names: np.ndarray,
    channel_configs: dict[str, Any],
) -> FrequencyResult:
    source_entries = _source_entries_from_callback_entry(entry)
    channels = np.asarray(channel_names)
    if len(source_entries) != channels.size:
        raise BoundaryLabSolverError(
            "Metal multi-source result count does not match Boundary Lab channel count: "
            f"{len(source_entries)} result(s), {channels.size} channel(s)."
        )

    planes = list(entry.get("observation_planes") or ["horizontal", "vertical"])
    horizontal_pressure = []
    vertical_pressure = []
    sphere_pressure: list[np.ndarray] = []
    impedance_rows = []
    native_diagnostics: list[dict[str, Any]] = []
    for source_entry in source_entries:
        pressure = source_entry.get("observation_pressure_complex")
        if pressure is None:
            raise BoundaryLabSolverError("Metal result did not include observation_pressure_complex.")
        pressure_complex = np.asarray(pressure, dtype=np.complex64)
        # Boundary Lab's channel synthesis uses the conjugate time convention
        # from the Metal core. Conjugating the basis keeps SPL invariant while
        # making GUI delay/filter edits phase in the expected direction.
        horizontal_pressure.append(np.conj(_plane_pressure(pressure_complex, planes, "horizontal")))
        vertical_pressure.append(np.conj(_plane_pressure(pressure_complex, planes, "vertical")))
        sphere = source_entry.get("observation_sphere_pressure_complex")
        if sphere is not None:
            # Same conjugate convention as the arc planes so channel synthesis
            # combines the balloon basis consistently with the polar planes.
            sphere_pressure.append(np.conj(np.asarray(sphere, dtype=np.complex64)))
        impedance_rows.append(_impedance_pair(source_entry.get("impedance")))
        native_diagnostics.append(dict(source_entry.get("native_diagnostics") or {}))

    horizontal_matrix = np.vstack(horizontal_pressure).astype(np.complex64, copy=False)
    vertical_matrix = np.vstack(vertical_pressure).astype(np.complex64, copy=False)
    sphere_matrix = (
        np.vstack(sphere_pressure).astype(np.complex64, copy=False)
        if len(sphere_pressure) == channels.size and sphere_pressure
        else None
    )
    angles = np.asarray(entry.get("observation_angles_deg", []), dtype=np.float32)
    if angles.size != horizontal_matrix.shape[1]:
        angles = np.linspace(0.0, 180.0, horizontal_matrix.shape[1], dtype=np.float32)

    synthesized = _synthesize_boundary_lab_channel_basis(
        frequency_hz=float(frequency_hz),
        channel_names=channels,
        channel_configs=channel_configs,
        horizontal_pressure=horizontal_matrix,
        vertical_pressure=vertical_matrix,
        angles_deg=angles,
    )
    first = source_entries[0]
    timings = FrequencySolveTimings(
        assembly_s=sum(float(item.get("assembly_s", 0.0) or 0.0) for item in source_entries),
        solve_s=sum(float(item.get("dense_solve_s", item.get("solve_s", 0.0)) or 0.0) for item in source_entries),
        field_s=sum(float(item.get("field_s", 0.0) or 0.0) for item in source_entries),
    )
    diagnostics = SolverDiagnostics(
        convergence_info=first.get("lapack_info"),
        message=str(first.get("backend", "native_metal")),
    )
    result = FrequencyResult(
        freq_hz=float(frequency_hz),
        horizontal_spl_norm_db=synthesized["horizontal_spl_norm_db"],
        vertical_spl_norm_db=synthesized["vertical_spl_norm_db"],
        impedance=np.vstack(impedance_rows).astype(np.float32, copy=False),
        horizontal_spl_db=synthesized["horizontal_spl_db"],
        vertical_spl_db=synthesized["vertical_spl_db"],
        sphere_spl_norm_db=None,
        channel_names=channels,
        horizontal_pressure=horizontal_matrix,
        vertical_pressure=vertical_matrix,
        sphere_pressure=sphere_matrix,
        timings=timings,
        diagnostics=diagnostics,
    )
    _attach_result_extras(
        result,
        observation_pressure_complex=np.asarray(
            [item.get("observation_pressure_complex") for item in source_entries],
            dtype=np.complex64,
        ),
        native_diagnostics=native_diagnostics,
    )
    return result


def _source_entries_from_callback_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    raw_sources = entry.get("source_results")
    if raw_sources is None:
        return [entry]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise BoundaryLabSolverError("source_results must be a non-empty list.")
    return [dict(item) for item in raw_sources]


def _attach_result_extras(result: FrequencyResult, **values: Any) -> None:
    for name, value in values.items():
        try:
            object.__setattr__(result, name, value)
        except Exception:
            pass


def _synthesize_boundary_lab_channel_basis(
    *,
    frequency_hz: float,
    channel_names: np.ndarray,
    channel_configs: dict[str, Any],
    horizontal_pressure: np.ndarray,
    vertical_pressure: np.ndarray,
    angles_deg: np.ndarray,
) -> dict[str, np.ndarray]:
    names = [str(name) for name in np.asarray(channel_names).tolist()]
    weights = np.asarray(
        [
            _boundary_lab_channel_drive(channel_configs.get(name), frequency_hz)
            for name in names
        ],
        dtype=np.complex64,
    )
    horizontal_summed = np.sum(horizontal_pressure * weights[:, np.newaxis], axis=0)
    vertical_summed = np.sum(vertical_pressure * weights[:, np.newaxis], axis=0)
    horizontal_spl = _pressure_to_spl(horizontal_summed)
    vertical_spl = _pressure_to_spl(vertical_summed)
    on_axis_idx = int(np.argmin(np.abs(np.asarray(angles_deg, dtype=np.float32)))) if angles_deg.size else 0
    on_axis_ref = horizontal_spl[on_axis_idx]
    return {
        "horizontal_spl_db": horizontal_spl.astype(np.float32, copy=False),
        "vertical_spl_db": vertical_spl.astype(np.float32, copy=False),
        "horizontal_spl_norm_db": (horizontal_spl - on_axis_ref).astype(np.float32, copy=False),
        "vertical_spl_norm_db": (vertical_spl - on_axis_ref).astype(np.float32, copy=False),
    }


def _pressure_to_spl(pressure: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (20.0 * np.log10(np.abs(pressure) / REFERENCE_PRESSURE_PA)).astype(np.float32, copy=False)


def _plane_pressure(pressure: np.ndarray, planes: list[str], plane: str) -> np.ndarray:
    if plane in planes:
        return np.asarray(pressure[planes.index(plane)], dtype=np.complex64)
    return np.zeros(pressure.shape[-1], dtype=np.complex64)


def _impedance_pair(value: Any) -> np.ndarray:
    z = complex(0.0 if value is None else value)
    return np.asarray([float(np.real(z)), float(np.imag(z))], dtype=np.float32)


def _plane_spl(spl: np.ndarray, planes: list[str], plane: str) -> np.ndarray:
    if plane in planes:
        return np.asarray(spl[planes.index(plane)], dtype=np.float32)
    return np.zeros(spl.shape[-1], dtype=np.float32)


def _impedance_array(value: Any) -> np.ndarray:
    z = complex(0.0 if value is None else value)
    return np.asarray([[float(np.real(z)), float(np.imag(z))]], dtype=np.float32)

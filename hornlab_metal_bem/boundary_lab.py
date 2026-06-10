from __future__ import annotations

from dataclasses import dataclass, field
import queue
import platform
import threading
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from scipy import signal

from .config import ObservationConfig, SolveConfig, VelocityMode

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
        observation_pressure_complex: np.ndarray | None = None
        native_diagnostics: dict[str, Any] = field(default_factory=dict)
        timings: FrequencySolveTimings = field(default_factory=FrequencySolveTimings)
        diagnostics: SolverDiagnostics | None = None

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
        supports_flat_target_normalization: bool = False
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
            supports_spherical_sampling=False,
            supports_burton_miller=False,
            supports_flat_target_normalization=False,
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
        return SolveMetadata(
            polar_angle_deg=_boundary_lab_angles(cfg).astype(np.float32, copy=False),
            radiator_names=np.asarray(_radiator_names(cfg)),
            sphere_metadata=None,
        )

    def solve_stream(
        self,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[FrequencyResult]:
        from . import solve_frequencies as _solve_frequencies

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

        def should_stop() -> bool:
            return self._stop or (stop_requested is not None and stop_requested())

        def on_frequency_result(index: int, frequency_hz: float, entry: dict[str, Any]) -> bool:
            result_queue.put(_frequency_result_from_log_entry(frequency_hz, entry))
            return not should_stop()

        overrides = {**self.default_overrides, "on_frequency_result": on_frequency_result}
        solve_config, translated_frequencies = solve_config_from_boundary_lab(cfg, **overrides)
        frequencies = frequencies_hz if frequencies_hz is not None else translated_frequencies
        if frequencies is None:
            raise BoundaryLabSolverError("No frequencies available for Metal solve.")

        def run() -> None:
            try:
                _solve_frequencies(mesh_path, frequencies, solve_config)
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
        "mesh_scale": float(
            _first(simulation_config, "mesh_scale", "scale_factor", default=1.0)
        ),
        "air_density": float(_first(simulation_config, "rho", "air_density", default=1.2041)),
        "native_symmetry_plane": _coerce_symmetry_plane(
            _first(simulation_config, "symmetry", default="off")
        ),
        "observation": observation,
        "metal_native_assembly_mode": "corrected",
    }
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


def _radiator_drive(radiator: Any, frequency_hz: float) -> complex:
    return _level_polarity_delay_filter_drive(radiator, frequency_hz)


def _channel_drive(channel: Any, frequency_hz: float) -> complex:
    return _level_polarity_delay_filter_drive(channel, frequency_hz)


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
        "horizontal_spl_db": None,
        "vertical_spl_db": None,
        "sphere_spl_norm_db": None,
        "observation_pressure_complex": pressure_complex,
        "native_diagnostics": native_diagnostics,
        "timings": timings,
        "diagnostics": diagnostics,
    }
    try:
        return FrequencyResult(**kwargs)
    except TypeError:
        observation_pressure_complex = kwargs.pop("observation_pressure_complex")
        native_diagnostics = kwargs.pop("native_diagnostics")
        result = FrequencyResult(**kwargs)
        for name, value in (
            ("observation_pressure_complex", observation_pressure_complex),
            ("native_diagnostics", native_diagnostics),
        ):
            try:
                object.__setattr__(result, name, value)
            except Exception:
                pass
        return result


def _plane_spl(spl: np.ndarray, planes: list[str], plane: str) -> np.ndarray:
    if plane in planes:
        return np.asarray(spl[planes.index(plane)], dtype=np.float32)
    return np.zeros(spl.shape[-1], dtype=np.float32)


def _impedance_array(value: Any) -> np.ndarray:
    z = complex(0.0 if value is None else value)
    return np.asarray([[float(np.real(z)), float(np.imag(z))]], dtype=np.float32)

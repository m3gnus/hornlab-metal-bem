from __future__ import annotations

from dataclasses import dataclass, field
import platform
from typing import Any, Iterable

import numpy as np

from .config import ObservationConfig, SolveConfig


BACKEND_ID = "hornlab_metal"
__all__ = [
    "BACKEND_ID",
    "BoundaryLabAdapterError",
    "BoundaryLabBackend",
    "BoundaryLabSession",
    "create_backend",
    "is_apple_silicon",
    "solve_config_from_boundary_lab",
]


class BoundaryLabAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class BoundaryLabBackend:
    """Duck-typed Boundary Lab solver backend."""

    backend_id: str = BACKEND_ID
    label: str = "HornLab Metal BEM"
    default_overrides: dict[str, Any] = field(default_factory=dict)

    def create_session(
        self,
        simulation_config: Any | None = None,
    ) -> "BoundaryLabSession":
        return BoundaryLabSession(
            simulation_config=simulation_config,
            default_overrides=dict(self.default_overrides),
        )

    def supports(self, simulation_config: Any | None = None) -> bool:
        return is_apple_silicon()


@dataclass
class BoundaryLabSession:
    """Small session wrapper matching Boundary Lab's expected lifecycle shape."""

    simulation_config: Any | None = None
    default_overrides: dict[str, Any] = field(default_factory=dict)

    def solve(
        self,
        mesh: Any,
        simulation_config: Any | None = None,
        **overrides: Any,
    ):
        from . import solve as _solve
        from . import solve_frequencies as _solve_frequencies

        config_source = (
            self.simulation_config if simulation_config is None else simulation_config
        )
        merged_overrides = {**self.default_overrides, **overrides}
        solve_config, frequencies_hz = solve_config_from_boundary_lab(
            config_source,
            **merged_overrides,
        )
        if frequencies_hz is not None:
            return _solve_frequencies(mesh, frequencies_hz, solve_config)
        return _solve(mesh, solve_config)

    def close(self) -> None:
        return None

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
        angle_min_deg=float(
            _first(
                simulation_config,
                "angle_min_deg",
                "min_angle_deg",
                default=0.0,
            )
        ),
        angle_max_deg=float(
            _first(
                simulation_config,
                "angle_max_deg",
                "max_angle_deg",
                default=180.0,
            )
        ),
        angle_count=int(
            _first(simulation_config, "angle_count", "n_angles", default=37)
        ),
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
                "freq_min_hz",
                "frequency_min_hz",
                "min_frequency_hz",
                default=500.0,
            )
        ),
        "freq_max_hz": float(
            _first(
                simulation_config,
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
        "mesh_scale": float(_first(simulation_config, "mesh_scale", default=1.0)),
        "observation": observation,
        "assembly_backend": "metal",
        "experimental_metal_backend": True,
        "metal_backend_fallback": "error",
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
    raise BoundaryLabAdapterError("frequencies_hz must be a number or sequence")


def _coerce_velocity_sources(source: Any | None) -> dict[int, float]:
    value = _first(source, "velocity_sources", default=None)
    if value is not None:
        return {int(k): float(v) for k, v in dict(value).items()}
    source_tag = int(_first(source, "source_tag", "driver_tag", default=2))
    source_weight = float(
        _first(source, "source_weight", "velocity_weight", default=1.0)
    )
    return {source_tag: source_weight}

"""Non-routing Python adapter for the experimental native Metal helper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .native import (
    MetalNativeRuntimeConfig,
    MetalNativeRuntimeStatus,
    MetalNativeStandardSession,
    discover_native_runtime,
)


@dataclass(frozen=True)
class DenseBieSystem:
    """Dense standard-Neumann assembly artifact from the native helper."""

    session_id: str
    frequency_hz: float
    matrix_real_f32: Any
    matrix_imag_f32: Any
    rhs_real_f32: Any
    rhs_imag_f32: Any
    matrix_shape: tuple[int, int]
    rhs_shape: tuple[int]
    matrix_layout: str


class MetalBemBackend:
    """Factory for package-owned native Metal validation contexts.

    This adapter deliberately does not participate in production backend
    resolution. Callers must instantiate it directly while the Metal path is
    still behind parity and benchmark gates.
    """

    def __init__(self, runtime_config: MetalNativeRuntimeConfig | None = None) -> None:
        self.runtime_config = runtime_config

    def is_available(self, *, run_smoke_test: bool = False) -> MetalNativeRuntimeStatus:
        return discover_native_runtime(
            self.runtime_config,
            run_smoke_test=run_smoke_test,
        )

    def create_context(
        self,
        *,
        grid: Any | None = None,
        physical_tags: Any | None = None,
        p1_space: Any | None = None,
        dp0_space: Any | None = None,
        geometry_buffers: Any | None = None,
        work_dir: Any | None = None,
        session_id: str | None = None,
        keep_artifacts: bool = False,
    ) -> "MetalBemContext":
        session = MetalNativeStandardSession.create_session(
            grid=grid,
            physical_tags=physical_tags,
            p1_space=p1_space,
            dp0_space=dp0_space,
            geometry_buffers=geometry_buffers,
            runtime_config=self.runtime_config,
            work_dir=work_dir,
            session_id=session_id,
            keep_artifacts=keep_artifacts,
        )
        return MetalBemContext(session)


class MetalBemContext:
    """Session-scoped native Metal validation context."""

    def __init__(self, session: MetalNativeStandardSession) -> None:
        self.session = session

    @property
    def session_id(self) -> str:
        return self.session.info.session_id

    def validate_contract(self) -> dict[str, Any]:
        return self.session.validate_contract()

    def assemble_standard_neumann(
        self,
        frequency_hz: float,
        k_real: float,
        neumann_dp0: NDArray[Any],
        *,
        operation_id: str | None = None,
    ) -> DenseBieSystem:
        result = self.session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.asarray(neumann_dp0),
            operation_id=operation_id,
        )
        return DenseBieSystem(
            session_id=result.session_id,
            frequency_hz=result.frequency_hz,
            matrix_real_f32=result.matrix_real_f32,
            matrix_imag_f32=result.matrix_imag_f32,
            rhs_real_f32=result.rhs_real_f32,
            rhs_imag_f32=result.rhs_imag_f32,
            matrix_shape=result.matrix_shape,
            rhs_shape=result.rhs_shape,
            matrix_layout=result.matrix_layout,
        )

    def evaluate_field_batch(
        self,
        frequency_hz: float,
        k_real: float,
        pressure_p1: NDArray[Any],
        neumann_dp0: NDArray[Any],
        observation_points: NDArray[Any],
        *,
        batch_id: str = "batch",
        operation_id: str | None = None,
    ) -> Any:
        return self.session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            pressure_p1,
            neumann_dp0,
            observation_points,
            batch_id=batch_id,
            operation_id=operation_id,
        )

    def close(self) -> None:
        self.session.close()

    cleanup = close

    def __enter__(self) -> "MetalBemContext":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

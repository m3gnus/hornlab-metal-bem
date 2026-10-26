"""BEAT worker protocol v1 and system_result v2 over the native Metal solver.

``MetalEngineWorker`` has the client surface of BEAT's ``EngineWorker``
(``worker_info``, ``ensure_started``, ``submit``, ``terminate``) and emits the
same event stream: ``ready`` on start, then per submission zero or more
``status`` and ``result`` events and exactly one terminal ``completed``,
``cancelled`` or ``failed`` event. Result events carry BEAT's typed
``system_result`` v2 records, so a client written against BEAT's worker can
switch to this package by constructing a different worker object.

The solve runs in-process through :func:`solve_compiled`; there is no Julia
process. Results are emitted once the whole sweep has been solved, because
every motion-basis column shares each frequency's factorization.
"""

from __future__ import annotations

import base64
import copy
import json
import queue
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .compiled import (
    NEGATIVE_TIME,
    POSITIVE_TIME,
    CompiledResult,
    _package_version,
    solve_compiled,
)

PROTOCOL = {"name": "beat-worker", "version": 1}
SYSTEM_RESULT_VERSION = 2

# Quantities BEAT always emits in complex128: they come out of the float64
# host network/force integration rather than the BEM precision.
_FLOAT64_QUANTITIES = {
    "radiation_impedance_matrix",
    "diaphragm_velocity",
    "voice_coil_current",
}


def _metal_availability() -> dict[str, Any]:
    try:
        from .metal import discover_native_runtime

        status = discover_native_runtime(run_smoke_test=False)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - platform specific
        return {"available": False, "reason": str(exc)}
    if status.available:
        return {"available": True}
    return {"available": False, "reason": "; ".join(status.unavailable_reasons)}


def worker_announcement(
    availability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``ready`` event, accepted by BEAT's ``validate_worker_ready``.

    ``engine.name`` is the protocol's required token; ``implementation``
    identifies this package so results are never mistaken for BEAT's own.
    """
    version = _package_version()
    return {
        "type": "ready",
        "protocol": dict(PROTOCOL),
        "engine": {"name": "BEAT Engine", "version": version},
        "implementation": {"name": "hornlab-metal-bem", "version": version},
        "contracts": {
            "system_request": [1],
            "compiled_system": [1, 2, 3],
            "system_result": [SYSTEM_RESULT_VERSION],
            "field_array": [1],
            "mesh_data": [1],
        },
        "operations": ["solve"],
        "precisions": ["float32"],
        "solve_kinds": ["exterior_bem"],
        "exterior_source_profiles": ["uniform_normal", "rigid_translation"],
        "exterior_component_kinds": [
            "ideal_velocity_source",
            "electrodynamic_transducer",
        ],
        "optional_output_quantities": ["radiation_impedance_matrix"],
        "phasor_conventions": [NEGATIVE_TIME, POSITIVE_TIME],
        "cancellation": "marker_file",
        "request_transports": ["file", "inline_json"],
        "backends": {
            "metal": dict(availability if availability is not None else _metal_availability())
        },
    }


def _complex_array_wire(values: np.ndarray, dtype: str) -> dict[str, Any]:
    array = np.ascontiguousarray(values, dtype="<c8" if dtype == "complex64" else "<c16")
    return {
        "encoding": "base64",
        "dtype": dtype,
        "shape": list(array.shape),
        "order": "C",
        "byte_order": "little",
        "content_base64": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _at(value: Any, index: int) -> Any:
    return value[index] if isinstance(value, list) else value


def result_records(result: CompiledResult) -> list[dict[str, Any]]:
    """One BEAT ``system_result`` v2 record per requested frequency.

    Quantity order, IDs, units, axes, dtypes and metadata follow BEAT's
    exterior solver; frequency-dependent diagnostics are split per record.
    """
    precision = result.diagnostics.get("precision", "float32")
    scalar = np.float32 if precision == "float32" else np.float64
    records = []
    for i, frequency in enumerate(result.frequencies_hz):
        quantities = []
        for q in result.quantities.values():
            dtype = (
                "complex128"
                if q.quantity in _FLOAT64_QUANTITIES or precision == "float64"
                else "complex64"
            )
            metadata = {
                key: (_at(value, i) if key in ("reciprocity_max_rel", "passivity_min_eig") else value)
                for key, value in q.metadata.items()
            }
            quantities.append(
                {
                    "id": q.id,
                    "quantity": q.quantity,
                    "unit": q.unit,
                    "target_id": q.target_id,
                    "axes": list(q.axes[1:]),
                    "values": _complex_array_wire(q.values[i], dtype),
                    "metadata": copy.deepcopy(metadata),
                }
            )
        diagnostics = copy.deepcopy(result.diagnostics)
        wall = diagnostics.pop("wall_time_s", 0.0)
        # The sweep is solved as one batch; its wall time is reported once,
        # like BEAT's one-off mesh setup time.
        diagnostics["timings"] = {"solve_s": wall if i == 0 else 0.0}
        network = diagnostics.get("exterior_lumped_network")
        if network is not None:
            network["residual_max_abs"] = _at(network["residual_max_abs"], i)
        records.append(
            {
                "schema_version": SYSTEM_RESULT_VERSION,
                "freq_hz": float(scalar(frequency)),
                "excitation_port_ids": list(result.excitation_port_ids),
                "quantities": quantities,
                "diagnostics": diagnostics,
            }
        )
    return records


def solve_events(
    request: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
    cancel_cb: Callable[[], None] | None = None,
    native_formulation: str = "standard",
    complex_k_shift: float = 0.005,
    native_progress_cb: Callable[[int, float, dict], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Run one solve and yield BEAT worker events, ending in a terminal one.

    Request errors and solver failures become ``failed`` events with BEAT's
    ``worker_request_failed`` code; cancellation becomes ``cancelled``.
    """
    yield {"type": "status", "message": "Solving on hornlab-metal-bem native Metal"}
    try:
        native_options = {}
        if native_formulation != "standard" or complex_k_shift != 0.005:
            native_options = {"native_formulation": native_formulation, "complex_k_shift": complex_k_shift}
        if native_progress_cb is not None:
            native_options["native_progress_cb"] = native_progress_cb
        result = solve_compiled(dict(request), base_dir=base_dir, cancel_cb=cancel_cb, **native_options)
        records = result_records(result)
    except InterruptedError:
        yield {"type": "cancelled", "solved_count": 0}
        return
    except Exception as exc:  # noqa: BLE001 - every job error is a failed event, as in BEAT
        yield {
            "type": "failed",
            "code": "worker_request_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return
    for record in records:
        yield {"type": "result", "result": record}
    yield {"type": "completed", "solved_count": len(records)}


_DONE = object()


class MetalSubmission(Iterator[dict[str, Any]]):
    """One job's event stream, with its own cancellation state.

    The solve runs on a thread owned by the submission, so ``close()`` may be
    called from any thread, including while another thread is blocked in
    ``next()``. ``close()`` requests cancellation, which stops the native
    helper at its next frequency checkpoint, and returns only after the job
    has stopped: nothing the submission owns is still running, and no further
    event is published. Repeated and late ``close()`` calls are no-ops and
    never touch another submission.
    """

    def __init__(self, worker: MetalEngineWorker, run, status_callback) -> None:
        self._worker = worker
        self._run = run
        self._status_callback = status_callback
        self._events: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._produce, name="hornlab-metal-bem-solve", daemon=True
        )

    def _start(self) -> None:
        self._thread.start()

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise InterruptedError("Compiled BEAT submission closed")

    def _produce(self) -> None:
        try:
            for event in self._run(self._check_cancel):
                if self._cancel.is_set():
                    break
                self._events.put(event)
        finally:
            self._events.put(_DONE)
            self._worker._release(self)

    @property
    def running(self) -> bool:
        """Whether this submission's solve thread is still alive."""
        return self._thread.is_alive()

    def __iter__(self) -> MetalSubmission:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise StopIteration
        event = self._events.get()
        if event is _DONE or self._closed:
            self._events.put(_DONE)  # wake any other blocked reader
            raise StopIteration
        if event["type"] == "status" and self._status_callback is not None:
            self._status_callback(event["message"])
        return event

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True  # cancellation requested, not shutdown finished
                self._cancel.set()
                self._events.put(_DONE)
        # Every external closer must wait, including a concurrent/repeated
        # close while another caller is already joining this thread.
        if self._thread.ident is not None and threading.current_thread() is not self._thread:
            self._thread.join()


class MetalEngineWorker:
    """In-process stand-in for ``beat_engine.EngineWorker``.

    ``submit`` takes a request path (relative mesh files resolve against
    ``base_dir``, the working directory by default, as in BEAT) or an inline
    request mapping, and returns a :class:`MetalSubmission`. One submission
    runs at a time: a new ``submit`` waits until the previous job has finished
    or been closed. Drain or close every submission. ``terminate()`` closes
    the active submission (waiting for it to stop) and forgets the
    announcement. ``bem_field`` is not offered.
    """

    def __init__(
        self,
        *,
        base_dir: str | Path | None = None,
        availability: Mapping[str, Any] | None = None,
        native_formulation: str = "standard",
        complex_k_shift: float = 0.005,
        native_progress_cb: Callable[[int, float, dict], None] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.native_formulation = native_formulation
        self.complex_k_shift = complex_k_shift
        self.native_progress_cb = native_progress_cb
        self._availability = availability
        self._info: dict[str, Any] | None = None
        self._available = threading.Condition()
        self._active: MetalSubmission | None = None
        self._terminating = False

    @property
    def worker_info(self) -> dict[str, Any] | None:
        """A copy of the ready announcement, if started."""
        with self._available:
            return copy.deepcopy(self._info)

    def ensure_started(
        self, *, status_callback: Callable[[str], None] | None = None
    ) -> None:
        with self._available:
            while self._terminating:
                self._available.wait()
            started = self._info is None
            if started:
                self._info = worker_announcement(self._availability)
        if started and status_callback is not None:
            status_callback("hornlab-metal-bem worker ready")

    def submit(
        self,
        request_path: str | Path | Mapping[str, Any],
        *,
        status_callback: Callable[[str], None] | None = None,
        operation: str = "solve",
    ) -> MetalSubmission:
        base = self.base_dir if self.base_dir is not None else Path.cwd()

        def run(cancel_cb):
            if operation != "solve":
                yield {
                    "type": "failed",
                    "code": "worker_request_failed",
                    "error": f"operation {operation!r} is not supported by hornlab-metal-bem",
                }
                return
            if isinstance(request_path, Mapping):
                request = dict(request_path)
            else:
                try:
                    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    yield {
                        "type": "failed",
                        "code": "worker_request_failed",
                        "error": f"cannot read request: {exc}",
                    }
                    return
            yield from solve_events(request, base_dir=base, cancel_cb=cancel_cb,
                                    native_formulation=self.native_formulation,
                                    complex_k_shift=self.complex_k_shift,
                                    native_progress_cb=self.native_progress_cb)

        submission = MetalSubmission(self, run, status_callback)
        with self._available:
            while self._active is not None or self._terminating:
                self._available.wait()
            self.ensure_started()
            self._active = submission
            # Publish and start atomically with respect to terminate().
            submission._start()
        return submission

    def _release(self, submission: MetalSubmission) -> None:
        with self._available:
            if self._active is submission:
                self._active = None
                self._available.notify_all()

    def terminate(self) -> None:
        with self._available:
            while self._terminating:
                self._available.wait()
            self._terminating = True
            active = self._active
        try:
            if active is not None:
                active.close()
        finally:
            with self._available:
                self._info = None
                self._terminating = False
                self._available.notify_all()

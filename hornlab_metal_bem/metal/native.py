"""Discovery and execution for the native Swift/Metal helper."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


_DEFAULT_SWIFT_ENV_VAR = "HORNLAB_METAL_BEM_SWIFT"
_DEFAULT_HELPER_ENV_VAR = "HORNLAB_METAL_BEM_NATIVE"
_DEFAULT_NATIVE_ENTRYPOINT = "HornlabMetalBemNative.swift"
_DEFAULT_NATIVE_PACKAGE_DIR = "native_helper"
_DEFAULT_NATIVE_BINARY_NAME = "HornlabMetalBemNative"
_DEFAULT_SMOKE_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class MetalNativeRuntimeConfig:
    """Discovery inputs for the package-owned Swift/Metal helper."""

    helper_executable: str | None = None
    helper_env_var: str = _DEFAULT_HELPER_ENV_VAR
    swift_executable: str | None = None
    swift_env_var: str = _DEFAULT_SWIFT_ENV_VAR
    backend_dir: Path | None = None
    native_entrypoint: str = _DEFAULT_NATIVE_ENTRYPOINT
    native_package_dir: str = _DEFAULT_NATIVE_PACKAGE_DIR
    native_binary_name: str = _DEFAULT_NATIVE_BINARY_NAME
    smoke_timeout_s: float = _DEFAULT_SMOKE_TIMEOUT_S

    # Wall-clock limit for one helper operation (assembly/solve/field batch).
    # None means unbounded, matching dense solves whose runtime scales with
    # mesh size; set a limit to turn a wedged GPU/helper hang into an error.
    operation_timeout_s: float | None = None

    @property
    def resolved_backend_dir(self) -> Path:
        if self.backend_dir is not None:
            return Path(self.backend_dir)
        return Path(__file__).resolve().parent


@dataclass(frozen=True)
class MetalNativeRuntimeStatus:
    """Result of native Swift/Metal helper discovery."""

    available: bool
    platform_system: str
    platform_machine: str
    is_macos: bool
    is_apple_silicon: bool
    swift_path: str | None
    swift_source: str | None
    helper_executable_path: Path | None
    helper_source: str | None
    backend_dir: Path
    native_entrypoint: Path
    native_package_dir: Path
    helper_assets_present: bool
    smoke_test_ran: bool
    smoke_test_ok: bool
    smoke_test_error: str | None
    reasons: tuple[str, ...]

    @property
    def unavailable_reasons(self) -> tuple[str, ...]:
        return self.reasons


@dataclass(frozen=True)
class MetalNativeSessionInfo:
    """Created-session metadata for the Swift/Metal native helper path."""

    session_id: str
    work_dir: Path
    manifest_path: Path
    geometry_dir: Path
    runtime_status: MetalNativeRuntimeStatus


def discover_native_runtime(
    config: MetalNativeRuntimeConfig | None = None,
    *,
    run_smoke_test: bool = False,
) -> MetalNativeRuntimeStatus:
    """Inspect native Swift/Metal helper prerequisites."""
    if config is None:
        config = MetalNativeRuntimeConfig()

    system = platform.system()
    machine = platform.machine()
    normalized_machine = machine.lower()
    is_macos = system == "Darwin"
    is_apple_silicon = is_macos and normalized_machine in {"arm64", "aarch64"}

    swift_path, swift_source = _find_swift(config)
    backend_dir = config.resolved_backend_dir
    native_entrypoint = backend_dir / config.native_entrypoint
    native_package_dir = backend_dir / config.native_package_dir
    helper_executable_path, helper_source = _find_helper_executable(
        config,
        native_package_dir,
    )
    helper_assets_present = (
        helper_executable_path is not None
        or native_entrypoint.is_file()
        or (native_package_dir / "Package.swift").is_file()
    )

    reasons: list[str] = []
    if not is_macos:
        reasons.append("Native Metal helper requires macOS.")
    elif not is_apple_silicon:
        reasons.append("Native Metal helper requires Apple Silicon macOS.")

    if helper_executable_path is None and swift_path is None:
        reasons.append(
            "Native helper executable not found and Swift executable not found "
            f"via {config.swift_env_var} or PATH."
        )

    if not helper_assets_present:
        reasons.append(
            "Packaged Swift/Metal helper is not installed under "
            f"{backend_dir}."
        )

    smoke_test_ran = False
    smoke_test_ok = False
    smoke_test_error: str | None = None
    if run_smoke_test and not reasons and (
        helper_executable_path is not None or swift_path is not None
    ):
        smoke_test_ran = True
        smoke_test_ok, smoke_test_error = _run_native_smoke_test(
            helper_executable_path,
            swift_path,
            native_entrypoint,
            timeout_s=config.smoke_timeout_s,
        )
        if not smoke_test_ok:
            reasons.append(
                "Packaged Swift/Metal helper smoke test failed"
                + (f": {smoke_test_error}" if smoke_test_error else ".")
            )

    return MetalNativeRuntimeStatus(
        available=not reasons,
        platform_system=system,
        platform_machine=machine,
        is_macos=is_macos,
        is_apple_silicon=is_apple_silicon,
        swift_path=swift_path,
        swift_source=swift_source,
        helper_executable_path=helper_executable_path,
        helper_source=helper_source,
        backend_dir=backend_dir,
        native_entrypoint=native_entrypoint,
        native_package_dir=native_package_dir,
        helper_assets_present=helper_assets_present,
        smoke_test_ran=smoke_test_ran,
        smoke_test_ok=smoke_test_ok,
        smoke_test_error=smoke_test_error,
        reasons=tuple(reasons),
    )


def assert_native_runtime_available(
    config: MetalNativeRuntimeConfig | None = None,
    *,
    run_smoke_test: bool = False,
) -> MetalNativeRuntimeStatus:
    """Return native helper discovery status or raise with clear blockers."""
    status = discover_native_runtime(config, run_smoke_test=run_smoke_test)
    if status.available:
        return status

    raise RuntimeError(
        "Swift/Metal native helper is unavailable: "
        + "; ".join(status.unavailable_reasons)
    )


def validate_session_with_native_helper(
    session_manifest_path: Path,
    result_path: Path,
    config: MetalNativeRuntimeConfig | None = None,
) -> dict[str, Any]:
    """Run the native helper's contract validator for one session manifest."""
    status = assert_native_runtime_available(config, run_smoke_test=True)
    command = _native_helper_command(
        status,
        "validate_session",
        str(session_manifest_path),
        str(result_path),
    )
    try:
        result = subprocess.run(
            command,
            cwd=status.backend_dir,
            check=False,
            capture_output=True,
            text=True,
            timeout=(config or MetalNativeRuntimeConfig()).smoke_timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Failed to launch Swift/Metal native helper: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        message = stderr or stdout or f"Swift helper exited with {result.returncode}"
        raise RuntimeError(f"Swift/Metal native helper failed: {message}")
    if not Path(result_path).is_file():
        raise RuntimeError(f"Swift/Metal native helper did not write {result_path}")
    return json.loads(Path(result_path).read_text(encoding="utf-8"))


class MetalNativeStandardSession:
    """Session wrapper for native helper contract validation."""

    def __init__(
        self,
        info: MetalNativeSessionInfo,
        geometry_payload: Any,
        *,
        owns_work_dir: bool,
        runtime_config: MetalNativeRuntimeConfig | None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.info = info
        self.geometry_payload = geometry_payload
        self._owns_work_dir = owns_work_dir
        self._runtime_config = runtime_config
        self._extra_env = dict(extra_env) if extra_env else None
        self._closed = False

    @classmethod
    def create_session(
        cls,
        grid: Any | None = None,
        physical_tags: Any | None = None,
        p1_space: Any | None = None,
        dp0_space: Any | None = None,
        *,
        geometry_buffers: Any | None = None,
        runtime_config: MetalNativeRuntimeConfig | None = None,
        work_dir: Path | None = None,
        session_id: str | None = None,
        keep_artifacts: bool = False,
        regular_triangle_order: int = 4,
        duffy_1d_order: int = 4,
        precision: str = "complex64",
        symmetry_plane: str | None = None,
        runtime_status: MetalNativeRuntimeStatus | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> "MetalNativeStandardSession":
        """Create a Python-written session after native helper discovery.

        Pass an already-validated ``runtime_status`` to skip the discovery
        smoke test (a helper subprocess) when the caller just ran one.
        """
        from .geometry import build_metal_geometry_buffers
        from .geometry import validate_native_symmetry_plane
        from .session import GeometryPayload, write_geometry_buffers, write_json_manifest

        # The native helper currently hardcodes its quadrature rules; these
        # manifest fields are recorded but not yet honored, so reject silent
        # accuracy expectations until the helper reads them.
        if regular_triangle_order != 4 or duffy_1d_order != 4:
            raise ValueError(
                "regular_triangle_order and duffy_1d_order are fixed at 4; "
                "the native helper does not yet honor quadrature overrides"
            )

        status = runtime_status
        if status is None or not status.available:
            status = discover_native_runtime(runtime_config, run_smoke_test=True)
        if not status.available:
            raise RuntimeError(
                "Swift/Metal native helper is unavailable for session creation: "
                + "; ".join(status.unavailable_reasons)
            )

        if geometry_buffers is None:
            if grid is None or physical_tags is None or p1_space is None:
                raise ValueError(
                    "grid, physical_tags, and p1_space are required when "
                    "geometry_buffers is not provided"
                )
            geometry_buffers = build_metal_geometry_buffers(
                grid,
                physical_tags,
                p1_space,
                dp0_space,
            )
        symmetry_plane = validate_native_symmetry_plane(
            geometry_buffers,
            symmetry_plane,
        )

        session_id = session_id or f"native-metal-{uuid4().hex[:12]}"
        owns_work_dir = work_dir is None and not keep_artifacts
        created_temp_dir = work_dir is None
        root = (
            Path(tempfile.mkdtemp(prefix="hornlab-native-metal-session-"))
            if work_dir is None
            else Path(work_dir)
        )
        try:
            geometry_dir = root / "geometry"
            mesh = write_geometry_buffers(
                geometry_buffers,
                geometry_dir,
                relative_to=root,
            )
            payload = GeometryPayload(
                session_id=session_id,
                mesh=mesh,
                p1_dof_count=geometry_buffers.p1_dof_count,
                dp0_dof_count=geometry_buffers.dp0_dof_count,
                regular_triangle_order=regular_triangle_order,
                duffy_1d_order=duffy_1d_order,
                precision=precision,
                symmetry_plane=symmetry_plane,
            )
            manifest_path = write_json_manifest(payload, root / "session.json")
        except BaseException:
            if created_temp_dir:
                shutil.rmtree(root, ignore_errors=True)
            raise
        info = MetalNativeSessionInfo(
            session_id=session_id,
            work_dir=root,
            manifest_path=manifest_path,
            geometry_dir=geometry_dir,
            runtime_status=status,
        )
        return cls(
            info,
            payload,
            owns_work_dir=owns_work_dir,
            runtime_config=runtime_config,
            extra_env=extra_env,
        )

    def validate_contract(self, *, result_name: str = "native-result.json") -> dict[str, Any]:
        """Run the Swift helper's session contract validator."""
        self._ensure_open()
        return validate_session_with_native_helper(
            self.info.manifest_path,
            self.info.work_dir / result_name,
            self._runtime_config,
        )

    def assemble_standard_neumann(
        self,
        frequency_hz: float,
        k_real: float,
        neumann_dp0: NDArray[Any],
        *,
        operation_id: str | None = None,
    ) -> Any:
        """Assemble a standard-Neumann system through the native helper.

        This remains an experimental helper route for promotion validation. It
        is not wired into production solve routing.
        """
        from .session import (
            AssemblyPayload,
            BinaryArrayDescriptor,
            DenseAssemblyResult,
            read_json_manifest,
            write_json_manifest,
        )

        self._ensure_open()
        neumann = _require_complex_vector(
            "neumann_dp0",
            neumann_dp0,
            self.geometry_payload.dp0_dof_count,
        )
        op_id = operation_id or _operation_id("native-assembly", frequency_hz)
        op_dir = self.info.work_dir / op_id
        inputs_dir = op_dir / "inputs"
        outputs_dir = op_dir / "outputs"
        neumann_desc = _write_complex_vector(
            neumann,
            inputs_dir / "neumann",
            relative_to=self.info.work_dir,
        )
        n = self.geometry_payload.p1_dof_count
        outputs = {
            "A_real_f32": BinaryArrayDescriptor(
                path=(outputs_dir / "A_re_f32.bin").relative_to(
                    self.info.work_dir
                ).as_posix(),
                shape=(n, n),
                dtype="float32",
            ),
            "A_imag_f32": BinaryArrayDescriptor(
                path=(outputs_dir / "A_im_f32.bin").relative_to(
                    self.info.work_dir
                ).as_posix(),
                shape=(n, n),
                dtype="float32",
            ),
            "rhs_real_f32": BinaryArrayDescriptor(
                path=(outputs_dir / "rhs_re_f32.bin").relative_to(
                    self.info.work_dir
                ).as_posix(),
                shape=(n,),
                dtype="float32",
            ),
            "rhs_imag_f32": BinaryArrayDescriptor(
                path=(outputs_dir / "rhs_im_f32.bin").relative_to(
                    self.info.work_dir
                ).as_posix(),
                shape=(n,),
                dtype="float32",
            ),
        }
        payload = AssemblyPayload(
            session_id=self.info.session_id,
            frequency_hz=frequency_hz,
            k_real_f32=float(np.float32(k_real)),
            neumann_dp0=neumann_desc,
            outputs=outputs,
        )
        payload_path = write_json_manifest(payload, op_dir / "assembly.json")
        result_path = op_dir / "assembly-result.json"
        self._run_native_helper(
            "assemble_standard_neumann",
            payload_path=payload_path,
            result_path=result_path,
        )
        result = read_json_manifest(result_path)
        return DenseAssemblyResult(
            session_id=str(result["session_id"]),
            frequency_hz=float(result["frequency_hz"]),
            matrix_real_f32=self.info.work_dir / result["matrix_real_f32"],
            matrix_imag_f32=self.info.work_dir / result["matrix_imag_f32"],
            rhs_real_f32=self.info.work_dir / result["rhs_real_f32"],
            rhs_imag_f32=self.info.work_dir / result["rhs_imag_f32"],
            matrix_shape=tuple(int(v) for v in result["matrix_shape"]),
            rhs_shape=tuple(int(v) for v in result["rhs_shape"]),
            matrix_layout=str(result["matrix_layout"]),
        )

    def assemble_standard_neumann_batch(
        self,
        frequency_hz: NDArray[Any],
        k_real: NDArray[Any],
        neumann_dp0: NDArray[Any],
        *,
        operation_id: str = "assembly-batch",
    ) -> list[Any]:
        """Assemble multiple standard-Neumann systems in one native helper run."""
        from .session import (
            BatchAssemblyPayload,
            BinaryArrayDescriptor,
            DenseAssemblyResult,
            read_json_manifest,
            write_json_manifest,
        )

        self._ensure_open()
        frequencies = np.asarray(frequency_hz, dtype=np.float64)
        k_values = np.asarray(k_real, dtype=np.float32)
        neumann_values = np.asarray(neumann_dp0)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequency_hz must be a non-empty 1D array")
        if k_values.shape != frequencies.shape:
            raise ValueError("k_real must have the same shape as frequency_hz")
        if neumann_values.shape != (
            frequencies.size,
            self.geometry_payload.dp0_dof_count,
        ):
            raise ValueError(
                "neumann_dp0 must have shape "
                f"{(frequencies.size, self.geometry_payload.dp0_dof_count)}, "
                f"got {neumann_values.shape}"
            )
        if not np.issubdtype(neumann_values.dtype, np.complexfloating):
            raise ValueError("neumann_dp0 must be complex")
        if not np.all(np.isfinite(neumann_values)):
            raise ValueError("neumann_dp0 must contain only finite values")

        op_dir = self.info.work_dir / operation_id
        inputs_dir = op_dir / "inputs"
        outputs_root = op_dir / "outputs"
        n = self.geometry_payload.p1_dof_count
        cases: list[dict[str, Any]] = []
        for idx, (freq, kval) in enumerate(zip(frequencies, k_values)):
            case_id = f"case-{idx:04d}-{float(freq):.6g}hz"
            case_input = inputs_dir / case_id
            case_output = outputs_root / case_id
            neumann = _write_complex_vector(
                np.ascontiguousarray(neumann_values[idx], dtype=np.complex64),
                case_input / "neumann",
                relative_to=self.info.work_dir,
            )
            outputs = {
                "matrix_layout": "row_major_c",
                "A_real_f32": BinaryArrayDescriptor(
                    path=(case_output / "A_re_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n, n),
                    dtype="float32",
                ).to_manifest(),
                "A_imag_f32": BinaryArrayDescriptor(
                    path=(case_output / "A_im_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n, n),
                    dtype="float32",
                ).to_manifest(),
                "rhs_real_f32": BinaryArrayDescriptor(
                    path=(case_output / "rhs_re_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n,),
                    dtype="float32",
                ).to_manifest(),
                "rhs_imag_f32": BinaryArrayDescriptor(
                    path=(case_output / "rhs_im_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n,),
                    dtype="float32",
                ).to_manifest(),
            }
            cases.append(
                {
                    "case_id": case_id,
                    "frequency_hz": float(freq),
                    "k_real_f32": float(np.float32(kval)),
                    "neumann_dp0": {
                        key: descriptor.to_manifest()
                        for key, descriptor in neumann.items()
                    },
                    "outputs": outputs,
                }
            )

        payload = BatchAssemblyPayload(
            session_id=self.info.session_id,
            cases=tuple(cases),
        )
        payload_path = write_json_manifest(payload, op_dir / "assembly-batch.json")
        result_path = op_dir / "assembly-batch-result.json"
        self._run_native_helper(
            "assemble_standard_neumann_batch",
            payload_path=payload_path,
            result_path=result_path,
        )
        result = read_json_manifest(result_path)
        case_results = _case_results_from_manifest(
            result,
            expected_count=int(frequencies.size),
            op="assemble_standard_neumann_batch",
        )
        systems = []
        for case_result in case_results:
            systems.append(
                DenseAssemblyResult(
                    session_id=str(case_result["session_id"]),
                    frequency_hz=float(case_result["frequency_hz"]),
                    matrix_real_f32=self.info.work_dir
                    / case_result["matrix_real_f32"],
                    matrix_imag_f32=self.info.work_dir
                    / case_result["matrix_imag_f32"],
                    rhs_real_f32=self.info.work_dir / case_result["rhs_real_f32"],
                    rhs_imag_f32=self.info.work_dir / case_result["rhs_imag_f32"],
                    matrix_shape=tuple(
                        int(v) for v in case_result["matrix_shape"]
                    ),
                    rhs_shape=tuple(int(v) for v in case_result["rhs_shape"]),
                    matrix_layout=str(case_result["matrix_layout"]),
                )
            )
        return systems

    def assemble_solve_standard_neumann_batch(
        self,
        frequency_hz: NDArray[Any],
        k_real: NDArray[Any],
        neumann_dp0: NDArray[Any],
        *,
        operation_id: str = "assembly-solve-batch",
    ) -> list[Any]:
        """Assemble and directly solve standard-Neumann systems in one helper run."""
        from .session import (
            BatchAssemblySolvePayload,
            BinaryArrayDescriptor,
            DenseSolveResult,
            read_json_manifest,
            write_json_manifest,
        )

        self._ensure_open()
        frequencies = np.asarray(frequency_hz, dtype=np.float64)
        k_values = np.asarray(k_real, dtype=np.float32)
        neumann_values = np.asarray(neumann_dp0)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequency_hz must be a non-empty 1D array")
        if k_values.shape != frequencies.shape:
            raise ValueError("k_real must have the same shape as frequency_hz")
        if neumann_values.shape != (
            frequencies.size,
            self.geometry_payload.dp0_dof_count,
        ):
            raise ValueError(
                "neumann_dp0 must have shape "
                f"{(frequencies.size, self.geometry_payload.dp0_dof_count)}, "
                f"got {neumann_values.shape}"
            )
        if not np.issubdtype(neumann_values.dtype, np.complexfloating):
            raise ValueError("neumann_dp0 must be complex")
        if not np.all(np.isfinite(neumann_values)):
            raise ValueError("neumann_dp0 must contain only finite values")

        op_dir = self.info.work_dir / operation_id
        inputs_dir = op_dir / "inputs"
        outputs_root = op_dir / "outputs"
        n = self.geometry_payload.p1_dof_count
        cases: list[dict[str, Any]] = []
        for idx, (freq, kval) in enumerate(zip(frequencies, k_values)):
            case_id = f"case-{idx:04d}-{float(freq):.6g}hz"
            case_input = inputs_dir / case_id
            case_output = outputs_root / case_id
            neumann = _write_complex_vector(
                np.ascontiguousarray(neumann_values[idx], dtype=np.complex64),
                case_input / "neumann",
                relative_to=self.info.work_dir,
            )
            outputs = {
                "pressure_real_f32": BinaryArrayDescriptor(
                    path=(case_output / "pressure_re_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n,),
                    dtype="float32",
                ).to_manifest(),
                "pressure_imag_f32": BinaryArrayDescriptor(
                    path=(case_output / "pressure_im_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n,),
                    dtype="float32",
                ).to_manifest(),
            }
            cases.append(
                {
                    "case_id": case_id,
                    "frequency_hz": float(freq),
                    "k_real_f32": float(np.float32(kval)),
                    "neumann_dp0": {
                        key: descriptor.to_manifest()
                        for key, descriptor in neumann.items()
                    },
                    "outputs": outputs,
                }
            )

        payload = BatchAssemblySolvePayload(
            session_id=self.info.session_id,
            cases=tuple(cases),
        )
        payload_path = write_json_manifest(
            payload,
            op_dir / "assembly-solve-batch.json",
        )
        result_path = op_dir / "assembly-solve-batch-result.json"
        self._run_native_helper(
            "assemble_solve_standard_neumann_batch",
            payload_path=payload_path,
            result_path=result_path,
        )
        result = read_json_manifest(result_path)
        case_results = _case_results_from_manifest(
            result,
            expected_count=int(frequencies.size),
            op="assemble_solve_standard_neumann_batch",
        )
        solved = []
        for case_result in case_results:
            solved.append(
                DenseSolveResult(
                    session_id=str(case_result["session_id"]),
                    frequency_hz=float(case_result["frequency_hz"]),
                    pressure_real_f32=self.info.work_dir
                    / case_result["pressure_real_f32"],
                    pressure_imag_f32=self.info.work_dir
                    / case_result["pressure_imag_f32"],
                    shape=tuple(int(v) for v in case_result["shape"]),
                    assembly_s=float(case_result["assembly_seconds"]),
                    dense_solve_s=float(case_result["dense_solve_seconds"]),
                    lapack_info=int(case_result["lapack_info"]),
                )
            )
        return solved

    def assemble_solve_evaluate_standard_neumann_batch(
        self,
        frequency_hz: NDArray[Any],
        k_real: NDArray[Any],
        neumann_dp0: NDArray[Any],
        observation_points: NDArray[Any],
        *,
        batch_id: str = "batch",
        operation_id: str = "assembly-solve-field-batch",
        source_tags: list[int] | tuple[int, ...] | None = None,
        impedance_source_tag: int | None = None,
        write_surface_pressure: bool = True,
        write_batched_field: bool = False,
        on_case_result: Any | None = None,
    ) -> list[Any]:
        """Assemble, solve, and evaluate field in one resident helper run.

        When ``on_case_result`` is provided, the helper streams per-case
        result manifests to disk as each case completes and
        ``on_case_result(index, result)`` fires per case while the batch is
        still running. Returning exactly ``False`` from the callback
        terminates the helper; the cases solved so far are returned. Streamed
        results do not carry whole-batch diagnostics (the batch is still in
        flight when each case completes).
        """
        from .session import (
            BatchAssemblySolveFieldPayload,
            BinaryArrayDescriptor,
            read_json_manifest,
            write_binary_array,
            write_json_manifest,
            _require_observation_points_3xn,
        )

        self._ensure_open()
        if on_case_result is not None and write_batched_field:
            raise ValueError(
                "on_case_result streaming requires per-case field outputs; "
                "disable write_batched_field"
            )
        frequencies = np.asarray(frequency_hz, dtype=np.float64)
        k_values = np.asarray(k_real, dtype=np.float32)
        neumann_values = np.asarray(neumann_dp0)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequency_hz must be a non-empty 1D array")
        if k_values.shape != frequencies.shape:
            raise ValueError("k_real must have the same shape as frequency_hz")
        if neumann_values.shape != (
            frequencies.size,
            self.geometry_payload.dp0_dof_count,
        ):
            raise ValueError(
                "neumann_dp0 must have shape "
                f"{(frequencies.size, self.geometry_payload.dp0_dof_count)}, "
                f"got {neumann_values.shape}"
            )
        if not np.issubdtype(neumann_values.dtype, np.complexfloating):
            raise ValueError("neumann_dp0 must be complex")
        if not np.all(np.isfinite(neumann_values)):
            raise ValueError("neumann_dp0 must contain only finite values")

        points_3xn = _require_observation_points_3xn(observation_points)
        n_obs = int(points_3xn.shape[1])
        op_dir = self.info.work_dir / operation_id
        inputs_dir = op_dir / "inputs"
        outputs_root = op_dir / "outputs"
        source_tag_values = (
            [int(tag) for tag in source_tags] if source_tags is not None else None
        )
        if source_tag_values is not None and any(tag < 0 for tag in source_tag_values):
            raise ValueError("source_tags must be non-negative integers")
        impedance_tag_value = (
            int(impedance_source_tag) if impedance_source_tag is not None else None
        )
        if impedance_tag_value is not None and impedance_tag_value < 0:
            raise ValueError("impedance_source_tag must be a non-negative integer")
        obs_desc = write_binary_array(
            points_3xn,
            inputs_dir / "obs_points_3xn_f32.bin",
            dtype=np.float32,
            relative_to=self.info.work_dir,
        )
        n = self.geometry_payload.p1_dof_count
        batch_outputs: dict[str, Any] | None = None
        if write_batched_field:
            batch_outputs = {
                "observation_pressure_real_f32": BinaryArrayDescriptor(
                    path=(outputs_root / "obs_pressure_re_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(frequencies.size, n_obs),
                    dtype="float32",
                ).to_manifest(),
                "observation_pressure_imag_f32": BinaryArrayDescriptor(
                    path=(outputs_root / "obs_pressure_im_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(frequencies.size, n_obs),
                    dtype="float32",
                ).to_manifest(),
            }
        cases: list[dict[str, Any]] = []
        for idx, (freq, kval) in enumerate(zip(frequencies, k_values)):
            case_id = f"case-{idx:04d}-{float(freq):.6g}hz"
            case_input = inputs_dir / case_id
            case_output = outputs_root / case_id
            neumann = _write_complex_vector(
                np.ascontiguousarray(neumann_values[idx], dtype=np.complex64),
                case_input / "neumann",
                relative_to=self.info.work_dir,
            )
            outputs: dict[str, Any] = {}
            if not write_batched_field:
                outputs["observation_pressure_real_f32"] = BinaryArrayDescriptor(
                    path=(case_output / "obs_pressure_re_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n_obs,),
                    dtype="float32",
                ).to_manifest()
                outputs["observation_pressure_imag_f32"] = BinaryArrayDescriptor(
                    path=(case_output / "obs_pressure_im_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n_obs,),
                    dtype="float32",
                ).to_manifest()
            if write_surface_pressure:
                outputs["pressure_real_f32"] = BinaryArrayDescriptor(
                    path=(case_output / "pressure_re_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n,),
                    dtype="float32",
                ).to_manifest()
                outputs["pressure_imag_f32"] = BinaryArrayDescriptor(
                    path=(case_output / "pressure_im_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n,),
                    dtype="float32",
                ).to_manifest()
            cases.append(
                {
                    "case_id": case_id,
                    "frequency_hz": float(freq),
                    "k_real_f32": float(np.float32(kval)),
                    "neumann_dp0": {
                        key: descriptor.to_manifest()
                        for key, descriptor in neumann.items()
                    },
                    "observation_points": obs_desc.to_manifest(),
                    "outputs": outputs,
                    **(
                        {"source_tags": source_tag_values}
                        if source_tag_values is not None
                        else {}
                    ),
                    **(
                        {"impedance_source_tag": impedance_tag_value}
                        if impedance_tag_value is not None
                        else {}
                    ),
                }
            )

        case_results_dir = op_dir / "case-results"
        payload = BatchAssemblySolveFieldPayload(
            session_id=self.info.session_id,
            batch_id=batch_id,
            cases=tuple(cases),
            batch_outputs=batch_outputs,
            case_results_dir=(
                case_results_dir.relative_to(self.info.work_dir).as_posix()
                if on_case_result is not None
                else None
            ),
        )
        payload_path = write_json_manifest(
            payload,
            op_dir / "assembly-solve-field-batch.json",
        )
        result_path = op_dir / "assembly-solve-field-batch-result.json"

        if on_case_result is not None:
            return self._stream_assemble_solve_evaluate(
                payload_path=payload_path,
                result_path=result_path,
                case_results_dir=case_results_dir,
                expected_count=int(frequencies.size),
                on_case_result=on_case_result,
            )

        self._run_native_helper(
            "assemble_solve_evaluate_standard_neumann_batch",
            payload_path=payload_path,
            result_path=result_path,
        )
        result = read_json_manifest(result_path)
        case_results = _case_results_from_manifest(
            result,
            expected_count=int(frequencies.size),
            op="assemble_solve_evaluate_standard_neumann_batch",
        )
        batch_diagnostics = _native_batch_diagnostics(result)
        return [
            self._dense_solve_field_result(case_result, batch_diagnostics)
            for case_result in case_results
        ]

    def _dense_solve_field_result(
        self,
        case_result: dict[str, Any],
        batch_diagnostics: dict[str, Any],
    ) -> Any:
        from .session import DenseSolveFieldResult

        pressure_real = case_result.get("pressure_real_f32")
        pressure_imag = case_result.get("pressure_imag_f32")
        return DenseSolveFieldResult(
            session_id=str(case_result["session_id"]),
            batch_id=str(case_result["batch_id"]),
            frequency_hz=float(case_result["frequency_hz"]),
            pressure_real_f32=(
                self.info.work_dir / str(pressure_real)
                if pressure_real is not None
                else None
            ),
            pressure_imag_f32=(
                self.info.work_dir / str(pressure_imag)
                if pressure_imag is not None
                else None
            ),
            pressure_shape=tuple(int(v) for v in case_result["pressure_shape"]),
            field_real_f32=self.info.work_dir
            / case_result["observation_pressure_real_f32"],
            field_imag_f32=self.info.work_dir
            / case_result["observation_pressure_imag_f32"],
            field_shape=tuple(int(v) for v in case_result["field_shape"]),
            assembly_s=float(case_result["assembly_seconds"]),
            dense_solve_s=float(case_result["dense_solve_seconds"]),
            field_s=float(case_result["field_seconds"]),
            lapack_info=int(case_result["lapack_info"]),
            field_row_index=(
                int(case_result["field_row_index"])
                if case_result.get("field_row_index") is not None
                else None
            ),
            field_batch_shape=(
                tuple(int(v) for v in case_result["field_batch_shape"])
                if case_result.get("field_batch_shape") is not None
                else None
            ),
            impedance=_complex_from_manifest(case_result.get("impedance")),
            surface_pressure_avg=_complex_map_from_manifest(
                case_result.get("surface_pressure_avg")
            ),
            diagnostics=_native_case_diagnostics(
                case_result,
                batch_diagnostics=batch_diagnostics,
            ),
        )

    def _stream_assemble_solve_evaluate(
        self,
        *,
        payload_path: Path,
        result_path: Path,
        case_results_dir: Path,
        expected_count: int,
        on_case_result: Any,
    ) -> list[Any]:
        """Run one batch helper invocation, firing callbacks per case.

        Per-case results are tailed from ``case_results_dir`` while the
        helper is still solving later frequencies. A ``False`` return from
        ``on_case_result`` terminates the helper early; the cases already
        solved are returned and their outputs stay on disk. Helpers that
        predate streamed case results fall back to the one-shot protocol:
        callbacks then fire from the final batch manifest after exit.
        """
        from .session import read_json_manifest

        op = "assemble_solve_evaluate_standard_neumann_batch"
        solved_fields: list[Any] = []

        def consume_available_cases() -> bool:
            """Fire callbacks for ready case files; True when stop requested."""
            while len(solved_fields) < expected_count:
                case_path = case_results_dir / f"case-{len(solved_fields):04d}.json"
                if not case_path.is_file():
                    return False
                case_result = json.loads(case_path.read_text(encoding="utf-8"))
                solved = self._dense_solve_field_result(case_result, {})
                solved_fields.append(solved)
                if on_case_result(len(solved_fields) - 1, solved) is False:
                    return True
            return False

        completed = self._run_native_helper_streaming(
            op,
            payload_path=payload_path,
            result_path=result_path,
            poll=consume_available_cases,
        )
        if not completed:
            return solved_fields

        result = read_json_manifest(result_path)
        case_results = _case_results_from_manifest(
            result,
            expected_count=expected_count,
            op=op,
        )
        batch_diagnostics = _native_batch_diagnostics(result)
        # One-shot fallback: a helper without streamed case results writes
        # only the final manifest, so fire the remaining callbacks from it.
        for index in range(len(solved_fields), expected_count):
            solved = self._dense_solve_field_result(
                case_results[index],
                batch_diagnostics,
            )
            solved_fields.append(solved)
            if on_case_result(index, solved) is False:
                break
        return solved_fields

    def evaluate_standard_exterior(
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
        """Evaluate exterior pressure via the native helper reference path."""
        from .session import (
            BinaryArrayDescriptor,
            FieldPayload,
            FieldResult,
            read_json_manifest,
            write_binary_array,
            write_json_manifest,
            _require_observation_points_3xn,
        )

        self._ensure_open()
        pressure = _require_complex_vector(
            "pressure_p1",
            pressure_p1,
            self.geometry_payload.p1_dof_count,
        )
        neumann = _require_complex_vector(
            "neumann_dp0",
            neumann_dp0,
            self.geometry_payload.dp0_dof_count,
        )
        points_3xn = _require_observation_points_3xn(observation_points)
        n_obs = int(points_3xn.shape[1])
        op_id = operation_id or _operation_id("native-field", frequency_hz)
        op_dir = self.info.work_dir / op_id
        inputs_dir = op_dir / "inputs"
        outputs_dir = op_dir / "outputs"

        pressure_desc = _write_complex_vector(
            pressure,
            inputs_dir / "pressure",
            real_name="pressure_re_f32.bin",
            imag_name="pressure_im_f32.bin",
            relative_to=self.info.work_dir,
        )
        neumann_desc = _write_complex_vector(
            neumann,
            inputs_dir / "neumann",
            real_name="neumann_re_f32.bin",
            imag_name="neumann_im_f32.bin",
            relative_to=self.info.work_dir,
        )
        obs_desc = write_binary_array(
            points_3xn,
            inputs_dir / "obs_points_3xn_f32.bin",
            dtype=np.float32,
            relative_to=self.info.work_dir,
        )
        output = {
            "pressure_real_f32": BinaryArrayDescriptor(
                path=(outputs_dir / "obs_pressure_re_f32.bin").relative_to(
                    self.info.work_dir
                ).as_posix(),
                shape=(n_obs,),
                dtype="float32",
            ),
            "pressure_imag_f32": BinaryArrayDescriptor(
                path=(outputs_dir / "obs_pressure_im_f32.bin").relative_to(
                    self.info.work_dir
                ).as_posix(),
                shape=(n_obs,),
                dtype="float32",
            ),
        }
        payload = FieldPayload(
            session_id=self.info.session_id,
            batch_id=batch_id,
            frequency_hz=frequency_hz,
            k_real_f32=float(np.float32(k_real)),
            pressure_p1=pressure_desc,
            neumann_dp0=neumann_desc,
            observation_points=obs_desc,
            output=output,
        )
        payload_path = write_json_manifest(payload, op_dir / "field.json")
        result_path = op_dir / "field-result.json"
        self._run_native_helper(
            "evaluate_standard_exterior",
            payload_path=payload_path,
            result_path=result_path,
        )
        result = read_json_manifest(result_path)
        return FieldResult(
            session_id=str(result["session_id"]),
            batch_id=str(result["batch_id"]),
            frequency_hz=float(result["frequency_hz"]),
            pressure_real_f32=self.info.work_dir / result["pressure_real_f32"],
            pressure_imag_f32=self.info.work_dir / result["pressure_imag_f32"],
            shape=tuple(int(v) for v in result["shape"]),
        )

    def evaluate_standard_exterior_batch(
        self,
        frequency_hz: NDArray[Any],
        k_real: NDArray[Any],
        pressure_p1: NDArray[Any],
        neumann_dp0: NDArray[Any],
        observation_points: NDArray[Any],
        *,
        batch_id: str = "batch",
        operation_id: str = "field-batch",
    ) -> list[Any]:
        """Evaluate exterior pressure for multiple frequencies in one helper run."""
        from .session import (
            BatchFieldPayload,
            BinaryArrayDescriptor,
            FieldResult,
            read_json_manifest,
            write_binary_array,
            write_json_manifest,
            _require_observation_points_3xn,
        )

        self._ensure_open()
        frequencies = np.asarray(frequency_hz, dtype=np.float64)
        k_values = np.asarray(k_real, dtype=np.float32)
        pressure_values = np.asarray(pressure_p1)
        neumann_values = np.asarray(neumann_dp0)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequency_hz must be a non-empty 1D array")
        if k_values.shape != frequencies.shape:
            raise ValueError("k_real must have the same shape as frequency_hz")
        if pressure_values.shape != (
            frequencies.size,
            self.geometry_payload.p1_dof_count,
        ):
            raise ValueError(
                "pressure_p1 must have shape "
                f"{(frequencies.size, self.geometry_payload.p1_dof_count)}, "
                f"got {pressure_values.shape}"
            )
        if neumann_values.shape != (
            frequencies.size,
            self.geometry_payload.dp0_dof_count,
        ):
            raise ValueError(
                "neumann_dp0 must have shape "
                f"{(frequencies.size, self.geometry_payload.dp0_dof_count)}, "
                f"got {neumann_values.shape}"
            )
        if not np.issubdtype(pressure_values.dtype, np.complexfloating):
            raise ValueError("pressure_p1 must be complex")
        if not np.issubdtype(neumann_values.dtype, np.complexfloating):
            raise ValueError("neumann_dp0 must be complex")
        if not np.all(np.isfinite(pressure_values)) or not np.all(
            np.isfinite(neumann_values)
        ):
            raise ValueError("field inputs must contain only finite values")

        points_3xn = _require_observation_points_3xn(observation_points)
        n_obs = int(points_3xn.shape[1])
        op_dir = self.info.work_dir / operation_id
        inputs_dir = op_dir / "inputs"
        outputs_root = op_dir / "outputs"
        obs_desc = write_binary_array(
            points_3xn,
            inputs_dir / "obs_points_3xn_f32.bin",
            dtype=np.float32,
            relative_to=self.info.work_dir,
        )

        cases: list[dict[str, Any]] = []
        for idx, (freq, kval) in enumerate(zip(frequencies, k_values)):
            case_id = f"case-{idx:04d}-{float(freq):.6g}hz"
            case_input = inputs_dir / case_id
            case_output = outputs_root / case_id
            pressure_desc = _write_complex_vector(
                np.ascontiguousarray(pressure_values[idx], dtype=np.complex64),
                case_input / "pressure",
                real_name="pressure_re_f32.bin",
                imag_name="pressure_im_f32.bin",
                relative_to=self.info.work_dir,
            )
            neumann_desc = _write_complex_vector(
                np.ascontiguousarray(neumann_values[idx], dtype=np.complex64),
                case_input / "neumann",
                real_name="neumann_re_f32.bin",
                imag_name="neumann_im_f32.bin",
                relative_to=self.info.work_dir,
            )
            output = {
                "pressure_real_f32": BinaryArrayDescriptor(
                    path=(case_output / "obs_pressure_re_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n_obs,),
                    dtype="float32",
                ).to_manifest(),
                "pressure_imag_f32": BinaryArrayDescriptor(
                    path=(case_output / "obs_pressure_im_f32.bin").relative_to(
                        self.info.work_dir
                    ).as_posix(),
                    shape=(n_obs,),
                    dtype="float32",
                ).to_manifest(),
            }
            cases.append(
                {
                    "case_id": case_id,
                    "frequency_hz": float(freq),
                    "k_real_f32": float(np.float32(kval)),
                    "pressure_p1": {
                        key: descriptor.to_manifest()
                        for key, descriptor in pressure_desc.items()
                    },
                    "neumann_dp0": {
                        key: descriptor.to_manifest()
                        for key, descriptor in neumann_desc.items()
                    },
                    "output": output,
                }
            )

        payload = BatchFieldPayload(
            session_id=self.info.session_id,
            batch_id=batch_id,
            observation_points=obs_desc,
            cases=tuple(cases),
        )
        payload_path = write_json_manifest(payload, op_dir / "field-batch.json")
        result_path = op_dir / "field-batch-result.json"
        self._run_native_helper(
            "evaluate_standard_exterior_batch",
            payload_path=payload_path,
            result_path=result_path,
        )
        result = read_json_manifest(result_path)
        case_results = _case_results_from_manifest(
            result,
            expected_count=int(frequencies.size),
            op="evaluate_standard_exterior_batch",
        )
        fields = []
        for case_result in case_results:
            fields.append(
                FieldResult(
                    session_id=str(case_result["session_id"]),
                    batch_id=str(case_result["batch_id"]),
                    frequency_hz=float(case_result["frequency_hz"]),
                    pressure_real_f32=self.info.work_dir
                    / case_result["pressure_real_f32"],
                    pressure_imag_f32=self.info.work_dir
                    / case_result["pressure_imag_f32"],
                    shape=tuple(int(v) for v in case_result["shape"]),
                )
            )
        return fields

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_work_dir and self.info.work_dir.exists():
            shutil.rmtree(self.info.work_dir)
        self._closed = True

    cleanup = close

    def __enter__(self) -> "MetalNativeStandardSession":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MetalNativeStandardSession is closed")

    def _run_native_helper(
        self,
        op: str,
        *,
        payload_path: Path,
        result_path: Path,
    ) -> None:
        status = self.info.runtime_status
        if not status.available:
            raise RuntimeError(
                "Swift/Metal native helper is unavailable for execution: "
                + "; ".join(status.unavailable_reasons)
            )
        command = [
            *_native_helper_command(
                status,
                op,
                str(self.info.manifest_path),
                str(payload_path),
                str(result_path),
            )
        ]
        timeout_s = (
            self._runtime_config or MetalNativeRuntimeConfig()
        ).operation_timeout_s
        env = None
        if self._extra_env:
            env = {**os.environ, **self._extra_env}
        try:
            result = subprocess.run(
                command,
                cwd=status.backend_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Swift/Metal native helper timed out after {timeout_s}s during {op}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Failed to launch Swift/Metal native helper: {exc}"
            ) from exc

        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            message = stderr or stdout or f"Swift helper exited with {result.returncode}"
            raise RuntimeError(f"Swift/Metal native helper failed during {op}: {message}")
        if not result_path.is_file():
            raise RuntimeError(f"Swift/Metal native helper did not write {result_path}")

    def _run_native_helper_streaming(
        self,
        op: str,
        *,
        payload_path: Path,
        result_path: Path,
        poll: Any,
    ) -> bool:
        """Run one helper operation while polling for streamed results.

        ``poll()`` is invoked repeatedly while the helper runs and once more
        after it exits; returning True requests an early stop, upon which the
        helper process is terminated. Returns True when the helper ran to
        completion, False when it was stopped early.
        """
        status = self.info.runtime_status
        if not status.available:
            raise RuntimeError(
                "Swift/Metal native helper is unavailable for execution: "
                + "; ".join(status.unavailable_reasons)
            )
        command = _native_helper_command(
            status,
            op,
            str(self.info.manifest_path),
            str(payload_path),
            str(result_path),
        )
        timeout_s = (
            self._runtime_config or MetalNativeRuntimeConfig()
        ).operation_timeout_s
        env = None
        if self._extra_env:
            env = {**os.environ, **self._extra_env}
        try:
            process = subprocess.Popen(
                command,
                cwd=status.backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Failed to launch Swift/Metal native helper: {exc}"
            ) from exc

        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        try:
            while True:
                if poll():
                    process.terminate()
                    try:
                        process.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return False
                if process.poll() is not None:
                    break
                if deadline is not None and time.monotonic() > deadline:
                    process.kill()
                    process.wait()
                    raise RuntimeError(
                        f"Swift/Metal native helper timed out after {timeout_s}s "
                        f"during {op}"
                    )
                time.sleep(0.005)
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise

        stdout, stderr = process.communicate()
        if process.returncode != 0:
            message = (
                (stderr or "").strip()
                or (stdout or "").strip()
                or f"Swift helper exited with {process.returncode}"
            )
            raise RuntimeError(
                f"Swift/Metal native helper failed during {op}: {message}"
            )
        if not result_path.is_file():
            raise RuntimeError(
                f"Swift/Metal native helper did not write {result_path}"
            )
        # Consume case files written between the last poll and process exit.
        return not poll()


def _find_swift(
    config: MetalNativeRuntimeConfig,
) -> tuple[str | None, str | None]:
    if config.swift_executable:
        return config.swift_executable, "explicit"

    env_path = os.environ.get(config.swift_env_var)
    if env_path:
        return env_path, config.swift_env_var

    path_swift = shutil.which("swift")
    if path_swift:
        return path_swift, "PATH"

    return None, None


def _find_helper_executable(
    config: MetalNativeRuntimeConfig,
    native_package_dir: Path,
) -> tuple[Path | None, str | None]:
    if config.helper_executable:
        helper = Path(config.helper_executable)
        if not helper.is_file():
            raise ValueError(
                f"helper_executable {helper} does not exist or is not a file"
            )
        return helper, "explicit"

    env_path = os.environ.get(config.helper_env_var)
    if env_path:
        helper = Path(env_path)
        if helper.is_file():
            return helper, config.helper_env_var
        logger.warning(
            "%s points at %s, which does not exist; falling back to "
            "compiled-package discovery",
            config.helper_env_var,
            helper,
        )

    candidates = (
        native_package_dir / ".build" / "release" / config.native_binary_name,
        native_package_dir / ".build" / "debug" / config.native_binary_name,
    )
    main_source = (
        native_package_dir / "Sources" / config.native_binary_name / "main.swift"
    )
    for candidate in candidates:
        if candidate.is_file():
            if (
                main_source.is_file()
                and candidate.stat().st_mtime < main_source.stat().st_mtime
            ):
                logger.warning(
                    "Native helper binary %s is older than %s; numeric changes "
                    "in the source are not active until `swift build -c release`",
                    candidate,
                    main_source,
                )
            return candidate, "swift-package"

    return None, None


def _native_helper_command(
    status: MetalNativeRuntimeStatus,
    *args: str,
) -> list[str]:
    if status.helper_executable_path is not None:
        return [str(status.helper_executable_path), *args]
    if status.swift_path is None:
        raise RuntimeError("Swift executable is unavailable for helper script fallback")
    return [status.swift_path, str(status.native_entrypoint), *args]


def _run_native_smoke_test(
    helper_executable_path: Path | None,
    swift_path: str | None,
    native_entrypoint: Path,
    *,
    timeout_s: float,
) -> tuple[bool, str | None]:
    command = (
        [str(helper_executable_path), "--smoke"]
        if helper_executable_path is not None
        else [str(swift_path), str(native_entrypoint), "--smoke"]
    )
    try:
        result = subprocess.run(
            command,
            cwd=native_entrypoint.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)

    if result.returncode == 0:
        return True, None

    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    message = stderr or stdout or f"Swift helper exited with {result.returncode}"
    return False, message.splitlines()[-1]


def _operation_id(prefix: str, frequency_hz: float) -> str:
    freq = f"{float(frequency_hz):.6f}".replace(".", "p").replace("-", "m")
    return f"{prefix}-{freq}-{uuid4().hex[:8]}"


def _require_complex_vector(
    name: str,
    value: NDArray[Any],
    expected_len: int,
) -> NDArray[np.complex64]:
    array = np.asarray(value)
    if array.shape != (expected_len,):
        raise ValueError(
            f"{name} must have shape {(expected_len,)}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a complex vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.complex64)


def _write_complex_vector(
    vector: NDArray[np.complex64],
    directory: Path,
    *,
    real_name: str = "neumann_re_f32.bin",
    imag_name: str = "neumann_im_f32.bin",
    relative_to: Path,
) -> dict[str, Any]:
    from .session import write_binary_array

    return {
        "real_f32": write_binary_array(
            np.ascontiguousarray(np.real(vector), dtype=np.float32),
            directory / real_name,
            dtype=np.float32,
            relative_to=relative_to,
        ),
        "imag_f32": write_binary_array(
            np.ascontiguousarray(np.imag(vector), dtype=np.float32),
            directory / imag_name,
            dtype=np.float32,
            relative_to=relative_to,
        ),
    }


def _case_results_from_manifest(
    result: dict[str, Any],
    *,
    expected_count: int,
    op: str,
) -> list[Any]:
    cases = result.get("cases")
    if not isinstance(cases, list):
        raise RuntimeError(f"Swift/Metal native helper {op} result missing cases")
    if len(cases) != expected_count:
        raise RuntimeError(
            f"Swift/Metal native helper {op} returned {len(cases)} case(s), "
            f"expected {expected_count}"
        )
    return cases


def _native_batch_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "implementation",
        "session_id",
        "batch_id",
        "symmetry_plane",
        "case_count",
        "assembly_seconds",
        "regular_assembly_seconds",
        "dense_solve_seconds",
        "field_seconds",
        "resident_context_seconds",
        "resident_duffy_reduction_plan_seconds",
        "wall_seconds",
        "resident_reuse",
    )
    return {key: result[key] for key in keys if key in result}


def _native_case_diagnostics(
    case_result: dict[str, Any],
    *,
    batch_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "case_id",
        "implementation",
        "assembly_implementation",
        "solve_implementation",
        "field_implementation",
        "assembly_mode",
        "field_mode",
        "regular_assembly_seconds",
        "lapack_info",
        "dense_solve_rcond",
        "dense_solve_condition_1norm",
        "symmetry_plane",
        "pressure_shape",
        "field_shape",
        "field_row_index",
        "field_batch_shape",
        "field_output_layout",
        "duffy_corrections",
        "metal_dispatch",
        "field_metal_dispatch",
    )
    diagnostics = {key: case_result[key] for key in keys if key in case_result}
    if batch_diagnostics:
        diagnostics["batch"] = dict(batch_diagnostics)
    return diagnostics


def _complex_from_manifest(value: Any) -> complex | None:
    if value is None:
        return None
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(part, (int, float)) for part in value)
    ):
        return complex(float(value[0]), float(value[1]))
    raise ValueError(f"complex manifest value must be [real, imag], got {value!r}")


def _complex_map_from_manifest(value: Any) -> dict[int, complex] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("complex map manifest value must be an object")
    result: dict[int, complex] = {}
    for key, raw in value.items():
        if raw is None:
            raise ValueError(f"complex map manifest entry {key!r} must not be null")
        result[int(key)] = _complex_from_manifest(raw)
    return result

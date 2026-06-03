"""Language-neutral session and IPC contract for native Metal backends.

The module owns the Python-side file contract for JSON manifests plus
little-endian binary buffers consumed by the Swift/Metal native helper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .geometry import MetalGeometryBuffers


METAL_STANDARD_SCHEMA = "hornlab.metal.standard.v1"
INDEX_BASE = 0
MATRIX_LAYOUT_ROW_MAJOR_C = "row_major_c"


@dataclass(frozen=True)
class BinaryArrayDescriptor:
    """Manifest descriptor for one raw C-contiguous binary array."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    byte_order: str = "little"
    order: str = "C"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _validate_relative_manifest_path(self.path))
        object.__setattr__(self, "shape", _validate_shape(self.shape))
        _validate_dtype(self.dtype)
        if self.byte_order != "little":
            raise ValueError("Metal IPC binary arrays must be little-endian")
        if self.order != "C":
            raise ValueError("Metal IPC binary arrays must be C-contiguous")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_order": self.byte_order,
            "order": self.order,
        }


@dataclass(frozen=True)
class GeometryPayload:
    """Create-session payload for resident standard-BIE geometry buffers."""

    session_id: str
    mesh: dict[str, BinaryArrayDescriptor]
    p1_dof_count: int
    dp0_dof_count: int
    regular_triangle_order: int = 4
    duffy_1d_order: int = 4
    precision: str = "complex64"
    symmetry_plane: str | None = None
    schema: str = METAL_STANDARD_SCHEMA
    op: str = "create_session"
    index_base: int = INDEX_BASE
    matrix_layout: str = MATRIX_LAYOUT_ROW_MAJOR_C

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "op": self.op,
            "session_id": self.session_id,
            "precision": self.precision,
            "index_base": self.index_base,
            "matrix_layout": self.matrix_layout,
            "mesh": {
                key: descriptor.to_manifest()
                for key, descriptor in self.mesh.items()
            },
            "space": {
                "basis_trial": "P1",
                "basis_test": "P1",
                "source_basis": "DP0",
                "p1_dof_count": self.p1_dof_count,
                "dp0_dof_count": self.dp0_dof_count,
            },
            "quadrature": {
                "regular_triangle_order": self.regular_triangle_order,
                "duffy_1d_order": self.duffy_1d_order,
            },
            "assembly_scope": {
                "formulation": "standard_neumann",
                "basis_trial": "P1",
                "basis_test": "P1",
                "source_basis": "DP0",
                "symmetry_plane": self.symmetry_plane,
            },
        }


@dataclass(frozen=True)
class AssemblyPayload:
    """Manifest payload for one future standard Neumann assembly request."""

    session_id: str
    frequency_hz: float
    k_real_f32: float
    neumann_dp0: dict[str, BinaryArrayDescriptor]
    outputs: dict[str, BinaryArrayDescriptor]
    schema: str = METAL_STANDARD_SCHEMA
    op: str = "assemble_standard_neumann"
    index_base: int = INDEX_BASE
    matrix_layout: str = MATRIX_LAYOUT_ROW_MAJOR_C

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "op": self.op,
            "session_id": self.session_id,
            "frequency_hz": float(self.frequency_hz),
            "k_real_f32": float(np.float32(self.k_real_f32)),
            "index_base": self.index_base,
            "neumann_dp0": {
                key: descriptor.to_manifest()
                for key, descriptor in self.neumann_dp0.items()
            },
            "outputs": {
                "matrix_layout": self.matrix_layout,
                **{
                    key: descriptor.to_manifest()
                    for key, descriptor in self.outputs.items()
                },
            },
        }


@dataclass(frozen=True)
class BatchAssemblyPayload:
    """Manifest payload for a resident-helper standard Neumann assembly batch."""

    session_id: str
    cases: tuple[dict[str, Any], ...]
    schema: str = METAL_STANDARD_SCHEMA
    op: str = "assemble_standard_neumann_batch"
    index_base: int = INDEX_BASE
    matrix_layout: str = MATRIX_LAYOUT_ROW_MAJOR_C

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "op": self.op,
            "session_id": self.session_id,
            "index_base": self.index_base,
            "matrix_layout": self.matrix_layout,
            "cases": list(self.cases),
        }


@dataclass(frozen=True)
class BatchAssemblySolvePayload:
    """Manifest payload for resident standard Neumann assembly and dense solve."""

    session_id: str
    cases: tuple[dict[str, Any], ...]
    schema: str = METAL_STANDARD_SCHEMA
    op: str = "assemble_solve_standard_neumann_batch"
    index_base: int = INDEX_BASE
    matrix_layout: str = MATRIX_LAYOUT_ROW_MAJOR_C

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "op": self.op,
            "session_id": self.session_id,
            "index_base": self.index_base,
            "matrix_layout": self.matrix_layout,
            "cases": list(self.cases),
        }


@dataclass(frozen=True)
class BatchAssemblySolveFieldPayload:
    """Manifest payload for resident assembly, dense solve, and field evaluation."""

    session_id: str
    batch_id: str
    cases: tuple[dict[str, Any], ...]
    batch_outputs: dict[str, Any] | None = None
    schema: str = METAL_STANDARD_SCHEMA
    op: str = "assemble_solve_evaluate_standard_neumann_batch"
    index_base: int = INDEX_BASE
    matrix_layout: str = MATRIX_LAYOUT_ROW_MAJOR_C

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "op": self.op,
            "session_id": self.session_id,
            "batch_id": self.batch_id,
            "index_base": self.index_base,
            "matrix_layout": self.matrix_layout,
            "cases": list(self.cases),
        }
        if self.batch_outputs is not None:
            payload["batch_outputs"] = self.batch_outputs
        return payload


@dataclass(frozen=True)
class FieldPayload:
    """Manifest payload for one future exterior field evaluation request."""

    session_id: str
    batch_id: str
    frequency_hz: float
    k_real_f32: float
    pressure_p1: dict[str, BinaryArrayDescriptor]
    neumann_dp0: dict[str, BinaryArrayDescriptor]
    observation_points: BinaryArrayDescriptor
    output: dict[str, BinaryArrayDescriptor]
    schema: str = METAL_STANDARD_SCHEMA
    op: str = "evaluate_standard_exterior"
    index_base: int = INDEX_BASE

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "op": self.op,
            "session_id": self.session_id,
            "batch_id": self.batch_id,
            "frequency_hz": float(self.frequency_hz),
            "k_real_f32": float(np.float32(self.k_real_f32)),
            "index_base": self.index_base,
            "pressure_p1": {
                key: descriptor.to_manifest()
                for key, descriptor in self.pressure_p1.items()
            },
            "neumann_dp0": {
                key: descriptor.to_manifest()
                for key, descriptor in self.neumann_dp0.items()
            },
            "observation_points": self.observation_points.to_manifest(),
            "output": {
                key: descriptor.to_manifest()
                for key, descriptor in self.output.items()
            },
        }


@dataclass(frozen=True)
class BatchFieldPayload:
    """Manifest payload for a resident-helper exterior field batch."""

    session_id: str
    batch_id: str
    observation_points: BinaryArrayDescriptor | None
    cases: tuple[dict[str, Any], ...]
    schema: str = METAL_STANDARD_SCHEMA
    op: str = "evaluate_standard_exterior_batch"
    index_base: int = INDEX_BASE

    def to_manifest(self) -> dict[str, Any]:
        manifest = {
            "schema": self.schema,
            "op": self.op,
            "session_id": self.session_id,
            "batch_id": self.batch_id,
            "index_base": self.index_base,
            "cases": list(self.cases),
        }
        if self.observation_points is not None:
            manifest["observation_points"] = self.observation_points.to_manifest()
        return manifest


@dataclass(frozen=True)
class DenseAssemblyResult:
    """Dense matrix/RHS result descriptor returned by packaged helpers."""

    session_id: str
    frequency_hz: float
    matrix_real_f32: Path
    matrix_imag_f32: Path
    rhs_real_f32: Path
    rhs_imag_f32: Path
    matrix_shape: tuple[int, int]
    rhs_shape: tuple[int]
    schema: str = METAL_STANDARD_SCHEMA
    matrix_layout: str = MATRIX_LAYOUT_ROW_MAJOR_C


@dataclass(frozen=True)
class DenseSolveResult:
    """Dense solved pressure descriptor returned by packaged helpers."""

    session_id: str
    frequency_hz: float
    pressure_real_f32: Path
    pressure_imag_f32: Path
    shape: tuple[int]
    assembly_s: float
    dense_solve_s: float
    lapack_info: int
    schema: str = METAL_STANDARD_SCHEMA


@dataclass(frozen=True)
class DenseSolveFieldResult:
    """Solved surface pressure plus exterior field descriptors."""

    session_id: str
    batch_id: str
    frequency_hz: float
    pressure_real_f32: Path | None
    pressure_imag_f32: Path | None
    pressure_shape: tuple[int]
    field_real_f32: Path
    field_imag_f32: Path
    field_shape: tuple[int]
    assembly_s: float
    dense_solve_s: float
    field_s: float
    lapack_info: int
    field_row_index: int | None = None
    field_batch_shape: tuple[int, int] | None = None
    impedance: complex | None = None
    surface_pressure_avg: dict[int, complex] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    schema: str = METAL_STANDARD_SCHEMA


@dataclass(frozen=True)
class FieldResult:
    """Future exterior-pressure result descriptor."""

    session_id: str
    batch_id: str
    frequency_hz: float
    pressure_real_f32: Path
    pressure_imag_f32: Path
    shape: tuple[int]
    schema: str = METAL_STANDARD_SCHEMA


def write_json_manifest(payload: Any, path: Path) -> Path:
    """Write a deterministic UTF-8 JSON manifest for a payload dataclass."""
    manifest = payload_to_manifest(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_json_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest written by ``write_json_manifest``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def payload_to_manifest(payload: Any) -> dict[str, Any]:
    """Convert a supported payload object to its JSON-ready dict."""
    if hasattr(payload, "to_manifest"):
        manifest = payload.to_manifest()
    elif isinstance(payload, dict):
        manifest = payload
    else:
        raise TypeError(f"Unsupported manifest payload type: {type(payload)!r}")

    if manifest.get("schema") != METAL_STANDARD_SCHEMA:
        raise ValueError(f"manifest schema must be {METAL_STANDARD_SCHEMA!r}")
    if manifest.get("index_base") != INDEX_BASE:
        raise ValueError("Metal session manifests must use index_base=0")
    _validate_manifest_contract(manifest)
    return manifest


def write_binary_array(
    array: NDArray[Any],
    path: Path,
    *,
    dtype: str | np.dtype,
    relative_to: Path | None = None,
) -> BinaryArrayDescriptor:
    """Write one C-contiguous little-endian numeric array to ``path``."""
    little_dtype = np.dtype(dtype).newbyteorder("<")
    output = np.ascontiguousarray(array, dtype=little_dtype)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.tofile(path)
    manifest_path = path.name if relative_to is None else path.relative_to(relative_to)
    return BinaryArrayDescriptor(
        path=manifest_path.as_posix()
        if isinstance(manifest_path, Path)
        else str(manifest_path),
        shape=tuple(int(dim) for dim in output.shape),
        dtype=_manifest_dtype_name(little_dtype),
    )


def write_geometry_buffers(
    buffers: MetalGeometryBuffers,
    directory: Path,
    *,
    relative_to: Path | None = None,
) -> dict[str, BinaryArrayDescriptor]:
    """Write all current Metal geometry buffers and return manifest entries."""
    directory = Path(directory)
    return {
        "vertices_f32": write_binary_array(
            buffers.vertices_3xn_f32,
            directory / "vertices_3xn_f32.bin",
            dtype=np.float32,
            relative_to=relative_to,
        ),
        "triangles_i32": write_binary_array(
            buffers.triangles_3xm_i32,
            directory / "triangles_3xm_i32.bin",
            dtype=np.int32,
            relative_to=relative_to,
        ),
        "physical_tags_i32": write_binary_array(
            buffers.physical_tags_i32,
            directory / "physical_tags_i32.bin",
            dtype=np.int32,
            relative_to=relative_to,
        ),
        "p1_local2global_i32": write_binary_array(
            buffers.p1_local2global_i32,
            directory / "p1_local2global_i32.bin",
            dtype=np.int32,
            relative_to=relative_to,
        ),
        "triangle_areas_f32": write_binary_array(
            buffers.triangle_areas_f32,
            directory / "triangle_areas_f32.bin",
            dtype=np.float32,
            relative_to=relative_to,
        ),
        "triangle_normals_3xm_f32": write_binary_array(
            buffers.triangle_normals_3xm_f32,
            directory / "triangle_normals_3xm_f32.bin",
            dtype=np.float32,
            relative_to=relative_to,
        ),
    }


def _manifest_dtype_name(dtype: np.dtype) -> str:
    if dtype.kind == "f" and dtype.itemsize == 4:
        return "float32"
    if dtype.kind == "i" and dtype.itemsize == 4:
        return "int32"
    raise TypeError(f"Unsupported Metal IPC array dtype: {dtype}")


def _validate_relative_manifest_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("Metal IPC descriptor path must be a non-empty string")
    posix = PurePosixPath(path)
    if posix.is_absolute():
        raise ValueError("Metal IPC descriptor paths must be relative")
    if ".." in posix.parts:
        raise ValueError("Metal IPC descriptor paths must not contain '..'")
    return posix.as_posix()


def _validate_shape(shape: Any) -> tuple[int, ...]:
    if not isinstance(shape, tuple):
        shape = tuple(shape)
    if not shape:
        raise ValueError("Metal IPC descriptor shape must not be empty")
    normalized = tuple(int(dim) for dim in shape)
    if any(dim <= 0 for dim in normalized):
        raise ValueError("Metal IPC descriptor shape dimensions must be positive")
    return normalized


def _validate_dtype(dtype: str) -> None:
    if dtype not in {"float32", "int32"}:
        raise ValueError("Metal IPC descriptor dtype must be float32 or int32")


def _validate_manifest_contract(manifest: dict[str, Any]) -> None:
    op = manifest.get("op")
    if op == "create_session":
        _validate_geometry_manifest(manifest)
    elif op == "assemble_standard_neumann":
        _validate_assembly_manifest(manifest)
    elif op == "assemble_standard_neumann_batch":
        _validate_batch_assembly_manifest(manifest)
    elif op == "assemble_solve_standard_neumann_batch":
        _validate_batch_assembly_solve_manifest(manifest)
    elif op == "assemble_solve_evaluate_standard_neumann_batch":
        _validate_batch_assembly_solve_field_manifest(manifest)
    elif op == "evaluate_standard_exterior":
        _validate_field_manifest(manifest)
    elif op == "evaluate_standard_exterior_batch":
        _validate_batch_field_manifest(manifest)
    else:
        raise ValueError(f"Unsupported Metal session op: {op!r}")


def _validate_descriptor(
    descriptor: Any,
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...] | None = None,
    rank: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{name} descriptor must be an object")
    required = {"path", "shape", "dtype", "byte_order", "order"}
    missing = required.difference(descriptor)
    if missing:
        raise ValueError(f"{name} descriptor missing keys: {sorted(missing)}")
    _validate_relative_manifest_path(str(descriptor["path"]))
    actual_shape = _validate_shape(descriptor["shape"])
    if rank is not None and len(actual_shape) != rank:
        raise ValueError(f"{name} descriptor must have rank {rank}")
    if shape is not None and actual_shape != shape:
        raise ValueError(
            f"{name} descriptor shape must be {shape}, got {actual_shape}"
        )
    if descriptor["dtype"] != dtype:
        raise ValueError(f"{name} descriptor dtype must be {dtype}")
    _validate_dtype(str(descriptor["dtype"]))
    if descriptor["byte_order"] != "little":
        raise ValueError(f"{name} descriptor must be little-endian")
    if descriptor["order"] != "C":
        raise ValueError(f"{name} descriptor must be C-contiguous")
    return actual_shape


def _require_descriptor_group(
    group: Any,
    *,
    name: str,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(group, dict):
        raise ValueError(f"{name} must be an object")
    missing = set(keys).difference(group)
    if missing:
        raise ValueError(f"{name} missing keys: {sorted(missing)}")
    return group


def _validate_geometry_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("matrix_layout") != MATRIX_LAYOUT_ROW_MAJOR_C:
        raise ValueError("create_session manifest must use row_major_c layout")
    mesh = _require_descriptor_group(
        manifest.get("mesh"),
        name="mesh",
        keys=(
            "vertices_f32",
            "triangles_i32",
            "physical_tags_i32",
            "p1_local2global_i32",
            "triangle_areas_f32",
            "triangle_normals_3xm_f32",
        ),
    )
    space = manifest.get("space")
    if not isinstance(space, dict):
        raise ValueError("space must be an object")

    vertices_shape = _validate_descriptor(
        mesh["vertices_f32"], name="mesh.vertices_f32", dtype="float32", rank=2
    )
    triangles_shape = _validate_descriptor(
        mesh["triangles_i32"], name="mesh.triangles_i32", dtype="int32", rank=2
    )
    if vertices_shape[0] != 3:
        raise ValueError("mesh.vertices_f32 must have shape (3, n_vertices)")
    if triangles_shape[0] != 3:
        raise ValueError("mesh.triangles_i32 must have shape (3, n_triangles)")

    n_triangles = triangles_shape[1]
    _validate_descriptor(
        mesh["physical_tags_i32"],
        name="mesh.physical_tags_i32",
        dtype="int32",
        shape=(n_triangles,),
    )
    _validate_descriptor(
        mesh["p1_local2global_i32"],
        name="mesh.p1_local2global_i32",
        dtype="int32",
        shape=(n_triangles, 3),
    )
    _validate_descriptor(
        mesh["triangle_areas_f32"],
        name="mesh.triangle_areas_f32",
        dtype="float32",
        shape=(n_triangles,),
    )
    _validate_descriptor(
        mesh["triangle_normals_3xm_f32"],
        name="mesh.triangle_normals_3xm_f32",
        dtype="float32",
        shape=(3, n_triangles),
    )

    p1_dof_count = int(space.get("p1_dof_count", 0))
    dp0_dof_count = int(space.get("dp0_dof_count", 0))
    if p1_dof_count <= 0:
        raise ValueError("space.p1_dof_count must be positive")
    if dp0_dof_count != n_triangles:
        raise ValueError("space.dp0_dof_count must equal n_triangles")


def _validate_assembly_manifest(manifest: dict[str, Any]) -> None:
    _validate_assembly_case_manifest(manifest)


def _validate_assembly_case_manifest(manifest: dict[str, Any]) -> None:
    neumann = _require_descriptor_group(
        manifest.get("neumann_dp0"),
        name="neumann_dp0",
        keys=("real_f32", "imag_f32"),
    )
    n_shape = _validate_descriptor(
        neumann["real_f32"],
        name="neumann_dp0.real_f32",
        dtype="float32",
        rank=1,
    )
    _validate_descriptor(
        neumann["imag_f32"],
        name="neumann_dp0.imag_f32",
        dtype="float32",
        shape=n_shape,
    )

    outputs = _require_descriptor_group(
        manifest.get("outputs"),
        name="outputs",
        keys=(
            "matrix_layout",
            "A_real_f32",
            "A_imag_f32",
            "rhs_real_f32",
            "rhs_imag_f32",
        ),
    )
    if outputs["matrix_layout"] != MATRIX_LAYOUT_ROW_MAJOR_C:
        raise ValueError("assembly outputs must use row_major_c matrix_layout")
    matrix_shape = _validate_descriptor(
        outputs["A_real_f32"],
        name="outputs.A_real_f32",
        dtype="float32",
        rank=2,
    )
    if matrix_shape[0] != matrix_shape[1]:
        raise ValueError("assembly matrix output must be square")
    _validate_descriptor(
        outputs["A_imag_f32"],
        name="outputs.A_imag_f32",
        dtype="float32",
        shape=matrix_shape,
    )
    rhs_shape = (matrix_shape[0],)
    _validate_descriptor(
        outputs["rhs_real_f32"],
        name="outputs.rhs_real_f32",
        dtype="float32",
        shape=rhs_shape,
    )
    _validate_descriptor(
        outputs["rhs_imag_f32"],
        name="outputs.rhs_imag_f32",
        dtype="float32",
        shape=rhs_shape,
    )


def _validate_batch_assembly_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("matrix_layout") != MATRIX_LAYOUT_ROW_MAJOR_C:
        raise ValueError("batch assembly manifest must use row_major_c layout")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("batch assembly manifest must contain non-empty cases")
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"batch assembly case {idx} must be an object")
        _validate_assembly_case_manifest(case)


def _validate_assembly_solve_case_manifest(manifest: dict[str, Any]) -> None:
    neumann = _require_descriptor_group(
        manifest.get("neumann_dp0"),
        name="neumann_dp0",
        keys=("real_f32", "imag_f32"),
    )
    neumann_shape = _validate_descriptor(
        neumann["real_f32"],
        name="neumann_dp0.real_f32",
        dtype="float32",
        rank=1,
    )
    _validate_descriptor(
        neumann["imag_f32"],
        name="neumann_dp0.imag_f32",
        dtype="float32",
        shape=neumann_shape,
    )

    outputs = _require_descriptor_group(
        manifest.get("outputs"),
        name="outputs",
        keys=("pressure_real_f32", "pressure_imag_f32"),
    )
    pressure_shape = _validate_descriptor(
        outputs["pressure_real_f32"],
        name="outputs.pressure_real_f32",
        dtype="float32",
        rank=1,
    )
    _validate_descriptor(
        outputs["pressure_imag_f32"],
        name="outputs.pressure_imag_f32",
        dtype="float32",
        shape=pressure_shape,
    )


def _validate_batch_assembly_solve_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("matrix_layout") != MATRIX_LAYOUT_ROW_MAJOR_C:
        raise ValueError("batch assembly-solve manifest must use row_major_c layout")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("batch assembly-solve manifest must contain non-empty cases")
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"batch assembly-solve case {idx} must be an object")
        _validate_assembly_solve_case_manifest(case)


def _validate_assembly_solve_field_case_manifest(
    manifest: dict[str, Any],
    *,
    uses_batch_field_outputs: bool = False,
) -> None:
    neumann = _require_descriptor_group(
        manifest.get("neumann_dp0"),
        name="neumann_dp0",
        keys=("real_f32", "imag_f32"),
    )
    neumann_shape = _validate_descriptor(
        neumann["real_f32"],
        name="neumann_dp0.real_f32",
        dtype="float32",
        rank=1,
    )
    _validate_descriptor(
        neumann["imag_f32"],
        name="neumann_dp0.imag_f32",
        dtype="float32",
        shape=neumann_shape,
    )
    obs_shape = _validate_descriptor(
        manifest.get("observation_points"),
        name="observation_points",
        dtype="float32",
        rank=2,
    )
    if obs_shape[0] != 3:
        raise ValueError("observation_points must have shape (3, n_obs)")
    outputs = _require_descriptor_group(
        manifest.get("outputs"),
        name="outputs",
        keys=()
        if uses_batch_field_outputs
        else (
            "observation_pressure_real_f32",
            "observation_pressure_imag_f32",
        ),
    )
    has_pressure_real = "pressure_real_f32" in outputs
    has_pressure_imag = "pressure_imag_f32" in outputs
    if has_pressure_real != has_pressure_imag:
        raise ValueError(
            "outputs pressure_real_f32 and pressure_imag_f32 must be provided together"
        )
    if has_pressure_real:
        pressure_shape = _validate_descriptor(
            outputs["pressure_real_f32"],
            name="outputs.pressure_real_f32",
            dtype="float32",
            rank=1,
        )
        _validate_descriptor(
            outputs["pressure_imag_f32"],
            name="outputs.pressure_imag_f32",
            dtype="float32",
            shape=pressure_shape,
        )
    if not uses_batch_field_outputs:
        field_shape = _validate_descriptor(
            outputs["observation_pressure_real_f32"],
            name="outputs.observation_pressure_real_f32",
            dtype="float32",
            shape=(obs_shape[1],),
        )
        _validate_descriptor(
            outputs["observation_pressure_imag_f32"],
            name="outputs.observation_pressure_imag_f32",
            dtype="float32",
            shape=field_shape,
        )


def _validate_batch_assembly_solve_field_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("matrix_layout") != MATRIX_LAYOUT_ROW_MAJOR_C:
        raise ValueError(
            "batch assembly-solve-field manifest must use row_major_c layout"
        )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(
            "batch assembly-solve-field manifest must contain non-empty cases"
        )
    uses_batch_field_outputs = "batch_outputs" in manifest
    if uses_batch_field_outputs:
        batch_outputs = _require_descriptor_group(
            manifest.get("batch_outputs"),
            name="batch_outputs",
            keys=(
                "observation_pressure_real_f32",
                "observation_pressure_imag_f32",
            ),
        )
        field_shape = _validate_descriptor(
            batch_outputs["observation_pressure_real_f32"],
            name="batch_outputs.observation_pressure_real_f32",
            dtype="float32",
            rank=2,
        )
        if field_shape[0] != len(cases):
            raise ValueError(
                "batch_outputs observation pressure first dimension must match "
                "case count"
            )
        _validate_descriptor(
            batch_outputs["observation_pressure_imag_f32"],
            name="batch_outputs.observation_pressure_imag_f32",
            dtype="float32",
            shape=field_shape,
        )
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(
                f"batch assembly-solve-field case {idx} must be an object"
            )
        _validate_assembly_solve_field_case_manifest(
            case,
            uses_batch_field_outputs=uses_batch_field_outputs,
        )


def _validate_field_manifest(manifest: dict[str, Any]) -> None:
    _validate_field_case_manifest(manifest)


def _validate_field_case_manifest(manifest: dict[str, Any]) -> None:
    pressure = _require_descriptor_group(
        manifest.get("pressure_p1"),
        name="pressure_p1",
        keys=("real_f32", "imag_f32"),
    )
    pressure_shape = _validate_descriptor(
        pressure["real_f32"],
        name="pressure_p1.real_f32",
        dtype="float32",
        rank=1,
    )
    _validate_descriptor(
        pressure["imag_f32"],
        name="pressure_p1.imag_f32",
        dtype="float32",
        shape=pressure_shape,
    )

    neumann = _require_descriptor_group(
        manifest.get("neumann_dp0"),
        name="neumann_dp0",
        keys=("real_f32", "imag_f32"),
    )
    neumann_shape = _validate_descriptor(
        neumann["real_f32"],
        name="neumann_dp0.real_f32",
        dtype="float32",
        rank=1,
    )
    _validate_descriptor(
        neumann["imag_f32"],
        name="neumann_dp0.imag_f32",
        dtype="float32",
        shape=neumann_shape,
    )

    obs_shape = _validate_descriptor(
        manifest.get("observation_points"),
        name="observation_points",
        dtype="float32",
        rank=2,
    )
    if obs_shape[0] != 3:
        raise ValueError("observation_points must have shape (3, n_obs)")

    output = _require_descriptor_group(
        manifest.get("output"),
        name="output",
        keys=("pressure_real_f32", "pressure_imag_f32"),
    )
    expected = (obs_shape[1],)
    _validate_descriptor(
        output["pressure_real_f32"],
        name="output.pressure_real_f32",
        dtype="float32",
        shape=expected,
    )
    _validate_descriptor(
        output["pressure_imag_f32"],
        name="output.pressure_imag_f32",
        dtype="float32",
        shape=expected,
    )


def _validate_batch_field_manifest(manifest: dict[str, Any]) -> None:
    shared_obs_shape = None
    if manifest.get("observation_points") is not None:
        shared_obs_shape = _validate_descriptor(
            manifest.get("observation_points"),
            name="observation_points",
            dtype="float32",
            rank=2,
        )
        if shared_obs_shape[0] != 3:
            raise ValueError("observation_points must have shape (3, n_obs)")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("batch field manifest must contain non-empty cases")
    for idx, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"batch field case {idx} must be an object")
        if shared_obs_shape is None:
            _validate_field_case_manifest(case)
            continue
        case_with_obs = dict(case)
        case_with_obs["observation_points"] = manifest["observation_points"]
        _validate_field_case_manifest(case_with_obs)


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
    real_name: str,
    imag_name: str,
    relative_to: Path,
) -> dict[str, BinaryArrayDescriptor]:
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


def _require_observation_points_3xn(value: NDArray[Any]) -> NDArray[np.float32]:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"observation_points must be 2D, got {points.shape}")
    if points.shape[0] == 3:
        points_3xn = points
    elif points.shape[1] == 3:
        points_3xn = points.T
    else:
        raise ValueError(
            "observation_points must have shape (3, n) or (n, 3), "
            f"got {points.shape}"
        )
    if points_3xn.shape[1] == 0:
        raise ValueError("observation_points must contain at least one point")
    if not np.all(np.isfinite(points_3xn)):
        raise ValueError("observation_points must contain only finite values")
    return np.ascontiguousarray(points_3xn, dtype=np.float32)

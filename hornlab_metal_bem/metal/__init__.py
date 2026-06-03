"""Language-neutral Metal backend contract with optional validation helpers."""
from __future__ import annotations

from .geometry import (
    MetalGeometryBuffers,
    MetalGeometryError,
    build_metal_geometry_buffers,
)
from .backend import DenseBieSystem, MetalBemBackend, MetalBemContext
from .native import (
    MetalNativeRuntimeConfig,
    MetalNativeRuntimeStatus,
    MetalNativeSessionInfo,
    MetalNativeStandardSession,
    assert_native_runtime_available,
    discover_native_runtime,
    validate_session_with_native_helper,
)
from .session import (
    INDEX_BASE,
    MATRIX_LAYOUT_ROW_MAJOR_C,
    METAL_STANDARD_SCHEMA,
    AssemblyPayload,
    BinaryArrayDescriptor,
    DenseAssemblyResult,
    FieldPayload,
    FieldResult,
    GeometryPayload,
    payload_to_manifest,
    read_json_manifest,
    write_binary_array,
    write_geometry_buffers,
    write_json_manifest,
)

__all__ = [
    "INDEX_BASE",
    "MATRIX_LAYOUT_ROW_MAJOR_C",
    "METAL_STANDARD_SCHEMA",
    "AssemblyPayload",
    "BinaryArrayDescriptor",
    "DenseAssemblyResult",
    "DenseBieSystem",
    "FieldPayload",
    "FieldResult",
    "GeometryPayload",
    "MetalGeometryBuffers",
    "MetalGeometryError",
    "MetalBemBackend",
    "MetalBemContext",
    "MetalNativeRuntimeConfig",
    "MetalNativeRuntimeStatus",
    "MetalNativeSessionInfo",
    "MetalNativeStandardSession",
    "assert_native_runtime_available",
    "build_metal_geometry_buffers",
    "discover_native_runtime",
    "payload_to_manifest",
    "read_json_manifest",
    "validate_session_with_native_helper",
    "write_binary_array",
    "write_geometry_buffers",
    "write_json_manifest",
]

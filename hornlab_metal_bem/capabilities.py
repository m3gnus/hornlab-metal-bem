"""Versioned package capabilities, independent of this host's Metal readiness."""

from __future__ import annotations

from dataclasses import fields
from importlib import metadata
from typing import Any

from .config import (
    BIEFormulation,
    NATIVE_GROUND_PLANES,
    NATIVE_SYMMETRY_PLANES,
    SolveConfig,
    SourceMotion,
)

# Change this when the report's structure changes, not when a feature is enabled.
CAPABILITY_SCHEMA_VERSION = 1

# Version of the public SolveConfig request contract, not the native helper IPC.
# Change this for incompatible request names, values, or semantics.
REQUEST_SCHEMA_VERSION = 1


def capabilities() -> dict[str, Any]:
    """Return fresh, JSON-serialisable metadata without starting the solver.

    Request fields come from the public dataclass, and mode values come from
    the same constants its validator uses. Availability is a separate query:
    ``hornlab_metal_bem.metal.discover_native_runtime``.
    """
    request_fields = [field.name for field in fields(SolveConfig) if field.init]
    try:
        package_version = metadata.version("hornlab-metal-bem")
    except metadata.PackageNotFoundError:
        package_version = "unknown"

    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "package": "hornlab-metal-bem",
        "package_version": package_version,
        "request_fields": request_fields,
        "features": {
            "source_motion": [SourceMotion.NORMAL, SourceMotion.AXIAL],
            "source_axes": "source_axes" in request_fields,
            "infinite_baffle": {
                "supported": "aperture_tag" in request_fields,
                "field": "aperture_tag",
                "formulations": [BIEFormulation.STANDARD, BIEFormulation.COMPLEX_K],
            },
            "formulation": [
                BIEFormulation.STANDARD,
                BIEFormulation.COMPLEX_K,
                BIEFormulation.BURTON_MILLER,
            ],
            "complex_k_shift": "complex_k_shift" in request_fields,
            "frame_override": "frame_override" in request_fields,
            "ground_plane": {
                "supported": "ground_plane" in request_fields,
                "planes": list(NATIVE_GROUND_PLANES),
                "composes_with_symmetry": False,
                "composes_with_infinite_baffle": False,
                "formulations": [BIEFormulation.STANDARD, BIEFormulation.COMPLEX_K],
            },
            "native_symmetry_plane": [None, *NATIVE_SYMMETRY_PLANES],
            "on_frequency_result": "on_frequency_result" in request_fields,
            "return_surface_traces": "return_surface_traces" in request_fields,
            "require_closed_mesh": "require_closed_mesh" in request_fields,
            "workers": "workers" in request_fields,
        },
        "conventions": {
            "time_convention": "exp(-i*omega*t)",
            "outgoing_wave": "exp(+i*k*r)",
            "neumann_coefficient": "q = +i*rho*omega*v_n",
        },
    }

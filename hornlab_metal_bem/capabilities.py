"""Shared BEM capability contract; package support is separate from readiness."""

from __future__ import annotations

from dataclasses import fields
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .config import (
    BIEFormulation,
    NATIVE_GROUND_PLANES,
    NATIVE_SYMMETRY_PLANES,
    SolveConfig,
    SourceMotion,
)

# This named schema is independent of BEAT's provider schema.
CAPABILITY_SCHEMA_VERSION = 1
# Version of the Python SolveConfig contract, not a native/wire protocol.
REQUEST_SCHEMA_VERSION = 1


def _feature(
    request_fields,
    *names,
    values=None,
    requires=None,
    refuses=(),
    composes_with=(),
    supported=None,
):
    return {
        "supported": all(name in request_fields for name in names)
        if supported is None
        else supported,
        "request_fields": list(names),
        "values": values,
        "requires": {} if requires is None else requires,
        "refuses": list(refuses),
        "composes_with": list(composes_with),
    }


def _refusal(identifier, reason, *names):
    return {"id": identifier, "request_fields": list(names), "reason": reason}


def _supports(**kwargs):
    """Read support from real config/image guards without a mesh or runtime."""
    try:
        SolveConfig(**kwargs)
    except (ValueError, NotImplementedError):
        return False
    return True


def capabilities() -> dict[str, Any]:
    """Return fresh package support metadata, without probing runtime readiness."""
    request_fields = [item.name for item in fields(SolveConfig) if item.init]
    try:
        package_version = version("hornlab-metal-bem")
    except PackageNotFoundError:
        package_version = None
    symmetry = [
        plane
        for plane in NATIVE_SYMMETRY_PLANES
        if _supports(native_symmetry_plane=plane)
    ]
    ground = list(NATIVE_GROUND_PLANES)
    image_formulations = [
        value
        for value in [
            BIEFormulation.STANDARD,
            BIEFormulation.COMPLEX_K,
            BIEFormulation.BURTON_MILLER,
        ]
        if _supports(ground_plane=ground[0], formulation=value)
    ]
    compositions = [
        {"native_symmetry_plane": plane, "ground_plane": gp}
        for plane in symmetry
        for gp in ground
        if _supports(native_symmetry_plane=plane, ground_plane=gp)
    ]

    def f(*names, **kwargs):
        return _feature(request_fields, *names, **kwargs)

    r = _refusal
    ib_refusals = [
        r(
            "ib_burton_miller",
            "Coupled infinite baffle refuses Burton-Miller.",
            "formulation",
        ),
        r(
            "ib_surface_traces",
            "Coupled infinite baffle cannot retain generic surface traces.",
            "return_surface_traces",
        ),
        r(
            "ib_xy_symmetry",
            "Coupled infinite baffle refuses xy symmetry.",
            "native_symmetry_plane",
        ),
        r(
            "ib_assembly",
            "Coupled infinite baffle requires corrected native assembly.",
            "metal_native_assembly_mode",
        ),
        r(
            "ib_ground",
            "Coupled infinite baffle refuses ground images.",
            "ground_plane",
        ),
    ]
    formulation_refusals = [
        r(
            "bm_robin",
            "Burton-Miller refuses Robin impedance data.",
            "impedance_sources",
        ),
        r(
            "bm_infinite_baffle",
            "Burton-Miller refuses coupled infinite baffle.",
            "aperture_tag",
        ),
        r(
            "bm_ground",
            "Burton-Miller refuses ground-plane image assembly.",
            "ground_plane",
        ),
        r(
            "bm_robin_callback",
            "Burton-Miller refuses Robin impedance callbacks.",
            "impedance_source_callback",
        ),
        r("bm_chief", "Burton-Miller refuses CHIEF points.", "chief_points"),
    ]
    ground_refusals = [
        r("ground_burton_miller", "Ground images refuse Burton-Miller.", "formulation"),
        r(
            "ground_symmetry",
            "Ground images refuse all native symmetry.",
            "native_symmetry_plane",
        ),
        r(
            "ground_infinite_baffle",
            "Ground images refuse coupled infinite baffle.",
            "aperture_tag",
        ),
    ]
    return {
        "schema": "hornlab-bem-capabilities",
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "package": "hornlab-metal-bem",
        "package_version": package_version,
        "request_fields": request_fields,
        "conventions": {
            "time_convention": "exp(-i*omega*t)",
            "outgoing_wave": "exp(+i*k*r)",
        },
        "features": {
            "source_motion": f(
                "source_motion", values=[SourceMotion.NORMAL, SourceMotion.AXIAL]
            ),
            "source_axes": f(
                "source_axes",
                requires={
                    "motion": "source_motion='axial' or an AxialProfile for each tag.",
                    "coverage": "Every axial velocity-source tag must have an axis.",
                    "geometry": "Axes must be finite, nonzero, and lie in the symmetry subspace.",
                },
            ),
            "formulation": f(
                "formulation",
                values=[
                    BIEFormulation.STANDARD,
                    BIEFormulation.COMPLEX_K,
                    BIEFormulation.BURTON_MILLER,
                ],
                refuses=formulation_refusals,
            ),
            "complex_k_shift": f("complex_k_shift"),
            "frame_override": f("frame_override"),
            "infinite_baffle": f(
                "aperture_tag",
                requires={
                    "formulation": list(image_formulations),
                    "metal_native_assembly_mode": ["corrected"],
                },
                refuses=ib_refusals,
                composes_with=[
                    {"feature": "native_symmetry", "values": ["yz", "xz", "yz+xz"]}
                ],
            ),
            "native_symmetry": f(
                "native_symmetry_plane",
                values=symmetry,
                requires={
                    "formulation": list(
                        [
                            BIEFormulation.STANDARD,
                            BIEFormulation.COMPLEX_K,
                            BIEFormulation.BURTON_MILLER,
                        ]
                    )
                },
                refuses=[],
                composes_with=[{"feature": "ground_plane", "values": compositions}]
                if compositions
                else [],
            ),
            "ground_plane": f(
                "ground_plane",
                values=ground,
                requires={"formulation": image_formulations},
                refuses=ground_refusals,
                composes_with=[{"feature": "native_symmetry", "values": compositions}]
                if compositions
                else [],
            ),
            "explicit_frequencies": f(values=["solve_frequencies"], supported=True),
            "on_frequency_result": f("on_frequency_result", requires={}, refuses=[]),
            "return_surface_traces": f(
                "return_surface_traces",
                refuses=[
                    r(
                        "traces_infinite_baffle",
                        "Surface traces refuse coupled infinite baffle.",
                        "aperture_tag",
                    )
                ],
            ),
            "require_closed_mesh": f("require_closed_mesh"),
            "workers": f("workers", refuses=[]),
            "assembly_backend": f(
                "metal_native_assembly_mode",
                values=["corrected", "optimized", "reference", "parity"],
            ),
        },
    }

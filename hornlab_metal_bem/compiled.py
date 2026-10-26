"""Opt-in compatibility with BEAT's compiled exterior request envelope.

This module is mirrored in the two legacy solvers to avoid making either a
runtime dependency of the other. The tests and release handoff check both
copies. No legacy entry point calls this adapter.

The supported subset follows BEAT's exterior solve path: one unbounded air
region over one or more closed BEM surfaces, ideal velocity sources (uniform
normal or rigid translation, with boundary motion weights) and, under
compiled-system v3, exterior electrodynamic transducers coupled through a
lumped electrical/mechanical network. Off, x, xy and rigid-ground images map
onto the native symmetry and ground-plane solves. Every motion-basis column is
an ordinary native solve; impedances, boundary traces, transducer velocities
and currents are reduced from those columns on the host in float64.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from .config import (
    AxialProfile, ObservationConfig, SolveConfig, VelocityMode,
    _validate_source_axis_symmetry,
)
from .observation import ObservationFrame
from .source_identity import solver_source_identity

_IS_METAL = __package__ == "hornlab_metal_bem"

NEGATIVE_TIME = "exp(-i omega t)"
POSITIVE_TIME = "exp(+i omega t)"
DEFAULT_REFERENCE_VOLTAGE_V = 2.83
_BEAT_AXIS_SYMMETRY_TOLERANCE = 1e-8

# BEAT image modes -> native keyword, image-plane coordinate axes, and the
# number of physical radiator copies (ground images are not real radiators).
_SYMMETRY_KWARGS = {
    "off": {},
    # The adapter runs BEAT's own domain/closure checks, so a representative
    # may be a cut body or a whole driver whose mirrors are further drivers.
    "x": {"native_symmetry_plane": "yz", "native_symmetry_complete_images": True},
    "xy": {"native_symmetry_plane": "yz+xz", "native_symmetry_complete_images": True},
    "ground": {"ground_plane": "xz"},
}
_IMAGE_AXES = {"off": (), "x": (0,), "xy": (0, 1), "ground": (1,)}
_PHYSICAL_COPIES = {"off": 1, "x": 2, "xy": 4, "ground": 1}
_FRACTIONAL_AXES = {"off": (), "x": ("x",), "xy": ("x", "y"), "ground": ()}

_OUTPUTS = {
    "exterior_pressure": ("Pa", ("frequency", "excitation", "observation")),
    "bem_boundary_pressure": ("Pa", ("frequency", "excitation", "bem_node")),
    "bem_boundary_neumann": ("Pa/m", ("frequency", "excitation", "bem_face")),
    "radiation_impedance": ("N*s/m", ("frequency", "radiator")),
    "radiation_impedance_matrix": (
        "N*s/m",
        ("frequency", "receiver_component", "source_component"),
    ),
    "diaphragm_velocity": ("m/s", ("frequency", "excitation", "transducer")),
    "voice_coil_current": ("A", ("frequency", "excitation", "transducer")),
}
_TRANSDUCER_SCALARS = (
    "re_ohm",
    "le_h",
    "bl_n_per_a",
    "mmd_kg",
    "cms_m_per_n",
    "rms_n_s_per_m",
)
_TRANSDUCER_OPTIONAL = {
    "motion_profile",
    "boundary_motion_signs",
    "boundary_motion_weights",
    "symmetry_role",
    "surface_completion_factor",
    "physical_driver_orbit_count",
    "fractional_symmetry_axes",
    "semi_inductance",
    "lumped_sealed_rear_chamber",
}
_SEMI_INDUCTANCE = ("re_prime_ohm", "leb_h", "le_h", "ke_semi_h", "rss_ohm")
_SEALED_CHAMBER = ("volume_m3", "projected_area_m2")


@dataclass(frozen=True)
class CompiledQuantity:
    """One requested output, with a leading frequency axis.

    ``axes`` names every axis of ``values``; the remaining axes, units and
    metadata follow BEAT's exterior result quantity of the same name.
    Frequency-dependent metadata (reciprocity, passivity) is a list with one
    value per frequency.
    """

    id: str
    quantity: str
    unit: str
    axes: tuple[str, ...]
    values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    target_id: str | None = None


@dataclass(frozen=True)
class CompiledResult:
    """Response bases in the request's phasor convention.

    ``pressure`` maps each ``exterior_pressure`` output to an array with axes
    (frequency, excitation, point) in Pa per 1 m/s ideal-port velocity or per
    reference voltage on a transducer port. ``quantities`` holds every
    requested output, pressure included. Arrays retain the request's
    frequency, excitation and point order. ``native_results`` holds one
    native solve per motion-basis column (``motion_basis_component_ids``),
    always in exp(-i omega t). This is a Python result, not BEAT's
    worker/result wire protocol.
    """

    frequencies_hz: np.ndarray
    excitation_port_ids: tuple[str, ...]
    pressure: dict[str, np.ndarray]
    native_results: tuple[Any, ...]
    time_convention: str = NEGATIVE_TIME
    quantities: dict[str, CompiledQuantity] = field(default_factory=dict)
    motion_basis_component_ids: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)


def compiled_capabilities() -> dict[str, Any]:
    """Package support only; does not import BEAT or probe a runtime."""
    return {
        "request_schema_version": 1,
        "compiled_system_versions": [1, 2, 3],
        "backend": "metal" if _IS_METAL else "cpu",
        "precisions": ["float32"] if _IS_METAL else ["float32", "float64"],
        "components": ["ideal_velocity_source", "electrodynamic_transducer"],
        "motion_profiles": ["uniform_normal", "rigid_translation"],
        "outputs": list(_OUTPUTS),
        "symmetry": list(_SYMMETRY_KWARGS),
        "phasor_conventions": [NEGATIVE_TIME, POSITIVE_TIME],
        "limitations": [
            "one exterior region of closed outward BEM surfaces",
            "rigid and moving boundaries only",
            "independent unit-velocity or reference-voltage bases",
            "undriven transducers are shorted",
            "no FEM regions, interfaces, losses, Robin walls or passive radiators",
        ],
        "time_convention": NEGATIVE_TIME,
        "requires": "official beat-engine contract validator (optional for legacy APIs)",
    }


def _package_version() -> str:
    return solver_source_identity()["version"]


def _mesh_provenance(spec: dict, base_dir: Path) -> dict:
    from .metal.native import _file_sha256

    if "mesh_data" in spec:
        # Hash the received payload, before scaling/translation/tag remapping.
        payload = json.dumps(spec["mesh_data"], separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        source = "memory"
    else:
        path = (base_dir / spec["file"]).resolve()
        if not path.is_relative_to(base_dir):
            _refuse("mesh file must remain inside base_dir")
        digest = _file_sha256(path)
        source = "file"
    return {
        "id": spec["id"], "source": source, "sha256": digest,
        # Local filenames are deliberately omitted from archived identity.
        "file": None,
        "scale_to_m": spec["scale_to_m"],
        "translation_m": copy.deepcopy(spec["translation_m"]),
    }


def _run_provenance(request, meshes, results, config, source_identity) -> dict:
    """BEAT Run Provenance v1, with Python/native-helper runtime identity.

    Source identity is captured at solve entry; unknown revision/device values
    stay null. Runtime identity comes from the executed sweep, never a fresh
    discovery that could identify a different helper after the solve.
    """
    selected_runtime = copy.deepcopy(getattr(results[0], "native_runtime", {}))
    source_identity = selected_runtime.pop("source", source_identity)
    runtime = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "machine": platform.machine(), "kernel": platform.system(),
        "helper": None,
        **selected_runtime,
    }
    effective = {
        key: getattr(config, key)
        for key in (
            "formulation", "complex_k_shift", "velocity_mode", "air_density", "speed_of_sound",
            "native_symmetry_plane", "ground_plane", "ground_plane_min_clearance_m",
            "source_axis_symmetry_tolerance", "metal_native_assembly_mode",
            "dense_solve_dtype", "dense_solve_implementation",
        )
    }
    return {
        "schema_version": 1,
        "engine": source_identity,
        "runtime": runtime,
        "execution": {
            "backend": "metal", "precision": config.dense_solve_dtype,
            "device": None,
            "device_query_error": "Native helper does not report device identity",
            "solver_options": copy.deepcopy(request["solver_options"]),
            "native_options": effective,
        },
        "meshes": meshes,
    }


def _refuse(message: str) -> None:
    raise ValueError(f"Legacy compiled BEAT adapter: {message}")


def _empty(value: Any, label: str) -> None:
    if value:
        _refuse(f"{label} is not supported")


def _keys(value: dict, allowed: set[str], label: str) -> None:
    extra = set(value) - allowed
    if extra:
        _refuse(f"unsupported {label}: {sorted(extra)}")


def _numeric_array(value: Any, label: str) -> np.ndarray:
    def numeric(v):
        return (
            all(numeric(x) for x in v)
            if isinstance(v, list)
            else type(v) in (int, float)
        )

    if not numeric(value):
        _refuse(f"{label} must contain JSON numbers")
    a = np.asarray(value)
    if a.dtype.kind not in "ifu" or a.dtype.kind == "b":
        _refuse(f"{label} must contain JSON numbers")
    return a.astype(float)


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        _refuse(f"{label} must be a finite number")
    return float(value)


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    v = _number(value, label)
    if v < 0 or (v == 0 and not allow_zero):
        _refuse(f"{label} must be {'non-negative' if allow_zero else 'positive'}")
    return v


def _count(value: Any, label: str) -> int:
    if type(value) not in (int, float) or value not in (1, 2, 4):
        _refuse(f"{label} must be 1, 2 or 4")
    return int(value)


def _axis(value: Any) -> tuple[float, float, float]:
    a = _numeric_array(value, "motion_axis")
    if a.shape != (3,) or not np.isfinite(a).all() or not np.any(a):
        _refuse("motion_axis must be a finite nonzero three-vector")
    a = a / np.max(np.abs(a))
    return tuple(float(x) for x in a / np.linalg.norm(a))


def _require_in_planes(axis, symmetry: str, label: str) -> None:
    _validate_source_axis_symmetry(
        axis, _SYMMETRY_KWARGS[symmetry].get("native_symmetry_plane"),
        tolerance=_BEAT_AXIS_SYMMETRY_TOLERANCE, label=f"{label} motion_axis",
    )


def _boundary_map(raw: Any, boundary_ids: list[str], label: str) -> dict:
    if not isinstance(raw, dict):
        _refuse(f"{label} must be an object")
    unrelated = set(raw) - set(boundary_ids)
    if unrelated:
        _refuse(f"{label} reference unrelated boundaries: {sorted(unrelated)}")
    return raw


def _optional_model(raw: Any, names: tuple[str, ...], label: str) -> dict | None:
    """Parse BEAT's optional ``{enabled, ...}`` lumped sub-models."""
    if not isinstance(raw, dict):
        _refuse(f"{label} must be an object")
    _keys(raw, {*names, "enabled"}, f"{label} parameters")
    enabled = raw.get("enabled", False)
    if type(enabled) is not bool:
        _refuse(f"{label} enabled must be boolean")
    values = {n: _positive(raw[n], f"{label} {n}") for n in names if n in raw}
    if not enabled:
        return None
    if len(values) != len(names):
        _refuse(f"enabled {label} is missing: {sorted(set(names) - set(values))}")
    return values


def _ideal_column(component, bounds, tag_of, symmetry, contract_version) -> dict:
    p = component["parameters"]
    _keys(
        p,
        {"motion_profile", "motion_axis", "boundary_motion_weights"},
        "component parameters",
    )
    motion = p.get("motion_profile", "uniform_normal")
    if motion not in {"uniform_normal", "rigid_translation"}:
        _refuse("motion_profile must be uniform_normal or rigid_translation")
    if motion == "rigid_translation" and contract_version < 2:
        _refuse("rigid_translation requires compiled-system v2")
    if motion == "uniform_normal" and "motion_axis" in p:
        _refuse("normal motion cannot carry an axis")
    axis = None
    if motion == "rigid_translation":
        axis = _axis(p.get("motion_axis"))
        _require_in_planes(axis, symmetry, "rigid-translation source")
    weights = _boundary_map(
        p.get("boundary_motion_weights", {}),
        component["boundary_ids"],
        "boundary_motion_weights",
    )
    coefficients = {}
    for bid in component["boundary_ids"]:
        if bounds[bid]["kind"] != "moving":
            _refuse("ideal sources may own moving boundaries only")
        coefficients[tag_of[bid]] = _positive(
            weights.get(bid, 1.0), "boundary motion weight"
        )
    copies = _PHYSICAL_COPIES[symmetry]
    return {
        "component_id": component["id"],
        "kind": "ideal_velocity_source",
        "coefficients": coefficients,
        "axis": axis,
        # Ideal rows integrate force over every real copy; orbit stays 1.
        "force_scale": copies,
        "completion": copies,
        "orbit": 1,
    }


def _transducer_column(component, bounds, tag_of, symmetry) -> dict:
    cid = component["id"]
    p = component["parameters"]
    missing = [n for n in (*_TRANSDUCER_SCALARS, "motion_axis") if n not in p]
    if missing:
        _refuse(f"electrodynamic component {cid!r} is missing: {missing}")
    _keys(
        p,
        {*_TRANSDUCER_SCALARS, "motion_axis", *_TRANSDUCER_OPTIONAL},
        "electrodynamic parameters",
    )
    if p.get("motion_profile", "rigid_translation") != "rigid_translation":
        _refuse("electrodynamic components require rigid_translation motion")
    axis = _axis(p["motion_axis"])
    completion = _count(
        p.get("surface_completion_factor", 1), "surface_completion_factor"
    )
    orbit = _count(
        p.get("physical_driver_orbit_count", 1), "physical_driver_orbit_count"
    )
    role = p.get("symmetry_role", "complete_representative")
    if role != ("fractional_driver" if completion > 1 else "complete_representative"):
        _refuse("symmetry_role is inconsistent with surface_completion_factor")
    fractional = p.get("fractional_symmetry_axes", [])
    if (
        not isinstance(fractional, list)
        or len(set(fractional)) != len(fractional)
        or not set(fractional) <= set(_FRACTIONAL_AXES[symmetry])
    ):
        _refuse("fractional_symmetry_axes must be unique active x/y symmetry axes")
    if completion != 2 ** len(fractional):
        _refuse(
            "surface_completion_factor must equal 2 raised to the number of "
            "fractional_symmetry_axes"
        )
    _require_in_planes(axis, symmetry, f"transducer {cid!r}")
    if completion * orbit != _PHYSICAL_COPIES[symmetry]:
        _refuse(
            f"transducer {cid!r} represents {completion * orbit} symmetry images, "
            f"but {symmetry} symmetry requires {_PHYSICAL_COPIES[symmetry]}"
        )
    signs = _boundary_map(
        p.get("boundary_motion_signs", {}), component["boundary_ids"],
        "boundary_motion_signs",
    )
    weights = _boundary_map(
        p.get("boundary_motion_weights", {}), component["boundary_ids"],
        "boundary_motion_weights",
    )
    coefficients = {}
    for bid in component["boundary_ids"]:
        if bounds[bid]["kind"] != "moving":
            _refuse(f"transducer {cid!r} boundary {bid!r} is not moving")
        sign = signs.get(bid, 1)
        if type(sign) not in (int, float) or sign not in (-1, 1):
            _refuse("electrodynamic boundary motion signs must be -1 or +1")
        coefficients[tag_of[bid]] = float(sign) * _positive(
            weights.get(bid, 1.0), "boundary motion weight"
        )
    lem = {n: _number(p[n], n) for n in _TRANSDUCER_SCALARS}
    for n in ("re_ohm", "bl_n_per_a", "mmd_kg", "cms_m_per_n"):
        if lem[n] <= 0:
            _refuse(f"{n} must be greater than zero")
    for n in ("le_h", "rms_n_s_per_m"):
        if lem[n] < 0:
            _refuse(f"{n} must not be negative")
    if "semi_inductance" in p:
        lem["semi_inductance"] = _optional_model(
            p["semi_inductance"], _SEMI_INDUCTANCE, "semi_inductance"
        )
    if "lumped_sealed_rear_chamber" in p:
        lem["sealed_chamber"] = _optional_model(
            p["lumped_sealed_rear_chamber"], _SEALED_CHAMBER,
            "lumped_sealed_rear_chamber",
        )
    return {
        "component_id": cid,
        "kind": "electrodynamic_transducer",
        "coefficients": coefficients,
        "axis": axis,
        # Completion enters the force rows; orbit only the row weights.
        "force_scale": completion,
        "completion": completion,
        "orbit": orbit,
        "lem": lem,
    }


def _column_configs(basis: list[dict], kwargs: dict) -> list[dict]:
    """Native velocity settings for each motion-basis column."""
    tags = sorted({t for col in basis for t in col["coefficients"]})
    axes = {
        t: col["axis"] for col in basis if col["axis"] for t in col["coefficients"]
    }
    if len(axes) == len(tags):
        motion = {"source_motion": "axial", "source_axes": axes}
    elif not axes:
        motion = {"source_motion": "normal", "source_axes": None}
    else:
        # Mixed profiles: per-tag axial overrides on a normal default.
        motion = {
            "source_motion": "normal",
            "source_axes": axes,
            "source_velocity_profiles": {t: AxialProfile() for t in axes},
        }
    return [
        {
            **kwargs,
            **motion,
            "velocity_sources": {
                t: float(col["coefficients"].get(t, 0.0)) for t in tags
            },
        }
        for col in basis
    ]


def _plan(request: dict) -> dict:
    # BEAT remains authoritative for syntax/versions/references. Import lazily:
    # the old packages and Boundary Lab never need BEAT for their existing API.
    try:
        from beat_engine.beat_contract import validate_solve_request
    except ImportError as exc:
        raise RuntimeError(
            "solve_compiled requires official beat-engine installed alongside this package"
        ) from exc
    validate_solve_request(request)
    system = request["compiled_system"]
    version = system["contract_version"]
    if version not in (1, 2, 3):
        _refuse("only compiled-system v1, v2 and v3 are supported")
    _empty(system["interfaces"], "FEM/BEM interfaces")
    if len(system["regions"]) != 1 or system["regions"][0]["kind"] != "unbounded_air":
        _refuse("requires exactly one unbounded air region; FEM regions are not supported")
    region = system["regions"][0]
    meshes = {m["id"]: m for m in system["meshes"]}
    if not region["mesh_ids"] or set(region["mesh_ids"]) != set(meshes):
        _refuse("the exterior region must own every mesh")
    if any(meshes[mid]["purpose"] != "bem_surface" for mid in region["mesh_ids"]):
        _refuse("requires exterior BEM surfaces")
    _empty(region["loss_model"], "region losses")
    _empty(region["volume_groups"], "volume groups")
    if not _IS_METAL and region["sound_speed_m_per_s"] != 343.0:
        _refuse("BEM++ supports sound_speed_m_per_s=343 only")
    options = request["solver_options"]
    _keys(
        options,
        {
            "precision",
            "bem_backend",
            "symmetry",
            "phasor_convention",
            "regular_quadrature_mode",
            "quadrature_order",
            "singular_order",
            "ground_plane_min_clearance_m",
            "transducer_reference_voltage_v",
        },
        "solver options",
    )
    symmetry = str(options.get("symmetry", "off")).strip().lower()
    if symmetry not in _SYMMETRY_KWARGS:
        _refuse("symmetry must be off, x, xy or ground")
    clearance = _positive(
        options.get("ground_plane_min_clearance_m", 0.0),
        "ground_plane_min_clearance_m",
        allow_zero=True,
    )
    if clearance and symmetry != "ground":
        _refuse("ground_plane_min_clearance_m requires ground symmetry")

    # Physical tags may repeat across meshes; give each (mesh, tag) a unique
    # solver tag, keeping the original numbers wherever they do not collide.
    declared: dict[str, list[int]] = {mid: [] for mid in region["mesh_ids"]}
    for b in system["boundaries"]:
        g = b["group"]
        if b["region_id"] != region["id"] or g["mesh_id"] not in meshes or g["dimension"] != 2:
            _refuse("boundary must own a surface group on an exterior mesh")
        if b["kind"] not in {"rigid", "moving"}:
            _refuse("only rigid and moving boundaries are supported")
        _empty(b["parameters"], "boundary parameters/Robin walls")
        if g["tag"] in declared[g["mesh_id"]]:
            _refuse("each physical surface tag needs exactly one boundary")
        declared[g["mesh_id"]].append(g["tag"])
    taken = {t for tags in declared.values() for t in tags}
    next_tag, used, mesh_tags = max(taken, default=0) + 1, set(), {}
    for mid in region["mesh_ids"]:
        mesh_tags[mid] = {}
        for t in declared[mid]:
            if t in used:
                t_new, next_tag = next_tag, next_tag + 1
            else:
                t_new = t
            used.add(t_new)
            mesh_tags[mid][t] = t_new
    bounds = {b["id"]: b for b in system["boundaries"]}
    tag_of = {
        bid: mesh_tags[b["group"]["mesh_id"]][b["group"]["tag"]]
        for bid, b in bounds.items()
    }

    components = {c["id"]: c for c in system["components"]}
    ports = {p["id"]: p for p in system["excitation_ports"]}
    owners: set[str] = set()
    for c in system["components"]:
        if c["kind"] not in {"ideal_velocity_source", "electrodynamic_transducer"}:
            _refuse(f"{c['kind']} components are not supported")
        for bid in c["boundary_ids"]:
            if bid in owners:
                _refuse("moving boundaries must belong to exactly one component")
            owners.add(bid)
    if owners != {bid for bid, b in bounds.items() if b["kind"] == "moving"}:
        _refuse("every moving boundary must belong to exactly one component")
    for p in ports.values():
        kind = components[p["component_id"]]["kind"]
        expected = "voltage" if kind == "electrodynamic_transducer" else "normal_velocity"
        if p["kind"] != expected:
            _refuse(f"{kind} excitation ports must be {expected} ports")
    port_ids = list(request["excitation_port_ids"])

    # Motion basis: every transducer in compiled order (undriven ones too, so
    # they load the network), then each requested ideal source once.
    basis = [
        _transducer_column(c, bounds, tag_of, symmetry)
        for c in system["components"]
        if c["kind"] == "electrodynamic_transducer"
    ]
    transducer_count = len(basis)
    for pid in port_ids:
        c = components[ports[pid]["component_id"]]
        if c["kind"] == "ideal_velocity_source" and all(
            col["component_id"] != c["id"] for col in basis
        ):
            basis.append(_ideal_column(c, bounds, tag_of, symmetry, version))
    for c in system["components"]:
        if c["kind"] == "ideal_velocity_source" and not any(
            col["component_id"] == c["id"] for col in basis
        ):
            # Unrequested ideal sources stay rigid; still validate them.
            _ideal_column(c, bounds, tag_of, symmetry, version)
    column_of = {col["component_id"]: i for i, col in enumerate(basis)}
    voltage = _positive(
        options.get("transducer_reference_voltage_v", DEFAULT_REFERENCE_VOLTAGE_V),
        "transducer_reference_voltage_v",
    )

    frequencies = np.array(request["frequencies_hz"], dtype=float)
    unique, inverse = np.unique(frequencies, return_inverse=True)
    order = np.argsort(np.unique(inverse, return_index=True)[1])
    unique_frequencies = unique[order]
    frequency_index = np.argsort(order)[inverse]

    backend = "metal" if _IS_METAL else "cpu"
    if options.get("bem_backend", backend) != backend:
        _refuse(f"bem_backend must be {backend}")
    if _IS_METAL and any(
        options.get(key, 4) != 4 for key in ("quadrature_order", "singular_order")
    ):
        # BEAT's default orders are accepted so unmodified BEAT requests run;
        # the native corrected assembly keeps its own fixed rules either way.
        _refuse(
            "Metal uses fixed native quadrature; only BEAT's default quadrature_order "
            "and singular_order (4) are accepted"
        )
    precision = options.get("precision", "float32" if _IS_METAL else "float64")
    if precision not in compiled_capabilities()["precisions"]:
        _refuse(f"unsupported precision {precision}")
    phasor = options.get("phasor_convention", NEGATIVE_TIME)
    if phasor not in (NEGATIVE_TIME, POSITIVE_TIME):
        _refuse(f"phasor_convention must be {NEGATIVE_TIME} or {POSITIVE_TIME}")
    if options.get("regular_quadrature_mode", "fixed") != "fixed":
        _refuse("only fixed regular quadrature is supported")
    for key in ("quadrature_order", "singular_order"):
        v = options.get(key, 4)
        if type(v) is not int or not 1 <= v <= 6:
            _refuse(f"{key} must be an integer in [1, 6]")

    output_points, slices, cursor, outputs = [], {}, 0, []
    for output in request["outputs"]:
        quantity, o = output["quantity"], output["options"]
        if quantity not in _OUTPUTS:
            _refuse(f"unsupported output quantity {quantity!r}")
        if quantity in ("diaphragm_velocity", "voice_coil_current") and not transducer_count:
            _refuse(f"{quantity} requires an electrodynamic transducer")
        targets = list(output["target_ids"])
        if quantity == "radiation_impedance_matrix":
            if len(set(targets)) != len(targets) or any(t not in column_of for t in targets):
                _refuse("radiation_impedance_matrix targets must be unique motion-basis components")
        elif targets:
            _refuse(f"{quantity} outputs must be untargeted")
        if quantity == "radiation_impedance" and any(
            c["kind"] == "ideal_velocity_source" and c["id"] not in column_of
            for c in system["components"]
        ):
            _refuse("radiation_impedance requires every ideal source to be excited")
        if quantity != "exterior_pressure":
            _keys(o, set(), f"{quantity} output options")
            outputs.append(output)
            continue
        _keys(o, {"points_m"}, "exterior pressure output options")
        pts = _numeric_array(o.get("points_m"), "pressure points")
        if (
            pts.ndim != 2
            or pts.shape[1] != 3
            or len(pts) == 0
            or not np.isfinite(pts).all()
        ):
            _refuse("pressure points must be a nonempty finite (N,3) array")
        output_points.append(pts)
        slices[output["id"]] = slice(cursor, cursor + len(pts))
        cursor += len(pts)
        outputs.append(output)
    if not outputs:
        _refuse("requires at least one supported output")
    if cursor > 100_000:
        _refuse("pressure point count exceeds 100000")
    surface_needed = transducer_count > 0 or any(
        o["quantity"]
        in ("bem_boundary_pressure", "radiation_impedance", "radiation_impedance_matrix")
        for o in outputs
    )
    kwargs = {
        "velocity_mode": VelocityMode.VELOCITY,
        "air_density": region["density_kg_per_m3"],
        "frame_override": ObservationFrame(
            axis=np.array([0.0, 0.0, 1.0]),
            origin=np.zeros(3),
            u=np.array([1.0, 0.0, 0.0]),
            v=np.array([0.0, 1.0, 0.0]),
            mouth_center=np.zeros(3),
            source_center=np.zeros(3),
        ),
        **_SYMMETRY_KWARGS[symmetry],
    }
    if clearance:
        kwargs["ground_plane_min_clearance_m"] = clearance
    if surface_needed:
        kwargs["return_surface_pressure"] = True
    if _IS_METAL:
        kwargs.update(
            speed_of_sound=region["sound_speed_m_per_s"],
            source_axis_symmetry_tolerance=_BEAT_AXIS_SYMMETRY_TOLERANCE,
            dense_solve_dtype="float32",
            metal_native_assembly_mode="corrected",
            mesh_merge_tol=0.0,
        )
    else:
        kwargs.update(
            precision="double" if precision == "float64" else "single",
            assembly_backend="numba",
            require_closed_mesh=True,
            slp_dlp_quadrature=options.get("quadrature_order", 4),
            hyp_adlp_quadrature=options.get("quadrature_order", 4),
            slp_dlp_singular_quadrature=options.get("singular_order", 4),
        )
    column_kwargs = _column_configs(basis, kwargs)
    # Construct now so impossible native options fail before reading a mesh.
    for ck in column_kwargs:
        SolveConfig(**ck, observation=ObservationConfig(planes=["compiled"]))
    return {
        "meshes": [meshes[mid] for mid in region["mesh_ids"]],
        "mesh_tags": mesh_tags,
        "symmetry": symmetry,
        "clearance": clearance,
        "basis": basis,
        "transducer_count": transducer_count,
        "column_of": column_of,
        "components": [c["id"] for c in system["components"]],
        "ideal_components": [
            c["id"] for c in system["components"] if c["kind"] == "ideal_velocity_source"
        ],
        "ports": ports,
        "port_ids": port_ids,
        "voltage": voltage,
        "density": float(region["density_kg_per_m3"]),
        "sound_speed": float(region["sound_speed_m_per_s"]),
        "frequencies": frequencies,
        "unique_frequencies": unique_frequencies,
        "frequency_index": frequency_index,
        "phasor": phasor,
        "precision": precision,
        "kwargs": kwargs,
        "column_kwargs": column_kwargs,
        "outputs": outputs,
        "output_points": output_points,
        "slices": slices,
    }


def _packed_array(raw: dict) -> np.ndarray:
    return (
        np.frombuffer(base64.b64decode(raw["data"], validate=True), dtype=raw["dtype"])
        .reshape(raw["shape"])
        .copy()
    )


def _read_mesh(spec: dict, base_dir: Path):
    """One compiled mesh in its source units: points, faces, tags, names."""
    if "mesh_data" in spec:
        data = spec["mesh_data"]
        if any(b["type"] != "triangle" for b in data["cells"]):
            _refuse("packed mesh supports triangle cells only")
        points = _packed_array(data["points"])
        faces = np.vstack([_packed_array(b["connectivity"]) for b in data["cells"]])
        tags = np.concatenate(
            [_packed_array(b["physical_tags"]) for b in data["cells"]]
        )
        return points, faces, tags, data["physical_names"]
    import meshio

    source = (base_dir / spec["file"]).resolve()
    if not source.is_relative_to(base_dir):
        _refuse("mesh file must remain inside base_dir")
    m = meshio.read(source)
    if any(c.type not in {"triangle", "line", "vertex"} for c in m.cells):
        _refuse("file mesh supports linear triangles, lines and vertices only")
    faces = m.cells_dict.get("triangle")
    tags = m.cell_data_dict.get("gmsh:physical", {}).get("triangle")
    if faces is None or tags is None:
        _refuse("file mesh needs physically tagged triangles")
    return m.points, faces, tags, m.field_data


def _surface(plan: dict, base_dir: Path) -> dict:
    """Combine, place and validate the exterior surface in the global frame."""
    all_points, all_faces, all_tags, offset = [], [], [], 0
    for spec in plan["meshes"]:
        points, faces, tags, names = _read_mesh(spec, base_dir)
        if any(str(n).strip().lower() == "mouth_aperture" for n in names):
            _refuse("coupled infinite-baffle mesh semantics are not supported")
        points = (
            np.asarray(points, dtype=float) * spec["scale_to_m"] + spec["translation_m"]
        )
        faces, tags = np.asarray(faces, dtype=np.int64), np.asarray(tags, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            _refuse("mesh coordinates must be finite (N,3)")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or not len(faces)
            or faces.min() < 0
            or faces.max() >= len(points)
        ):
            _refuse("mesh requires valid triangle indices")
        tag_map = plan["mesh_tags"][spec["id"]]
        if tags.shape != (len(faces),) or set(tags.tolist()) != set(tag_map):
            _refuse("mesh triangle tags must match all declared boundaries exactly")
        all_points.append(points)
        all_faces.append(faces + offset)
        all_tags.append(np.array([tag_map[t] for t in tags.tolist()], dtype=np.int64))
        offset += len(points)
    points = np.vstack(all_points)
    faces = np.vstack(all_faces)
    tags = np.concatenate(all_tags)

    symmetry = plan["symmetry"]
    axes = _IMAGE_AXES[symmetry]
    tolerance = max(1e-9, 1e-6 * float(np.linalg.norm(np.ptp(points, axis=0))))
    used = np.unique(faces)
    if symmetry in ("x", "xy"):
        # Snap the cut onto its plane, as BEAT does, then require the
        # positive fundamental domain.
        for a in axes:
            near = np.abs(points[:, a]) <= tolerance
            points[near, a] = 0.0
            if points[used, a].min() < 0.0:
                _refuse(
                    f"mesh is not in the positive {'XY'[a]} fundamental domain "
                    f"for {symmetry} symmetry"
                )
    elif symmetry == "ground":
        if points[used, 1].min() < -tolerance:
            _refuse("rigid-ground symmetry requires the whole radiator at Y >= 0")
        if np.any(np.all(np.abs(points[faces, 1]) <= tolerance, axis=1)):
            _refuse("rigid-ground symmetry cannot image triangles lying flat on Y=0")
        if plan["clearance"] and points[used, 1].min() < plan["clearance"]:
            _refuse("mesh violates ground_plane_min_clearance_m")
    tri = points[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    doubled_area = np.linalg.norm(cross, axis=1)
    if np.any(doubled_area == 0):
        _refuse("mesh contains degenerate triangles")
    if symmetry in ("x", "xy") and any(
        np.all(points[faces, a] == 0.0, axis=1).any() for a in axes
    ):
        _refuse("faces must not lie wholly on an active symmetry plane")

    edge_owners: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for index, f in enumerate(faces.tolist()):
        for a, b in zip(f, f[1:] + f[:1]):
            edge_owners.setdefault((min(a, b), max(a, b)), []).append(
                (index, 1 if a < b else -1)
            )
    for (a, b), owners in edge_owners.items():
        # Reflection supplies the other incidence of a seam on an image plane.
        on_plane = len(owners) == 1 and any(
            points[a, ax] == 0.0 and points[b, ax] == 0.0 for ax in axes
        )
        if not on_plane and (len(owners) != 2 or owners[0][1] == owners[1][1]):
            _refuse("mesh must be a closed consistently oriented manifold")
    neighbors = [set() for _ in faces]
    for v in edge_owners.values():
        if len(v) == 2:
            neighbors[v[0][0]].add(v[1][0])
            neighbors[v[1][0]].add(v[0][0])
    unseen = set(range(len(faces)))
    while unseen:
        pending, shell = [unseen.pop()], []
        while pending:
            i = pending.pop()
            shell.append(i)
            for n in neighbors[i] & unseen:
                unseen.remove(n)
                pending.append(n)
        # Virtual caps on the image planes contribute nothing about an origin
        # lying in every active plane.
        origin = tri[shell].mean(axis=(0, 1))
        origin[list(axes)] = 0.0
        t = tri[shell] - origin
        if np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))) <= 0:
            _refuse("each closed shell must have outward winding")
    return {
        "points": points,
        "faces": faces,
        "tags": tags,
        "used": used,
        "normals": cross / doubled_area[:, None],
        "areas": doubled_area / 2.0,
    }


def _write_surface(surface: dict, tags: set[int], target: Path) -> None:
    import meshio

    # No repair/merge is performed. Work in the exact global metre frame.
    meshio.write(
        target,
        meshio.Mesh(
            surface["points"],
            [("triangle", surface["faces"])],
            cell_data={
                "gmsh:physical": [surface["tags"]],
                "gmsh:geometrical": [surface["tags"]],
            },
            field_data={f"surface_{t}": np.array([t, 2]) for t in sorted(tags)},
        ),
        file_format="gmsh22",
        binary=False,
    )


def _face_velocity(surface: dict, column: dict) -> np.ndarray:
    """Signed unit-motion normal velocity on every face (float64)."""
    v = np.zeros(len(surface["faces"]))
    for tag, coefficient in column["coefficients"].items():
        v[surface["tags"] == tag] = coefficient
    if column["axis"] is not None:
        v *= surface["normals"] @ np.asarray(column["axis"])
    return v


def _force_matrix(surface: dict, velocities: np.ndarray, basis: list[dict]) -> np.ndarray:
    """Nodal force coefficients: ``force_scale * v_f * A_f / 3`` per vertex."""
    force = np.zeros((len(surface["points"]), len(basis)))
    for j, col in enumerate(basis):
        nodal = col["force_scale"] * velocities[:, j] * surface["areas"] / 3.0
        for k in range(3):
            np.add.at(force[:, j], surface["faces"][:, k], nodal)
    return force


def _electrical_impedance(lem: dict, s: complex) -> complex:
    model = lem.get("semi_inductance")
    if model is None:
        return lem["re_ohm"] + s * lem["le_h"]
    admittance = (
        1 / model["rss_ohm"]
        + 1 / (s * model["le_h"])
        + 1 / (np.sqrt(s) * model["ke_semi_h"])
    )
    return model["re_prime_ohm"] + s * model["leb_h"] + 1 / admittance


def _mechanical_impedance(lem: dict, omega: float, density: float, c: float) -> complex:
    # exp(-i omega t): mass reactance is -i omega m, stiffness +i/(omega C).
    z = complex(lem["rms_n_s_per_m"], -omega * lem["mmd_kg"] + 1 / (omega * lem["cms_m_per_n"]))
    chamber = lem.get("sealed_chamber")
    if chamber is not None:
        stiffness = density * c**2 * chamber["projected_area_m2"] ** 2 / chamber["volume_m3"]
        z += 1j * stiffness / omega
    return z


def _network(plan: dict, z: np.ndarray | None, omega: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Unit-motion amplitudes ``U`` (basis x port) and coil currents.

    Transducer rows balance mechanical, radiation and Lorentz forces; voltage
    rows close each coil, shorted (V=0) unless its port drives it.
    """
    d, ports = plan["transducer_count"], plan["port_ids"]
    u = np.zeros((len(plan["basis"]), len(ports)), dtype=np.complex128)
    if d == 0:
        for k, pid in enumerate(ports):
            u[plan["column_of"][plan["ports"][pid]["component_id"]], k] = 1.0
        return u, np.zeros((0, len(ports)), dtype=np.complex128), 0.0
    lems = [col["lem"] for col in plan["basis"][:d]]
    s = -1j * omega
    zm = [_mechanical_impedance(lem, omega, plan["density"], plan["sound_speed"]) for lem in lems]
    ze = [_electrical_impedance(lem, s) for lem in lems]
    bl = np.diag([lem["bl_n_per_a"] for lem in lems])
    network = np.block([[z[:d, :d] + np.diag(zm), -bl], [bl, np.diag(ze)]])
    rhs = np.zeros((2 * d, len(ports)), dtype=np.complex128)
    for k, pid in enumerate(ports):
        j = plan["column_of"][plan["ports"][pid]["component_id"]]
        if j < d:
            rhs[d + j, k] = plan["voltage"]
        else:
            u[j, k] = 1.0
            rhs[:d, k] = -z[:d, j]
    solution = np.linalg.solve(network, rhs)
    u[:d] = solution[:d]
    residual = float(np.max(np.abs(network @ solution - rhs), initial=0.0))
    return u, solution[d:], residual


def _matrix_quantity(plan, output, z, velocities, surface) -> tuple[np.ndarray, dict]:
    ids = list(output["target_ids"]) or [
        cid for cid in plan["components"] if cid in plan["column_of"]
    ]
    cols = [plan["column_of"][cid] for cid in ids]
    selected = [plan["basis"][j] for j in cols]
    matrix = z[:, cols][:, :, cols]
    weights = np.array([col["orbit"] for col in selected], dtype=float)
    weighted = weights[None, :, None] * matrix
    reciprocity, passivity = [], []
    for w in weighted:
        scale = np.max(np.abs(w), initial=0.0)
        reciprocity.append(
            0.0 if scale == 0 else float(np.max(np.abs(w - w.T)) / scale)
        )
        passivity.append(float(np.linalg.eigvalsh((w + w.conj().T) / 2).min()))
    areas, ratios = [], []
    for j, col in zip(cols, selected):
        copies = col["completion"] * col["orbit"]
        contributions = velocities[:, j] * surface["areas"]
        area = copies * float(np.sum(contributions))
        absolute = copies * float(np.sum(np.abs(contributions)))
        areas.append(area)
        ratios.append(0.0 if absolute == 0 else abs(area) / absolute)
    metadata = {
        "component_ids": ids,
        "kinds": [col["kind"] for col in selected],
        "surface_completion_factors": [float(col["completion"]) for col in selected],
        "physical_driver_orbit_counts": [int(col["orbit"]) for col in selected],
        "effective_volume_area_m2": areas,
        "effective_volume_area_definition": (
            "signed integral of motion factor over physical moving surfaces; "
            "real symmetry copies included; ground images excluded"
        ),
        "effective_volume_area_zero_or_near_cancelling": [r <= 1e-2 for r in ratios],
        "effective_volume_area_cancellation_ratio": ratios,
        "effective_volume_area_cancellation_relative_tolerance": 1e-2,
        "row_weights": weights.tolist(),
        "definition": (
            "force_per_unit_velocity; per-row physical-copy weighting in row_weights"
        ),
        "phasor_convention": plan["phasor"],
        "reciprocity_max_rel": reciprocity,
        "passivity_min_eig": passivity,
    }
    return matrix, metadata


def validate_compiled(request: dict, *, base_dir: str | Path = ".") -> None:
    """Validate request and surface compatibility without starting a solve."""
    plan = _plan(request)
    _surface(plan, Path(base_dir).resolve())


def solve_compiled(
    request: dict,
    *,
    base_dir: str | Path = ".",
    cancel_cb: Callable[[], None] | None = None,
    native_formulation: str = "standard",
    complex_k_shift: float = 0.005,
    native_progress_cb: Callable[[int, float, dict], None] | None = None,
) -> CompiledResult:
    """Translate the supported BEAT envelope into unchanged native solves.

    Optional cancel_cb raises to abandon the solve. cancel_path is checked
    before submission and after each native frequency; no partial basis is
    returned on cancellation. Unsupported physics fails before mesh I/O.
    The optional native formulation is an explicit host-side extension, not
    part of the BEAT wire request; its default preserves BEAT semantics.
    """
    from . import load_mesh, solve_multi_source

    plan = _plan(request)
    if native_formulation not in {"standard", "complex_k"}:
        raise ValueError("native_formulation must be 'standard' or 'complex_k'")
    if (isinstance(complex_k_shift, bool) or not isinstance(complex_k_shift, (int, float))
            or not math.isfinite(complex_k_shift) or complex_k_shift < 0):
        raise ValueError("complex_k_shift must be finite and non-negative")
    for column in plan["column_kwargs"]:
        column.update(formulation=native_formulation, complex_k_shift=float(complex_k_shift))
    base = Path(base_dir).resolve()
    marker = request.get("cancel_path")
    cancel_path = (base / marker) if marker else None

    def check_cancel():
        if cancel_cb is not None:
            cancel_cb()
        if cancel_path is not None and cancel_path.exists():
            raise InterruptedError("Compiled BEAT solve cancelled")

    def on_frequency(index, frequency, entry):
        check_cancel()
        if native_progress_cb is not None:
            native_progress_cb(index, frequency, entry)

    check_cancel()
    started = time.perf_counter()
    source_identity = solver_source_identity()
    meshes = [_mesh_provenance(spec, base) for spec in plan["meshes"]]
    ids = tuple(plan["port_ids"])
    basis = plan["basis"]
    freqs = plan["unique_frequencies"]
    with TemporaryDirectory(prefix="legacy-beat-compiled-") as tmp:
        surface = _surface(plan, base)
        mesh_path = Path(tmp) / "surface.msh"
        all_tags = {t for tags in plan["mesh_tags"].values() for t in tags.values()}
        _write_surface(surface, all_tags, mesh_path)
        check_cancel()
        points = plan["output_points"] or [
            # Impedance-only requests still need one observation point; place
            # it well clear of the body inside every image domain.
            [surface["points"].max(axis=0) + np.ptp(surface["points"], axis=0) + 1.0]
        ]
        points = np.vstack(points)
        observation = ObservationConfig(
            planes=["compiled"],
            custom_points={"compiled": points},
            angle_count=len(points),
        )
        mesh_kwargs = {"validate": True, "merge_tol": 0.0, "repair_normals": False}
        if "native_symmetry_plane" in plan["kwargs"]:
            mesh_kwargs["native_symmetry_plane"] = plan["kwargs"]["native_symmetry_plane"]
        if not _IS_METAL:
            mesh_kwargs["require_closed"] = True
        loaded = load_mesh(mesh_path, **mesh_kwargs)
        configs = [
            SolveConfig(
                **column_kwargs,
                observation=observation,
                on_frequency_result=on_frequency,
            )
            for column_kwargs in plan["column_kwargs"]
        ]
        # One assembly and factorization per frequency serves every basis
        # column. The native multi-source API preserves its one-source path.
        results = solve_multi_source(
            loaded,
            [c.velocity_sources for c in configs],
            configs[0],
            freqs.copy(),
        )
        check_cancel()
        if len(results) != len(configs):
            raise RuntimeError("Legacy solver returned the wrong number of source results")
        for result in results:
            if not np.array_equal(result.frequencies_hz, freqs):
                raise RuntimeError(
                    "Legacy solver returned an incomplete or reordered frequency axis"
                )
            expected = (len(freqs), 1, len(points))
            if (
                result.pressure_complex.shape != expected
                or not np.isfinite(result.pressure_complex).all()
            ):
                raise RuntimeError(
                    "Legacy solver returned invalid pressure dimensions/values"
                )
            if plan["kwargs"].get("return_surface_pressure"):
                trace = getattr(result, "surface_pressure_complex", None)
                if (
                    trace is None
                    or trace.shape != (len(freqs), len(surface["used"]))
                    or not np.isfinite(trace).all()
                ):
                    raise RuntimeError(
                        "Legacy solver returned an invalid surface pressure trace"
                    )

    nf, nb, nv = len(freqs), len(basis), len(surface["points"])
    fields = np.stack([r.pressure_complex[:, 0, :] for r in results], axis=1)
    velocities = np.stack([_face_velocity(surface, col) for col in basis], axis=1)
    surface_p = z = None
    if plan["kwargs"].get("return_surface_pressure"):
        # Only loads, boundary pressure or a driver network need the traces.
        surface_p = np.zeros((nf, nv, nb), dtype=np.complex128)
        for j, r in enumerate(results):
            surface_p[:, surface["used"], j] = r.surface_pressure_complex
        force = _force_matrix(surface, velocities, basis)
        z = np.einsum("vi,fvj->fij", force, surface_p)
    omegas = 2 * np.pi * freqs
    u = np.zeros((nf, nb, len(ids)), dtype=np.complex128)
    current = np.zeros((nf, plan["transducer_count"], len(ids)), dtype=np.complex128)
    residuals = []
    for f, omega in enumerate(omegas):
        u[f], current[f], residual = _network(
            plan, None if z is None else z[f], float(omega)
        )
        residuals.append(residual)

    def response(basis_values):
        # (frequency, sample, basis) -> (frequency, excitation, sample)
        return np.einsum("fsj,fje->fes", basis_values, u)

    quantities = {}
    for output in plan["outputs"]:
        quantity, oid = output["quantity"], output["id"]
        unit, axes = _OUTPUTS[quantity]
        metadata: dict[str, Any] = {}
        if quantity == "exterior_pressure":
            values = response(fields[:, :, plan["slices"][oid]].transpose(0, 2, 1))
        elif quantity == "bem_boundary_pressure":
            values = response(surface_p)
        elif quantity == "bem_boundary_neumann":
            q = 1j * plan["density"] * omegas[:, None, None] * velocities[None]
            values = response(q)
        elif quantity == "radiation_impedance":
            # Ideal self loads only; a transducer-only system has an empty
            # radiator axis (its loads are in radiation_impedance_matrix).
            cols = [plan["column_of"][c] for c in plan["ideal_components"]]
            values = z[:, cols, cols].reshape(nf, len(cols))
        elif quantity == "radiation_impedance_matrix":
            values, metadata = _matrix_quantity(plan, output, z, velocities, surface)
        else:
            d = plan["transducer_count"]
            values = (u[:, :d, :] if quantity == "diaphragm_velocity" else current).transpose(0, 2, 1)
            metadata = {
                "component_ids": [col["component_id"] for col in basis[:d]],
                "surface_completion_factors": [float(col["completion"]) for col in basis[:d]],
                "physical_driver_orbit_counts": [int(col["orbit"]) for col in basis[:d]],
            }
        values = values[plan["frequency_index"]]
        if plan["phasor"] == POSITIVE_TIME:
            # Real drives: the opposite time factor conjugates every phasor.
            # Hermitian eigenvalues and the reciprocity ratio are unchanged.
            values = values.conj()
        if "reciprocity_max_rel" in metadata:
            for key in ("reciprocity_max_rel", "passivity_min_eig"):
                metadata[key] = [metadata[key][i] for i in plan["frequency_index"]]
        quantities[oid] = CompiledQuantity(
            oid,
            quantity,
            unit,
            axes,
            np.ascontiguousarray(values),
            metadata,
            output["target_ids"][0] if output["target_ids"] else None,
        )
    diagnostics = {
        "precision": "float32" if _IS_METAL else plan["precision"],
        "bem_backend": "metal" if _IS_METAL else "cpu",
        "symmetry": plan["symmetry"],
        "phasor_convention": plan["phasor"],
        "formulation": (f"native_corrected_{native_formulation}" if _IS_METAL
                        else f"bempp_{native_formulation}"),
        "motion_basis_component_ids": [col["component_id"] for col in basis],
        "factorization_count": 1,
        "wall_time_s": time.perf_counter() - started,
        "engine_provenance": _run_provenance(request, meshes, results, configs[0], source_identity),
    }
    if plan["transducer_count"]:
        diagnostics["transducer_reference_voltage_v"] = plan["voltage"]
        diagnostics["exterior_lumped_network"] = {
            "termination": "shorted",
            "precision": "float64",
            "reference_voltage_v": plan["voltage"],
            "residual_max_abs": [residuals[i] for i in plan["frequency_index"]],
        }
    return CompiledResult(
        plan["frequencies"].copy(),
        ids,
        {
            oid: q.values
            for oid, q in quantities.items()
            if q.quantity == "exterior_pressure"
        },
        tuple(results),
        plan["phasor"],
        quantities,
        tuple(col["component_id"] for col in basis),
        diagnostics,
    )

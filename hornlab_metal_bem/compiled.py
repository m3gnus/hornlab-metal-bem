"""Opt-in compatibility with BEAT's compiled exterior request envelope.

This module is mirrored in the two legacy solvers to avoid making either a
runtime dependency of the other. The tests and release handoff check both
copies. No legacy entry point calls this adapter.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from .config import ObservationConfig, SolveConfig, VelocityMode
from .observation import ObservationFrame

_IS_METAL = __package__ == "hornlab_metal_bem"


@dataclass(frozen=True)
class CompiledResult:
    """Pressure bases in Pa per 1 m/s, with axes (frequency, excitation, point).

    Arrays retain the request's frequency, excitation and point order and the
    exp(-i omega t) convention. Native results retain diagnostics and traces;
    this is a Python result, not BEAT's worker/result wire protocol.
    """

    frequencies_hz: np.ndarray
    excitation_port_ids: tuple[str, ...]
    pressure: dict[str, np.ndarray]
    native_results: tuple[Any, ...]
    time_convention: str = "exp(-i omega t)"


def compiled_capabilities() -> dict[str, Any]:
    """Package support only; does not import BEAT or probe a runtime."""
    return {
        "request_schema_version": 1,
        "compiled_system_versions": [1, 2],
        "backend": "metal" if _IS_METAL else "cpu",
        "precisions": ["float32"] if _IS_METAL else ["float32", "float64"],
        "components": ["ideal_velocity_source"],
        "motion_profiles": ["uniform_normal", "rigid_translation"],
        "outputs": ["exterior_pressure"],
        "symmetry": ["off"],
        "limitations": [
            "one exterior region and triangle mesh",
            "closed outward solids",
            "all sources use one motion profile",
            "independent unit-velocity bases",
            "no transducers, losses, Robin walls, interfaces or ground images",
        ],
        "time_convention": "exp(-i omega t)",
        "requires": "official beat-engine contract validator (optional for legacy APIs)",
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


def _axis(value: Any) -> tuple[float, float, float]:
    a = _numeric_array(value, "motion_axis")
    if a.shape != (3,) or not np.isfinite(a).all() or not np.any(a):
        _refuse("motion_axis must be a finite nonzero three-vector")
    a = a / np.max(np.abs(a))
    return tuple(a / np.linalg.norm(a))


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
    if system["contract_version"] not in (1, 2):
        _refuse("only compiled-system v1/v2 ideal velocity sources are supported")
    if len(system["meshes"]) != 1 or len(system["regions"]) != 1:
        _refuse("requires one mesh and one exterior air region")
    _empty(system["interfaces"], "FEM/BEM interfaces")
    mesh, region = system["meshes"][0], system["regions"][0]
    if mesh["purpose"] != "bem_surface" or region["kind"] != "unbounded_air":
        _refuse("requires an exterior BEM surface")
    if region["mesh_ids"] != [mesh["id"]]:
        _refuse("region must own the surface mesh")
    _empty(region["loss_model"], "region losses")
    _empty(region["volume_groups"], "volume groups")
    if not _IS_METAL and region["sound_speed_m_per_s"] != 343.0:
        _refuse("BEM++ supports sound_speed_m_per_s=343 only")
    bounds, tag_owner = {}, {}
    for b in system["boundaries"]:
        _empty(b["parameters"], "boundary parameters/Robin walls")
        g = b["group"]
        if (
            b["region_id"] != region["id"]
            or g["mesh_id"] != mesh["id"]
            or g["dimension"] != 2
        ):
            _refuse("boundary must own a surface group on the exterior mesh")
        if b["kind"] not in {"rigid", "moving"}:
            _refuse("only rigid and moving boundaries are supported")
        if g["tag"] in tag_owner:
            _refuse("each physical surface tag needs exactly one boundary")
        tag_owner[g["tag"]] = b["id"]
        bounds[b["id"]] = b
    components, owners, motions, axes = {}, set(), set(), {}
    for c in system["components"]:
        if c["kind"] != "ideal_velocity_source":
            _refuse(
                "only ideal_velocity_source components are supported; use official BEAT for coupled drivers"
            )
        p = c["parameters"]
        _keys(p, {"motion_profile", "motion_axis"}, "component parameters")
        motion = p.get("motion_profile", "uniform_normal")
        if motion not in {"uniform_normal", "rigid_translation"}:
            _refuse("motion_profile must be uniform_normal or rigid_translation")
        if motion == "rigid_translation" and system["contract_version"] < 2:
            _refuse("rigid_translation requires compiled-system v2")
        if motion == "uniform_normal" and "motion_axis" in p:
            _refuse("normal motion cannot carry an axis")
        tags = []
        for bid in c["boundary_ids"]:
            if bid in owners or bounds[bid]["kind"] != "moving":
                _refuse("moving boundaries must belong to exactly one source component")
            owners.add(bid)
            tag = bounds[bid]["group"]["tag"]
            tags.append(tag)
            if motion == "rigid_translation":
                axes[tag] = _axis(p.get("motion_axis"))
        if not tags:
            _refuse("source must own at least one moving boundary")
        components[c["id"]] = tags
        motions.add(motion)
    if not components or len(motions) != 1:
        _refuse("requires ideal sources with one common motion profile")
    if owners != {b["id"] for b in bounds.values() if b["kind"] == "moving"}:
        _refuse("every moving boundary must belong to a source")
    ports = {p["id"]: p for p in system["excitation_ports"]}
    for p in ports.values():
        if p["kind"] != "normal_velocity" or p["component_id"] not in components:
            _refuse("only normal_velocity ports on ideal sources are supported")
    frequencies = np.array(request["frequencies_hz"], dtype=float)
    if len(set(frequencies)) != len(frequencies):
        _refuse("duplicate frequencies are not supported by the legacy adapter")
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
        },
        "solver options",
    )
    backend = "metal" if _IS_METAL else "cpu"
    if options.get("bem_backend", backend) != backend:
        _refuse(f"bem_backend must be {backend}")
    if _IS_METAL and ({"quadrature_order", "singular_order"} & set(options)):
        _refuse(
            "Metal uses fixed native quadrature; BEAT quadrature-order overrides are not supported"
        )
    precision = options.get("precision", "float32" if _IS_METAL else "float64")
    if precision not in compiled_capabilities()["precisions"]:
        _refuse(f"unsupported precision {precision}")
    if options.get("symmetry", "off") != "off":
        _refuse("only full exterior meshes are supported")
    if options.get("phasor_convention", "exp(-i omega t)") != "exp(-i omega t)":
        _refuse("phasor_convention must be exp(-i omega t)")
    if options.get("regular_quadrature_mode", "fixed") != "fixed":
        _refuse("only fixed regular quadrature is supported")
    for key in ("quadrature_order", "singular_order"):
        v = options.get(key, 4)
        if type(v) is not int or not 1 <= v <= 6:
            _refuse(f"{key} must be an integer in [1, 6]")
    output_points, slices, cursor = [], {}, 0
    for output in request["outputs"]:
        if output["quantity"] != "exterior_pressure" or output["target_ids"]:
            _refuse("only untargeted exterior pressure outputs are supported")
        o = output["options"]
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
    if not output_points:
        _refuse("requires at least one exterior pressure output")
    if cursor > 100_000:
        _refuse("pressure point count exceeds 100000")
    kwargs = {
        "velocity_mode": VelocityMode.VELOCITY,
        "source_motion": "axial" if axes else "normal",
        "source_axes": axes or None,
        "air_density": region["density_kg_per_m3"],
        "observation": ObservationConfig(
            planes=["compiled"],
            custom_points={"compiled": np.vstack(output_points)},
            angle_count=cursor,
        ),
        "frame_override": ObservationFrame(
            axis=np.array([0.0, 0.0, 1.0]),
            origin=np.zeros(3),
            u=np.array([1.0, 0.0, 0.0]),
            v=np.array([0.0, 1.0, 0.0]),
            mouth_center=np.zeros(3),
            source_center=np.zeros(3),
        ),
    }
    if _IS_METAL:
        kwargs.update(
            speed_of_sound=region["sound_speed_m_per_s"],
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
    # Construct now so impossible native options fail before reading a mesh.
    SolveConfig(
        **kwargs,
        velocity_sources={tag: 0.0 for tag in axes}
        or {tag: 0.0 for c in components.values() for tag in c},
    )
    return {
        "mesh": mesh,
        "tags": set(tag_owner),
        "components": components,
        "ports": ports,
        "frequencies": frequencies,
        "kwargs": kwargs,
        "slices": slices,
    }


def _packed_array(raw: dict) -> np.ndarray:
    return (
        np.frombuffer(base64.b64decode(raw["data"], validate=True), dtype=raw["dtype"])
        .reshape(raw["shape"])
        .copy()
    )


def _mesh_file(plan: dict, base_dir: Path, target: Path) -> None:
    import meshio

    spec = plan["mesh"]
    if "mesh_data" in spec:
        data = spec["mesh_data"]
        if any(b["type"] != "triangle" for b in data["cells"]):
            _refuse("packed mesh supports triangle cells only")
        points = _packed_array(data["points"])
        faces = np.vstack([_packed_array(b["connectivity"]) for b in data["cells"]])
        tags = np.concatenate(
            [_packed_array(b["physical_tags"]) for b in data["cells"]]
        )
        names = data["physical_names"]
    else:
        source = (base_dir / spec["file"]).resolve()
        if not source.is_relative_to(base_dir):
            _refuse("mesh file must remain inside base_dir")
        m = meshio.read(source)
        if any(c.type not in {"triangle", "line", "vertex"} for c in m.cells):
            _refuse("file mesh supports linear triangles, lines and vertices only")
        points = m.points
        faces = m.cells_dict.get("triangle")
        tags = m.cell_data_dict.get("gmsh:physical", {}).get("triangle")
        if faces is None or tags is None:
            _refuse("file mesh needs physically tagged triangles")
        names = m.field_data
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
    if tags.shape != (len(faces),) or set(tags) != plan["tags"]:
        _refuse("mesh triangle tags must match all declared boundaries exactly")
    tri = points[faces]
    if np.any(
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        == 0
    ):
        _refuse("mesh contains degenerate triangles")
    edge_owners = {}
    for index, f in enumerate(faces):
        for a, b in zip(f, np.roll(f, -1)):
            edge_owners.setdefault(tuple(sorted((a, b))), []).append(
                (index, 1 if a < b else -1)
            )
    if any(len(v) != 2 or v[0][1] == v[1][1] for v in edge_owners.values()):
        _refuse("mesh must be a closed consistently oriented manifold")
    neighbors = [set() for _ in faces]
    for v in edge_owners.values():
        a, b = v[0][0], v[1][0]
        neighbors[a].add(b)
        neighbors[b].add(a)
    unseen = set(range(len(faces)))
    while unseen:
        pending, shell = [unseen.pop()], []
        while pending:
            i = pending.pop()
            shell.append(i)
            for n in neighbors[i] & unseen:
                unseen.remove(n)
                pending.append(n)
        t = tri[shell] - tri[shell].mean(axis=(0, 1))
        if np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))) <= 0:
            _refuse("each closed shell must have outward winding")
    # No repair/merge is performed. Work in the exact global metre frame.
    meshio.write(
        target,
        meshio.Mesh(
            points,
            [("triangle", faces)],
            cell_data={"gmsh:physical": [tags], "gmsh:geometrical": [tags]},
            field_data={f"surface_{t}": np.array([t, 2]) for t in sorted(plan["tags"])},
        ),
        file_format="gmsh22",
        binary=False,
    )


def solve_compiled(
    request: dict,
    *,
    base_dir: str | Path = ".",
    cancel_cb: Callable[[], None] | None = None,
) -> CompiledResult:
    """Translate the supported BEAT envelope into unchanged native solves.

    Optional cancel_cb raises to abandon the solve. cancel_path is checked
    before submission and after each native frequency; no partial basis is
    returned on cancellation. Unsupported physics fails before mesh I/O.
    """
    from . import load_mesh, solve_frequencies

    plan = _plan(request)
    base = Path(base_dir).resolve()
    marker = request.get("cancel_path")
    cancel_path = (base / marker) if marker else None

    def check_cancel():
        if cancel_cb is not None:
            cancel_cb()
        if cancel_path is not None and cancel_path.exists():
            raise InterruptedError("Compiled BEAT solve cancelled")

    def on_frequency(*args):
        check_cancel()

    check_cancel()
    ids = tuple(request["excitation_port_ids"])
    all_tags = {tag for c in plan["components"].values() for tag in c}
    results = []
    with TemporaryDirectory(prefix="legacy-beat-compiled-") as tmp:
        mesh_path = Path(tmp) / "surface.msh"
        _mesh_file(plan, base, mesh_path)
        check_cancel()
        mesh_kwargs = {"validate": True, "merge_tol": 0.0, "repair_normals": False}
        if not _IS_METAL:
            mesh_kwargs["require_closed"] = True
        loaded = load_mesh(mesh_path, **mesh_kwargs)
        for pid in ids:
            check_cancel()
            driven = set(plan["components"][plan["ports"][pid]["component_id"]])
            config = SolveConfig(
                **plan["kwargs"],
                velocity_sources={
                    tag: float(tag in driven) for tag in sorted(all_tags)
                },
                on_frequency_result=on_frequency,
            )
            result = solve_frequencies(loaded, plan["frequencies"].copy(), config)
            check_cancel()
            if not np.array_equal(result.frequencies_hz, plan["frequencies"]):
                raise RuntimeError(
                    "Legacy solver returned an incomplete or reordered frequency axis"
                )
            expected = (len(plan["frequencies"]), 1, config.observation.angle_count)
            if (
                result.pressure_complex.shape != expected
                or not np.isfinite(result.pressure_complex).all()
            ):
                raise RuntimeError(
                    "Legacy solver returned invalid pressure dimensions/values"
                )
            results.append(result)
    bases = np.stack([r.pressure_complex[:, 0, :] for r in results], axis=1)
    return CompiledResult(
        plan["frequencies"].copy(),
        ids,
        {oid: bases[:, :, sl].copy() for oid, sl in plan["slices"].items()},
        tuple(results),
    )

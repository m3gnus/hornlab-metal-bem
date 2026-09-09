#!/usr/bin/env python3
"""Compare CircSym with a matched quarter-domain full-3D Metal solve.

The harness intentionally accepts an external mesher config and quarter mesh so
the solver package does not acquire a mesher dependency. Mesh generation is not
part of either timed arm. Runs are paired and alternate order after one excluded
warm-up per arm.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAG_SOURCE = 2
_DIRECTIVITY_DB_FLOOR = -300.0
# This is deliberately much tighter than the older extent checks.  It measures
# the actual revolved surface represented by the quarter mesh, tag by tag,
# rather than only comparing its bounding box.  0.1 mm leaves a practical
# allowance for serialized mesh coordinates while exposing the old 3 mm mouth
# closure disagreement.
_GEOMETRY_DISTANCE_TOLERANCE_M = 1.0e-4
_GENERATING_CURVE_REFERENCE_REFINEMENT = 8


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _azimuth_int(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 8:
        raise argparse.ArgumentTypeError("must be at least 8")
    return parsed


def _refinement_factors(value: str) -> tuple[int, ...]:
    """Parse a strictly increasing, duplicate-free meridian refinement ladder."""
    try:
        factors = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be comma-separated positive integers, for example 1,2,4,8"
        ) from exc
    if not factors or any(factor < 1 for factor in factors):
        raise argparse.ArgumentTypeError("must contain positive integers")
    if tuple(sorted(set(factors))) != factors:
        raise argparse.ArgumentTypeError(
            "must be strictly increasing without duplicate factors"
        )
    return factors


def _complex_k_shifts(value: str) -> tuple[float, ...]:
    """Parse a strictly increasing positive complex-wavenumber shift ladder."""
    try:
        shifts = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be comma-separated positive shifts, for example 0.001,0.005,0.01"
        ) from exc
    if not shifts or any(not np.isfinite(shift) or shift <= 0.0 for shift in shifts):
        raise argparse.ArgumentTypeError("must contain finite positive shifts")
    if tuple(sorted(set(shifts))) != shifts:
        raise argparse.ArgumentTypeError(
            "must be strictly increasing without duplicate shifts"
        )
    return shifts


def _full_mesh_ladder(value: str) -> tuple[Path, ...]:
    """Parse caller-ordered external full meshes from coarse to fine."""
    paths = tuple(Path(item.strip()) for item in value.split(",") if item.strip())
    if len(paths) < 2:
        raise argparse.ArgumentTypeError(
            "must contain at least two comma-separated full mesh paths"
        )
    if len({path.resolve() for path in paths}) != len(paths):
        raise argparse.ArgumentTypeError("must not contain duplicate full mesh paths")
    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--quarter-mesh", type=Path, required=True)
    full_mesh_group = parser.add_mutually_exclusive_group()
    full_mesh_group.add_argument(
        "--full-mesh",
        type=Path,
        help=(
            "optional unsymmetrized full-domain 3-D mesh; runs a diagnostic-only "
            "Axisym/quarter/full comparison without changing qualification gates"
        ),
    )
    full_mesh_group.add_argument(
        "--full-mesh-ladder",
        type=_full_mesh_ladder,
        metavar="COARSE.msh,...,FINE.msh",
        help=(
            "caller-ordered external full-domain meshes, coarse to fine; reports "
            "consecutive and finest-rung diagnostics only"
        ),
    )
    full_mesh_group.add_argument(
        "--reflect-quarter-to-full",
        action="store_true",
        help=(
            "diagnostically expand --quarter-mesh through exact X/Y reflections "
            "and solve it with no native symmetry; mutually exclusive with --full-mesh"
        ),
    )
    parser.add_argument("--f1", type=_positive_float, default=100.0)
    parser.add_argument("--f2", type=_positive_float, default=20_000.0)
    parser.add_argument("--frequencies", type=_positive_int, default=40)
    parser.add_argument("--angles", type=_positive_int, default=37)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument(
        "--axisym-backend",
        choices=("cpu", "metal"),
        default="cpu",
        help="CircSym assembly and field executor (default: %(default)s)",
    )
    parser.add_argument(
        "--cpu-field",
        choices=("numpy", "numba"),
        default="numba",
        help="CircSym portable field executor (default: %(default)s)",
    )
    parser.add_argument(
        "--qualification-ratio",
        type=_positive_float,
        default=0.5,
        help="required warm median CircSym/quarter ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--azimuth-min",
        type=_azimuth_int,
        default=64,
        metavar="N",
        help="experimental CircSym azimuth-order floor (default: %(default)s)",
    )
    parser.add_argument("--max-pressure-rel-l2", type=_positive_float, default=0.02)
    parser.add_argument("--max-directivity-error-db", type=_positive_float, default=0.5)
    parser.add_argument("--max-phase-rms-deg", type=_positive_float, default=5.0)
    parser.add_argument(
        "--meridian-refinement-factor",
        type=_positive_int,
        default=1,
        help=(
            "uniform straight-segment subdivision used by the timed CircSym arm "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--meridian-refinement-factors",
        type=_refinement_factors,
        default=None,
        metavar="F1,F2,...",
        help=(
            "optional fixed-geometry CircSym convergence ladder; e.g. 1,2,4,8 "
            "compares 53/106/212/424 segments for a 53-segment base mesh"
        ),
    )
    parser.add_argument(
        "--resonance-comparison",
        action="store_true",
        help=(
            "run diagnostic paired real-k+CHIEF and complex-k shift-ladder solves; "
            "requires --chief-points"
        ),
    )
    parser.add_argument(
        "--chief-points",
        type=Path,
        metavar="JSON",
        help=(
            "caller-supplied JSON array of [x, y, z] points in the excluded solid "
            "interior enclosed by the exterior boundary (not horn air); caller must "
            "verify placement"
        ),
    )
    parser.add_argument(
        "--chief-weight",
        type=_positive_float,
        default=1.0,
        help="relative real-k CHIEF row weight (default: %(default)s)",
    )
    parser.add_argument(
        "--complex-k-shifts",
        type=_complex_k_shifts,
        default=(0.001, 0.005, 0.01),
        metavar="S1,S2,...",
        help="complex-k shifts for --resonance-comparison (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.f2 < args.f1:
        parser.error("--f2 must be greater than or equal to --f1")
    if args.angles < 2:
        parser.error("--angles must be at least 2")
    if args.qualification_ratio >= 1.0:
        parser.error("--qualification-ratio must be less than 1")
    if args.resonance_comparison and args.chief_points is None:
        parser.error("--resonance-comparison requires --chief-points JSON")
    return args


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("refusing to serialize non-finite complex result")
        return {"real": value.real, "imaginary": value.imag}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("refusing to serialize non-finite floating-point result")
        return value
    return value


def _load_chief_points(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load caller-owned CHIEF points without inferring any cavity geometry."""
    raw = path.read_bytes()
    try:
        # PowerShell's common UTF-8 output mode prepends a BOM.  It carries no
        # semantic content for this JSON payload, so accept it explicitly.
        points = np.asarray(json.loads(raw.decode("utf-8-sig")), dtype=np.float64)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(
            "--chief-points must be UTF-8 JSON containing an array shaped (m, 3)"
        ) from exc
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError("--chief-points must contain a non-empty array shaped (m, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("--chief-points must contain only finite coordinates")
    return points, {
        "source": "caller_supplied_json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "point_count": int(points.shape[0]),
    }


def _high_resolution_generating_curve_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Return the same geometry with only meridian target lengths refined.

    Full-3D meshing and CircSym panelization sample the same analytic/profile
    geometry differently.  A coarse CircSym chord is therefore not an adequate
    geometry-contract oracle for vertices from a finer 3-D surface mesh.  This
    reference changes only the three public millimetre target lengths, never
    formula, wall thickness, topology, source, or profile settings.
    """
    config = copy.deepcopy(raw_config)
    mesh = config.get("mesh")
    if mesh is None:
        mesh = {}
        config["mesh"] = mesh
    if not isinstance(mesh, dict):
        raise ValueError("benchmark config field 'mesh' must be an object")

    resolution_names = {
        "throat_res_mm": ("throat_res_mm", "throat_res", "throatResolution", 4.0),
        "mouth_res_mm": ("mouth_res_mm", "mouth_res", "mouthResolution", 26.0),
        "rear_res_mm": ("rear_res_mm", "rear_res", "rearResolution", 15.0),
    }
    for name, aliases_and_default in resolution_names.items():
        *aliases, default = aliases_and_default
        # Match the mesher's `_float(mesh, config, names=...)` precedence:
        # every accepted spelling in the nested mesh section wins over every
        # accepted top-level spelling.  Then write the canonical nested key so
        # the reference config itself is unambiguous.
        raw_value = default
        for source in (mesh, config):
            found = next(
                (source[alias] for alias in aliases if source.get(alias) is not None),
                None,
            )
            if found is not None:
                raw_value = found
                break
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"cannot build high-resolution generating curve: {name} must be numeric"
            ) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"cannot build high-resolution generating curve: {name} must be positive"
            )
        # The nested canonical spelling has precedence in the mesher and keeps
        # this reference independent of the input's historical alias spelling.
        mesh[name] = value / _GENERATING_CURVE_REFERENCE_REFINEMENT
    return config


def _expanded_full_mesh_from_quarter(quarter: Any) -> tuple[Any, dict[str, Any]]:
    """Build an unsymmetrized full mesh from exactly the quarter triangles.

    This is intentionally a diagnostic control, distinct from a separately
    remeshed ``--full-mesh`` input: it isolates native-symmetry, seam, and
    source-normalisation differences without introducing meshing variation.
    """
    from hornlab_metal_bem import LoadedMesh, MeshInfo
    from hornlab_metal_bem.mesh import make_pure_grid
    from hornlab_metal_bem.validation.native_symmetry import expand_quarter_mesh_xy

    vertices = np.asarray(quarter.grid.vertices, dtype=np.float64).T
    triangles = np.asarray(quarter.grid.elements, dtype=np.int64).T
    tags = np.asarray(quarter.physical_tags, dtype=np.int32).reshape(-1)
    expanded = expand_quarter_mesh_xy(vertices, triangles, tags)
    expanded_vertices = expanded.vertices_nx3
    full = LoadedMesh(
        grid=make_pure_grid(expanded_vertices, expanded.triangles_nx3),
        physical_tags=expanded.physical_tags,
        info=MeshInfo(
            n_vertices=int(expanded_vertices.shape[0]),
            n_triangles=int(expanded.triangles_nx3.shape[0]),
            physical_groups=dict(quarter.info.physical_groups),
            bounding_box_m=(
                np.min(expanded_vertices, axis=0),
                np.max(expanded_vertices, axis=0),
            ),
        ),
        coupled_ib_aperture_tag=quarter.coupled_ib_aperture_tag,
    )
    return full, {
        "source": "exact_xy_reflection_of_quarter_mesh",
        "quarter_triangles": int(quarter.info.n_triangles),
        "full_triangles": int(full.info.n_triangles),
        "triangle_image_count": 4,
        "native_symmetry_plane": None,
    }


def _build_inputs(
    config_path: Path,
    quarter_mesh_path: Path,
    angle_count: int,
    full_mesh_path: Path | None = None,
    reflect_quarter_to_full: bool = False,
    full_mesh_ladder_paths: tuple[Path, ...] | None = None,
):
    import hornlab_mesher as hm

    import hornlab_metal_bem as mb

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    meridian_build = hm.build_meridian(raw_config)
    meridian = meridian_build.as_metal_meridian()
    reference_build = hm.build_meridian(
        _high_resolution_generating_curve_config(raw_config)
    )
    reference_meridian = reference_build.as_metal_meridian()
    quarter = mb.load_mesh(
        quarter_mesh_path,
        scale=1.0,
        native_symmetry_plane="yz+xz",
    )
    full: Any | None = None
    full_provenance: dict[str, Any] | None = None
    full_meshes: list[tuple[Any, dict[str, Any]]] = []
    if full_mesh_ladder_paths is not None:
        for rung, path in enumerate(full_mesh_ladder_paths):
            loaded = mb.load_mesh(path, scale=1.0, native_symmetry_plane=None)
            full_meshes.append(
                (
                    loaded,
                    {
                        "source": "caller_supplied_full_mesh_ladder",
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "ladder_rung": rung,
                        "native_symmetry_plane": None,
                    },
                )
            )
        full, full_provenance = full_meshes[0]
    elif full_mesh_path is not None:
        full = mb.load_mesh(full_mesh_path, scale=1.0, native_symmetry_plane=None)
        full_provenance = {
            "source": "caller_supplied_full_mesh",
            "path": str(full_mesh_path.resolve()),
            "sha256": hashlib.sha256(full_mesh_path.read_bytes()).hexdigest(),
            "native_symmetry_plane": None,
        }
    elif reflect_quarter_to_full:
        full, full_provenance = _expanded_full_mesh_from_quarter(quarter)
    geometry_distance = _validate_matched_inputs(
        raw_config,
        meridian,
        quarter,
        generating_curve_reference=reference_meridian,
        generating_curve_reference_metadata=reference_build.metadata,
    )
    geometry_distance["generating_curve_reference_refinement"] = (
        _GENERATING_CURVE_REFERENCE_REFINEMENT
    )
    full_geometry_distance = None
    full_geometry_distances: list[dict[str, Any]] = []
    meshes_to_validate = full_meshes or (
        [(full, full_provenance)] if full is not None else []
    )
    for full_candidate, _ in meshes_to_validate:
        full_geometry_distances.append(_validate_full_matched_input(
            raw_config,
            meridian,
            full_candidate,
            generating_curve_reference=reference_meridian,
        ))
    if full_geometry_distances:
        full_geometry_distance = full_geometry_distances[0]
    meridian_source_area, quarter_source_area = _source_areas(meridian, quarter)
    quarter_velocity_scale = meridian_source_area / quarter_source_area
    full_source_area = _full_source_area(full) if full is not None else None
    full_velocity_scale = (
        meridian_source_area / full_source_area if full_source_area is not None else None
    )
    frame = mb.ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.zeros(3),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=np.zeros(3),
        source_center=np.zeros(3),
    )
    observation = mb.ObservationConfig(
        planes=["horizontal", "vertical"],
        distance_m=2.0,
        angle_min_deg=0.0,
        angle_max_deg=180.0,
        angle_count=angle_count,
    )
    common = {
        "velocity_sources": {TAG_SOURCE: 1.0},
        "observation": observation,
        "frame_override": frame,
        "formulation": "complex_k",
        "complex_k_shift": 0.005,
    }
    axisym_config = mb.SolveConfig(**common)
    quarter_config = mb.SolveConfig(
        **{
            **common,
            # Match physical volume velocity despite polygonization of the
            # quarter-domain source disc.
            "velocity_sources": {TAG_SOURCE: quarter_velocity_scale},
        },
        mesh_scale=1.0,
        native_symmetry_plane="yz+xz",
        metal_native_assembly_mode="corrected",
        dense_solve_implementation="cgetrf_cgetrs",
    )
    full_config = (
        mb.SolveConfig(
            **{
                **common,
                "velocity_sources": {TAG_SOURCE: full_velocity_scale},
            },
            mesh_scale=1.0,
            # An explicitly full mesh must be solved directly.  Do not infer
            # a symmetry plane from its appearance or its file name.
            native_symmetry_plane=None,
            metal_native_assembly_mode="corrected",
            dense_solve_implementation="cgetrf_cgetrs",
        )
        if full is not None
        else None
    )
    full_mesh_ladder = None
    if full_meshes:
        full_mesh_ladder = []
        for (full_candidate, provenance), geometry_distance_candidate in zip(
            full_meshes, full_geometry_distances, strict=True
        ):
            source_area = _full_source_area(full_candidate)
            velocity_scale = meridian_source_area / source_area
            config = replace(
                full_config,
                velocity_sources={TAG_SOURCE: velocity_scale},
            )
            full_mesh_ladder.append(
                {
                    "mesh": full_candidate,
                    "config": config,
                    "provenance": provenance,
                    "source_area_m2": source_area,
                    "velocity_scale_for_equal_volume_velocity": velocity_scale,
                    "geometry_contract": geometry_distance_candidate,
                }
            )
    return (
        meridian,
        quarter,
        full,
        axisym_config,
        quarter_config,
        full_config,
        geometry_distance,
        full_geometry_distance,
        full_source_area,
        full_velocity_scale,
        full_provenance,
        full_mesh_ladder,
    )


def _source_areas(meridian: Any, quarter: Any) -> tuple[float, float]:
    """Return full-domain source areas for the meridian and quarter meshes."""
    vertices = np.asarray(quarter.grid.vertices, dtype=np.float64).T
    triangles = np.asarray(quarter.grid.elements, dtype=np.int64).T
    corners = vertices[triangles]
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    quarter_source_area = 4.0 * float(
        np.sum(triangle_areas[np.asarray(quarter.physical_tags) == TAG_SOURCE])
    )
    meridian_geom = meridian.segment_geometry()
    meridian_source_area = float(
        np.sum(
            meridian_geom.area_weights[
                np.asarray(meridian.physical_tags) == TAG_SOURCE
            ]
        )
    )
    return meridian_source_area, quarter_source_area


def _full_source_area(full: Any) -> float:
    """Return the actual source area of an unsymmetrized full-domain mesh."""
    vertices = np.asarray(full.grid.vertices, dtype=np.float64).T
    triangles = np.asarray(full.grid.elements, dtype=np.int64).T
    corners = vertices[triangles]
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    return float(
        np.sum(triangle_areas[np.asarray(full.physical_tags) == TAG_SOURCE])
    )


def _point_to_segment_distances(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Return each 2-D point's distance to its nearest closed line segment."""
    deltas = ends - starts
    lengths_squared = np.sum(np.square(deltas), axis=1)
    # MeridianMesh rejects zero-length segments, so this denominator is safe.
    offsets = points[:, None, :] - starts[None, :, :]
    parameters = np.clip(
        np.sum(offsets * deltas[None, :, :], axis=2) / lengths_squared[None, :],
        0.0,
        1.0,
    )
    closest = starts[None, :, :] + parameters[:, :, None] * deltas[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1)


def _nearest_sample_distances(points: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Return each point's distance to the nearest sampled point."""
    return np.min(
        np.linalg.norm(points[:, None, :] - samples[None, :, :], axis=2), axis=1
    )


def _quarter_edge_scale(quarter: Any) -> dict[str, float]:
    """Return representation-scale context for non-gating curve diagnostics."""
    vertices = np.asarray(quarter.grid.vertices, dtype=np.float64).T
    triangles = np.asarray(quarter.grid.elements, dtype=np.int64).T
    corners = vertices[triangles]
    edges = np.concatenate(
        (
            np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1),
            np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1),
            np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1),
        )
    )
    return {
        "min_edge_m": float(np.min(edges)),
        "median_edge_m": float(np.median(edges)),
        "p95_edge_m": float(np.percentile(edges, 95.0)),
    }


def _meridian_quarter_geometry_distance(meridian: Any, quarter: Any) -> dict[str, Any]:
    """Measure a 3-D surface against its tagged meridian generating curve.

    Each quarter-triangle corner is projected to ``(rho, z)`` and compared only
    against meridian segments with the same physical tag.  That catches a
    geometrically different closure even when radial and axial extents agree.
    It intentionally uses corners, not triangle centroids: azimuthal facets are
    a representation error of the 3-D mesh, not a disagreement in the intended
    generating curve.
    """
    vertices = np.asarray(quarter.grid.vertices, dtype=np.float64).T
    triangles = np.asarray(quarter.grid.elements, dtype=np.int64).T
    triangle_tags = np.asarray(quarter.physical_tags, dtype=np.int32).reshape(-1)
    if triangles.shape[0] != triangle_tags.size:
        raise ValueError("quarter mesh triangle tags do not match its triangles")

    per_tag: dict[str, dict[str, float | int]] = {}
    all_distances: list[np.ndarray] = []
    for tag in np.unique(triangle_tags):
        triangle_indices = np.flatnonzero(triangle_tags == tag)
        projected = vertices[triangles[triangle_indices]].reshape(-1, 3)
        points = np.column_stack(
            (np.hypot(projected[:, 0], projected[:, 1]), projected[:, 2])
        )
        # Repeated corners are deliberately retained: this weights the
        # diagnostic by triangle incidence and identifies the offending tag.
        segment_indices = np.flatnonzero(
            np.asarray(meridian.physical_tags, dtype=np.int32) == tag
        )
        if segment_indices.size == 0:
            raise ValueError(f"quarter mesh tag {int(tag)} is absent from the meridian")
        segments = np.asarray(meridian.segments, dtype=np.int64)[segment_indices]
        starts = np.asarray(meridian.nodes, dtype=np.float64)[segments[:, 0]]
        ends = np.asarray(meridian.nodes, dtype=np.float64)[segments[:, 1]]
        distances = _point_to_segment_distances(points, starts, ends)
        all_distances.append(distances)
        per_tag[str(int(tag))] = {
            "sample_count": int(distances.size),
            "max_distance_m": float(np.max(distances)),
            "rms_distance_m": float(np.sqrt(np.mean(np.square(distances)))),
            "p95_distance_m": float(np.percentile(distances, 95.0)),
        }

    combined = np.concatenate(all_distances)
    return {
        "method": "tagged_3d_triangle_corners_projected_to_rho_z",
        "sample_count": int(combined.size),
        "max_distance_m": float(np.max(combined)),
        "rms_distance_m": float(np.sqrt(np.mean(np.square(combined)))),
        "p95_distance_m": float(np.percentile(combined, 95.0)),
        "per_tag": per_tag,
    }


def _flat_mouth_closure_diagnostics(
    meridian: Any, quarter: Any, metadata: dict[str, Any] | None
) -> dict[str, Any]:
    """Verify the explicit freestanding flat mouth-closure contract.

    The full wall's curve distance is intentionally measured against a finer
    generating curve below.  The closure is different: it is a contractually
    straight span between the inner and outer mouth rings, so it gets a separate
    strict endpoint/collinearity check rather than inheriting any coarse-chord
    allowance.
    """
    if not metadata:
        return {"checked": False, "reason": "no meridian-build metadata"}
    try:
        source_count = int(metadata["sourceSegmentCount"])
        inner_count = int(metadata["innerSegmentCount"])
        closure_count = int(metadata["mouthRimSegmentCount"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("meridian metadata is missing mouth-closure segment counts") from exc
    if closure_count == 0:
        return {"checked": False, "reason": "configuration has no freestanding mouth closure"}
    if closure_count < 0:
        raise ValueError("meridian metadata has a negative mouth-closure segment count")

    first = source_count + inner_count
    last = first + closure_count
    segments = np.asarray(meridian.segments, dtype=np.int64)[first:last]
    if segments.shape[0] != closure_count:
        raise ValueError("meridian mouth-closure metadata does not match its segments")
    nodes = np.asarray(meridian.nodes, dtype=np.float64)
    closure_nodes = np.vstack((nodes[segments[0, 0]], nodes[segments[:, 1]]))
    chord_distance = _point_to_segment_distances(
        closure_nodes,
        closure_nodes[:1],
        closure_nodes[-1:],
    )

    vertices = np.asarray(quarter.grid.vertices, dtype=np.float64).T
    triangles = np.asarray(quarter.grid.elements, dtype=np.int64).T
    tags = np.asarray(quarter.physical_tags, dtype=np.int32).reshape(-1)
    wall_triangles = triangles[tags == 1]
    if wall_triangles.size == 0:
        raise ValueError("quarter mesh has no rigid-wall vertices for mouth closure")
    wall_vertices = vertices[wall_triangles].reshape(-1, 3)
    wall_points = np.column_stack(
        (np.hypot(wall_vertices[:, 0], wall_vertices[:, 1]), wall_vertices[:, 2])
    )
    endpoint_distance = _nearest_sample_distances(closure_nodes[[0, -1]], wall_points)
    wall_triangle_vertices = vertices[wall_triangles]
    wall_projected_triangles = np.stack(
        (
            np.hypot(wall_triangle_vertices[:, :, 0], wall_triangle_vertices[:, :, 1]),
            wall_triangle_vertices[:, :, 2],
        ),
        axis=2,
    )
    closure_triangle_distance = _point_to_segment_distances(
        wall_projected_triangles.reshape(-1, 2),
        closure_nodes[:1],
        closure_nodes[-1:],
    ).reshape(-1, 3)
    closure_triangles = np.all(
        closure_triangle_distance <= _GEOMETRY_DISTANCE_TOLERANCE_M, axis=1
    )
    chord_delta = closure_nodes[-1] - closure_nodes[0]
    chord_length_squared = float(np.dot(chord_delta, chord_delta))
    if chord_length_squared <= 0.0:
        raise ValueError("freestanding mouth closure has zero chord length")
    closure_parameter = np.sum(
        (wall_projected_triangles - closure_nodes[0]) * chord_delta,
        axis=2,
    ) / chord_length_squared
    parameter_tolerance = _GEOMETRY_DISTANCE_TOLERANCE_M / np.sqrt(
        chord_length_squared
    )
    on_closure_span = (
        (closure_triangle_distance <= _GEOMETRY_DISTANCE_TOLERANCE_M)
        & (closure_parameter >= -parameter_tolerance)
        & (closure_parameter <= 1.0 + parameter_tolerance)
    )
    span_minimum = np.min(
        np.where(on_closure_span, closure_parameter, np.inf), axis=1
    )
    span_maximum = np.max(
        np.where(on_closure_span, closure_parameter, -np.inf), axis=1
    )
    # Adjacent inner/outer wall triangles commonly contain two azimuthal
    # samples of one mouth-ring endpoint.  They project to the same chord
    # parameter and are not closure faces.  A candidate must span a resolvable
    # portion of the chord as well as having two vertices on it.
    closure_region_triangles = (
        (np.count_nonzero(on_closure_span, axis=1) >= 2)
        & ((span_maximum - span_minimum) > parameter_tolerance)
    )
    closure_region_distance = closure_triangle_distance[closure_region_triangles]
    flat_closure_triangles = closure_region_triangles & np.all(
        closure_triangle_distance <= _GEOMETRY_DISTANCE_TOLERANCE_M, axis=1
    )
    flat_intervals = sorted(
        (
            float(np.min(closure_parameter[index])),
            float(np.max(closure_parameter[index])),
        )
        for index in np.flatnonzero(flat_closure_triangles)
    )
    flat_coverage_min = flat_intervals[0][0] if flat_intervals else np.inf
    flat_coverage_max = flat_intervals[-1][1] if flat_intervals else -np.inf
    covered_until = -np.inf
    largest_gap = 0.0
    gap_free = bool(flat_intervals)
    for interval_start, interval_end in flat_intervals:
        if covered_until == -np.inf:
            covered_until = interval_end
            continue
        gap = interval_start - covered_until
        largest_gap = max(largest_gap, max(0.0, gap))
        if gap > parameter_tolerance:
            gap_free = False
        covered_until = max(covered_until, interval_end)
    if not flat_intervals:
        largest_gap = np.inf
    flat_coverage_full_span = bool(
        gap_free
        and flat_coverage_min <= parameter_tolerance
        and covered_until >= 1.0 - parameter_tolerance
    )
    diagnostics = {
        "checked": True,
        "contract": "flat_inner_to_outer_mouth_span",
        "closure_segment_count": closure_count,
        "reference_closure_max_distance_to_straight_chord_m": float(
            np.max(chord_distance)
        ),
        "quarter_wall_endpoint_max_distance_m": float(np.max(endpoint_distance)),
        "quarter_wall_triangles_on_flat_closure": int(
            np.count_nonzero(closure_triangles)
        ),
        "quarter_wall_closure_region_triangle_count": int(
            np.count_nonzero(closure_region_triangles)
        ),
        "quarter_wall_closure_region_max_distance_m": float(
            np.max(closure_region_distance)
            if closure_region_distance.size
            else np.inf
        ),
        "flat_closure_parameter_min": flat_coverage_min,
        "flat_closure_parameter_max": flat_coverage_max,
        "flat_closure_parameter_intervals": flat_intervals,
        "flat_closure_largest_gap": largest_gap,
        "flat_closure_covers_full_span": flat_coverage_full_span,
        "tolerance_m": _GEOMETRY_DISTANCE_TOLERANCE_M,
    }
    diagnostics["passes"] = bool(
        diagnostics["reference_closure_max_distance_to_straight_chord_m"]
        <= _GEOMETRY_DISTANCE_TOLERANCE_M
        and diagnostics["quarter_wall_endpoint_max_distance_m"]
        <= _GEOMETRY_DISTANCE_TOLERANCE_M
        and diagnostics["quarter_wall_triangles_on_flat_closure"] > 0
        and diagnostics["quarter_wall_closure_region_max_distance_m"]
        <= _GEOMETRY_DISTANCE_TOLERANCE_M
        and diagnostics["flat_closure_covers_full_span"]
    )
    return diagnostics


def _geometry_contract_diagnostics(
    coarse_meridian: Any,
    generating_curve_reference: Any,
    quarter: Any,
    generating_curve_reference_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Separate coarse panel chord error from geometry-contract validation."""
    coarse_distance = _meridian_quarter_geometry_distance(coarse_meridian, quarter)
    reference_distance = _meridian_quarter_geometry_distance(
        generating_curve_reference, quarter
    )
    closure = _flat_mouth_closure_diagnostics(
        generating_curve_reference, quarter, generating_curve_reference_metadata
    )
    edge_scale = _quarter_edge_scale(quarter)
    for distance in (coarse_distance, reference_distance):
        distance["max_distance_over_median_quarter_edge"] = float(
            distance["max_distance_m"] / max(edge_scale["median_edge_m"], 1.0e-30)
        )
        distance["p95_distance_over_median_quarter_edge"] = float(
            distance["p95_distance_m"] / max(edge_scale["median_edge_m"], 1.0e-30)
        )
    return {
        # These retain the independent surface-sampling gap, but never gate:
        # a full-3D curve mesh and a coarse CircSym chord can both correctly
        # represent the same profile while their sampled vertices differ.
        "coarse_meridian_panel_distance": coarse_distance,
        "quarter_edge_scale": edge_scale,
        "generating_curve_reference_distance": reference_distance,
        "mouth_closure": closure,
        "mouth_closure_tolerance_m": _GEOMETRY_DISTANCE_TOLERANCE_M,
    }


def _validate_matched_inputs(
    raw_config: dict[str, Any],
    meridian: Any,
    quarter: Any,
    *,
    generating_curve_reference: Any | None = None,
    generating_curve_reference_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_benchmark_geometry_config(raw_config)

    meridian_tags = {int(tag) for tag in np.unique(meridian.physical_tags)}
    quarter_tags = {int(tag) for tag in np.unique(quarter.physical_tags)}
    if meridian_tags != {1, TAG_SOURCE} or quarter_tags != {1, TAG_SOURCE}:
        raise ValueError(
            "matched benchmark requires only rigid-wall tag 1 and source tag 2; "
            f"got meridian={sorted(meridian_tags)}, quarter={sorted(quarter_tags)}"
        )

    vertices = np.asarray(quarter.grid.vertices, dtype=np.float64).T
    if np.min(vertices[:, :2]) < -1.0e-6:
        raise ValueError("quarter mesh must lie in the non-negative X/Y quadrant")
    meridian_rho_max = float(np.max(meridian.nodes[:, 0]))
    quarter_rho_max = float(np.max(np.hypot(vertices[:, 0], vertices[:, 1])))
    if not np.isclose(quarter_rho_max, meridian_rho_max, rtol=0.01, atol=1.0e-5):
        raise ValueError("quarter mesh radial extent does not match the meridian")
    if not np.allclose(
        [np.min(vertices[:, 2]), np.max(vertices[:, 2])],
        [np.min(meridian.nodes[:, 1]), np.max(meridian.nodes[:, 1])],
        rtol=0.01,
        atol=5.0e-4,
    ):
        raise ValueError("quarter mesh axial extent does not match the meridian")

    meridian_source_area, quarter_source_area = _source_areas(meridian, quarter)
    if not np.isclose(
        quarter_source_area, meridian_source_area, rtol=0.05, atol=1.0e-10
    ):
        raise ValueError("quarter mesh source area does not match the meridian")
    if generating_curve_reference is None:
        # Direct callers can still retrieve extent and area validation, but the
        # benchmark itself always supplies the high-resolution reference above.
        return {
            "geometry_contract_checked": False,
            "coarse_meridian_panel_distance": _meridian_quarter_geometry_distance(
                meridian, quarter
            ),
        }
    geometry_distance = _geometry_contract_diagnostics(
        meridian,
        generating_curve_reference,
        quarter,
        generating_curve_reference_metadata,
    )
    closure = geometry_distance["mouth_closure"]
    if closure["checked"] and not closure["passes"]:
        raise ValueError(
            "quarter mesh does not satisfy the flat freestanding mouth-closure "
            f"contract within {_GEOMETRY_DISTANCE_TOLERANCE_M:.6g} m"
        )
    return {"geometry_contract_checked": True, **geometry_distance}


def _validate_benchmark_geometry_config(raw_config: dict[str, Any]) -> None:
    """Validate the shared analytic geometry assumptions of all 3-D arms."""
    if str(raw_config.get("mode", "")).strip().lower() != "freestanding":
        raise ValueError("matched benchmark currently requires mode='freestanding'")
    cross_section = dict(raw_config.get("cross_section") or {})
    morph = dict(raw_config.get("morph") or {})
    if not np.isclose(float(cross_section.get("exponent", 2.0)), 2.0) or not np.isclose(
        float(cross_section.get("aspectRatio", 1.0)), 1.0
    ):
        raise ValueError("matched benchmark requires a circular cross section")
    if not np.isclose(float(morph.get("morphTarget", 0.0)), 0.0):
        raise ValueError("matched benchmark does not support morphed geometry")


def _validate_full_mesh_watertight(full: Any) -> None:
    """Reject reduced, open, or inconsistently oriented caller full meshes."""
    triangles = np.asarray(full.grid.elements, dtype=np.int64).T
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("full mesh must contain triangular elements")
    if triangles.size == 0:
        raise ValueError("full mesh must contain at least one triangle")
    if np.any(triangles < 0) or np.any(triangles >= full.grid.vertices.shape[1]):
        raise ValueError("full mesh contains an out-of-range vertex index")
    if np.any(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 2] == triangles[:, 0])
    ):
        raise ValueError("full mesh contains a degenerate triangle index")

    edge_counts: Counter[tuple[int, int]] = Counter()
    oriented_edges: dict[tuple[int, int], list[int]] = {}
    for triangle in triangles:
        for start, end in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            edge = (min(start, end), max(start, end))
            edge_counts[edge] += 1
            oriented_edges.setdefault(edge, []).append(1 if start < end else -1)
    nonmanifold = [edge for edge, count in edge_counts.items() if count != 2]
    if nonmanifold:
        raise ValueError(
            "full mesh must be watertight: every undirected edge must occur exactly "
            f"twice; found {len(nonmanifold)} offending edges (first {nonmanifold[0]})"
        )
    inconsistent = [
        edge for edge, directions in oriented_edges.items() if sum(directions) != 0
    ]
    if inconsistent:
        raise ValueError(
            "full mesh must have consistently oriented closed faces; shared edge "
            f"{inconsistent[0]} has matching traversal directions"
        )


def _validate_full_matched_input(
    raw_config: dict[str, Any],
    meridian: Any,
    full: Any,
    *,
    generating_curve_reference: Any | None = None,
) -> dict[str, Any]:
    """Validate an explicitly full domain without attributing symmetry to it."""
    _validate_benchmark_geometry_config(raw_config)
    _validate_full_mesh_watertight(full)

    meridian_tags = {int(tag) for tag in np.unique(meridian.physical_tags)}
    full_tags = {int(tag) for tag in np.unique(full.physical_tags)}
    if meridian_tags != {1, TAG_SOURCE} or full_tags != {1, TAG_SOURCE}:
        raise ValueError(
            "full comparison requires only rigid-wall tag 1 and source tag 2; "
            f"got meridian={sorted(meridian_tags)}, full={sorted(full_tags)}"
        )
    vertices = np.asarray(full.grid.vertices, dtype=np.float64).T
    # A caller must not accidentally label a reduced mesh as full. This permits
    # arbitrary topology but requires the physical revolution to occupy both
    # sides of both mirror planes in the fixed benchmark coordinate frame.
    if (
        np.min(vertices[:, 0]) >= -1.0e-6
        or np.max(vertices[:, 0]) <= 1.0e-6
        or np.min(vertices[:, 1]) >= -1.0e-6
        or np.max(vertices[:, 1]) <= 1.0e-6
    ):
        raise ValueError("full mesh must span both X and Y sides of the benchmark frame")
    meridian_rho_max = float(np.max(meridian.nodes[:, 0]))
    full_rho_max = float(np.max(np.hypot(vertices[:, 0], vertices[:, 1])))
    if not np.isclose(full_rho_max, meridian_rho_max, rtol=0.01, atol=1.0e-5):
        raise ValueError("full mesh radial extent does not match the meridian")
    if not np.allclose(
        [np.min(vertices[:, 2]), np.max(vertices[:, 2])],
        [np.min(meridian.nodes[:, 1]), np.max(meridian.nodes[:, 1])],
        rtol=0.01,
        atol=5.0e-4,
    ):
        raise ValueError("full mesh axial extent does not match the meridian")

    meridian_source_area = float(
        np.sum(
            meridian.segment_geometry().area_weights[
                np.asarray(meridian.physical_tags) == TAG_SOURCE
            ]
        )
    )
    full_source_area = _full_source_area(full)
    if not np.isclose(
        full_source_area, meridian_source_area, rtol=0.05, atol=1.0e-10
    ):
        raise ValueError("full mesh source area does not match the meridian")
    if generating_curve_reference is None:
        return {
            "geometry_contract_checked": False,
            "coarse_meridian_panel_distance": _meridian_quarter_geometry_distance(
                meridian, full
            ),
        }
    reference_distance = _meridian_quarter_geometry_distance(
        generating_curve_reference, full
    )
    if reference_distance["max_distance_m"] > _GEOMETRY_DISTANCE_TOLERANCE_M:
        raise ValueError(
            "full mesh does not match the high-resolution generating curve within "
            f"{_GEOMETRY_DISTANCE_TOLERANCE_M:.6g} m"
        )
    return {
        "geometry_contract_checked": True,
        "coarse_meridian_panel_distance": _meridian_quarter_geometry_distance(
            meridian, full
        ),
        "generating_curve_reference_distance": reference_distance,
        "full_source_area_m2": full_source_area,
    }


def _subdivide_meridian(meridian: Any, factor: int):
    """Subdivide every existing straight segment without rebuilding its geometry.

    The result retains each source/wall tag and its original normal exactly.
    It preserves the original node identities and inserts only interior nodes,
    keeping a closed one-trace meridian connected for CircSym validation.
    """
    if factor < 1:
        raise ValueError("meridian refinement factor must be positive")
    if factor == 1:
        return meridian

    from hornlab_metal_bem import MeridianMesh

    # Keep the original node identities.  CircSym's closed-body validator
    # relies on one connected trace from the first axis endpoint to the last;
    # independent child polylines with duplicated endpoints have identical
    # geometry but are topologically open.
    nodes: list[np.ndarray] = [node.copy() for node in np.asarray(meridian.nodes)]
    segments: list[tuple[int, int]] = []
    tags: list[int] = []
    normals: list[np.ndarray] = []
    source_nodes = np.asarray(meridian.nodes, dtype=np.float64)
    source_segments = np.asarray(meridian.segments, dtype=np.int64)
    source_tags = np.asarray(meridian.physical_tags, dtype=np.int32)
    source_normals = np.asarray(meridian.normals, dtype=np.float64)
    for segment, tag, normal in zip(source_segments, source_tags, source_normals):
        start = source_nodes[segment[0]]
        end = source_nodes[segment[1]]
        previous_node = int(segment[0])
        for part in range(1, factor):
            nodes.append(start + (end - start) * (part / factor))
            current_node = len(nodes) - 1
            segments.append((previous_node, current_node))
            tags.append(int(tag))
            normals.append(normal)
            previous_node = current_node
        segments.append((previous_node, int(segment[1])))
        tags.append(int(tag))
        normals.append(normal)
    return MeridianMesh(
        nodes=np.asarray(nodes, dtype=np.float64),
        segments=np.asarray(segments, dtype=np.int32),
        physical_tags=np.asarray(tags, dtype=np.int32),
        normals=np.asarray(normals, dtype=np.float64),
    )


def _meridian_surface_area_by_tag(meridian: Any) -> dict[str, float]:
    geometry = meridian.segment_geometry()
    tags = np.asarray(meridian.physical_tags, dtype=np.int32)
    return {
        str(int(tag)): float(np.sum(geometry.area_weights[tags == tag]))
        for tag in np.unique(tags)
    }


def _meridian_refinement_provenance(base: Any, refined: Any, factor: int) -> dict[str, Any]:
    """Evidence that a convergence rung preserved the original panel geometry."""
    base_areas = _meridian_surface_area_by_tag(base)
    refined_areas = _meridian_surface_area_by_tag(refined)
    area_relative_error_by_tag = {
        tag: abs(refined_areas[tag] - area) / max(abs(area), 1.0e-30)
        for tag, area in base_areas.items()
    }
    return {
        "factor": factor,
        "segment_count": refined.segment_count,
        "method": "uniform_subdivision_of_existing_straight_segments",
        "base_segment_count": base.segment_count,
        "expected_segment_count": base.segment_count * factor,
        "surface_area_relative_error_by_tag": area_relative_error_by_tag,
    }


def _requested_convergence_ladder_factors(
    requested: tuple[int, ...] | None, candidate_factor: int
) -> tuple[int, ...] | None:
    """Return an explicit ladder, never inventing a diagnostic rung to solve."""
    if requested is None:
        return None
    return tuple(sorted(set(requested + (candidate_factor,))))


def _finite_directivity_db(value: Any, *, label: str) -> np.ndarray:
    """Return a finite directivity array with a documented deep-null floor.

    A true zero pressure legitimately becomes ``-inf`` dB.  A finite floor
    keeps its difference observable and the JSON report standards-compliant;
    NaN and positive infinity instead identify an invalid solve and fail closed.
    """
    directivity = np.asarray(value, dtype=np.float64)
    if np.any(np.isnan(directivity)) or np.any(np.isposinf(directivity)):
        raise RuntimeError(f"{label} directivity contains NaN or positive infinity")
    return np.maximum(directivity, _DIRECTIVITY_DB_FLOOR)


def _max_directivity_difference(
    candidate_db: np.ndarray, reference_db: np.ndarray, mask: np.ndarray
) -> float:
    """Return a finite masked difference, including an empty-mask convention."""
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(candidate_db[mask] - reference_db[mask])))


def _accuracy_by_frequency(
    axisym: Any, quarter: Any, frequencies_hz: Any
) -> list[dict[str, Any]]:
    """Expose frequency-local errors so a few high-frequency rows cannot hide."""
    axisym_pressure = np.asarray(axisym.pressure_complex, dtype=np.complex128)
    quarter_pressure = np.asarray(quarter.pressure_complex, dtype=np.complex128)
    axisym_db = _finite_directivity_db(axisym.directivity_db, label="candidate")
    quarter_db = _finite_directivity_db(quarter.directivity_db, label="reference")
    if axisym_pressure.shape != quarter_pressure.shape:
        raise RuntimeError("matched comparison returned different pressure shapes")
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.shape != (axisym_pressure.shape[0],):
        raise RuntimeError("frequency grid does not match pressure rows")
    rows: list[dict[str, float]] = []
    for index in range(axisym_pressure.shape[0]):
        ap = np.ravel(axisym_pressure[index])
        qp = np.ravel(quarter_pressure[index])
        adb = np.ravel(axisym_db[index])
        qdb = np.ravel(quarter_db[index])
        pressure_scale = max(float(np.linalg.norm(qp)), 1.0e-30)
        # Keep the historical intersection mask for continuity, but also
        # report a fixed-reference mask. The intersection can discard a lobe
        # precisely when the candidate moves or loses it, hiding the error.
        reference_db_mask = qdb >= -40.0
        candidate_db_mask = adb >= -40.0
        db_mask = candidate_db_mask & reference_db_mask
        union_db_mask = candidate_db_mask | reference_db_mask
        candidate_only_db_mask = candidate_db_mask & ~reference_db_mask
        amplitude_floor = 1.0e-4 * float(np.max(np.abs(qp)))
        phase_diagnostics = _phase_region_diagnostics(ap, qp, amplitude_floor)
        phase_diagnostics_1e3 = _phase_region_diagnostics(
            ap, qp, 1.0e-3 * float(np.max(np.abs(qp)))
        )
        phase_diagnostics_1e2 = _phase_region_diagnostics(
            ap, qp, 1.0e-2 * float(np.max(np.abs(qp)))
        )
        rows.append(
            {
                "frequency_hz": float(frequencies[index]),
                "pressure_relative_l2": float(
                    np.linalg.norm(ap - qp) / pressure_scale
                ),
                "directivity_max_abs_db_above_minus_40": float(
                    _max_directivity_difference(adb, qdb, db_mask)
                ),
                "directivity_max_abs_db_reference_mask_above_minus_40": float(
                    _max_directivity_difference(adb, qdb, reference_db_mask)
                ),
                "directivity_max_abs_db_union_mask_above_minus_40": float(
                    _max_directivity_difference(adb, qdb, union_db_mask)
                ),
                "directivity_reference_mask_sample_count_above_minus_40": int(
                    np.count_nonzero(reference_db_mask)
                ),
                "directivity_candidate_mask_sample_count_above_minus_40": int(
                    np.count_nonzero(candidate_db_mask)
                ),
                "directivity_intersection_mask_sample_count_above_minus_40": int(
                    np.count_nonzero(db_mask)
                ),
                "directivity_union_mask_sample_count_above_minus_40": int(
                    np.count_nonzero(union_db_mask)
                ),
                "directivity_candidate_only_mask_sample_count_above_minus_40": int(
                    np.count_nonzero(candidate_only_db_mask)
                ),
                "directivity_candidate_only_max_abs_db_above_minus_40": float(
                    _max_directivity_difference(adb, qdb, candidate_only_db_mask)
                ),
                "directivity_reference_mask_coverage_by_candidate": float(
                    np.count_nonzero(db_mask)
                    / max(np.count_nonzero(reference_db_mask), 1)
                ),
                # This field is the official qualification metric.  Keep its
                # name and reference-only -80 dB mask stable; the following
                # fields merely explain whether a phase error lives in a null.
                "phase_rms_degrees_above_floor": phase_diagnostics[
                    "significant_field_phase_rms_degrees"
                ],
                "significant_field_sample_count": phase_diagnostics[
                    "significant_field_sample_count"
                ],
                "significant_field_valid_phase_sample_count": phase_diagnostics[
                    "significant_field_valid_phase_sample_count"
                ],
                "significant_field_undefined_zero_reference_sample_count": (
                    phase_diagnostics[
                        "significant_field_undefined_zero_reference_sample_count"
                    ]
                ),
                "null_region_phase_rms_degrees": phase_diagnostics[
                    "null_region_phase_rms_degrees"
                ],
                "null_region_sample_count": phase_diagnostics[
                    "null_region_sample_count"
                ],
                "null_region_valid_phase_sample_count": phase_diagnostics[
                    "null_region_valid_phase_sample_count"
                ],
                "null_region_undefined_zero_reference_sample_count": phase_diagnostics[
                    "null_region_undefined_zero_reference_sample_count"
                ],
                "phase_amplitude_floor_relative_to_quarter_peak": 1.0e-4,
                # These are tighter significant-field diagnostics only.  They
                # must never replace the established 1e-4 qualification mask.
                "phase_rms_degrees_above_1e-3_reference_peak": (
                    phase_diagnostics_1e3["significant_field_phase_rms_degrees"]
                ),
                "phase_sample_count_above_1e-3_reference_peak": (
                    phase_diagnostics_1e3["significant_field_sample_count"]
                ),
                "phase_valid_sample_count_above_1e-3_reference_peak": (
                    phase_diagnostics_1e3["significant_field_valid_phase_sample_count"]
                ),
                "phase_undefined_zero_reference_sample_count_above_1e-3_reference_peak": (
                    phase_diagnostics_1e3[
                        "significant_field_undefined_zero_reference_sample_count"
                    ]
                ),
                "phase_rms_degrees_above_1e-2_reference_peak": (
                    phase_diagnostics_1e2["significant_field_phase_rms_degrees"]
                ),
                "phase_sample_count_above_1e-2_reference_peak": (
                    phase_diagnostics_1e2["significant_field_sample_count"]
                ),
                "phase_valid_sample_count_above_1e-2_reference_peak": (
                    phase_diagnostics_1e2["significant_field_valid_phase_sample_count"]
                ),
                "phase_undefined_zero_reference_sample_count_above_1e-2_reference_peak": (
                    phase_diagnostics_1e2[
                        "significant_field_undefined_zero_reference_sample_count"
                    ]
                ),
            }
        )
    return rows


def _timed(call: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - started
    if not np.all(np.isfinite(result.pressure_complex)):
        raise RuntimeError("solver returned non-finite pressure")
    return result, {
        "wall_seconds": elapsed,
        "timings": _jsonable(dict(result.timings)),
    }


def _accuracy(axisym: Any, quarter: Any) -> dict[str, float]:
    axisym_pressure = np.asarray(axisym.pressure_complex, dtype=np.complex128)
    quarter_pressure = np.asarray(quarter.pressure_complex, dtype=np.complex128)
    if axisym_pressure.shape != quarter_pressure.shape:
        raise RuntimeError(
            "matched comparison returned different pressure shapes: "
            f"{axisym_pressure.shape} versus {quarter_pressure.shape}"
        )
    scale = max(float(np.linalg.norm(quarter_pressure)), 1.0e-30)
    pressure_rel_l2 = float(np.linalg.norm(axisym_pressure - quarter_pressure) / scale)
    magnitude_scale = max(float(np.linalg.norm(np.abs(quarter_pressure))), 1.0e-30)
    pressure_magnitude_rel_l2 = float(
        np.linalg.norm(np.abs(axisym_pressure) - np.abs(quarter_pressure))
        / magnitude_scale
    )
    axisym_db = _finite_directivity_db(axisym.directivity_db, label="candidate")
    quarter_db = _finite_directivity_db(quarter.directivity_db, label="reference")
    if axisym_db.shape != quarter_db.shape:
        raise RuntimeError(
            "matched comparison returned different directivity shapes: "
            f"{axisym_db.shape} versus {quarter_db.shape}"
        )
    directivity_max_abs_db = float(np.max(np.abs(axisym_db - quarter_db)))
    directivity_mask = (axisym_db >= -40.0) & (quarter_db >= -40.0)
    reference_mask = quarter_db >= -40.0
    candidate_mask = axisym_db >= -40.0
    union_mask = candidate_mask | reference_mask
    directivity_max_abs_db_above_minus_40 = _max_directivity_difference(
        axisym_db, quarter_db, directivity_mask
    )
    phase_diagnostics = _phase_region_diagnostics(
        axisym_pressure,
        quarter_pressure,
        1.0e-4 * float(np.max(np.abs(quarter_pressure))),
    )
    phase_rms_deg = phase_diagnostics["significant_field_phase_rms_degrees"]
    return {
        "pressure_relative_l2": pressure_rel_l2,
        "pressure_magnitude_relative_l2": pressure_magnitude_rel_l2,
        "directivity_max_abs_db": directivity_max_abs_db,
        "directivity_max_abs_db_above_minus_40": (
            directivity_max_abs_db_above_minus_40
        ),
        "directivity_max_abs_db_reference_mask_above_minus_40": (
            _max_directivity_difference(axisym_db, quarter_db, reference_mask)
        ),
        "directivity_max_abs_db_union_mask_above_minus_40": (
            _max_directivity_difference(axisym_db, quarter_db, union_mask)
        ),
        "phase_rms_degrees_above_floor": phase_rms_deg,
    }


def _phase_region_diagnostics(
    candidate_pressure: np.ndarray,
    reference_pressure: np.ndarray,
    amplitude_floor: float,
) -> dict[str, float | int]:
    """Split phase error into the existing significant field and reference nulls.

    The significant-field mask is exactly the legacy qualification mask:
    ``abs(reference) >= 1e-4 * max(abs(reference))``.  Null-region values are
    intentionally diagnostic only; they are not used by the qualification gate.
    """
    candidate = np.ravel(np.asarray(candidate_pressure, dtype=np.complex128))
    reference = np.ravel(np.asarray(reference_pressure, dtype=np.complex128))
    if candidate.shape != reference.shape:
        raise RuntimeError("phase comparison returned different pressure shapes")
    significant = np.abs(reference) >= amplitude_floor
    nonzero_reference = np.abs(reference) > 0.0

    def region_metrics(mask: np.ndarray, name: str) -> dict[str, float | int]:
        valid = mask & nonzero_reference
        if not np.any(valid):
            rms = 0.0
        else:
            phase_delta = np.angle(candidate[valid] / reference[valid])
            rms = float(np.sqrt(np.mean(np.square(np.rad2deg(phase_delta)))))
        return {
            f"{name}_sample_count": int(np.count_nonzero(mask)),
            f"{name}_valid_phase_sample_count": int(np.count_nonzero(valid)),
            f"{name}_undefined_zero_reference_sample_count": int(
                np.count_nonzero(mask & ~nonzero_reference)
            ),
            f"{name}_phase_rms_degrees": rms,
        }

    significant_metrics = region_metrics(significant, "significant_field")
    null_metrics = region_metrics(~significant, "null_region")
    return {
        "significant_field_sample_count": significant_metrics[
            "significant_field_sample_count"
        ],
        "significant_field_valid_phase_sample_count": significant_metrics[
            "significant_field_valid_phase_sample_count"
        ],
        "significant_field_undefined_zero_reference_sample_count": significant_metrics[
            "significant_field_undefined_zero_reference_sample_count"
        ],
        "significant_field_phase_rms_degrees": significant_metrics[
            "significant_field_phase_rms_degrees"
        ],
        "null_region_sample_count": null_metrics["null_region_sample_count"],
        "null_region_valid_phase_sample_count": null_metrics[
            "null_region_valid_phase_sample_count"
        ],
        "null_region_undefined_zero_reference_sample_count": null_metrics[
            "null_region_undefined_zero_reference_sample_count"
        ],
        "null_region_phase_rms_degrees": null_metrics["null_region_phase_rms_degrees"],
    }


def _axisym_parity(candidate: Any, reference: Any) -> dict[str, float]:
    metrics = _accuracy(candidate, reference)
    candidate_impedance = np.asarray(candidate.impedance, dtype=np.complex128)
    reference_impedance = np.asarray(reference.impedance, dtype=np.complex128)
    if candidate_impedance.shape != reference_impedance.shape:
        raise RuntimeError("axisymmetric parity returned different impedance shapes")
    scale = max(float(np.linalg.norm(reference_impedance)), 1.0e-30)
    metrics["impedance_relative_l2"] = float(
        np.linalg.norm(candidate_impedance - reference_impedance) / scale
    )
    return metrics


def _median_wall(records: list[dict[str, Any]]) -> float:
    return float(statistics.median(float(record["wall_seconds"]) for record in records))


def _passes_overall_qualification(
    *,
    speed_ratio: bool,
    half_second: bool,
    compact_cpu_parity: bool,
    numerical_gate: bool,
    per_frequency_numerical_gate: bool,
) -> bool:
    return (
        speed_ratio
        and half_second
        and compact_cpu_parity
        and numerical_gate
        and per_frequency_numerical_gate
    )


def _per_frequency_numerical_gate(
    rows: list[dict[str, Any]], numerical_gate: dict[str, float]
) -> dict[str, Any]:
    """Apply existing budgets to every requested frequency independently.

    This is deliberately additive to the historical aggregate L2 gate. It
    makes the unresolved upper band visible instead of allowing a low-energy
    failing row to be diluted by all other frequencies.
    """
    failures: list[dict[str, Any]] = []
    for row in rows:
        checks = {
            "pressure_relative_l2": (
                row["pressure_relative_l2"]
                <= numerical_gate["max_pressure_relative_l2"]
            ),
            "directivity_reference_mask": (
                row["directivity_max_abs_db_reference_mask_above_minus_40"]
                <= numerical_gate["max_directivity_error_db_above_minus_40"]
            ),
            # Keep the legacy overlap check visible, retain the fixed-reference
            # check, and add a union check.  The union catches a new candidate
            # lobe that the fixed-reference mask cannot see.
            "directivity_legacy_intersection_mask": (
                row["directivity_max_abs_db_above_minus_40"]
                <= numerical_gate["max_directivity_error_db_above_minus_40"]
            ),
            "directivity_union_mask": (
                row["directivity_max_abs_db_union_mask_above_minus_40"]
                <= numerical_gate["max_directivity_error_db_above_minus_40"]
            ),
            "phase_rms_degrees": (
                row["phase_rms_degrees_above_floor"]
                <= numerical_gate["max_phase_rms_degrees_above_floor"]
            ),
        }
        if not all(checks.values()):
            failures.append(
                {
                    "frequency_hz": row["frequency_hz"],
                    "failed_checks": [name for name, passed in checks.items() if not passed],
                    "metrics": {
                        "pressure_relative_l2": row["pressure_relative_l2"],
                        "directivity_max_abs_db_reference_mask_above_minus_40": (
                            row[
                                "directivity_max_abs_db_reference_mask_above_minus_40"
                            ]
                        ),
                        "directivity_max_abs_db_above_minus_40": row[
                            "directivity_max_abs_db_above_minus_40"
                        ],
                        "directivity_max_abs_db_union_mask_above_minus_40": row[
                            "directivity_max_abs_db_union_mask_above_minus_40"
                        ],
                        "phase_rms_degrees_above_floor": row[
                            "phase_rms_degrees_above_floor"
                        ],
                    },
                }
            )
    return {
        "all_frequencies_pass": not failures,
        "frequency_count": len(rows),
        "failed_frequency_count": len(failures),
        "failures": failures,
        "directivity_metrics": [
            "legacy_intersection_mask_above_minus_40_db",
            "fixed_reference_mask_above_minus_40_db",
            "union_mask_above_minus_40_db",
        ],
        "qualification_effect": "additive_to_aggregate_gate",
    }


def _run_meridian_convergence_ladder(
    *,
    base_meridian: Any,
    factors: tuple[int, ...],
    candidate_factor: int,
    candidate_result: Any,
    frequencies: np.ndarray,
    axisym_config: Any,
    quarter_result: Any,
    solve: Callable[[Any, np.ndarray, Any], Any],
) -> dict[str, Any]:
    """Solve a fixed-geometry CircSym ladder and compare each rung to the finest.

    This is deliberately separate from the timed arm.  Users who need a speed
    result at a converged resolution select that rung with
    ``--meridian-refinement-factor``; requesting a ladder alone cannot make a
    coarse timing look like a qualified fine-resolution timing.
    """
    meshes = {factor: _subdivide_meridian(base_meridian, factor) for factor in factors}
    results: dict[int, Any] = {}
    wall_seconds: dict[int, float | None] = {}
    for factor, mesh in meshes.items():
        if factor == candidate_factor:
            results[factor] = candidate_result
            wall_seconds[factor] = None
            continue
        started = time.perf_counter()
        results[factor] = solve(mesh, frequencies, axisym_config)
        wall_seconds[factor] = time.perf_counter() - started
    reference_factor = factors[-1]
    reference = results[reference_factor]
    rows: list[dict[str, Any]] = []
    for factor in factors:
        result = results[factor]
        rows.append(
            {
                "refinement": _meridian_refinement_provenance(
                    base_meridian, meshes[factor], factor
                ),
                "additional_wall_seconds": wall_seconds[factor],
                "axisymmetric_vs_finest_axisymmetric": _accuracy(result, reference),
                "axisymmetric_vs_quarter_3d": _accuracy(result, quarter_result),
                "axisymmetric_vs_quarter_3d_by_frequency": _accuracy_by_frequency(
                    result, quarter_result, frequencies
                ),
            }
        )
    return {
        "enabled": len(factors) > 1,
        "factors": factors,
        "reference_factor": reference_factor,
        "timed_candidate_factor": candidate_factor,
        "rows": rows,
    }


def _run_resonance_comparison(
    *,
    enabled: bool,
    chief_points: np.ndarray | None,
    chief_provenance: dict[str, Any] | None,
    chief_weight: float,
    complex_k_shifts: tuple[float, ...],
    meridian: Any,
    quarter: Any,
    frequencies: np.ndarray,
    axisym_config: Any,
    quarter_config: Any,
    solve_axisym: Callable[[Any, np.ndarray, Any], Any],
    solve_quarter: Callable[[Any, np.ndarray, Any], Any],
) -> dict[str, Any]:
    """Compare a caller-specified real-k+CHIEF solve against complex-k shifts.

    CHIEF points are deliberately an explicit input. They must lie in the
    excluded solid interior enclosed by the exterior boundary (not the horn air
    cavity), and only the caller can verify that placement for arbitrary meshes.
    These rows are diagnostic evidence only and do not feed the existing
    Axisymmetric qualification predicate.
    """
    if not enabled:
        return {
            "enabled": False,
            "qualification_effect": "none",
            "reason": "enable with --resonance-comparison and caller-supplied --chief-points",
        }
    if chief_points is None or chief_provenance is None:
        raise ValueError("resonance comparison requires caller-supplied CHIEF points")

    real_k_axisym_config = replace(
        axisym_config,
        formulation="standard",
        chief_points=chief_points,
        chief_weight=chief_weight,
    )
    real_k_quarter_config = replace(
        quarter_config,
        formulation="standard",
        chief_points=chief_points,
        chief_weight=chief_weight,
    )
    real_k_axisym = solve_axisym(meridian, frequencies, real_k_axisym_config)
    real_k_quarter = solve_quarter(quarter, frequencies, real_k_quarter_config)

    shift_rows: list[dict[str, Any]] = []
    for shift in complex_k_shifts:
        shift_axisym_config = replace(
            axisym_config,
            formulation="complex_k",
            complex_k_shift=shift,
            chief_points=None,
        )
        shift_quarter_config = replace(
            quarter_config,
            formulation="complex_k",
            complex_k_shift=shift,
            chief_points=None,
        )
        shift_axisym = solve_axisym(meridian, frequencies, shift_axisym_config)
        shift_quarter = solve_quarter(quarter, frequencies, shift_quarter_config)
        shift_rows.append(
            {
                "complex_k_shift": shift,
                "axisymmetric_vs_quarter_3d": _accuracy(shift_axisym, shift_quarter),
                "axisymmetric_vs_quarter_3d_by_frequency": _accuracy_by_frequency(
                    shift_axisym, shift_quarter, frequencies
                ),
                "axisymmetric_vs_real_k_chief": _accuracy(
                    shift_axisym, real_k_axisym
                ),
                "quarter_3d_vs_real_k_chief": _accuracy(
                    shift_quarter, real_k_quarter
                ),
            }
        )
    return {
        "enabled": True,
        "qualification_effect": "none",
        "chief": {**chief_provenance, "weight": chief_weight},
        "real_k_chief": {
            "formulation": "standard",
            "axisymmetric_vs_quarter_3d": _accuracy(real_k_axisym, real_k_quarter),
            "axisymmetric_vs_quarter_3d_by_frequency": _accuracy_by_frequency(
                real_k_axisym, real_k_quarter, frequencies
            ),
        },
        "complex_k_shift_ladder": shift_rows,
    }


def _full_3d_comparison_diagnostics(
    *,
    full: Any | None,
    full_result: Any | None,
    axisym_result: Any,
    quarter_result: Any,
    frequencies: np.ndarray,
    full_provenance: dict[str, Any] | None,
    full_source_area: float | None,
    full_velocity_scale: float | None,
    full_geometry_distance: dict[str, Any] | None,
    full_median_seconds: float | None,
) -> dict[str, Any]:
    """Report the optional full-domain control without changing any gate."""
    if full is None or full_result is None:
        return {
            "enabled": False,
            "qualification_effect": "none",
            "reason": "provide --full-mesh or --reflect-quarter-to-full",
        }
    if full_provenance is None or full_source_area is None or full_velocity_scale is None:
        raise RuntimeError("full-domain comparison is missing input provenance")
    return {
        "enabled": True,
        # This wording is intentionally machine-readable so downstream result
        # consumers cannot accidentally interpret a full-mesh diagnostic as a
        # replacement for the established quarter qualification oracle.
        "qualification_effect": "none",
        "full_mesh": {
            **full_provenance,
            "vertices": int(full.info.n_vertices),
            "triangles": int(full.info.n_triangles),
            "source_area_m2": full_source_area,
            "velocity_scale_for_equal_volume_velocity": full_velocity_scale,
            "geometry_contract": full_geometry_distance,
        },
        "timing": {
            # Full solves are one diagnostic pass after the paired timed arms,
            # so this is intentionally not presented as a comparable median.
            "full_3d_diagnostic_wall_seconds": full_median_seconds,
            "timing_role": "after_paired_axisymmetric_quarter_qualification_arms",
        },
        "quarter_3d_vs_full_3d": _accuracy(quarter_result, full_result),
        "quarter_3d_vs_full_3d_by_frequency": _accuracy_by_frequency(
            quarter_result, full_result, frequencies
        ),
        "axisymmetric_vs_full_3d": _accuracy(axisym_result, full_result),
        "axisymmetric_vs_full_3d_by_frequency": _accuracy_by_frequency(
            axisym_result, full_result, frequencies
        ),
    }


def _run_full_mesh_convergence_ladder(
    *,
    rungs: list[dict[str, Any]] | None,
    candidate_result: Any | None,
    axisym_result: Any,
    quarter_result: Any,
    frequencies: np.ndarray,
    solve: Callable[[Any, np.ndarray, Any], Any],
) -> dict[str, Any]:
    """Compare caller-ordered independently remeshed full-domain rungs.

    The caller explicitly declares order with ``--full-mesh-ladder``.  We do
    not infer refinement from triangle count: local sizing and topology can
    make that unsafe.  This remains evidence about the full reference only;
    it never changes the Axisym qualification predicate.
    """
    if rungs is None:
        return {
            "enabled": False,
            "qualification_effect": "none",
            "reason": "request --full-mesh-ladder with caller-ordered coarse-to-fine meshes",
        }
    if candidate_result is None:
        raise RuntimeError("full mesh ladder requires its first rung result")
    if len(rungs) < 2:
        raise RuntimeError("full mesh ladder requires at least two rungs")
    results = [candidate_result]
    timing_records: list[dict[str, Any] | None] = [None]
    for rung in rungs[1:]:
        result, timing = _timed(
            lambda rung=rung: solve(rung["mesh"], frequencies, rung["config"])
        )
        results.append(result)
        timing_records.append(timing)
    finest = results[-1]
    rows: list[dict[str, Any]] = []
    for index, (rung, result, timing) in enumerate(
        zip(rungs, results, timing_records, strict=True)
    ):
        mesh = rung["mesh"]
        row = {
            "rung": index,
            "mesh": {
                **rung["provenance"],
                "vertices": int(mesh.info.n_vertices),
                "triangles": int(mesh.info.n_triangles),
                "source_area_m2": rung["source_area_m2"],
                "velocity_scale_for_equal_volume_velocity": rung[
                    "velocity_scale_for_equal_volume_velocity"
                ],
                "geometry_contract": rung["geometry_contract"],
            },
            "additional_timing": timing,
            "full_3d_vs_finest_full_3d": _accuracy(result, finest),
            "full_3d_vs_finest_full_3d_by_frequency": _accuracy_by_frequency(
                result, finest, frequencies
            ),
        }
        if index == 0:
            row["full_3d_vs_previous_full_3d"] = None
            row["full_3d_vs_previous_full_3d_by_frequency"] = None
        else:
            previous = results[index - 1]
            # The finer rung is always the reference, matching the
            # coarse-to-fine caller contract and the per-frequency masks.
            row["full_3d_vs_previous_full_3d"] = _accuracy(previous, result)
            row["full_3d_vs_previous_full_3d_by_frequency"] = _accuracy_by_frequency(
                previous, result, frequencies
            )
        rows.append(row)
    return {
        "enabled": True,
        "qualification_effect": "none",
        "ordering": "caller_supplied_coarse_to_fine",
        "reference_rung": len(rungs) - 1,
        "timed_candidate_rung": 0,
        "axisymmetric_vs_finest_full_3d": _accuracy(axisym_result, finest),
        "axisymmetric_vs_finest_full_3d_by_frequency": _accuracy_by_frequency(
            axisym_result, finest, frequencies
        ),
        "quarter_3d_vs_finest_full_3d": _accuracy(quarter_result, finest),
        "quarter_3d_vs_finest_full_3d_by_frequency": _accuracy_by_frequency(
            quarter_result, finest, frequencies
        ),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    chief_points: np.ndarray | None = None
    chief_provenance: dict[str, Any] | None = None
    if args.chief_points is not None:
        chief_points, chief_provenance = _load_chief_points(args.chief_points)
    os.environ["HORNLAB_CIRCSYM_ASSEMBLY_BACKEND"] = args.axisym_backend
    os.environ["HORNLAB_CIRCSYM_FIELD_BACKEND"] = args.axisym_backend
    os.environ["HORNLAB_CIRCSYM_CPU_FIELD_BACKEND"] = args.cpu_field
    os.environ["HORNLAB_CIRCSYM_AZIMUTH_POINTS_MIN"] = str(args.azimuth_min)

    import hornlab_metal_bem as mb

    (
        base_meridian,
        quarter,
        full,
        axisym_config,
        quarter_config,
        full_config,
        geometry_distance,
        full_geometry_distance,
        full_source_area,
        full_velocity_scale,
        full_provenance,
        full_mesh_ladder,
    ) = _build_inputs(
        args.config,
        args.quarter_mesh,
        args.angles,
        args.full_mesh,
        args.reflect_quarter_to_full,
        args.full_mesh_ladder,
    )
    ladder_factors = _requested_convergence_ladder_factors(
        args.meridian_refinement_factors, args.meridian_refinement_factor
    )
    meridian = _subdivide_meridian(base_meridian, args.meridian_refinement_factor)
    meridian_source_area, quarter_source_area = _source_areas(meridian, quarter)
    quarter_velocity_scale = meridian_source_area / quarter_source_area
    frequencies = np.geomspace(args.f1, args.f2, args.frequencies)

    def axisym_call():
        return mb.solve_circsym_frequencies(meridian, frequencies, axisym_config)

    def quarter_call():
        return mb.solve_frequencies(quarter, frequencies, quarter_config)

    def full_call():
        if full is None or full_config is None:
            raise RuntimeError("full-domain solve requested without a full mesh")
        return mb.solve_frequencies(full, frequencies, full_config)

    axisym_warmup, axisym_cold = _timed(axisym_call)
    quarter_warmup, quarter_cold = _timed(quarter_call)
    records: dict[str, list[dict[str, Any]]] = {"axisymmetric": [], "quarter_3d": []}
    last_results = {"axisymmetric": axisym_warmup, "quarter_3d": quarter_warmup}
    for repeat_index in range(args.repeats):
        order = ("axisymmetric", "quarter_3d")
        if repeat_index % 2:
            order = tuple(reversed(order))
        for name in order:
            call = axisym_call if name == "axisymmetric" else quarter_call
            result, record = _timed(call)
            record["repeat_index"] = repeat_index
            record["order_in_pair"] = order.index(name)
            records[name].append(record)
            last_results[name] = result

    axisym_median = _median_wall(records["axisymmetric"])
    quarter_median = _median_wall(records["quarter_3d"])
    # Full-domain controls are deliberately outside the paired qualification
    # loop.  A large dense full solve can perturb memory pressure and GPU/CPU
    # scheduling, making the Axisym/quarter timing ratio incomparable.
    full_diagnostic_timing: dict[str, Any] | None = None
    if full is not None:
        full_result, full_diagnostic_timing = _timed(full_call)
        last_results["full_3d"] = full_result
    full_median = (
        float(full_diagnostic_timing["wall_seconds"])
        if full_diagnostic_timing is not None
        else None
    )
    ratio = axisym_median / quarter_median
    axisym_candidate = last_results["axisymmetric"]
    os.environ["HORNLAB_CIRCSYM_ASSEMBLY_BACKEND"] = "cpu"
    os.environ["HORNLAB_CIRCSYM_FIELD_BACKEND"] = "cpu"
    compact_cpu_reference = axisym_call()
    os.environ["HORNLAB_CIRCSYM_ASSEMBLY_BACKEND"] = args.axisym_backend
    os.environ["HORNLAB_CIRCSYM_FIELD_BACKEND"] = args.axisym_backend
    os.environ["HORNLAB_CIRCSYM_AZIMUTH_POINTS_MIN"] = "64"
    axisym_reference = axisym_call()
    os.environ["HORNLAB_CIRCSYM_AZIMUTH_POINTS_MIN"] = str(args.azimuth_min)
    cross_solver_difference = _accuracy(
        last_results["axisymmetric"], last_results["quarter_3d"]
    )
    full_3d_comparison = _full_3d_comparison_diagnostics(
        full=full,
        full_result=last_results.get("full_3d"),
        axisym_result=last_results["axisymmetric"],
        quarter_result=last_results["quarter_3d"],
        frequencies=frequencies,
        full_provenance=full_provenance,
        full_source_area=full_source_area,
        full_velocity_scale=full_velocity_scale,
        full_geometry_distance=full_geometry_distance,
        full_median_seconds=full_median,
    )
    full_3d_convergence_ladder = _run_full_mesh_convergence_ladder(
        rungs=full_mesh_ladder,
        candidate_result=last_results.get("full_3d"),
        axisym_result=last_results["axisymmetric"],
        quarter_result=last_results["quarter_3d"],
        frequencies=frequencies,
        solve=mb.solve_frequencies,
    )
    numerical_gate = {
        "max_pressure_relative_l2": args.max_pressure_rel_l2,
        "max_directivity_error_db_above_minus_40": args.max_directivity_error_db,
        "max_phase_rms_degrees_above_floor": args.max_phase_rms_deg,
    }
    cross_solver_difference_by_frequency = _accuracy_by_frequency(
        last_results["axisymmetric"], last_results["quarter_3d"], frequencies
    )
    passes_numerical_gate = (
        cross_solver_difference["pressure_relative_l2"] <= args.max_pressure_rel_l2
        and cross_solver_difference["directivity_max_abs_db_above_minus_40"]
        <= args.max_directivity_error_db
        and cross_solver_difference[
            "directivity_max_abs_db_reference_mask_above_minus_40"
        ]
        <= args.max_directivity_error_db
        and cross_solver_difference[
            "directivity_max_abs_db_union_mask_above_minus_40"
        ]
        <= args.max_directivity_error_db
        and cross_solver_difference["phase_rms_degrees_above_floor"]
        <= args.max_phase_rms_deg
    )
    per_frequency_numerical_gate = _per_frequency_numerical_gate(
        cross_solver_difference_by_frequency, numerical_gate
    )
    passes_per_frequency_numerical_gate = per_frequency_numerical_gate[
        "all_frequencies_pass"
    ]
    passes_speed_ratio = ratio < args.qualification_ratio
    passes_half_second = axisym_median <= 0.5
    compact_cpu_parity = _axisym_parity(axisym_candidate, compact_cpu_reference)
    passes_compact_cpu_parity = (
        compact_cpu_parity["pressure_relative_l2"] < 5.0e-5
        and compact_cpu_parity["directivity_max_abs_db"] < 0.01
        and compact_cpu_parity["impedance_relative_l2"] < 5.0e-5
    )
    try:
        package_version = importlib.metadata.version("hornlab-metal-bem")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source-tree"
    from hornlab_metal_bem.metal.native import discover_native_runtime

    native_runtime = discover_native_runtime(run_smoke_test=True)
    helper_build_flavor = None
    if native_runtime.helper_executable_path is not None:
        helper_parts = native_runtime.helper_executable_path.parts
        if "release" in helper_parts:
            helper_build_flavor = "release"
        elif "debug" in helper_parts:
            helper_build_flavor = "debug"
    if ladder_factors is None:
        meridian_convergence_ladder = {
            "enabled": False,
            "timed_candidate_factor": args.meridian_refinement_factor,
            "reason": "request --meridian-refinement-factors to run additional rungs",
        }
    else:
        meridian_convergence_ladder = _run_meridian_convergence_ladder(
            base_meridian=base_meridian,
            factors=ladder_factors,
            candidate_factor=args.meridian_refinement_factor,
            candidate_result=axisym_candidate,
            frequencies=frequencies,
            axisym_config=axisym_config,
            quarter_result=last_results["quarter_3d"],
            solve=mb.solve_circsym_frequencies,
        )
    resonance_comparison = _run_resonance_comparison(
        enabled=args.resonance_comparison,
        chief_points=chief_points,
        chief_provenance=chief_provenance,
        chief_weight=args.chief_weight,
        complex_k_shifts=args.complex_k_shifts,
        meridian=meridian,
        quarter=quarter,
        frequencies=frequencies,
        axisym_config=axisym_config,
        quarter_config=quarter_config,
        solve_axisym=mb.solve_circsym_frequencies,
        solve_quarter=mb.solve_frequencies,
    )
    payload = {
        "benchmark": "axisymmetric_vs_quarter_3d",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "package_version": package_version,
        "process_load_average": os.getloadavg(),
        "native_runtime": {
            "helper_source": native_runtime.helper_source,
            "helper_name": (
                native_runtime.helper_executable_path.name
                if native_runtime.helper_executable_path is not None
                else None
            ),
            "apple_silicon": native_runtime.is_apple_silicon,
            "build_flavor": helper_build_flavor,
            "smoke_test_ok": native_runtime.smoke_test_ok,
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "HORNLAB_CIRCSYM_ASSEMBLY_BACKEND",
                "HORNLAB_CIRCSYM_FIELD_BACKEND",
                "HORNLAB_CIRCSYM_CPU_FIELD_BACKEND",
                "HORNLAB_CIRCSYM_CPU_REMAINDER_BACKEND",
                "HORNLAB_CIRCSYM_AZIMUTH_POINTS_MIN",
                "HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE",
                "HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE",
                "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE",
                "HORNLAB_METAL_BEM_NATIVE_SOLVE_CONCURRENCY",
                "NUMBA_THREADING_LAYER",
                "NUMBA_NUM_THREADS",
            )
        },
        "inputs": {
            "config": str(args.config.resolve()),
            "quarter_mesh": str(args.quarter_mesh.resolve()),
            "frequency_count": args.frequencies,
            "frequencies_hz": frequencies,
            "angle_count": args.angles,
            "base_meridian_segments": base_meridian.segment_count,
            "meridian_segments": meridian.segment_count,
            "meridian_refinement": _meridian_refinement_provenance(
                base_meridian, meridian, args.meridian_refinement_factor
            ),
            "quarter_triangles": quarter.info.n_triangles,
            "quarter_symmetry": "yz+xz",
            "axisymmetric_azimuth_min": args.azimuth_min,
            "meridian_source_area_m2": meridian_source_area,
            "quarter_source_area_m2": quarter_source_area,
            "quarter_velocity_scale_for_equal_volume_velocity": (
                quarter_velocity_scale
            ),
            "quarter_meridian_geometry_distance": geometry_distance,
            "full_mesh": (
                str(args.full_mesh.resolve()) if args.full_mesh is not None else None
            ),
            "full_mesh_ladder": (
                [str(path.resolve()) for path in args.full_mesh_ladder]
                if args.full_mesh_ladder is not None
                else None
            ),
            "reflect_quarter_to_full": args.reflect_quarter_to_full,
        },
        "benchmark_provenance": {
            "quarter_3d": {
                # The dense dtype is a config property because solve_frequencies
                # installs it in the native helper environment per call.
                "dense_solve_dtype_config": quarter_config.dense_solve_dtype,
                "dense_solve_implementation_config": (
                    quarter_config.dense_solve_implementation
                ),
                "native_near_quadrature_environment": os.environ.get(
                    "HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE"
                ),
            }
        },
        "axisymmetric_backend": {
            "assembly": args.axisym_backend,
            "field": (
                f"cpu-{args.cpu_field}"
                if args.axisym_backend == "cpu"
                else "metal-frequency-batch"
            ),
        },
        "warmup_excluded": {
            "axisymmetric": axisym_cold,
            "quarter_3d": quarter_cold,
            "full_3d": None,
        },
        "runs": records,
        "summary": {
            "axisymmetric_median_seconds": axisym_median,
            "quarter_3d_median_seconds": quarter_median,
            "full_3d_diagnostic_wall_seconds": full_median,
            "axisymmetric_over_quarter_ratio": ratio,
            "faster_than_quarter": ratio < 1.0,
            "qualification_ratio": args.qualification_ratio,
            "passes_speed_ratio": passes_speed_ratio,
            "passes_half_second_target": passes_half_second,
            "passes_compact_cpu_parity": passes_compact_cpu_parity,
            "passes_numerical_gate": passes_numerical_gate,
            "passes_per_frequency_numerical_gate": passes_per_frequency_numerical_gate,
            "passes_overall_qualification": _passes_overall_qualification(
                speed_ratio=passes_speed_ratio,
                half_second=passes_half_second,
                compact_cpu_parity=passes_compact_cpu_parity,
                numerical_gate=passes_numerical_gate,
                per_frequency_numerical_gate=passes_per_frequency_numerical_gate,
            ),
        },
        "numerical_gate": numerical_gate,
        "cross_solver_difference": cross_solver_difference,
        "cross_solver_difference_by_frequency": cross_solver_difference_by_frequency,
        "per_frequency_numerical_gate": per_frequency_numerical_gate,
        "full_3d_comparison": full_3d_comparison,
        "full_3d_convergence_ladder": full_3d_convergence_ladder,
        "meridian_convergence_ladder": meridian_convergence_ladder,
        "resonance_comparison": resonance_comparison,
        "candidate_vs_64_point_axisymmetric": _accuracy(
            last_results["axisymmetric"], axisym_reference
        ),
        "candidate_vs_compact_cpu": compact_cpu_parity,
        "compact_cpu_parity_gate": {
            "max_pressure_relative_l2": 5.0e-5,
            "max_directivity_error_db": 0.01,
            "max_impedance_relative_l2": 5.0e-5,
        },
    }
    if args.json:
        print(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False))
    else:
        summary = payload["summary"]
        print(
            "Axisymmetric median "
            f"{summary['axisymmetric_median_seconds']:.3f} s; quarter 3-D median "
            f"{summary['quarter_3d_median_seconds']:.3f} s; ratio "
            f"{summary['axisymmetric_over_quarter_ratio']:.3f}"
        )
        print(
            "qualification: "
            + ("PASS" if summary["passes_overall_qualification"] else "FAIL")
        )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())

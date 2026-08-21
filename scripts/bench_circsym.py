#!/usr/bin/env python3
"""Benchmark the production CircSym sweep on canonical horn meridians."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TAG_WALL = 1
TAG_SOURCE = 2
TAG_APERTURE = 4
RELEVANT_ENV = (
    "HORNLAB_CIRCSYM_ASSEMBLY_BACKEND",
    "HORNLAB_CIRCSYM_FIELD_BACKEND",
    "HORNLAB_CIRCSYM_CPU_REMAINDER_BACKEND",
    "HORNLAB_CIRCSYM_ASSEMBLY_THREADS",
    "HORNLAB_CIRCSYM_FIELD_THREADS",
)


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        choices=("freestanding", "infinite-baffle"),
        default="freestanding",
        help="canonical horn mounting (default: %(default)s)",
    )
    parser.add_argument("--f1", type=_positive_float, default=400.0, metavar="HZ")
    parser.add_argument("--f2", type=_positive_float, default=16_000.0, metavar="HZ")
    parser.add_argument(
        "--frequencies", type=_positive_int, default=24, metavar="N"
    )
    parser.add_argument("--angles", type=_positive_int, default=37, metavar="N")
    parser.add_argument("--repeat", type=_positive_int, default=2, metavar="N")
    parser.add_argument(
        "--target-edge-mm", type=_positive_float, default=6.0, metavar="MM"
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "metal"),
        default="auto",
        help="assembly and field backend (default: %(default)s)",
    )
    parser.add_argument(
        "--cpu-remainder",
        choices=("auto", "c", "numba", "numpy"),
        default="auto",
        help="portable CPU remainder kernel (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.f2 < args.f1:
        parser.error("--f2 must be greater than or equal to --f1")
    if args.angles < 2:
        parser.error("--angles must be at least 2")
    return args


def _resampled_meridian(
    control_points: np.ndarray,
    edge_tags: list[int],
    *,
    target_edge_m: float,
):
    from hornlab_metal_bem import MeridianMesh

    points = [np.asarray(control_points[0], dtype=np.float64)]
    tags: list[int] = []
    for start, end, tag in zip(
        control_points[:-1], control_points[1:], edge_tags, strict=True
    ):
        length = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(length / target_edge_m)))
        for index in range(1, count + 1):
            points.append(start + (end - start) * (index / count))
            tags.append(int(tag))
    return MeridianMesh.from_polyline(np.asarray(points), np.asarray(tags))


def build_fixture(name: str, *, target_edge_m: float):
    """Build a small, closed production-like horn without a mesher dependency."""

    if name == "freestanding":
        points = np.asarray(
            [
                [0.0, -0.1500],
                [0.0127, -0.1500],
                [0.1200, 0.0],
                [0.1260, -0.0030],
                [0.0187, -0.1560],
                [0.0, -0.1560],
            ],
            dtype=np.float64,
        )
        return _resampled_meridian(
            points,
            [TAG_SOURCE, TAG_WALL, TAG_WALL, TAG_WALL, TAG_WALL],
            target_edge_m=target_edge_m,
        )
    if name == "infinite-baffle":
        points = np.asarray(
            [
                [0.0, -0.0800],
                [0.0127, -0.0800],
                [0.0500, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
        return _resampled_meridian(
            points,
            [TAG_SOURCE, TAG_WALL, TAG_APERTURE],
            target_edge_m=target_edge_m,
        )
    raise ValueError(f"unknown CircSym benchmark fixture: {name}")


def solve_fixture(
    name: str,
    frequencies_hz: np.ndarray,
    *,
    target_edge_m: float,
    angle_count: int,
):
    from hornlab_metal_bem import solve_circsym_frequencies
    from hornlab_metal_bem.config import ObservationConfig, SolveConfig, VelocityMode

    meridian = build_fixture(name, target_edge_m=target_edge_m)
    kwargs: dict[str, Any] = {}
    if name == "infinite-baffle":
        kwargs["circsym_aperture_tag"] = TAG_APERTURE
    config = SolveConfig(
        velocity_sources={TAG_SOURCE: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        formulation="complex_k",
        complex_k_shift=0.005,
        observation=ObservationConfig(
            distance_m=2.0,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=angle_count,
            planes=["horizontal"],
            origin="mouth",
        ),
        **kwargs,
    )
    return meridian, solve_circsym_frequencies(meridian, frequencies_hz, config)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imaginary": value.imag}
    return value


def _run(
    fixture: str,
    frequencies_hz: np.ndarray,
    *,
    target_edge_m: float,
    angle_count: int,
) -> tuple[Any, Any, dict[str, Any]]:
    started = time.perf_counter()
    meridian, result = solve_fixture(
        fixture,
        frequencies_hz,
        target_edge_m=target_edge_m,
        angle_count=angle_count,
    )
    elapsed = time.perf_counter() - started
    diagnostics = list(getattr(result, "native_diagnostics", []) or [])
    return meridian, result, {
        "wall_seconds": elapsed,
        "seconds_per_frequency": elapsed / len(frequencies_hz),
        "native_timings": _jsonable(dict(getattr(result, "timings", {}) or {})),
        "assembly_backends": sorted(
            {
                str(item.get("assembly_backend"))
                for item in diagnostics
                if isinstance(item, dict) and item.get("assembly_backend")
            }
        ),
        "field_backends": sorted(
            {
                str(item.get("field_backend"))
                for item in diagnostics
                if isinstance(item, dict) and item.get("field_backend")
            }
        ),
        "azimuth_quadrature_points": [
            int(item["azimuth_quadrature_points"])
            for item in diagnostics
            if isinstance(item, dict) and item.get("azimuth_quadrature_points")
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["HORNLAB_CIRCSYM_ASSEMBLY_BACKEND"] = args.backend
    os.environ["HORNLAB_CIRCSYM_FIELD_BACKEND"] = args.backend
    os.environ["HORNLAB_CIRCSYM_CPU_REMAINDER_BACKEND"] = args.cpu_remainder
    frequencies = np.geomspace(args.f1, args.f2, args.frequencies)

    records: list[dict[str, Any]] = []
    meridian = None
    result = None
    for _ in range(args.repeat):
        meridian, result, record = _run(
            args.fixture,
            frequencies,
            target_edge_m=args.target_edge_mm * 0.001,
            angle_count=args.angles,
        )
        records.append(record)
    assert meridian is not None and result is not None

    try:
        version = importlib.metadata.version("hornlab-metal-bem")
    except importlib.metadata.PackageNotFoundError:
        version = "source-tree"
    payload = {
        "benchmark": "circsym_sweep",
        "fixture": args.fixture,
        "package_version": version,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "frequency_count": len(frequencies),
        "frequencies_hz": frequencies.tolist(),
        "angle_count": args.angles,
        "target_edge_mm": args.target_edge_mm,
        "meridian_segments": meridian.segment_count,
        "environment": {name: os.environ.get(name) for name in RELEVANT_ENV},
        "warmup": records[0],
        "runs": records[1:],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"CircSym {args.fixture}: {meridian.segment_count} segments, "
            f"{len(frequencies)} frequencies"
        )
        for index, record in enumerate(records):
            label = "warm-up" if index == 0 else f"run {index}"
            print(
                f"  {label}: {record['wall_seconds']:.3f} s "
                f"({record['seconds_per_frequency']:.3f} s/frequency)"
            )
        print("  assembly:", ", ".join(records[-1]["assembly_backends"]) or "unknown")
        print("  field:", ", ".join(records[-1]["field_backends"]) or "unknown")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())

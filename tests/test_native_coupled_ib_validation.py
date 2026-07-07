from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j1

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem.circsym import MeridianMesh
from hornlab_metal_bem.config import ObservationConfig, SolveConfig, VelocityMode
from hornlab_metal_bem.mesh import LoadedMesh, make_pure_grid
from hornlab_metal_bem.metal import discover_native_runtime
from hornlab_metal_bem.observation import ObservationFrame
from hornlab_metal_bem.result import MeshInfo

TAG_THROAT = 2
TAG_WALL = 3
TAG_APERTURE = 4

_TRIANGLE_QX = np.array(
    [
        0.4459484909159651,
        0.0915762135097710,
        0.1081030181680700,
        0.4459484909159651,
        0.8168475729804590,
        0.0915762135097710,
    ],
    dtype=np.float64,
)
_TRIANGLE_QY = np.array(
    [
        0.4459484909159651,
        0.0915762135097700,
        0.4459484909159651,
        0.1081030181680700,
        0.0915762135097700,
        0.8168475729804580,
    ],
    dtype=np.float64,
)
_TRIANGLE_QW = np.array(
    [
        0.5 * 0.2233815896780110,
        0.5 * 0.1099517436553220,
        0.5 * 0.2233815896780110,
        0.5 * 0.2233815896780110,
        0.5 * 0.1099517436553220,
        0.5 * 0.1099517436553220,
    ],
    dtype=np.float64,
)


def _triangulated_disc(
    radius: float,
    *,
    rings: int,
    sectors: int,
    z: float = 0.0,
    normal_sign: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, z)]
    ring_indices: list[list[int]] = []
    for ring in range(1, rings + 1):
        row: list[int] = []
        r = radius * ring / rings
        for sector in range(sectors):
            theta = 2.0 * np.pi * sector / sectors
            row.append(len(vertices))
            vertices.append((r * np.cos(theta), r * np.sin(theta), z))
        ring_indices.append(row)

    triangles: list[list[int]] = []
    first = ring_indices[0]
    for sector in range(sectors):
        nxt = (sector + 1) % sectors
        tri = [0, first[sector], first[nxt]]
        triangles.append(tri if normal_sign > 0 else [tri[0], tri[2], tri[1]])

    for ring in range(1, rings):
        inner = ring_indices[ring - 1]
        outer = ring_indices[ring]
        for sector in range(sectors):
            nxt = (sector + 1) % sectors
            tris = [
                [inner[sector], outer[sector], outer[nxt]],
                [inner[sector], outer[nxt], inner[nxt]],
            ]
            if normal_sign < 0:
                tris = [[a, c, b] for a, b, c in tris]
            triangles.extend(tris)

    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int32)


def _rayleigh_pressure_uniform_disc(
    vertices: np.ndarray,
    triangles: np.ndarray,
    points: np.ndarray,
    k: float,
) -> np.ndarray:
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    source_points = []
    weights = []
    for xi, eta, weight in zip(_TRIANGLE_QX, _TRIANGLE_QY, _TRIANGLE_QW, strict=True):
        source_points.append((1.0 - xi - eta) * v0 + xi * v1 + eta * v2)
        weights.append(weight * 2.0 * areas)
    sources = np.concatenate(source_points, axis=0)
    source_weights = np.concatenate(weights, axis=0)

    out = np.empty(points.shape[0], dtype=np.complex128)
    for point_index, point in enumerate(points):
        distance = np.linalg.norm(sources - point[None, :], axis=1)
        green = np.exp(1j * k * distance) / (4.0 * np.pi * distance)
        out[point_index] = 2.0 * np.sum(green * source_weights)
    return out


def _airy_directivity(ka: float, angles_deg: np.ndarray) -> np.ndarray:
    x = ka * np.sin(np.deg2rad(angles_deg))
    out = np.ones_like(x, dtype=np.float64)
    mask = np.abs(x) > 1.0e-12
    out[mask] = 2.0 * j1(x[mask]) / x[mask]
    return np.abs(out)


def _first_crossing_deg(
    angles_deg: np.ndarray,
    values_db: np.ndarray,
    target_db: float,
) -> float:
    for i in range(1, angles_deg.size):
        y0 = float(values_db[i - 1] - target_db)
        y1 = float(values_db[i] - target_db)
        if y0 == 0.0:
            return float(angles_deg[i - 1])
        if y0 * y1 <= 0.0:
            frac = y0 / (y0 - y1)
            return float(angles_deg[i - 1] + frac * (angles_deg[i] - angles_deg[i - 1]))
    raise AssertionError(f"no {target_db} dB crossing")


def _resample_polyline(points: np.ndarray, target_edge: float) -> np.ndarray:
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:], strict=True):
        count = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / target_edge)))
        for step in range(1, count + 1):
            out.append(a + (b - a) * (step / count))
    return np.asarray(out, dtype=np.float64)


def _straight_channel_meridian(
    radius: float,
    depth: float,
    *,
    target_edge: float,
) -> MeridianMesh:
    cap = _resample_polyline(np.array([[0.0, -depth], [radius, -depth]]), target_edge)
    wall = _resample_polyline(np.array([[radius, -depth], [radius, 0.0]]), target_edge)
    aperture = _resample_polyline(np.array([[radius, 0.0], [0.0, 0.0]]), target_edge)
    points = np.vstack([cap, wall[1:], aperture[1:]])
    tags = np.concatenate(
        [
            np.full(len(cap) - 1, TAG_THROAT, dtype=np.int32),
            np.full(len(wall) - 1, TAG_WALL, dtype=np.int32),
            np.full(len(aperture) - 1, TAG_APERTURE, dtype=np.int32),
        ]
    )
    return MeridianMesh.from_polyline(points, tags)


def _straight_channel_mesh(
    radius: float,
    depth: float,
    *,
    rings: int,
    sectors: int,
) -> LoadedMesh:
    top_vertices, top_triangles = _triangulated_disc(
        radius,
        rings=rings,
        sectors=sectors,
        z=0.0,
        normal_sign=1,
    )
    bottom_vertices, bottom_triangles = _triangulated_disc(
        radius,
        rings=rings,
        sectors=sectors,
        z=-depth,
        normal_sign=-1,
    )

    vertices = np.vstack([top_vertices, bottom_vertices])
    bottom_offset = top_vertices.shape[0]
    triangles = [*top_triangles.tolist()]
    tags = [TAG_APERTURE] * top_triangles.shape[0]
    triangles.extend((bottom_triangles + bottom_offset).tolist())
    tags.extend([TAG_THROAT] * bottom_triangles.shape[0])

    top_outer_start = 1 + (rings - 1) * sectors
    bottom_outer_start = bottom_offset + top_outer_start
    for sector in range(sectors):
        nxt = (sector + 1) % sectors
        top0 = top_outer_start + sector
        top1 = top_outer_start + nxt
        bottom0 = bottom_outer_start + sector
        bottom1 = bottom_outer_start + nxt
        triangles.append([bottom0, bottom1, top1])
        triangles.append([bottom0, top1, top0])
        tags.extend([TAG_WALL, TAG_WALL])

    triangles_arr = np.asarray(triangles, dtype=np.int32)
    tags_arr = np.asarray(tags, dtype=np.int32)
    grid = make_pure_grid(vertices, triangles_arr)
    return LoadedMesh(
        grid=grid,
        physical_tags=tags_arr,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles_arr.shape[0],
            physical_groups={
                TAG_THROAT: "throat",
                TAG_WALL: "wall",
                TAG_APERTURE: "aperture",
            },
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
    )


def _z_axis_frame(depth: float) -> ObservationFrame:
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        origin=origin,
        u=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        v=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        mouth_center=origin,
        source_center=np.array([0.0, 0.0, -depth], dtype=np.float64),
    )


def test_uniform_full3d_rayleigh_disc_matches_baffled_piston_analytic():
    radius = 0.05
    ka = 3.0
    k = ka / radius
    distance = 5.0
    angles_deg = np.linspace(0.0, 70.0, 29)
    vertices, triangles = _triangulated_disc(radius, rings=20, sectors=128)
    points = np.column_stack(
        [
            distance * np.sin(np.deg2rad(angles_deg)),
            np.zeros_like(angles_deg),
            distance * np.cos(np.deg2rad(angles_deg)),
        ]
    )

    pressure = _rayleigh_pressure_uniform_disc(vertices, triangles, points, k)
    on_axis_expected = (
        np.exp(1j * k * np.hypot(distance, radius)) - np.exp(1j * k * distance)
    ) / (1j * k)
    relative_on_axis_error = abs(pressure[0] - on_axis_expected) / abs(
        on_axis_expected
    )

    directivity_db = 20.0 * np.log10(np.maximum(np.abs(pressure / pressure[0]), 1e-12))
    airy_db = 20.0 * np.log10(np.maximum(_airy_directivity(ka, angles_deg), 1e-12))

    assert relative_on_axis_error < 8.0e-4
    assert np.max(np.abs(directivity_db - airy_db)) < 0.03
    assert abs(
        _first_crossing_deg(angles_deg, directivity_db, -6.0)
        - _first_crossing_deg(angles_deg, airy_db, -6.0)
    ) < 0.3


def test_native_coupled_ib_straight_circular_channel_matches_circsym(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    radius = 0.04
    depth = 0.003
    frequencies_hz = np.array([800.0, 1600.0], dtype=np.float64)
    observation = ObservationConfig(
        distance_m=1.5,
        angle_min_deg=0.0,
        angle_max_deg=90.0,
        angle_count=10,
        planes=["horizontal"],
        origin="mouth",
    )
    native_config = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        aperture_tag=TAG_APERTURE,
        observation=observation,
        metal_native_assembly_mode="corrected",
        dense_solve_dtype="float64",
    )
    circsym_config = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        circsym_aperture_tag=TAG_APERTURE,
        observation=observation,
    )

    native_result = metal_bem.solve_frequencies(
        _straight_channel_mesh(radius, depth, rings=5, sectors=32),
        frequencies_hz,
        native_config,
    )
    circsym_result = metal_bem.solve_circsym_frequencies(
        _straight_channel_meridian(radius, depth, target_edge=radius / 5.0),
        frequencies_hz,
        circsym_config,
    )

    native_directivity = native_result.directivity_db[:, 0, :]
    circsym_directivity = circsym_result.directivity_db[:, 0, :]
    max_error_db = float(
        np.max(np.abs(native_directivity[:, :-1] - circsym_directivity[:, :-1]))
    )

    assert all(entry.get("coupled_ib") is True for entry in native_result.native_diagnostics)
    assert all(
        entry.get("aperture_velocity_basis") == "DP0"
        for entry in native_result.native_diagnostics
    )
    assert native_directivity[:, -1].min() > -20.0
    assert max_error_db < 1.0

"""Sphere-grid (balloon) observation solves.

The frame-relative sphere grid must stay aligned with the polar arcs: a grid
point at (theta=90, phi=0) is by construction the same physical location as
the horizontal arc's 90-degree point, so both must return the same pressure
from the same solved system. CircSym gets the same parity check plus a
pulsating-sphere physics check (p ~ e^{ikd}/d about the sphere centre).
"""
from __future__ import annotations

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem._constants import SPEED_OF_SOUND
from hornlab_metal_bem.mesh import LoadedMesh, make_pure_grid
from hornlab_metal_bem.result import MeshInfo


def _require_native():
    from hornlab_metal_bem.metal import discover_native_runtime

    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )


def _octasphere(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Unit sphere from a subdivided octahedron (outward-oriented triangles)."""
    vertices = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    triangles = [
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
    ]
    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            cached = midpoint_cache.get(key)
            if cached is not None:
                return cached
            va = vertices[a]
            vb = vertices[b]
            mid = ((va[0] + vb[0]) / 2, (va[1] + vb[1]) / 2, (va[2] + vb[2]) / 2)
            norm = (mid[0] ** 2 + mid[1] ** 2 + mid[2] ** 2) ** 0.5
            vertices.append((mid[0] / norm, mid[1] / norm, mid[2] / norm))
            midpoint_cache[key] = len(vertices) - 1
            return midpoint_cache[key]

        next_triangles = []
        for a, b, c in triangles:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            next_triangles.extend(
                [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
            )
        triangles = next_triangles
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles, dtype=np.int32),
    )


def _capped_sphere_mesh() -> LoadedMesh:
    """Unit sphere with a driven cap at +z (tag 2), rigid elsewhere (tag 1)."""
    vertices, triangles = _octasphere(2)
    centroids = vertices[triangles].mean(axis=1)
    tags = np.ones(triangles.shape[0], dtype=np.int32)
    tags[centroids[:, 2] > 0.55] = 2
    assert np.count_nonzero(tags == 2) > 0
    bbox = (vertices.min(axis=0), vertices.max(axis=0))
    return LoadedMesh(
        grid=make_pure_grid(vertices, triangles),
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles.shape[0],
            physical_groups={1: "rigid", 2: "cap"},
            bounding_box_m=bbox,
        ),
    )


def _grid_index(n_phi: int, theta_index: int, phi_index: int) -> int:
    return theta_index * n_phi + phi_index


@pytest.mark.slow
def test_native_sphere_grid_matches_coincident_arc_points():
    _require_native()
    mesh = _capped_sphere_mesh()

    # 13 arc angles over 0..180 puts samples at 0 and 90 deg; the (7, 12) grid
    # has theta rows every 30 deg and phi columns every 30 deg, so grid
    # (theta=90, phi=0) coincides with the horizontal arc's 90-degree point
    # and (theta=90, phi=90) with the vertical arc's.
    observation = metal_bem.ObservationConfig(
        planes=["horizontal", "vertical"],
        angle_min_deg=0.0,
        angle_max_deg=180.0,
        angle_count=13,
        distance_m=2.0,
        sphere_grid=(7, 12),
    )
    config = metal_bem.native_config(observation=observation)
    result = metal_bem.solve_frequencies(mesh, [320.0, 640.0], config)

    n_points = 7 * 12
    assert result.sphere_pressure_complex is not None
    assert result.sphere_pressure_complex.shape == (2, n_points)
    assert result.sphere_points.shape == (n_points, 3)
    assert result.sphere_theta_deg.shape == (n_points,)
    assert result.sphere_phi_deg.shape == (n_points,)
    assert np.all(np.isfinite(result.sphere_pressure_complex.view(np.float64)))

    angles = np.asarray(result.observation_angles_deg)
    arc_90 = int(np.argmin(np.abs(angles - 90.0)))
    arc_0 = int(np.argmin(np.abs(angles)))
    h_plane = result.observation_planes.index("horizontal")
    v_plane = result.observation_planes.index("vertical")

    sphere = result.sphere_pressure_complex
    idx_h90 = _grid_index(12, 3, 0)
    idx_v90 = _grid_index(12, 3, 3)

    assert result.sphere_theta_deg[idx_h90] == pytest.approx(90.0)
    assert result.sphere_phi_deg[idx_h90] == pytest.approx(0.0)
    assert result.sphere_phi_deg[idx_v90] == pytest.approx(90.0)

    # Same coordinates, same solved system, same field kernel: near-exact.
    np.testing.assert_allclose(
        sphere[:, idx_h90],
        result.pressure_complex[:, h_plane, arc_90],
        rtol=5e-4,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        sphere[:, idx_v90],
        result.pressure_complex[:, v_plane, arc_90],
        rtol=5e-4,
        atol=1e-9,
    )
    # Every phi column of the theta=0 row is the same pole point.
    pole = sphere[:, :12]
    np.testing.assert_allclose(
        pole,
        np.repeat(
            result.pressure_complex[:, h_plane, arc_0][:, None], 12, axis=1
        ),
        rtol=5e-4,
        atol=1e-9,
    )

    # Grid points actually sit where the arcs sit.
    obs_points = np.asarray(result.observation_points)
    np.testing.assert_allclose(
        result.sphere_points[idx_h90], obs_points[h_plane, arc_90], atol=1e-9
    )


def test_circsym_sphere_grid_matches_arcs_and_point_source_decay():
    meridian = metal_bem.MeridianMesh.from_polyline(
        np.column_stack(
            [
                0.1 * np.sin(np.linspace(0.0, np.pi, 49)),
                0.1 * np.cos(np.linspace(0.0, np.pi, 49)),
            ]
        ),
        tags=2,
    )
    observation = metal_bem.ObservationConfig(
        planes=["horizontal", "vertical"],
        angle_min_deg=0.0,
        angle_max_deg=180.0,
        angle_count=13,
        distance_m=2.0,
        sphere_grid=(7, 12),
    )
    config = metal_bem.native_config(observation=observation)
    frequency = 1200.0
    result = metal_bem.solve_circsym_frequencies(meridian, [frequency], config)

    n_points = 7 * 12
    assert result.sphere_pressure_complex is not None
    assert result.sphere_pressure_complex.shape == (1, n_points)

    angles = np.asarray(result.observation_angles_deg)
    arc_90 = int(np.argmin(np.abs(angles - 90.0)))
    h_plane = result.observation_planes.index("horizontal")
    sphere = result.sphere_pressure_complex
    idx_h90 = _grid_index(12, 3, 0)
    np.testing.assert_allclose(
        sphere[0, idx_h90],
        result.pressure_complex[0, h_plane, arc_90],
        rtol=1e-6,
    )

    # Pulsating sphere: p ~ e^{+ikd}/d about the sphere centre (origin), so
    # normalizing out the propagation term must leave a constant.
    k = 2.0 * np.pi * frequency / SPEED_OF_SOUND
    d = np.linalg.norm(np.asarray(result.sphere_points), axis=1)
    normalized = sphere[0] * d * np.exp(-1j * k * d)
    magnitudes = np.abs(normalized)
    assert magnitudes.max() / magnitudes.min() == pytest.approx(1.0, abs=0.02)

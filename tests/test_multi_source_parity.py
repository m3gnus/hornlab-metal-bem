"""Multi-source (multi-RHS) solves must match sequential single-source solves.

The native helper assembles and factors each frequency's operator once and
back-substitutes one RHS per source; these tests pin that result to N
sequential ``solve_frequencies`` calls at float32 tolerance on a smoke mesh:
per-source ``pressure_complex``, ``surface_pressure_avg`` (including
cross-source zero-velocity tags), and ``impedance``.
"""
from __future__ import annotations

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
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


def _two_cap_sphere_mesh() -> LoadedMesh:
    """Unit sphere with a driven cap at +z (tag 2), -z (tag 3), rigid rest."""
    vertices, triangles = _octasphere(2)
    centroids = vertices[triangles].mean(axis=1)
    tags = np.ones(triangles.shape[0], dtype=np.int32)
    tags[centroids[:, 2] > 0.55] = 2
    tags[centroids[:, 2] < -0.55] = 3
    assert np.count_nonzero(tags == 2) > 0
    assert np.count_nonzero(tags == 3) > 0
    bbox = (vertices.min(axis=0), vertices.max(axis=0))
    return LoadedMesh(
        grid=make_pure_grid(vertices, triangles),
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles.shape[0],
            physical_groups={1: "rigid", 2: "cap_top", 3: "cap_bottom"},
            bounding_box_m=bbox,
        ),
    )


def _observation_config() -> metal_bem.ObservationConfig:
    points = np.array(
        [[0.0, 0.0, 2.2], [0.6, 0.0, 2.1], [0.0, -0.4, 2.4]],
        dtype=np.float64,
    )
    return metal_bem.ObservationConfig(
        planes=["probe"],
        angle_count=3,
        custom_points={"probe": points},
    )


# Both sources list BOTH tags so the sequential runs record the union
# surface_pressure_avg (the cross-term data a radiation matrix needs) and
# infer identical observation frames; the zero-velocity tag is undriven.
_SOURCES = [
    {2: 1.0, 3: 0.0},
    {3: 1.0, 2: 0.0},
]
_FREQUENCIES = [180.0, 240.0, 320.0]


def _assert_results_match(multi, sequential):
    np.testing.assert_allclose(
        multi.pressure_complex,
        sequential.pressure_complex,
        rtol=2.0e-4,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        multi.impedance,
        sequential.impedance,
        rtol=2.0e-4,
        atol=1.0e-6,
    )
    assert multi.surface_pressure_avg is not None
    assert sequential.surface_pressure_avg is not None
    assert set(multi.surface_pressure_avg) == set(sequential.surface_pressure_avg)
    for tag, values in sequential.surface_pressure_avg.items():
        np.testing.assert_allclose(
            multi.surface_pressure_avg[tag],
            values,
            rtol=2.0e-4,
            atol=1.0e-6,
        )
    np.testing.assert_allclose(
        multi.surface_pressure_complex,
        sequential.surface_pressure_complex,
        rtol=2.0e-4,
        atol=1.0e-6,
    )


def _configs(**overrides) -> metal_bem.SolveConfig:
    return metal_bem.native_config(
        observation=_observation_config(),
        return_surface_pressure=True,
        **overrides,
    )


@pytest.mark.slow
def test_multi_source_matches_sequential_solves():
    _require_native()
    mesh = _two_cap_sphere_mesh()
    config = _configs()

    multi_results = metal_bem.solve_multi_source(
        mesh,
        _SOURCES,
        config,
        frequencies_hz=_FREQUENCIES,
    )
    assert len(multi_results) == len(_SOURCES)

    for source, multi_result in zip(_SOURCES, multi_results):
        sequential = metal_bem.solve_frequencies(
            mesh,
            _FREQUENCIES,
            metal_bem.native_config(
                observation=_observation_config(),
                return_surface_pressure=True,
                velocity_sources=dict(source),
            ),
        )
        _assert_results_match(multi_result, sequential)
        # Multi-source diagnostics still acknowledge the shared factorization.
        assert multi_result.native_diagnostics
    # The shared-cost attribution: only source 0 carries assembly seconds.
    assert multi_results[0].timings["assembly_s"] > 0.0
    assert multi_results[1].timings["assembly_s"] == 0.0


@pytest.mark.slow
def test_multi_source_matches_sequential_float64_and_chief():
    _require_native()
    mesh = _two_cap_sphere_mesh()
    chief_points = np.array(
        [[0.15, 0.1, 0.0], [-0.2, 0.05, 0.1], [0.0, -0.15, -0.2]],
        dtype=np.float64,
    )

    multi_results = metal_bem.solve_multi_source(
        mesh,
        _SOURCES,
        _configs(dense_solve_dtype="float64", chief_points=chief_points),
        frequencies_hz=_FREQUENCIES,
    )

    for source, multi_result in zip(_SOURCES, multi_results):
        sequential = metal_bem.solve_frequencies(
            mesh,
            _FREQUENCIES,
            _configs(
                dense_solve_dtype="float64",
                chief_points=chief_points,
                velocity_sources=dict(source),
            ),
        )
        _assert_results_match(multi_result, sequential)
        for entry in multi_result.native_diagnostics:
            assert entry["dense_solve_dtype"] == "float64"
            assert entry["chief_points"] is True


def test_solve_multi_source_rejects_empty_and_velocity_source_callback():
    mesh = _two_cap_sphere_mesh()
    with pytest.raises(ValueError, match="at least one velocity dict"):
        metal_bem.solve_multi_source(mesh, [], _configs())

    from hornlab_metal_bem.sweep import run_sweep_native_metal_multi_source

    config = _configs(velocity_source_callback=lambda f: {2: 1.0})
    with pytest.raises(ValueError, match="velocity_source_callback"):
        run_sweep_native_metal_multi_source(
            mesh, np.array([100.0]), object(), config, [{2: 1.0}, {3: 1.0}]
        )

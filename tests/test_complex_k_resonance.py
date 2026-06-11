from __future__ import annotations

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem.mesh import LoadedMesh as MetalLoadedMesh
from hornlab_metal_bem.mesh import make_pure_grid
from hornlab_metal_bem.metal import discover_native_runtime
from hornlab_metal_bem.result import MeshInfo as MetalMeshInfo


def _unit_sphere_mesh() -> MetalLoadedMesh:
    vertices: list[np.ndarray] = []
    vertex_lookup: dict[tuple[float, float, float], int] = {}

    def add_vertex(point: np.ndarray) -> int:
        projected = np.asarray(point, dtype=np.float64)
        projected = projected / np.linalg.norm(projected)
        key = tuple(np.round(projected, 12))
        index = vertex_lookup.get(key)
        if index is None:
            index = len(vertices)
            vertex_lookup[key] = index
            vertices.append(projected)
        return index

    for point in (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ):
        add_vertex(np.array(point, dtype=np.float64))

    triangles = np.array(
        [
            [4, 0, 2],
            [4, 2, 1],
            [4, 1, 3],
            [4, 3, 0],
            [5, 2, 0],
            [5, 1, 2],
            [5, 3, 1],
            [5, 0, 3],
        ],
        dtype=np.int32,
    )

    for _ in range(2):
        refined: list[list[int]] = []
        for tri in triangles:
            a, b, c = (int(value) for value in tri)
            ab = add_vertex((vertices[a] + vertices[b]) * 0.5)
            bc = add_vertex((vertices[b] + vertices[c]) * 0.5)
            ca = add_vertex((vertices[c] + vertices[a]) * 0.5)
            refined.extend(
                [
                    [a, ab, ca],
                    [ab, b, bc],
                    [ca, bc, c],
                    [ab, bc, ca],
                ]
            )
        triangles = np.asarray(refined, dtype=np.int32)

    vertices_nx3 = np.asarray(vertices, dtype=np.float64)
    oriented = triangles.copy()
    for index, tri in enumerate(oriented):
        v0, v1, v2 = vertices_nx3[tri]
        centroid = (v0 + v1 + v2) / 3.0
        if np.dot(np.cross(v1 - v0, v2 - v0), centroid) < 0.0:
            oriented[index, [1, 2]] = oriented[index, [2, 1]]

    assert oriented.shape == (128, 3)
    centroids = vertices_nx3[oriented].mean(axis=1)
    tags = np.ones(oriented.shape[0], dtype=np.int32)
    tags[centroids[:, 2] > 0.75] = 2
    bbox = (vertices_nx3.min(axis=0), vertices_nx3.max(axis=0))
    return MetalLoadedMesh(
        grid=make_pure_grid(vertices_nx3, oriented),
        physical_tags=tags,
        info=MetalMeshInfo(
            n_vertices=vertices_nx3.shape[0],
            n_triangles=oriented.shape[0],
            physical_groups={1: "1", 2: "2"},
            bounding_box_m=bbox,
        ),
    )


@pytest.mark.slow
def test_complex_k_suppresses_unit_sphere_resonance_rcond():
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    mesh = _unit_sphere_mesh()
    observation = metal_bem.ObservationConfig(
        planes=["probe"],
        angle_count=2,
        custom_points={
            "probe": np.array(
                [
                    [0.0, 0.0, 2.2],
                    [0.6, 0.0, 2.1],
                ],
                dtype=np.float64,
            )
        },
    )
    frequencies = np.linspace(170.0, 185.0, 16)
    standard_config = metal_bem.native_config(
        velocity_sources={2: 1.0},
        observation=observation,
    )

    standard_result = metal_bem.solve_frequencies(mesh, frequencies, standard_config)
    standard_rcond = np.array(
        [
            float(diagnostics["dense_solve_rcond"])
            for diagnostics in standard_result.native_diagnostics
        ],
        dtype=np.float64,
    )

    assert np.all(np.isfinite(standard_rcond))
    min_standard = float(np.min(standard_rcond))
    assert min_standard < 0.5 * float(np.median(standard_rcond))

    resonance_frequency = float(frequencies[int(np.argmin(standard_rcond))])
    complex_config = metal_bem.native_config(
        formulation="complex_k",
        complex_k_shift=0.02,
        velocity_sources={2: 1.0},
        observation=observation,
    )
    complex_result = metal_bem.solve_frequencies(
        mesh,
        [resonance_frequency],
        complex_config,
    )
    complex_diagnostics = complex_result.native_diagnostics[0]
    complex_rcond = float(complex_diagnostics["dense_solve_rcond"])

    assert complex_rcond >= 4.0 * min_standard
    assert complex_diagnostics["complex_k"] is True

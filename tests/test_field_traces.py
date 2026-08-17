from __future__ import annotations

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem._constants import SPEED_OF_SOUND
from hornlab_metal_bem.field_traces import _total_neumann_from_surface_pressure
from hornlab_metal_bem.mesh import LoadedMesh, make_pure_grid
from hornlab_metal_bem.result import MeshInfo
from tests.test_multi_source_parity import _octasphere


_FREQUENCIES = np.array([160.0, 260.0, 420.0], dtype=np.float64)
_POINTS = np.array(
    [
        [0.0, 0.0, 2.2],
        [0.45, 0.2, 2.1],
        [-0.5, 0.3, 2.3],
        [0.2, -0.6, 2.2],
    ],
    dtype=np.float64,
)


def _require_native() -> None:
    from hornlab_metal_bem.metal import discover_native_runtime

    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )


def _compact_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(triangles)
    remap = np.full(vertices.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return vertices[used], remap[triangles]


def _sphere_mesh(
    symmetry_plane: str | None,
    *,
    robin: bool = False,
) -> LoadedMesh:
    vertices, triangles = _octasphere(1)
    keep = np.ones(triangles.shape[0], dtype=bool)
    triangle_vertices = vertices[triangles]
    if symmetry_plane in {"yz", "yz+xz"}:
        keep &= np.all(triangle_vertices[:, :, 0] >= -1.0e-12, axis=1)
    if symmetry_plane == "yz+xz":
        keep &= np.all(triangle_vertices[:, :, 1] >= -1.0e-12, axis=1)
    triangles = triangles[keep]
    centroids = vertices[triangles].mean(axis=1)

    tags = np.full(triangles.shape[0], 8 if robin else 1, dtype=np.int32)
    tags[centroids[:, 2] > 0.45] = 2
    if not robin:
        tags[centroids[:, 2] < -0.45] = 3
    vertices, triangles = _compact_mesh(vertices, triangles)
    return LoadedMesh(
        grid=make_pure_grid(vertices, triangles),
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles.shape[0],
            physical_groups=(
                {2: "source", 8: "robin"}
                if robin
                else {1: "rigid", 2: "top", 3: "bottom"}
            ),
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
    )


def _observation() -> metal_bem.ObservationConfig:
    return metal_bem.ObservationConfig(
        planes=["probe"],
        angle_count=_POINTS.shape[0],
        custom_points={"probe": _POINTS},
    )


def _k_real(frequency_hz: float) -> float:
    return float(np.float32(2.0 * np.pi * frequency_hz / SPEED_OF_SOUND))


def _evaluate_result(
    mesh: LoadedMesh,
    result: metal_bem.SolveResult,
    symmetry_plane: str | None,
) -> np.ndarray:
    assert result.surface_pressure_complex is not None
    assert result.surface_neumann_complex is not None
    rows = []
    for index, frequency_hz in enumerate(result.frequencies_hz):
        rows.append(
            metal_bem.evaluate_exterior_from_traces(
                mesh,
                float(frequency_hz),
                _k_real(float(frequency_hz)),
                result.surface_pressure_complex[index],
                result.surface_neumann_complex[index],
                result.observation_points.reshape(-1, 3),
                symmetry_plane=symmetry_plane,
            ).reshape(result.pressure_complex.shape[1:])
        )
    return np.stack(rows, axis=0)


def _assert_trace_parity(actual: np.ndarray, expected: np.ndarray) -> None:
    atol = 1.0e-8 * float(np.max(np.abs(expected)))
    np.testing.assert_allclose(actual, expected, rtol=2.0e-3, atol=atol)


def test_total_neumann_reconstruction_adds_only_robin_face_correction():
    driver = np.array(
        [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]],
        dtype=np.complex64,
    )
    pressure = np.array(
        [
            [1.0 + 1.0j, 2.0 + 2.0j, 4.0 + 4.0j, 8.0 + 8.0j],
            [2.0 - 1.0j, 3.0 - 2.0j, 5.0 - 4.0j, 9.0 - 8.0j],
        ],
        dtype=np.complex128,
    )
    local2global = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    tags = np.array([1, 8], dtype=np.int32)
    k_real = np.array([2.0, 3.0], dtype=np.float32)
    k_imag = np.array([0.1, 0.2], dtype=np.float32)
    betas = [{8: 0.25 + 0.5j}, {8: 0.5 - 0.25j}]

    total = _total_neumann_from_surface_pressure(
        driver,
        pressure,
        local2global,
        tags,
        k_real,
        k_imag,
        betas,
    )

    np.testing.assert_array_equal(total[:, 0], driver[:, 0])
    expected_robin = []
    pressure_f32 = pressure.astype(np.complex64)
    for index in range(2):
        p_avg = (
            pressure_f32[index, 1] + pressure_f32[index, 2] + pressure_f32[index, 3]
        ) / np.float32(3.0)
        i_k = np.complex64(complex(-float(k_imag[index]), float(k_real[index])))
        expected_robin.append(
            driver[index, 1] + i_k * np.complex64(betas[index][8]) * p_avg
        )
    np.testing.assert_array_equal(
        total[:, 1], np.asarray(expected_robin, dtype=np.complex128)
    )


@pytest.mark.slow
@pytest.mark.parametrize("symmetry_plane", [None, "yz", "yz+xz"])
def test_retained_traces_reproduce_solve_observation_pressure(symmetry_plane):
    _require_native()
    mesh = _sphere_mesh(symmetry_plane)
    common = dict(
        observation=_observation(),
        return_surface_traces=True,
        native_symmetry_plane=symmetry_plane,
    )
    if symmetry_plane is None:
        results = metal_bem.solve_multi_source(
            mesh,
            [{2: 1.0, 3: 0.0}, {3: 1.0, 2: 0.0}],
            metal_bem.native_config(**common),
            frequencies_hz=_FREQUENCIES,
        )
    else:
        results = [
            metal_bem.solve_frequencies(
                mesh,
                _FREQUENCIES,
                metal_bem.native_config(velocity_sources={2: 1.0}, **common),
            )
        ]

    for result in results:
        assert result.config.return_surface_pressure is False
        assert result.surface_pressure_complex is not None
        assert result.surface_neumann_complex is not None
        assert result.surface_pressure_complex.shape == (
            _FREQUENCIES.size,
            mesh.grid.vertices.shape[1],
        )
        assert result.surface_neumann_complex.shape == (
            _FREQUENCIES.size,
            mesh.grid.elements.shape[1],
        )
        reevaluated = _evaluate_result(mesh, result, symmetry_plane)
        _assert_trace_parity(reevaluated, result.pressure_complex)


@pytest.mark.slow
def test_robin_total_neumann_passes_but_driver_only_fails_parity():
    _require_native()
    mesh = _sphere_mesh(None, robin=True)
    result = metal_bem.solve_frequencies(
        mesh,
        _FREQUENCIES,
        metal_bem.native_config(
            velocity_sources={2: 1.0},
            impedance_sources={8: 0.35 + 0.08j},
            observation=_observation(),
            return_surface_traces=True,
        ),
    )

    total_field = _evaluate_result(mesh, result, None)
    _assert_trace_parity(total_field, result.pressure_complex)

    assert result.surface_neumann_complex is not None
    driver_only = result.surface_neumann_complex.copy()
    driver_only[:, mesh.physical_tags == 8] = 0.0
    driver_field = []
    for index, frequency_hz in enumerate(result.frequencies_hz):
        driver_field.append(
            metal_bem.evaluate_exterior_from_traces(
                mesh,
                float(frequency_hz),
                _k_real(float(frequency_hz)),
                result.surface_pressure_complex[index],
                driver_only[index],
                result.observation_points.reshape(-1, 3),
            ).reshape(result.pressure_complex.shape[1:])
        )
    driver_field_array = np.stack(driver_field, axis=0)
    relative_error = np.linalg.norm(
        driver_field_array - result.pressure_complex
    ) / np.linalg.norm(result.pressure_complex)
    assert relative_error > 1.0e-2

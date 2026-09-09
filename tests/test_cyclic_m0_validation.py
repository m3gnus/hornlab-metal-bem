from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_metal_bem.metal.geometry import build_metal_geometry_buffers
from hornlab_metal_bem.metal.native import (
    MetalNativeStandardSession,
    discover_native_runtime,
)
from hornlab_metal_bem.validation.cyclic_m0 import (
    expand_cyclic_m0_pressure,
    expand_triangle_orbit_values,
    load_and_reduce_native_cyclic_m0,
    reduce_cyclic_m0_representative_rows,
    revolve_meridian_p1,
)
from hornlab_metal_bem.validation.native_symmetry import orbit_reduce_matrix_rhs


def _cylinder(*, sectors: int = 6):
    return revolve_meridian_p1(
        np.array(
            [
                [0.0, 0.0],
                [0.4, 0.0],
                [0.4, 0.7],
                [0.0, 0.7],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        sectors=sectors,
        segment_tags=np.array([11, 12, 13, 14], dtype=np.int32),
    )


def test_revolve_meridian_builds_closed_cyclic_p1_mesh_and_orbits():
    mesh = _cylinder(sectors=6)

    assert mesh.vertices_nx3.shape == (14, 3)
    assert mesh.triangles_nx3.shape == (24, 3)
    assert [len(orbit) for orbit in mesh.vertex_orbits] == [1, 6, 6, 1]
    assert [len(orbit) for orbit in mesh.triangle_orbits] == [6, 6, 6, 6]
    assert mesh.triangle_orbit_segments.tolist() == [0, 1, 1, 2]
    assert set(mesh.physical_tags.tolist()) == {11, 12, 13}

    edges = np.sort(
        np.concatenate(
            [
                mesh.triangles_nx3[:, [0, 1]],
                mesh.triangles_nx3[:, [1, 2]],
                mesh.triangles_nx3[:, [2, 0]],
            ],
            axis=0,
        ),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)

    points = mesh.vertices_nx3[mesh.triangles_nx3]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    centroids = np.mean(points, axis=1)
    # Radial side normals and axial cap normals all point out of the cylinder.
    side = mesh.physical_tags == 12
    bottom = mesh.physical_tags == 11
    top = mesh.physical_tags == 13
    assert np.all(np.sum(normals[side, :2] * centroids[side, :2], axis=1) > 0.0)
    assert np.all(normals[bottom, 2] < 0.0)
    assert np.all(normals[top, 2] > 0.0)


def test_revolve_meridian_trims_nearly_repeated_closing_point():
    exact = _cylinder(sectors=6)
    near_closed = revolve_meridian_p1(
        np.array(
            [
                [0.0, 0.0],
                [0.4, 0.0],
                [0.4, 0.7],
                [0.0, 0.7],
                [0.0, 1.0e-9],
            ],
            dtype=np.float64,
        ),
        sectors=6,
        segment_tags=np.array([11, 12, 13, 14], dtype=np.int32),
    )

    assert np.array_equal(near_closed.triangles_nx3, exact.triangles_nx3)
    assert np.array_equal(near_closed.physical_tags, exact.physical_tags)


def test_representative_row_reduction_is_exact_for_cyclic_system():
    sectors = 5
    rings = 3
    orbits = tuple(
        np.arange(ring * sectors, (ring + 1) * sectors, dtype=np.int64)
        for ring in range(rings)
    )
    matrix = np.empty((rings * sectors, rings * sectors), dtype=np.complex128)
    for row_ring in range(rings):
        for row_sector in range(sectors):
            row = row_ring * sectors + row_sector
            for col_ring in range(rings):
                for col_sector in range(sectors):
                    col = col_ring * sectors + col_sector
                    delta = (col_sector - row_sector) % sectors
                    matrix[row, col] = (
                        5.0 * (row_ring == col_ring and delta == 0)
                        + 0.1 * (1 + row_ring + col_ring)
                        + np.exp(2j * np.pi * delta / sectors) / (2.0 + delta)
                    )
    rhs = np.repeat(
        np.array([1.0 + 0.5j, -0.25 + 0.1j, 0.4 - 0.2j]), sectors
    )

    reduced = reduce_cyclic_m0_representative_rows(matrix, rhs, orbits)
    reduced_pressure = np.linalg.solve(reduced.matrix, reduced.rhs)
    expanded = expand_cyclic_m0_pressure(reduced_pressure, matrix.shape[0], orbits)

    assert np.linalg.norm(matrix @ expanded - rhs) / np.linalg.norm(rhs) < 1.0e-13
    full_pressure = np.linalg.solve(matrix, rhs)
    assert np.allclose(expanded, full_pressure, rtol=1.0e-12, atol=1.0e-12)

    galerkin_matrix, galerkin_rhs = orbit_reduce_matrix_rhs(matrix, rhs, orbits)
    row_sizes = np.asarray([len(orbit) for orbit in orbits], dtype=np.float64)
    assert np.allclose(
        galerkin_matrix / row_sizes[:, None], reduced.matrix, rtol=2.0e-6, atol=2.0e-6
    )
    assert np.allclose(
        galerkin_rhs / row_sizes, reduced.rhs, rtol=2.0e-6, atol=2.0e-6
    )


def test_axisymmetric_triangle_values_expand_over_every_sector():
    mesh = _cylinder(sectors=7)
    values = np.array([1.0 + 0.0j, 2.0 + 0.5j, 3.0 - 0.5j, 4.0 + 0.0j])

    expanded = expand_triangle_orbit_values(
        values,
        mesh.triangles_nx3.shape[0],
        mesh.triangle_orbits,
    )

    for value, orbit in zip(values, mesh.triangle_orbits, strict=True):
        assert np.all(expanded[orbit] == value)


def test_native_full_cyclic_assembly_reduces_to_same_p1_solution(
    monkeypatch,
    tmp_path,
):
    """Prove the contract against the corrected native triangle assembler."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    mesh = _cylinder(sectors=6)
    grid = SimpleNamespace(
        vertices=mesh.vertices_nx3.T,
        elements=mesh.triangles_nx3.T,
        number_of_elements=mesh.triangles_nx3.shape[0],
    )
    p1 = SimpleNamespace(
        local2global=mesh.triangles_nx3,
        global_dof_count=mesh.vertices_nx3.shape[0],
    )
    buffers = build_metal_geometry_buffers(
        grid,
        mesh.physical_tags,
        p1,
        SimpleNamespace(global_dof_count=mesh.triangles_nx3.shape[0]),
    )
    source_by_orbit = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64)
    neumann = expand_triangle_orbit_values(
        source_by_orbit,
        mesh.triangles_nx3.shape[0],
        mesh.triangle_orbits,
    )
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv(
        "HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "entrywise"
    )
    with MetalNativeStandardSession.create_session(
        geometry_buffers=buffers,
        work_dir=tmp_path / "cyclic-full-session",
        session_id="cyclic-full-test",
    ) as session:
        assembly = session.assemble_standard_neumann(
            20_000.0,
            2.0 * np.pi * 20_000.0 / 343.0,
            neumann,
            operation_id="full-assembly",
        )

    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32, dtype="<f4"
    )
    reduced = load_and_reduce_native_cyclic_m0(
        assembly,
        mesh.vertex_orbits,
        invariance_rtol=8.0e-5,
        invariance_atol=8.0e-6,
    )
    reduced_pressure = np.linalg.solve(reduced.matrix, reduced.rhs)
    expanded_pressure = expand_cyclic_m0_pressure(
        reduced_pressure,
        matrix.shape[0],
        mesh.vertex_orbits,
    )
    full_pressure = np.linalg.solve(matrix, rhs)

    assert np.linalg.norm(matrix @ expanded_pressure - rhs) / np.linalg.norm(rhs) < 2e-4
    assert np.linalg.norm(expanded_pressure - full_pressure) / np.linalg.norm(
        full_pressure
    ) < 3e-4

"""Float32 native Burton–Miller parity and exterior resonance controls.

Frozen before comparing the native results: fused matrix/RHS against M1's
highest available quadrature order relative L2 < 5e-3 / 2e-3 (different
singular quadrature); exterior field relative error < 2e-3, sphere analytic
relative error < 0.14, symmetry field relative spread < 0.002, and sphere/face
power within 0.5 dB. The sphere ceiling includes the coarse mesh error.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

import hornlab_metal_bem as bem
from hornlab_metal_bem.burton_miller_reference import (
    assemble_burton_miller_reference,
    evaluate_burton_miller_exterior,
)
from hornlab_metal_bem.mesh import LoadedMesh, make_pure_grid
from hornlab_metal_bem.metal import MetalNativeStandardSession, discover_native_runtime
from hornlab_metal_bem.metal.geometry import build_metal_geometry_buffers
from hornlab_metal_bem.observation import ObservationFrame
from hornlab_metal_bem.result import MeshInfo
from tests.test_complex_k_resonance import _unit_sphere_mesh
from tests.test_metal_native import _near_quadrature_geometry_buffers, _tiny_geometry_buffers


def _require_native():
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip("Swift/Metal native helper unavailable")


def _complex_files(real: Path, imag: Path, shape):
    return (np.fromfile(real, dtype="<f4") + 1j*np.fromfile(imag, dtype="<f4")).reshape(shape)


def _sphere_frame():
    origin = np.zeros(3)
    return ObservationFrame(
        axis=np.array([0., 0., 1.]), origin=origin,
        u=np.array([1., 0., 0.]), v=np.array([0., 1., 0.]),
        mouth_center=origin, source_center=origin,
    )


def _sphere_observation():
    return bem.ObservationConfig(
        planes=["horizontal"], angle_count=1,
        distance_m=3.0, sphere_grid=(19, 36),
    )


def test_native_fused_blocks_and_real_k_field_match_m1(tmp_path):
    _require_native()
    buffers = _tiny_geometry_buffers()
    vertices = buffers.vertices_3xn_f32.T
    triangles = buffers.triangles_3xm_i32.T
    q = np.array([1+0j, 0.3+0.2j], np.complex64)
    k = 1.7
    expected = assemble_burton_miller_reference(
        vertices, triangles, q, k, quadrature_order=12,
    )
    with MetalNativeStandardSession.create_session(
        geometry_buffers=buffers,
        work_dir=tmp_path / "bm-session", session_id="bm-parity",
    ) as session:
        assembled = session.assemble_standard_neumann(
            100.0, k, q, formulation="burton_miller",
        )
        matrix = _complex_files(
            assembled.matrix_real_f32, assembled.matrix_imag_f32,
            assembled.matrix_shape,
        )
        rhs = _complex_files(
            assembled.rhs_real_f32, assembled.rhs_imag_f32,
            assembled.rhs_shape,
        )
        assert np.linalg.norm(matrix + expected.lhs)/np.linalg.norm(expected.lhs) < 5e-3
        assert np.linalg.norm(rhs + expected.rhs)/np.linalg.norm(expected.rhs) < 2e-3
        pressure = np.linalg.solve(matrix, rhs)
        points = np.array([[0.2, 0.3, 3.0], [0.7, -0.2, 3.2]])
        field_result = session.evaluate_standard_exterior(
            100.0, k, pressure, q, points.T,
        )
        field = _complex_files(
            field_result.pressure_real_f32, field_result.pressure_imag_f32,
            field_result.shape,
        )
    reference = evaluate_burton_miller_exterior(
        vertices, triangles, pressure, q, points, k,
    )
    assert np.linalg.norm(field-reference)/np.linalg.norm(reference) < 2e-3


@pytest.mark.parametrize("k", [0.5, 1.5, np.pi, 3.25])
def test_native_pulsating_sphere_analytic_and_power(k):
    _require_native()
    config = bem.native_config(
        formulation="burton_miller", velocity_sources={1: 1., 2: 1.},
        observation=_sphere_observation(), frame_override=_sphere_frame(),
        air_density=1., speed_of_sound=1.,
    )
    result = bem.solve_frequencies(_unit_sphere_mesh(), [k/(2*np.pi)], config)
    # Default source mode is unit acceleration: q=-1 at rho=1. The M1
    # pulsating-sphere formula assumes q=ik, so scale its exact field.
    exact = (1j*k*np.exp(2j*k)/(3*(1j*k-1))) * (-1/(1j*k))
    assert abs(result.pressure_complex[0, 0, 0]/exact-1) < 0.14
    face = result.radiated_power_surface_w[0]
    sphere = result.radiated_power_sphere_w[0]
    assert face > 0 and sphere > 0
    assert abs(10*np.log10(sphere/face)) < 0.5
    assert result.native_diagnostics[0]["assembly_mode"] == "burton_miller"


def test_native_standard_resonance_control_spikes_but_bm_stays_accurate():
    _require_native()
    mesh = _unit_sphere_mesh()
    k = 3.24
    common = dict(
        velocity_sources={1: 1., 2: 1.},
        observation=_sphere_observation(), frame_override=_sphere_frame(),
        air_density=1., speed_of_sound=1.,
    )
    bm = bem.solve_frequencies(mesh, [k/(2*np.pi)], bem.native_config(
        formulation="burton_miller", **common,
    )).pressure_complex[0, 0, 0]
    standard = bem.solve_frequencies(mesh, [k/(2*np.pi)], bem.native_config(
        formulation="standard", **common,
    )).pressure_complex[0, 0, 0]
    exact = (1j*k*np.exp(2j*k)/(3*(1j*k-1))) * (-1/(1j*k))
    assert abs(bm/exact-1) < 0.14
    assert abs(standard/exact-1) > 0.7


def test_native_oscillating_sphere_dipole_analytic(tmp_path):
    _require_native()
    mesh = _unit_sphere_mesh()
    triangles = mesh.grid.elements.T
    buffers = build_metal_geometry_buffers(
        mesh.grid, mesh.physical_tags,
        SimpleNamespace(local2global=triangles, global_dof_count=mesh.grid.vertices.shape[1]),
    )
    k = 1.5
    q = (1j*k*buffers.triangle_normals_3xm_f32[2]).astype(np.complex64)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=buffers, work_dir=tmp_path / "dipole", session_id="dipole",
    ) as session:
        assembled = session.assemble_standard_neumann(
            100., k, q, formulation="burton_miller",
        )
        matrix = _complex_files(assembled.matrix_real_f32, assembled.matrix_imag_f32,
                                assembled.matrix_shape)
        rhs = _complex_files(assembled.rhs_real_f32, assembled.rhs_imag_f32,
                             assembled.rhs_shape)
        pressure = np.linalg.solve(matrix, rhs)
        field_result = session.evaluate_standard_exterior(
            100., k, pressure, q, np.array([[0.0], [0.0], [3.0]]),
        )
        field = _complex_files(field_result.pressure_real_f32,
                               field_result.pressure_imag_f32, field_result.shape)[0]
    h = lambda z: spherical_jn(1, z) + 1j*spherical_yn(1, z)
    hp = (spherical_jn(1, k, derivative=True)
          + 1j*spherical_yn(1, k, derivative=True))
    exact = 1j*h(3*k)/hp
    assert abs(field/exact-1) < 0.15


def test_float32_near_limit_refuses_almost_coincident_disjoint_faces(tmp_path):
    _require_native()
    buffers = _near_quadrature_geometry_buffers()
    vertices = buffers.vertices_3xn_f32.copy()
    vertices[2, 3:] = 1e-4
    buffers = replace(buffers, vertices_3xn_f32=vertices)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=buffers, work_dir=tmp_path / "near-limit", session_id="near-limit",
    ) as session:
        with pytest.raises(RuntimeError, match="gap below float32 near limit"):
            session.assemble_standard_neumann(
                100., 1.5, np.ones(2, np.complex64), formulation="burton_miller",
            )


def test_native_full_half_quarter_bm_fields_agree():
    _require_native()
    original = _unit_sphere_mesh()
    vertices = original.grid.vertices.T
    triangles = original.grid.elements.T
    fields = []
    for mode in (None, "yz", "yz+xz"):
        centers = vertices[triangles].mean(axis=1)
        keep = np.ones(len(triangles), bool)
        if mode is not None:
            keep &= centers[:, 0] > 0
        if mode == "yz+xz":
            keep &= centers[:, 1] > 0
        tri = triangles[keep]
        used, inverse = np.unique(tri, return_inverse=True)
        verts = vertices[used]
        tri = inverse.reshape(-1, 3)
        mesh = LoadedMesh(
            grid=make_pure_grid(verts, tri),
            physical_tags=np.ones(len(tri), np.int32),
            info=MeshInfo(
                n_vertices=len(verts), n_triangles=len(tri),
                physical_groups={1: "1"},
                bounding_box_m=(verts.min(axis=0), verts.max(axis=0)),
            ),
        )
        observation = bem.ObservationConfig(
            planes=["probe"], angle_count=1,
            custom_points={"probe": np.array([[0.4, 0.3, 3.]])},
        )
        config = bem.native_config(
            formulation="burton_miller", velocity_sources={1: 1.},
            observation=observation, native_symmetry_plane=mode,
            native_check_open_edges=False, air_density=1., speed_of_sound=1.,
        )
        result = bem.solve_frequencies(mesh, [1.5/(2*np.pi)], config)
        fields.append(result.pressure_complex[0, 0, 0])
    assert max(abs(value/fields[0]-1) for value in fields[1:]) < 0.002

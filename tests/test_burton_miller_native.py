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
from tests.test_metal_native import (
    _geometry_buffers_from_mesh,
    _ib_box_geometry_buffers,
    _near_quadrature_geometry_buffers,
    _tiny_geometry_buffers,
)


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


def test_touching_disjoint_faces_are_refused_but_thin_gaps_are_not(tmp_path):
    _require_native()
    buffers = _near_quadrature_geometry_buffers()
    # 1 m faces 1e-7 m apart share no vertex: the surface touches itself.
    vertices = buffers.vertices_3xn_f32.copy()
    vertices[2, 3:] = 1e-7
    with MetalNativeStandardSession.create_session(
        geometry_buffers=replace(buffers, vertices_3xn_f32=vertices),
        work_dir=tmp_path / "touching", session_id="touching",
    ) as session:
        with pytest.raises(RuntimeError, match="touches, intersects or coincides"):
            session.assemble_standard_neumann(
                100., 1.5, np.ones(2, np.complex64), formulation="burton_miller",
            )
    # The same faces 0.1 mm apart were refused by the old float32 near limit.
    vertices[2, 3:] = 1e-4
    with MetalNativeStandardSession.create_session(
        geometry_buffers=replace(buffers, vertices_3xn_f32=vertices),
        work_dir=tmp_path / "thin", session_id="thin",
    ) as session:
        assembled = session.assemble_standard_neumann(
            100., 1.5, np.ones(2, np.complex64), formulation="burton_miller",
        )
        matrix = _complex_files(assembled.matrix_real_f32, assembled.matrix_imag_f32,
                                assembled.matrix_shape)
    assert np.all(np.isfinite(matrix))


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


# --- Near, singular and image blocks against independent float64 references.
#
# Fixtures (tests/fixtures) hold M1-convention operators (lhs = M/2 - D +
# (i/k)H, rhs = -S - (i/k)(K' + M10/2)); the native helper stores minus them.
# bm_thin_wall_reference.npz is the review's converged reference: M1 kernels,
# outer Gauss order 144, source fan graded radially at the gap (changes below
# 3e-5 from order 96). bm_lip_reference.npz comes from an independent adaptive
# reference (isotropic subdivision of both faces, graded fan only for a
# coplanar point), converged in its outer depth; it shares no quadrature with
# the helper. Tolerance: 5e-3 relative L2 (0.5 %), the numerical contract.

_FIXTURES = Path(__file__).parent / "fixtures"


def _native_operator(tmp_path, vertices, triangles, k, *, symmetry_plane=None,
                     name="operator", drives=None):
    """Native fused matrix and RHS operator (one column per unit DP0 drive)."""
    vertices = np.asarray(vertices, float)
    triangles = np.asarray(triangles)
    buffers = _geometry_buffers_from_mesh(
        vertices.T, triangles, np.ones(len(triangles), np.int32),
    )
    drives = np.eye(len(triangles), dtype=np.complex64) if drives is None else drives
    columns = []
    with MetalNativeStandardSession.create_session(
        geometry_buffers=buffers, work_dir=tmp_path / name, session_id=name,
        symmetry_plane=symmetry_plane, check_open_edges=False,
    ) as session:
        for q in drives:
            assembled = session.assemble_standard_neumann(
                100., k, np.asarray(q, np.complex64), formulation="burton_miller",
            )
            columns.append(_complex_files(
                assembled.rhs_real_f32, assembled.rhs_imag_f32, assembled.rhs_shape,
            ))
        matrix = _complex_files(
            assembled.matrix_real_f32, assembled.matrix_imag_f32, assembled.matrix_shape,
        )
    return matrix, np.array(columns).T


def _relative(value, reference):
    return np.linalg.norm(value - reference)/np.linalg.norm(reference)


@pytest.mark.parametrize("k", [1.7, 17.0])
def test_thin_wall_opposing_faces_match_converged_reference(tmp_path, k):
    # Review reproduction: opposing 100 mm right triangles 1 mm apart. The
    # fixed near rule gave 6.84 % RHS / 0.52 % matrix error at k = 1.7 and
    # 5.24 % / 3.98 % at k = 17.
    _require_native()
    reference = np.load(_FIXTURES / "bm_thin_wall_reference.npz")
    key = f"k{k:g}".replace(".", "_")
    matrix, rhs = _native_operator(
        tmp_path, reference["vertices"], reference["triangles"], k,
    )
    assert _relative(-matrix, reference[f"{key}_lhs"]) < 1e-3
    assert _relative(-rhs, reference[f"{key}_rhs"]) < 1e-3


@pytest.mark.parametrize("thickness_mm", [1, 2])
@pytest.mark.parametrize("k", [1.7, 17.0])
def test_thin_baffle_lip_matches_independent_reference(tmp_path, thickness_mm, k):
    # A closed 20 x 10 mm slab 1 or 2 mm thick: the two faces carry opposite
    # diagonals, so projections cross edges, and the rim is 10:1 (1 mm) or
    # 5:1 (2 mm) slivers touching both faces.
    _require_native()
    reference = np.load(_FIXTURES / "bm_lip_reference.npz")
    key = f"t{thickness_mm}mm_k{k:g}".replace(".", "_")
    matrix, rhs = _native_operator(
        tmp_path, reference[f"t{thickness_mm}mm_vertices"],
        reference[f"t{thickness_mm}mm_triangles"], k,
    )
    assert _relative(-matrix, reference[f"{key}_lhs"]) < 5e-3
    assert _relative(-rhs, reference[f"{key}_rhs"]) < 5e-3


def _closed_panel(size, gap):
    vertices = np.array([
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
        [0, 0, gap], [size, 0, gap], [size, size, gap], [0, size, gap],
    ], float)
    triangles = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    return vertices, triangles


def test_closed_thin_panel_is_solved_not_refused(tmp_path):
    # Review reproduction: a closed 1.2 m x 1.2 m x 1 mm panel at 25 Hz has
    # 8 elements per wavelength and was refused by the float32 near limit.
    _require_native()
    size, gap, frequency = 1.2, 0.001, 25.0
    vertices, triangles = _closed_panel(size, gap)
    buffers = _geometry_buffers_from_mesh(vertices.T, triangles, np.ones(12, np.int32))
    k = 2*np.pi*frequency/343.0
    points = np.array([[0.3, 0.4, 3.0], [0.6, 0.6, -2.0], [4.0, 0.5, 0.5]])
    with MetalNativeStandardSession.create_session(
        geometry_buffers=buffers, work_dir=tmp_path / "panel", session_id="panel",
    ) as session:
        q = np.ones(12, np.complex64)
        assembled = session.assemble_standard_neumann(
            frequency, k, q, formulation="burton_miller",
        )
        matrix = _complex_files(assembled.matrix_real_f32, assembled.matrix_imag_f32,
                                assembled.matrix_shape)
        rhs = _complex_files(assembled.rhs_real_f32, assembled.rhs_imag_f32,
                             assembled.rhs_shape)
        pressure = np.linalg.solve(matrix.astype(complex), rhs.astype(complex))
        evaluated = session.evaluate_standard_exterior(frequency, k, pressure, q, points.T)
        field = _complex_files(evaluated.pressure_real_f32, evaluated.pressure_imag_f32,
                               evaluated.shape)
    assert np.all(np.isfinite(field))
    # Uniform q on a closed body at ka ~ 0.4: the field is monopole-like with
    # the net volume flux, so compare against -sum(q A) G(r) to 15 %.
    area = 2*size*size + 4*size*gap
    centre = np.array([size/2, size/2, gap/2])
    r = np.linalg.norm(points - centre, axis=1)
    monopole = -area*np.exp(1j*k*r)/(4*np.pi*r)
    assert np.max(np.abs(field/monopole - 1)) < 0.15


def _reflect(vertices, mask):
    signs = np.array([-1. if mask & bit else 1. for bit in (1, 2, 4)])
    return vertices*signs


@pytest.mark.parametrize("mode,masks", [("yz", [0, 1]), ("yz+xz", [0, 1, 2, 3])])
def test_native_image_blocks_match_full_mesh_assembly(tmp_path, mode, masks):
    # Review P3: a reversed reflected-curl sign passed the whole suite. Compare
    # the reduced half/quarter sphere operator, including seam (image
    # singular), image near and regular image pairs, with the full sphere
    # assembled without images. Row i of the reduced system is the image count
    # times the Galerkin row over the kept part; on the full mesh a seam
    # vertex's hat also covers its images, so the full row is that count over
    # the vertex's number of distinct images times the same row.
    _require_native()
    sphere = _unit_sphere_mesh()
    full_vertices = sphere.grid.vertices.T
    full_triangles = sphere.grid.elements.T
    centres = full_vertices[full_triangles].mean(axis=1)
    keep = centres[:, 0] > 0
    if mode == "yz+xz":
        keep &= centres[:, 1] > 0
    kept = np.flatnonzero(keep)
    used, inverse = np.unique(full_triangles[kept], return_inverse=True)
    vertices, triangles = full_vertices[used], inverse.reshape(-1, 3)
    lookup = {tuple(np.round(v, 9) + 0.): i for i, v in enumerate(full_vertices)}
    face_lookup = {frozenset(tri.tolist()): i for i, tri in enumerate(full_triangles)}
    image_vertex = np.array([
        [lookup[tuple(np.round(v, 9) + 0.)] for v in _reflect(vertices, mask)]
        for mask in masks
    ])
    image_face = np.array([
        [face_lookup[frozenset(image_vertex[m][tri].tolist())] for tri in triangles]
        for m in range(len(masks))
    ])
    rng = np.random.default_rng(3)
    q = (rng.normal(size=len(triangles)) + 1j*rng.normal(size=len(triangles))).astype(np.complex64)
    q_full = np.zeros(len(full_triangles), np.complex64)
    for m in range(len(masks)):
        q_full[image_face[m]] = q
    k = 1.5
    reduced, reduced_rhs = _native_operator(
        tmp_path, vertices, triangles, k, symmetry_plane=mode, name="reduced", drives=[q],
    )
    full, full_rhs = _native_operator(
        tmp_path, full_vertices, full_triangles, k, name="full", drives=[q_full],
    )
    rows = image_vertex[0]
    distinct_images = np.array([len(set(image_vertex[:, j])) for j in range(len(vertices))])
    expected = np.zeros_like(reduced)
    for j in range(len(vertices)):
        for column in set(image_vertex[:, j]):
            expected[:, j] += full[rows, column]
    scale = distinct_images[:, None]
    assert _relative(reduced, scale*expected) < 2e-4
    assert _relative(reduced_rhs, scale*full_rhs[rows]) < 2e-4


@pytest.mark.parametrize("mode", ["yz", "yz+xz"])
def test_native_image_blocks_match_m1_reference(tmp_path, mode):
    # The review's half/quarter probe against M1 (order 12, not fully
    # converged: 0.3 % matrix / 0.2 % RHS at the fix); a wrong image curl
    # sign gave 45.87 %.
    _require_native()
    from tests.test_metal_native import (
        _tiny_yz_parity_half_buffers, _tiny_yz_xz_parity_quarter_buffers,
    )
    buffers = (_tiny_yz_parity_half_buffers() if mode == "yz"
               else _tiny_yz_xz_parity_quarter_buffers())
    vertices = buffers.vertices_3xn_f32.T.astype(float)
    triangles = buffers.triangles_3xm_i32.T
    reference = assemble_burton_miller_reference(
        vertices, triangles, np.eye(len(triangles)), 1.7,
        symmetry_plane=mode, quadrature_order=12,
    )
    matrix, rhs = _native_operator(tmp_path, vertices, triangles, 1.7, symmetry_plane=mode)
    weight = {"yz": 2, "yz+xz": 4}[mode]
    assert _relative(-matrix/weight, reference.lhs) < 5e-3
    assert _relative(-rhs/weight, reference.rhs_operator) < 3e-3


def test_single_assembly_refuses_burton_miller_with_coupled_baffle(tmp_path):
    # Review P4: the batch path refused this, the single-assembly path did not.
    _require_native()
    buffers = _ib_box_geometry_buffers()
    with MetalNativeStandardSession.create_session(
        geometry_buffers=buffers, work_dir=tmp_path / "ib", session_id="ib", aperture_tag=7,
    ) as session:
        with pytest.raises(ValueError, match="coupled infinite-baffle"):
            session.assemble_standard_neumann(
                100., 1.7, np.ones(buffers.triangles_3xm_i32.shape[1], np.complex64),
                formulation="burton_miller",
            )

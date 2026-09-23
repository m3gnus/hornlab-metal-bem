"""Portable Burton--Miller sign, singular, analytic and symmetry controls."""
from __future__ import annotations

from functools import lru_cache
import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

from hornlab_metal_bem.burton_miller_reference import (
    _geometry, _rule, _source_rule_at_x,
    assemble_burton_miller_reference, evaluate_burton_miller_exterior,
)
from hornlab_metal_bem.config import BIEFormulation, SolveConfig
from hornlab_metal_bem.backends import AssemblyBackendUnavailable
from hornlab_metal_bem.sweep import should_route_native_metal, _k_values_for_native
from hornlab_metal_bem.metal.native import MetalNativeStandardSession
from tests.test_complex_k_resonance import _unit_sphere_mesh


def _sphere():
    mesh = _unit_sphere_mesh()
    return mesh.grid.vertices.T, mesh.grid.elements.T


@lru_cache(maxsize=None)
def _sphere_blocks(k: float):
    vertices, triangles = _sphere()
    return assemble_burton_miller_reference(
        vertices, triangles, np.zeros(len(triangles), complex), k,
        quadrature_order=3,
    )


def _field_for_q(k, q, *, standard=False):
    vertices, triangles = _sphere()
    blocks = _sphere_blocks(k)
    if standard:
        p = np.linalg.solve(
            blocks.mass / 2 - blocks.double_layer,
            -blocks.single_layer @ q,
        )
    else:
        p = np.linalg.solve(blocks.lhs, blocks.rhs_operator @ q)
    field = evaluate_burton_miller_exterior(
        vertices, triangles, p, q, np.array([[0., 0., 3.]]), k,
    )[0]
    return p, field


def _beat_formula_pair(test_face, source_face, tn, sn, tc, sc,
                       tj, sj, k, near, order=3):
    """Independent scalar translation of BEAT's pair formula, before fusion."""
    bary, weights = _rule(order)
    blocks = [np.zeros(3, complex), np.zeros((3, 3), complex),
              np.zeros(3, complex), np.zeros((3, 3), complex)]
    for ib, bx in enumerate(bary):
        x = bx @ test_face
        source = _source_rule_at_x(x, source_face, bary, weights) if near else None
        bys, wys = (bary, weights * sj) if source is None else source
        for by, wy in zip(bys, wys):
            y = by @ source_face
            vec = y - x
            r = np.linalg.norm(vec)
            g = np.exp(1j * k * r) / (4 * np.pi * r)
            dgdr = g * (1j * k - 1 / r) / r
            w = weights[ib] * tj * wy
            for a in range(3):
                blocks[0][a] += w * bx[a] * g
                blocks[2][a] += w * bx[a] * (-dgdr * np.dot(vec, tn))
                for b in range(3):
                    blocks[1][a, b] += w * bx[a] * by[b] * dgdr * np.dot(vec, sn)
                    blocks[3][a, b] += w * g * (
                        np.dot(tc[a], sc[b])
                        - k*k*np.dot(tn, sn)*bx[a]*by[b]
                    )
    return blocks


def test_beats_unfused_pair_blocks_for_coincident_adjacent_and_far():
    vertices = np.array([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        [0., 0., 1.], [5., 0., 0.], [6., 0., 0.], [5., 1., 0.],
    ])
    triangles = np.array([[0, 1, 2], [0, 3, 1], [4, 5, 6]])
    k = 1.7
    blocks = assemble_burton_miller_reference(
        vertices, triangles, np.ones(3), k, quadrature_order=3,
    )
    face, normal, curls, jac = _geometry(vertices, triangles)
    expected_s = np.zeros_like(blocks.single_layer)
    expected_d = np.zeros_like(blocks.double_layer)
    expected_kp = np.zeros_like(blocks.adjoint_double_layer)
    expected_h = np.zeros_like(blocks.hypersingular)
    for i in range(3):
        for j in range(3):
            separation = np.linalg.norm(face[i].mean(axis=0)-face[j].mean(axis=0))
            near = separation < 1.5*max(np.sqrt(jac[i]), np.sqrt(jac[j]))
            s, d, kp, h = _beat_formula_pair(
                face[i], face[j], normal[i], normal[j], curls[i], curls[j],
                jac[i], jac[j], k, near,
            )
            expected_s[triangles[i], j] += s
            expected_kp[triangles[i], j] += kp
            expected_d[np.ix_(triangles[i], triangles[j])] += d
            expected_h[np.ix_(triangles[i], triangles[j])] += h
    for actual, expected in (
        (blocks.single_layer, expected_s),
        (blocks.double_layer, expected_d),
        (blocks.adjoint_double_layer, expected_kp),
        (blocks.hypersingular, expected_h),
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    eta = 1j / k
    np.testing.assert_allclose(blocks.lhs, blocks.mass/2 - blocks.double_layer + eta*blocks.hypersingular)
    np.testing.assert_allclose(blocks.rhs_operator, -blocks.single_layer - eta*(blocks.adjoint_double_layer + blocks.mass_10/2))
    assert np.isfinite(blocks.hypersingular).all()


def test_reference_rejects_scalar_neumann_and_handles_small_triangle():
    vertices = 1e-8*np.eye(3)
    triangles = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="neumann_dp0"):
        assemble_burton_miller_reference(vertices, triangles, 1j, 1.)
    blocks = assemble_burton_miller_reference(vertices, triangles, np.array([1j]), 1.)
    assert np.isfinite(blocks.lhs).all()


@pytest.mark.parametrize("kwargs, message", [
    ({"aperture_tag": 3}, "coupled infinite-baffle"),
    ({"circsym_aperture_tag": 3}, "coupled infinite-baffle"),
    ({"impedance_sources": {4: 0.1}}, "Robin/impedance"),
    ({"impedance_source_callback": lambda _: {4: 0.1}}, "Robin/impedance"),
    ({"ground_plane": "xy"}, "full/half/quarter"),
    ({"chief_points": np.array([[0., 0., 0.]])}, "CHIEF"),
])
def test_bm_capability_refusals(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SolveConfig(formulation=BIEFormulation.BURTON_MILLER, **kwargs)


def test_bm_native_and_circsym_refuse_without_fallback():
    config = SolveConfig(formulation=BIEFormulation.BURTON_MILLER)
    assert SolveConfig().formulation == BIEFormulation.STANDARD
    with pytest.raises(AssemblyBackendUnavailable, match="reference-only"):
        should_route_native_metal(config)
    with pytest.raises(AssemblyBackendUnavailable, match="reference-only"):
        _k_values_for_native(np.array([100.]), config)
    with pytest.raises(ValueError, match="reference-only"):
        MetalNativeStandardSession.assemble_solve_evaluate_standard_neumann_batch(
            None, None, None, None, None, formulation="burton_miller",
        )
    from hornlab_metal_bem.circsym import solve_circsym, run_sweep_circsym
    with pytest.raises(ValueError, match="CircSym"):
        solve_circsym(None, config)
    with pytest.raises(ValueError, match="CircSym"):
        run_sweep_circsym(None, np.array([100.]), config)


@pytest.mark.parametrize("k, ceiling", [(0.5, .06), (1.5, .07), (np.pi, .13), (3.25, .14)])
def test_pulsating_sphere_analytic_and_phase(k, ceiling):
    vertices, triangles = _sphere()
    q = 1j*k*np.ones(len(triangles))  # rho*c = 1, v_n = 1
    _, field = _field_for_q(k, q)
    exact = 1j*k*np.exp(1j*k*2)/(3*(1j*k-1))
    assert abs(field/exact - 1) < ceiling
    assert abs(np.angle(field/exact)) < .12


@pytest.mark.parametrize("k, ceiling", [(0.5, .12), (1.5, .08), (np.pi, .14)])
def test_oscillating_sphere_analytic(k, ceiling):
    vertices, triangles = _sphere()
    face, normal, _, _ = _geometry(vertices, triangles)
    q = 1j*k*normal[:, 2]
    _, field = _field_for_q(k, q)
    h = lambda z: spherical_jn(1, z) + 1j*spherical_yn(1, z)
    hp = lambda z: spherical_jn(1, z, True) + 1j*spherical_yn(1, z, True)
    exact = 1j*h(3*k)/hp(k)
    assert abs(field/exact - 1) < ceiling


def test_standard_spikes_at_sphere_interior_dirichlet_resonance():
    _, triangles = _sphere()
    for k, standard_floor, bm_ceiling in ((np.pi, .18, .13), (3.25, .7, .14)):
        q = 1j*k*np.ones(len(triangles))
        _, bm = _field_for_q(k, q)
        _, standard = _field_for_q(k, q, standard=True)
        exact = 1j*k*np.exp(1j*k*2)/(3*(1j*k-1))
        assert abs(bm/exact - 1) < bm_ceiling
        assert abs(standard/exact - 1) > standard_floor


def test_full_half_quarter_symmetry_fields_agree():
    vertices, triangles = _sphere()
    k = 1.5
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
        verts, tri = vertices[used], inverse.reshape(-1, 3)
        q = 1j*k*np.ones(len(tri))
        blocks = assemble_burton_miller_reference(
            verts, tri, q, k, symmetry_plane=mode, quadrature_order=3,
        )
        p = np.linalg.solve(blocks.lhs, blocks.rhs)
        fields.append(evaluate_burton_miller_exterior(
            verts, tri, p, q, np.array([[.4, .3, 3.]]), k,
            symmetry_plane=mode,
        )[0])
    assert max(abs(f-fields[0])/abs(fields[0]) for f in fields[1:]) < .001


def test_driven_face_power_agrees_with_far_sphere_to_one_percent():
    vertices, triangles = _sphere()
    k = 1.5
    q = 1j*k*np.ones(len(triangles))
    p, _ = _field_for_q(k, q)
    _, _, _, jac = _geometry(vertices, triangles)
    face_power = .5*np.real(np.sum(jac/2*np.mean(p[triangles], axis=1)))
    count = 300
    index = np.arange(count)
    z = 1 - 2*(index+.5)/count
    phi = np.pi*(3-np.sqrt(5))*index
    radius = np.sqrt(1-z*z)
    points = 8*np.stack((radius*np.cos(phi), radius*np.sin(phi), z), axis=1)
    field = evaluate_burton_miller_exterior(vertices, triangles, p, q, points, k)
    sphere_power = .5*4*np.pi*8**2*np.mean(abs(field)**2)
    assert face_power > 0
    assert abs(sphere_power/face_power - 1) < .01

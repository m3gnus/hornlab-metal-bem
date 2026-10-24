"""Native qualification of the BEAT-exterior physics added to solve_compiled.

These run the real Swift/Metal helper (macOS CI) and compare against
references that do not reuse the adapter's own arithmetic:

- an oscillating rigid sphere driven as an electrodynamic transducer, against
  the analytic dipole radiation load and field;
- symmetry representatives (a cut half driver, whole-driver mirror copies and
  a cut driver with a mirror copy) against the explicitly mirrored full
  geometry solved without symmetry;
- two acoustically coupled drivers with one shorted and undriven;
- two meshes with colliding physical tags against one merged mesh.

The request builders also run on every platform through
``test_qualification_requests_plan_and_stage``.
"""

import base64

import numpy as np
import pytest

api = pytest.importorskip("hornlab_metal_bem")
adapter = pytest.importorskip("hornlab_metal_bem.compiled")
pytest.importorskip("beat_engine.beat_contract")

RHO, C = 1.21, 343.0
RADIUS = 0.05
LEM = {
    "re_ohm": 6.0,
    "le_h": 2e-4,
    "bl_n_per_a": 5.0,
    "mmd_kg": 0.01,
    "cms_m_per_n": 1e-3,
    "rms_n_s_per_m": 1.0,
}
FREQUENCIES = [300.0, 900.0]
POINTS = [[0.6, 0.35, 0.45], [1.2, 0.25, -0.3], [0.4, 0.9, 0.1]]


def native():
    from native_helper_guard import require_fresh_native_helper

    require_fresh_native_helper()


def packed(a, dtype):
    a = np.ascontiguousarray(a, dtype=dtype)
    return {
        "dtype": dtype,
        "shape": list(a.shape),
        "data": base64.b64encode(a.tobytes()).decode(),
    }


def uv_sphere(center, radius=RADIUS, n_theta=14, n_phi=32):
    """Outward UV sphere whose meridians include the X=0 and Y=0 planes."""
    center = np.asarray(center, dtype=float)
    points = [[0.0, 0.0, radius]]
    for i in range(1, n_theta):
        t = np.pi * i / n_theta
        for j in range(n_phi):
            p = 2 * np.pi * j / n_phi
            points.append(
                [radius * np.sin(t) * np.cos(p), radius * np.sin(t) * np.sin(p), radius * np.cos(t)]
            )
    points.append([0.0, 0.0, -radius])
    points = np.array(points)
    points[np.abs(points) < 1e-12] = 0.0

    def ring(i, j):
        return 1 + (i - 1) * n_phi + j % n_phi

    south = len(points) - 1
    faces = []
    for j in range(n_phi):
        faces.append([0, ring(1, j), ring(1, j + 1)])
        faces.append([south, ring(n_theta - 1, j + 1), ring(n_theta - 1, j)])
        for i in range(1, n_theta - 1):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            faces += [[a, c, d], [a, d, b]]
    faces = np.array(faces)
    tri = points[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", normal, tri.mean(axis=1)) < 0
    faces[flip] = faces[flip][:, ::-1]
    return points + center, faces


def mirror_faces(faces):
    return faces[:, ::-1]  # reflection reverses orientation


def build(bodies, components, *, symmetry="off", ports=None, outputs=None, phasor="exp(-i omega t)", version=3):
    """bodies: [(mesh_id, points, [(tag, faces, component_id)])]."""
    meshes, boundaries, comp_bounds = [], [], {}
    for mesh_id, points, parts in bodies:
        faces = np.vstack([f for _, f, _ in parts])
        tags = np.concatenate([np.full(len(f), t) for t, f, _ in parts])
        meshes.append(
            {
                "id": mesh_id,
                "name": mesh_id,
                "purpose": "bem_surface",
                "file": "",
                "scale_to_m": 1.0,
                "translation_m": [0.0, 0.0, 0.0],
                "mesh_data": {
                    "schema_version": 1,
                    "points": packed(points, "<f8"),
                    "cells": [
                        {
                            "type": "triangle",
                            "connectivity": packed(faces, "<i8"),
                            "physical_tags": packed(tags, "<i8"),
                        }
                    ],
                    "physical_names": {str(t): [int(t), 2] for t, _, _ in parts},
                },
            }
        )
        for t, _, cid in parts:
            bid = f"{mesh_id}:b{t}"
            boundaries.append(
                {
                    "id": bid,
                    "name": bid,
                    "region_id": "air",
                    "kind": "moving",
                    "group": {"mesh_id": mesh_id, "dimension": 2, "tag": int(t), "name": None},
                    "parameters": {},
                }
            )
            comp_bounds.setdefault(cid, []).append(bid)
    comps = [
        {"id": cid, "name": cid, "kind": kind, "boundary_ids": comp_bounds[cid], "parameters": params}
        for cid, kind, params in components
    ]
    port_defs = [
        {
            "id": f"p:{cid}",
            "name": cid,
            "component_id": cid,
            "kind": "voltage" if kind == "electrodynamic_transducer" else "normal_velocity",
        }
        for cid, kind, _ in components
    ]
    outputs = outputs or [
        {"id": "p", "quantity": "exterior_pressure", "target_ids": [], "options": {"points_m": POINTS}},
        {"id": "z", "quantity": "radiation_impedance_matrix", "target_ids": [], "options": {}},
        {"id": "u", "quantity": "diaphragm_velocity", "target_ids": [], "options": {}},
        {"id": "i", "quantity": "voice_coil_current", "target_ids": [], "options": {}},
    ]
    return {
        "schema_version": 1,
        "compiled_system": {
            "id": "sys",
            "name": "sys",
            "contract_version": version,
            "meshes": meshes,
            "regions": [
                {
                    "id": "air",
                    "name": "air",
                    "kind": "unbounded_air",
                    "mesh_ids": [m["id"] for m in meshes],
                    "volume_groups": [],
                    "sound_speed_m_per_s": C,
                    "density_kg_per_m3": RHO,
                    "loss_model": {},
                }
            ],
            "boundaries": boundaries,
            "interfaces": [],
            "components": comps,
            "excitation_ports": port_defs,
        },
        "frequencies_hz": FREQUENCIES,
        "excitation_port_ids": ports or [p["id"] for p in port_defs],
        "outputs": outputs,
        "solver_options": {
            "bem_backend": "metal",
            "precision": "float32",
            "symmetry": symmetry,
            "phasor_convention": phasor,
            "transducer_reference_voltage_v": 2.0,
        },
    }


def driver(**extra):
    return ("electrodynamic_transducer", {**LEM, "motion_axis": [0.0, 0.0, 1.0], **extra})


def single_sphere(phasor="exp(-i omega t)"):
    pts, faces = uv_sphere([0.0, 0.0, 0.0])
    return build([("m", pts, [(2, faces, "d")])], [("d", *driver())], phasor=phasor)


def half_sphere_x():
    pts, faces = uv_sphere([0.0, 0.0, 0.0])
    half = faces[np.all(pts[faces][:, :, 0] >= 0.0, axis=1)]
    params = {
        "surface_completion_factor": 2,
        "symmetry_role": "fractional_driver",
        "fractional_symmetry_axes": ["x"],
    }
    return build([("m", pts, [(2, half, "d")])], [("d", *driver(**params))], symmetry="x")


def whole_driver_copies(symmetry):
    """One complete sphere off the planes, plus the explicit mirrored model."""
    center = [0.15, 0.12 if symmetry == "xy" else 0.0, 0.0]
    pts, faces = uv_sphere(center)
    orbit = 4 if symmetry == "xy" else 2
    reduced = build(
        [("m", pts, [(2, faces, "d0")])],
        [("d0", *driver(physical_driver_orbit_count=orbit))],
        symmetry=symmetry,
    )
    signs = [(1, 1), (-1, 1), (1, -1), (-1, -1)][:orbit]
    parts, comps, all_pts, offset = [], [], [], 0
    for k, (sx, sy) in enumerate(signs):
        p = pts * [sx, sy, 1]
        f = faces if sx * sy > 0 else mirror_faces(faces)
        all_pts.append(p)
        parts.append((10 + k, f + offset, f"d{k}"))
        comps.append((f"d{k}", *driver()))
        offset += len(p)
    full = build([("m", np.vstack(all_pts), parts)], comps)
    return reduced, full


def cut_driver_with_copy():
    """Half sphere cut by X=0 at y=+d, mirrored to a second driver at y=-d."""
    pts, faces = uv_sphere([0.0, 0.15, 0.0])
    half = faces[np.all(pts[faces][:, :, 0] >= 0.0, axis=1)]
    params = {
        "surface_completion_factor": 2,
        "physical_driver_orbit_count": 2,
        "symmetry_role": "fractional_driver",
        "fractional_symmetry_axes": ["x"],
    }
    reduced = build([("m", pts, [(2, half, "d0")])], [("d0", *driver(**params))], symmetry="xy")
    mirror = pts * [1, -1, 1]
    full = build(
        [("m", np.vstack([pts, mirror]), [(10, faces, "d0"), (11, mirror_faces(faces) + len(pts), "d1")])],
        [("d0", *driver()), ("d1", *driver())],
    )
    return reduced, full


def coupled_pair():
    a, fa = uv_sphere([0.0, 0.0, 0.0])
    b, fb = uv_sphere([0.0, 0.0, 0.13])
    return build(
        [("m", np.vstack([a, b]), [(2, fa, "d0"), (3, fb + len(a), "d1")])],
        [("d0", *driver()), ("d1", *driver())],
        ports=["p:d0"],
    )


def two_meshes(colliding):
    a, fa = uv_sphere([0.0, 0.0, 0.0])
    b, fb = uv_sphere([0.0, 0.0, 0.2])
    ideal = ("ideal_velocity_source", {})
    outputs = [
        {"id": "p", "quantity": "exterior_pressure", "target_ids": [], "options": {"points_m": POINTS}},
        {"id": "z", "quantity": "radiation_impedance_matrix", "target_ids": [], "options": {}},
    ]
    if colliding:
        bodies = [("ma", a, [(2, fa, "s0")]), ("mb", b, [(2, fb, "s1")])]
    else:
        bodies = [("m", np.vstack([a, b]), [(2, fa, "s0"), (3, fb + len(a), "s1")])]
    return build(bodies, [("s0", *ideal), ("s1", *ideal)], outputs=outputs, version=1)


def h1(n, x):
    from scipy.special import spherical_jn, spherical_yn

    return spherical_jn(n, x) + 1j * spherical_yn(n, x)


def dh1(n, x):
    from scipy.special import spherical_jn, spherical_yn

    return spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)


ALL_REQUESTS = {
    "single": single_sphere,
    "half_x": half_sphere_x,
    "whole_x": lambda: whole_driver_copies("x")[0],
    "whole_x_full": lambda: whole_driver_copies("x")[1],
    "whole_xy": lambda: whole_driver_copies("xy")[0],
    "whole_xy_full": lambda: whole_driver_copies("xy")[1],
    "cut_copy": lambda: cut_driver_with_copy()[0],
    "cut_copy_full": lambda: cut_driver_with_copy()[1],
    "pair": coupled_pair,
    "meshes_colliding": lambda: two_meshes(True),
    "meshes_merged": lambda: two_meshes(False),
}


@pytest.mark.parametrize("name", sorted(ALL_REQUESTS))
def test_qualification_requests_plan_and_stage(name, tmp_path):
    w = ALL_REQUESTS[name]()
    plan = adapter._plan(w)
    surface = adapter._surface(plan, tmp_path)
    assert len(surface["faces"]) > 0
    if plan["symmetry"] in ("x", "xy"):
        assert plan["kwargs"]["native_symmetry_complete_images"] is True


def even(result, key):
    """The even drive of the full model: every port at once (sum of columns)."""
    return result.quantities[key].values.sum(axis=1)


@pytest.mark.slow
def test_oscillating_sphere_transducer_against_analytic_dipole():
    native()
    result = api.solve_compiled(single_sphere())
    q = result.quantities
    for f, freq in enumerate(FREQUENCIES):
        w = 2 * np.pi * freq
        k = w / C
        ka = k * RADIUS
        z_ref = 4 * np.pi * RADIUS**2 / 3 * 1j * RHO * C * h1(1, ka) / dh1(1, ka)
        z = q["z"].values[f, 0, 0]
        assert abs(z - z_ref) / abs(z_ref) < 0.05, (freq, z, z_ref)
        zm = LEM["rms_n_s_per_m"] + 1j * (-w * LEM["mmd_kg"] + 1 / (w * LEM["cms_m_per_n"]))
        ze = LEM["re_ohm"] - 1j * w * LEM["le_h"]
        bl = LEM["bl_n_per_a"]
        u_ref = 2.0 * bl / (bl**2 + ze * (zm + z_ref))
        i_ref = (zm + z_ref) * u_ref / bl
        np.testing.assert_allclose(q["u"].values[f, 0, 0], u_ref, rtol=5e-3)
        np.testing.assert_allclose(q["i"].values[f, 0, 0], i_ref, rtol=5e-3)
        pts = np.asarray(POINTS)
        r = np.linalg.norm(pts, axis=1)
        p_ref = 1j * RHO * C * u_ref / dh1(1, ka) * h1(1, k * r) * pts[:, 2] / r
        np.testing.assert_allclose(q["p"].values[f, 0], p_ref, rtol=0.06)


@pytest.mark.slow
def test_positive_time_phasor_is_the_conjugate():
    native()
    neg = api.solve_compiled(single_sphere())
    pos = api.solve_compiled(single_sphere("exp(+i omega t)"))
    for key in ("p", "z", "u", "i"):
        np.testing.assert_allclose(
            pos.quantities[key].values, neg.quantities[key].values.conj(), rtol=1e-5
        )


def assert_reduced_matches_full(reduced, full, rtol=2e-3):
    pts = np.asarray(POINTS)
    np.testing.assert_allclose(
        reduced.quantities["p"].values[:, 0], even(full, "p"), rtol=rtol, atol=1e-6 * np.abs(even(full, "p")).max()
    )
    np.testing.assert_allclose(reduced.quantities["u"].values[:, 0, 0], even(full, "u")[:, 0], rtol=rtol)
    np.testing.assert_allclose(reduced.quantities["i"].values[:, 0, 0], even(full, "i")[:, 0], rtol=rtol)
    # Per-copy load on driver 0 under the even drive: its row of the full matrix.
    np.testing.assert_allclose(
        reduced.quantities["z"].values[:, 0, 0],
        full.quantities["z"].values[:, 0, :].sum(axis=1),
        rtol=rtol,
    )
    assert np.all(pts[:, :2] > 0)


@pytest.mark.slow
def test_cut_half_driver_matches_full_sphere():
    native()
    reduced = api.solve_compiled(half_sphere_x())
    full = api.solve_compiled(single_sphere())
    assert_reduced_matches_full(reduced, full)


@pytest.mark.slow
@pytest.mark.parametrize("symmetry", ["x", "xy"])
def test_whole_driver_mirror_copies_match_explicit_drivers(symmetry):
    native()
    reduced_request, full_request = whole_driver_copies(symmetry)
    reduced = api.solve_compiled(reduced_request)
    full = api.solve_compiled(full_request)
    assert reduced.quantities["z"].metadata["row_weights"] == [4.0 if symmetry == "xy" else 2.0]
    assert_reduced_matches_full(reduced, full)


@pytest.mark.slow
def test_cut_driver_with_mirror_copy_matches_explicit_pair():
    native()
    reduced_request, full_request = cut_driver_with_copy()
    assert_reduced_matches_full(api.solve_compiled(reduced_request), api.solve_compiled(full_request))


@pytest.mark.slow
def test_coupled_pair_with_shorted_undriven_driver():
    native()
    result = api.solve_compiled(coupled_pair())
    z = result.quantities["z"]
    for f in range(len(FREQUENCIES)):
        m = z.values[f]
        assert abs(m[0, 1] - m[1, 0]) / abs(m[0, 1]) < 2e-2  # reciprocity
        assert z.metadata["passivity_min_eig"][f] > -1e-3 * np.abs(m).max()
    u = result.quantities["u"].values[:, 0, :]
    i = result.quantities["i"].values[:, 0, :]
    assert np.all(np.abs(u[:, 1]) > 0)  # the undriven driver is moved by coupling
    # Shorted coil: Bl u + Ze i = 0 on the undriven driver.
    w = 2 * np.pi * np.asarray(FREQUENCIES)
    ze = LEM["re_ohm"] - 1j * w * LEM["le_h"]
    np.testing.assert_allclose(LEM["bl_n_per_a"] * u[:, 1] + ze * i[:, 1], 0, atol=1e-9)
    # Its mechanical row balances the mutual load from the driven sphere.
    zm = LEM["rms_n_s_per_m"] + 1j * (-w * LEM["mmd_kg"] + 1 / (w * LEM["cms_m_per_n"]))
    lhs = (zm + z.values[:, 1, 1]) * u[:, 1] - LEM["bl_n_per_a"] * i[:, 1]
    np.testing.assert_allclose(lhs, -z.values[:, 1, 0] * u[:, 0], rtol=1e-6)


@pytest.mark.slow
def test_colliding_tags_across_meshes_match_merged_mesh():
    native()
    split = api.solve_compiled(two_meshes(True))
    merged = api.solve_compiled(two_meshes(False))
    for key in ("p", "z"):
        np.testing.assert_allclose(
            split.quantities[key].values, merged.quantities[key].values, rtol=1e-5
        )

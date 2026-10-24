"""BEAT-exterior features of the compiled adapter and its BEAT worker surface.

A synthetic linear "BEM" stands in for the native helper (no Metal here): each
basis column's surface pressure is ``ALPHA * sum(drive weights)`` at every
vertex and its field ``BETA * sum(drive weights)`` at every point. Host-side
force integration, the lumped network, symmetry accounting, phasor conversion
and wire encoding are then checked against closed forms.
"""

import base64
import copy

import numpy as np
import pytest
from test_compiled import (
    adapter,
    api,
    native_mesh_loading_is_not_needed_for_translation_tests,  # noqa: F401  (autouse)
    request,
)

beat_worker = pytest.importorskip(api.__name__ + ".beat_worker")
contract_worker = pytest.importorskip("beat_engine.beat_contract.worker")

ALPHA = 3.0 - 2.0j
BETA = 0.5 + 0.25j
LEM = {
    "re_ohm": 6.0,
    "le_h": 2e-4,
    "bl_n_per_a": 5.0,
    "mmd_kg": 0.01,
    "cms_m_per_n": 1e-3,
    "rms_n_s_per_m": 1.5,
}
pytestmark = pytest.mark.skipif(
    not adapter._IS_METAL, reason="BEAT worker surface ships with the Metal package"
)


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def solve(loaded, freqs, cfg):
        seen.append(cfg)
        drive = sum(complex(w) for w in cfg.velocity_sources.values())
        n_points = cfg.observation.angle_count
        result = {
            "frequencies_hz": freqs,
            "pressure_complex": np.full((len(freqs), 1, n_points), BETA * drive),
        }
        if cfg.return_surface_pressure:
            import meshio

            used = len(np.unique(meshio.read(loaded.staged_path).cells_dict["triangle"]))
            result["surface_pressure_complex"] = np.full((len(freqs), used), ALPHA * drive)
        from types import SimpleNamespace

        return SimpleNamespace(**result)

    monkeypatch.setattr(api, "solve_frequencies", solve)
    return seen


def geometry():
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ) * 0.001 + [0.2, 0.3, 0.4]
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    tri = points[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return np.linalg.norm(cross, axis=1) / 2, cross / np.linalg.norm(cross, axis=1)[:, None]


def transducer_request(phasor="exp(-i omega t)", ports=("pv", "pa")):
    w = request()
    s = w["compiled_system"]
    s["contract_version"] = 3
    s["components"] = [
        {
            "id": "drv",
            "name": "drv",
            "kind": "electrodynamic_transducer",
            "boundary_ids": ["b11"],
            "parameters": {**LEM, "motion_axis": [0.0, 0.0, -2.0]},
        },
        {
            "id": "a",
            "name": "a",
            "kind": "ideal_velocity_source",
            "boundary_ids": ["b12", "b13"],
            "parameters": {},
        },
    ]
    s["excitation_ports"] = [
        {"id": "pv", "name": "pv", "component_id": "drv", "kind": "voltage"},
        {"id": "pa", "name": "pa", "component_id": "a", "kind": "normal_velocity"},
    ]
    w["excitation_port_ids"] = list(ports)
    w["solver_options"].update(
        phasor_convention=phasor, transducer_reference_voltage_v=2.0
    )
    w["outputs"] += [
        {"id": q, "quantity": q, "target_ids": [], "options": {}}
        for q in (
            "diaphragm_velocity",
            "voice_coil_current",
            "radiation_impedance_matrix",
            "bem_boundary_pressure",
            "bem_boundary_neumann",
        )
    ]
    return w


@pytest.mark.parametrize("phasor", ["exp(-i omega t)", "exp(+i omega t)"])
def test_transducer_network_matches_closed_form(calls, phasor):
    result = api.solve_compiled(transducer_request(phasor))
    areas, normals = geometry()
    b_drv = normals[0] @ np.array([0.0, 0.0, -1.0]) * areas[0]  # signed area
    # Basis: transducer first, then the requested ideal source.
    assert result.motion_basis_component_ids == ("drv", "a")
    assert calls[0].source_velocity_profiles  # mixed axial/normal drive
    # Columns: drv drive weight is n.axis only through the face scale, so the
    # synthetic drive sum is its tag coefficient (1); ideal a drives two tags.
    z_dd, z_da = ALPHA * b_drv, ALPHA * b_drv * 2
    v = 2.0
    for f, freq in enumerate(result.frequencies_hz):
        w = 2 * np.pi * freq
        zm = LEM["rms_n_s_per_m"] + 1j * (-w * LEM["mmd_kg"] + 1 / (w * LEM["cms_m_per_n"]))
        ze = LEM["re_ohm"] - 1j * w * LEM["le_h"]
        bl = LEM["bl_n_per_a"]
        u_v = v * bl / (bl**2 + ze * (zm + z_dd))
        u_a = -z_da / (zm + z_dd + bl**2 / ze)
        i_v, i_a = (zm + z_dd) * u_v / bl, -bl * u_a / ze
        expected_u = np.array([u_v, u_a])
        expected_i = np.array([i_v, i_a])
        expected_p = BETA * np.array([u_v, u_a + 2.0])
        if phasor == "exp(+i omega t)":
            expected_u, expected_i, expected_p = map(
                np.conj, (expected_u, expected_i, expected_p)
            )
        q = result.quantities
        np.testing.assert_allclose(q["diaphragm_velocity"].values[f, :, 0], expected_u)
        np.testing.assert_allclose(q["voice_coil_current"].values[f, :, 0], expected_i)
        np.testing.assert_allclose(result.pressure["far"][f, :, 0], expected_p)
    assert result.time_convention == phasor
    assert result.diagnostics["transducer_reference_voltage_v"] == 2.0


def test_impedance_matrix_and_traces(calls):
    w = transducer_request()
    result = api.solve_compiled(w)
    areas, normals = geometry()
    q = result.quantities
    m = q["radiation_impedance_matrix"]
    assert m.axes == ("frequency", "receiver_component", "source_component")
    assert m.metadata["component_ids"] == ["drv", "a"]
    assert m.metadata["kinds"] == ["electrodynamic_transducer", "ideal_velocity_source"]
    signed = np.array([-normals[0][2] * areas[0], areas[1] + areas[2]])
    np.testing.assert_allclose(m.metadata["effective_volume_area_m2"], signed)
    drive = np.array([1.0, 2.0])  # synthetic drive sums of the two columns
    np.testing.assert_allclose(m.values[0], ALPHA * np.outer(signed, drive))
    # Neumann trace: i*rho*omega*v on moving faces, zero on the rigid face.
    neumann = q["bem_boundary_neumann"].values
    u = q["diaphragm_velocity"].values[:, :, 0]
    w0 = 2 * np.pi * result.frequencies_hz[0]
    np.testing.assert_allclose(
        neumann[0, 1, 1:3], 1j * 1.21 * w0 * np.array([1.0, 1.0]), rtol=1e-12
    )
    np.testing.assert_allclose(
        neumann[0, 0, 0], 1j * 1.21 * w0 * u[0, 0] * -normals[0][2], rtol=1e-12
    )
    assert np.all(neumann[:, :, 3] == 0)
    assert q["bem_boundary_pressure"].values.shape == (3, 2, 4)


def test_ideal_radiation_impedance_counts_symmetry_copies(calls):
    # Half model on X=0: drop the tetrahedron's X=0 face (tag 13); its open
    # rim lies on the image plane, which reflection closes.
    w = request()
    s = w["compiled_system"]
    d = s["meshes"][0]
    d["translation_m"] = [0.0, 0.3, 0.4]
    cells = d["mesh_data"]["cells"][0]
    faces = adapter._packed_array(cells["connectivity"])
    tags = adapter._packed_array(cells["physical_tags"])
    cells["connectivity"] = _packed(faces[tags != 13], "<i8")
    cells["physical_tags"] = _packed(tags[tags != 13], "<i8")
    s["boundaries"] = [b for b in s["boundaries"] if b["id"] != "b13"]
    s["boundaries"][-1]["kind"] = "moving"
    s["components"][1]["boundary_ids"] = ["b14"]
    w["solver_options"]["symmetry"] = "x"
    w["outputs"].append(
        {"id": "z", "quantity": "radiation_impedance", "target_ids": [], "options": {}}
    )
    result = api.solve_compiled(w)
    assert calls[0].native_symmetry_plane == "yz"
    areas, _ = geometry()
    # Compiled component order (a, b); each row counts both real copies.
    a_area, b_area = areas[0] + areas[1], areas[3]
    np.testing.assert_allclose(
        result.quantities["z"].values[0], 2 * ALPHA * np.array([2 * a_area, b_area])
    )
    w["compiled_system"]["meshes"][0]["translation_m"] = [-0.1, 0.3, 0.4]
    with pytest.raises(ValueError, match="fundamental domain"):
        api.solve_compiled(w)


def _packed(a, dtype):
    a = np.asarray(a, dtype=dtype)
    return {"dtype": dtype, "shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}


def test_ground_mode_maps_to_native_ground_plane(calls):
    w = request()
    w["solver_options"]["symmetry"] = "ground"
    w["solver_options"]["ground_plane_min_clearance_m"] = 0.1
    api.solve_compiled(w)
    assert calls[0].ground_plane == "xz"
    assert calls[0].ground_plane_min_clearance_m == 0.1
    w["compiled_system"]["meshes"][0]["translation_m"] = [0.2, -0.3, 0.4]
    with pytest.raises(ValueError, match="Y >= 0"):
        api.solve_compiled(w)


def test_repeated_frequencies_weights_and_mixed_profiles(calls):
    w = request()
    w["compiled_system"]["contract_version"] = 2
    w["frequencies_hz"] = [200.0, 80.0, 200.0]
    w["compiled_system"]["components"][0]["parameters"] = {
        "boundary_motion_weights": {"b12": 0.5}
    }
    w["compiled_system"]["components"][1]["parameters"] = {
        "motion_profile": "rigid_translation",
        "motion_axis": [0, 0, 3],
    }
    result = api.solve_compiled(w)
    assert [c.velocity_sources for c in calls] == [
        {11: 0.0, 12: 0.0, 13: 1.0},
        {11: 1.0, 12: 0.5, 13: 0.0},
    ]
    assert calls[0].source_axes == {13: (0.0, 0.0, 1.0)}
    assert set(calls[0].source_velocity_profiles) == {13}
    p = result.pressure["far"]
    assert p.shape == (3, 2, 2)
    np.testing.assert_array_equal(p[0], p[2])
    np.testing.assert_array_equal(result.frequencies_hz, [200, 80, 200])


def test_multiple_meshes_with_colliding_tags(calls):
    w = request()
    s = w["compiled_system"]
    second = copy.deepcopy(s["meshes"][0])
    second["id"] = "mesh2"
    second["translation_m"] = [1.0, 1.0, 1.0]
    s["meshes"].append(second)
    s["regions"][0]["mesh_ids"].append("mesh2")
    for t in (11, 12, 13, 14):
        s["boundaries"].append(
            {
                "id": f"m2b{t}",
                "name": str(t),
                "region_id": "air",
                "kind": "rigid",
                "group": {"mesh_id": "mesh2", "dimension": 2, "tag": t, "name": None},
                "parameters": {},
            }
        )
    api.solve_compiled(w)
    plan = adapter._plan(w)
    assert plan["mesh_tags"]["mesh"] == {11: 11, 12: 12, 13: 13, 14: 14}
    assert plan["mesh_tags"]["mesh2"] == {11: 15, 12: 16, 13: 17, 14: 18}


def test_worker_events_negotiate_and_decode_like_beat(calls, tmp_path):
    worker = beat_worker.MetalEngineWorker(availability={"available": True})
    worker.ensure_started()
    info = worker.worker_info
    contract_worker.validate_worker_ready(info)
    w = transducer_request("exp(+i omega t)")
    command = contract_worker.negotiate_submission(info, w, "solve")
    assert command["result_schema_version"] == 2
    direct = api.solve_compiled(w)
    events = list(worker.submit(w))
    assert [e["type"] for e in events] == ["status", "result", "result", "result", "completed"]
    assert events[-1]["solved_count"] == 3
    for f, event in enumerate(events[1:-1]):
        contract_worker.validate_worker_event(event)
        record = event["result"]
        assert record["excitation_port_ids"] == ["pv", "pa"]
        assert record["freq_hz"] == float(np.float32(w["frequencies_hz"][f]))
        d = record["diagnostics"]
        assert (d["precision"], d["bem_backend"], d["symmetry"], d["phasor_convention"]) == (
            "float32", "metal", "off", "exp(+i omega t)")
        assert [q["id"] for q in record["quantities"]] == [o["id"] for o in w["outputs"]]
        for item in record["quantities"]:
            values = item["values"]
            dtype = "<c8" if values["dtype"] == "complex64" else "<c16"
            array = np.frombuffer(
                base64.b64decode(values["content_base64"]), dtype=dtype
            ).reshape(values["shape"])
            expected = direct.quantities[item["id"]]
            assert item["axes"] == list(expected.axes[1:])
            assert item["target_id"] is None
            np.testing.assert_allclose(array, expected.values[f], rtol=1e-6)
            if item["quantity"] in ("exterior_pressure", "bem_boundary_pressure"):
                assert values["dtype"] == "complex64"
            if item["quantity"] in ("diaphragm_velocity", "radiation_impedance_matrix"):
                assert values["dtype"] == "complex128"
            if item["quantity"] == "radiation_impedance_matrix":
                assert isinstance(item["metadata"]["passivity_min_eig"], float)


def test_worker_reports_failures_and_cancellation(calls, tmp_path):
    worker = beat_worker.MetalEngineWorker(availability={"available": True})
    bad = request()
    bad["compiled_system"]["regions"][0]["loss_model"] = {"kind": "viscous"}
    events = list(worker.submit(bad))
    assert events[-1]["type"] == "failed"
    assert events[-1]["code"] == "worker_request_failed"
    w = request()
    marker = tmp_path / "cancel"
    marker.touch()
    w["cancel_path"] = str(marker)
    assert list(worker.submit(w))[-1] == {"type": "cancelled", "solved_count": 0}
    path = tmp_path / "request.json"
    import json

    path.write_text(json.dumps(request()))
    assert list(worker.submit(path))[-1]["type"] == "completed"
    assert list(worker.submit(request(), operation="bem_field"))[-1]["type"] == "failed"

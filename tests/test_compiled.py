"""The same BEAT requests exercise both independently installed legacy adapters."""

import base64
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# CI's dev extra supplies only the BEAT Python package; no Julia is launched.
pytest.importorskip("beat_engine.beat_contract")
_PACKAGE = (
    "hornlab_metal_bem"
    if (Path(__file__).resolve().parents[1] / "hornlab_metal_bem").is_dir()
    else "hornlab_bempp_bem"
)
api = importlib.import_module(_PACKAGE)
adapter = importlib.import_module(_PACKAGE + ".compiled")


@pytest.fixture(autouse=True)
def native_mesh_loading_is_not_needed_for_translation_tests(monkeypatch):
    monkeypatch.setattr(
        api,
        "load_mesh",
        lambda path, **kwargs: SimpleNamespace(staged_path=path, mesh_kwargs=kwargs),
    )


def packed(a, dtype):
    a = np.asarray(a, dtype=dtype)
    return {
        "dtype": dtype,
        "shape": list(a.shape),
        "data": base64.b64encode(a.tobytes()).decode(),
    }


def request():
    # Outward tetrahedron, two source components, one owning two tags.
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    tags = [11, 12, 13, 14]
    mesh = {
        "id": "mesh",
        "name": "mesh",
        "purpose": "bem_surface",
        "file": "",
        "scale_to_m": 0.001,
        "translation_m": [0.2, 0.3, 0.4],
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
            "physical_names": {str(t): [t, 2] for t in tags},
        },
    }
    boundaries = [
        {
            "id": f"b{t}",
            "name": str(t),
            "region_id": "air",
            "kind": "moving" if t != 14 else "rigid",
            "group": {"mesh_id": "mesh", "dimension": 2, "tag": t, "name": None},
            "parameters": {},
        }
        for t in tags
    ]
    components = [
        {
            "id": "a",
            "name": "a",
            "kind": "ideal_velocity_source",
            "boundary_ids": ["b11", "b12"],
            "parameters": {},
        },
        {
            "id": "b",
            "name": "b",
            "kind": "ideal_velocity_source",
            "boundary_ids": ["b13"],
            "parameters": {},
        },
    ]
    system = {
        "id": "sys",
        "name": "sys",
        "contract_version": 1,
        "meshes": [mesh],
        "regions": [
            {
                "id": "air",
                "name": "air",
                "kind": "unbounded_air",
                "mesh_ids": ["mesh"],
                "volume_groups": [],
                "sound_speed_m_per_s": 343.0,
                "density_kg_per_m3": 1.21,
                "loss_model": {},
            }
        ],
        "boundaries": boundaries,
        "interfaces": [],
        "components": components,
        "excitation_ports": [
            {"id": "pa", "name": "pa", "component_id": "a", "kind": "normal_velocity"},
            {"id": "pb", "name": "pb", "component_id": "b", "kind": "normal_velocity"},
        ],
    }
    return {
        "schema_version": 1,
        "compiled_system": system,
        "frequencies_hz": [200.0, 80.0, 120.0],
        "excitation_port_ids": ["pb", "pa"],
        "outputs": [
            {
                "id": "near",
                "quantity": "exterior_pressure",
                "target_ids": [],
                "options": {"points_m": [[0, 0, 2]]},
            },
            {
                "id": "far",
                "quantity": "exterior_pressure",
                "target_ids": [],
                "options": {"points_m": [[2, 0, 0], [0, 2, 0]]},
            },
        ],
        "solver_options": {
            "bem_backend": "metal" if adapter._IS_METAL else "cpu",
            "precision": "float32" if adapter._IS_METAL else "float64",
            "phasor_convention": "exp(-i omega t)",
            "symmetry": "off",
        },
    }


def test_units_identity_order_and_separate_basis(monkeypatch):
    import meshio

    wire = request()
    saved = copy.deepcopy(wire)
    calls = []

    def solve(path, freqs, cfg):
        assert path.mesh_kwargs == {
            "validate": True,
            "merge_tol": 0.0,
            "repair_normals": False,
            **({} if adapter._IS_METAL else {"require_closed": True}),
        }
        m = meshio.read(path.staged_path)
        assert np.allclose(
            m.points,
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
            * 0.001
            + [0.2, 0.3, 0.4],
        )
        assert cfg.air_density == 1.21
        assert cfg.velocity_mode == api.VelocityMode.VELOCITY
        assert cfg.source_axes is None
        assert np.array_equal(
            cfg.observation.custom_points["compiled"], [[0, 0, 2], [2, 0, 0], [0, 2, 0]]
        )
        calls.append(cfg.velocity_sources)
        n = len(calls)
        return SimpleNamespace(
            frequencies_hz=freqs,
            pressure_complex=np.full((3, 1, 3), n + 1j * n),
            solver_log=["retained"],
        )

    monkeypatch.setattr(api, "solve_frequencies", solve)
    result = api.solve_compiled(wire)
    assert calls == [{11: 0.0, 12: 0.0, 13: 1.0}, {11: 1.0, 12: 1.0, 13: 0.0}]
    assert result.excitation_port_ids == ("pb", "pa")
    assert np.array_equal(result.frequencies_hz, [200, 80, 120])
    assert result.pressure["near"].shape == (3, 2, 1)
    assert result.pressure["far"].shape == (3, 2, 2)
    assert np.all(result.pressure["far"][:, 0, :] == 1 + 1j)
    assert result.native_results[0].solver_log == ["retained"]
    assert result.time_convention == "exp(-i omega t)"
    assert wire == saved


def test_axes_and_velocity_basis_do_not_depend_on_observation_frame(monkeypatch):
    wire = request()
    wire["compiled_system"]["contract_version"] = 2
    for c in wire["compiled_system"]["components"]:
        c["parameters"] = {
            "motion_profile": "rigid_translation",
            "motion_axis": [2, 0, 0] if c["id"] == "a" else [0, -3, 0],
        }

    def solve(path, freqs, cfg):
        assert cfg.source_motion == "axial"
        assert cfg.source_axes == {
            11: (1.0, 0.0, 0.0),
            12: (1.0, 0.0, 0.0),
            13: (0.0, -1.0, 0.0),
        }
        return SimpleNamespace(
            frequencies_hz=freqs, pressure_complex=np.zeros((3, 1, 3), complex)
        )

    monkeypatch.setattr(api, "solve_frequencies", solve)
    api.solve_compiled(wire)


@pytest.mark.parametrize(
    "change",
    [
        lambda w: w.update(schema_version=True),
        lambda w: w["compiled_system"].update(contract_version=99),
        lambda w: w["solver_options"].update(symmetry="ground"),
        lambda w: w["solver_options"].update(phasor_convention="exp(+i omega t)"),
        lambda w: w["solver_options"].update(unknown_physics=True),
        lambda w: w["solver_options"].update(regular_quadrature_mode="wavelength"),
        lambda w: w["compiled_system"]["components"][0].update(
            kind="electrodynamic_transducer"
        ),
        lambda w: w["compiled_system"]["components"][0]["parameters"].update(mass=1),
        lambda w: w["compiled_system"]["boundaries"][0]["parameters"].update(
            admittance=1
        ),
        lambda w: w["compiled_system"]["regions"][0]["loss_model"].update(
            kind="viscous"
        ),
        lambda w: w["compiled_system"]["boundaries"][0].update(kind="impedance"),
        lambda w: w["outputs"][0].update(quantity="radiation_impedance"),
        lambda w: w["outputs"][0]["options"].update(domain="interior"),
        lambda w: w["outputs"][0]["options"].update(extra=True),
        lambda w: w["outputs"][0]["options"].update(points_m=[[0, float("nan"), 1]]),
        lambda w: w.update(frequencies_hz=[80, 80]),
        lambda w: w["solver_options"].update(quadrature_order=True),
        lambda w: w["compiled_system"]["boundaries"][1]["group"].update(tag=11),
        lambda w: w["compiled_system"]["components"][1].update(boundary_ids=["b11"]),
        lambda w: w["compiled_system"]["excitation_ports"][0].update(kind="voltage"),
    ],
)
def test_unsupported_or_malformed_physics_refused_before_mesh_io(monkeypatch, change):
    w = request()
    change(w)

    def no_read(*args):
        pytest.fail("unsupported request reached mesh I/O")

    monkeypatch.setattr(adapter, "_mesh_file", no_read)
    with pytest.raises((ValueError, KeyError)):
        api.solve_compiled(w)


@pytest.mark.parametrize(
    "defect",
    ["missing-tag", "open", "inward", "non-manifold", "zero-area", "infinite-baffle"],
)
def test_bad_mesh_never_reaches_native_solve(monkeypatch, defect):
    w = request()
    d = w["compiled_system"]["meshes"][0]["mesh_data"]
    faces = adapter._packed_array(d["cells"][0]["connectivity"])
    tags = adapter._packed_array(d["cells"][0]["physical_tags"])
    if defect == "missing-tag":
        tags[0] = 99
        d["physical_names"]["99"] = [99, 2]
    elif defect == "open":
        faces = faces[:3]
        tags = tags[:3]
    elif defect == "inward":
        faces = faces[:, [0, 2, 1]]
    elif defect == "non-manifold":
        faces = np.vstack([faces, faces[0]])
        tags = np.append(tags, tags[0])
    elif defect == "zero-area":
        faces[0] = [0, 0, 1]
    else:
        d["physical_names"]["mouth_aperture"] = d["physical_names"].pop("11")
    d["cells"][0]["connectivity"] = packed(faces, "<i8")
    d["cells"][0]["physical_tags"] = packed(tags, "<i8")
    monkeypatch.setattr(
        api, "solve_frequencies", lambda *args: pytest.fail("bad mesh submitted")
    )
    with pytest.raises(ValueError):
        api.solve_compiled(w)


def test_file_mesh_scale_translation_and_confinement(monkeypatch, tmp_path):
    import meshio

    w = request()
    spec = w["compiled_system"]["meshes"][0]
    d = spec.pop("mesh_data")
    faces = adapter._packed_array(d["cells"][0]["connectivity"])
    tags = adapter._packed_array(d["cells"][0]["physical_tags"])
    meshio.write(
        tmp_path / "source.msh",
        meshio.Mesh(
            adapter._packed_array(d["points"]),
            [("triangle", faces)],
            cell_data={"gmsh:physical": [tags], "gmsh:geometrical": [tags]},
        ),
        file_format="gmsh22",
        binary=False,
    )
    spec["file"] = "source.msh"

    def solve(path, freqs, cfg):
        assert np.allclose(meshio.read(path.staged_path).points[0], [0.2, 0.3, 0.4])
        return SimpleNamespace(
            frequencies_hz=freqs, pressure_complex=np.zeros((3, 1, 3), complex)
        )

    monkeypatch.setattr(api, "solve_frequencies", solve)
    api.solve_compiled(w, base_dir=tmp_path)
    spec["file"] = "../escape.msh"
    with pytest.raises(ValueError, match="inside base_dir"):
        api.solve_compiled(w, base_dir=tmp_path)


def test_cancel_before_mesh_and_between_frequencies(monkeypatch, tmp_path):
    w = request()
    marker = tmp_path / "cancel"
    w["cancel_path"] = str(marker)
    marker.touch()
    monkeypatch.setattr(
        api, "solve_frequencies", lambda *a: pytest.fail("cancelled solve submitted")
    )
    with pytest.raises(InterruptedError):
        api.solve_compiled(w)
    marker.unlink()

    def solve(path, freqs, cfg):
        marker.touch()
        cfg.on_frequency_result(0, freqs[0], {})
        pytest.fail("cancellation did not propagate")

    monkeypatch.setattr(api, "solve_frequencies", solve)
    with pytest.raises(InterruptedError):
        api.solve_compiled(w)
    marker.unlink()
    with pytest.raises(InterruptedError):
        api.solve_compiled(
            w, cancel_cb=lambda: (_ for _ in ()).throw(InterruptedError())
        )


@pytest.mark.parametrize("defect", ["partial", "reordered", "nan", "wrong-shape"])
def test_invalid_native_result_rejected(monkeypatch, defect):
    w = request()

    def solve(path, freqs, cfg):
        p = np.zeros((3, 1, 3), complex)
        if defect == "partial":
            freqs = freqs[:1]
            p = p[:1]
        if defect == "reordered":
            freqs = freqs[::-1]
        if defect == "nan":
            p[0, 0, 0] = np.nan
        if defect == "wrong-shape":
            p = p[:, :, 0]
        return SimpleNamespace(frequencies_hz=freqs, pressure_complex=p)

    monkeypatch.setattr(api, "solve_frequencies", solve)
    with pytest.raises(RuntimeError):
        api.solve_compiled(w)


def test_capabilities_and_backend_precision_guards():
    c = api.compiled_capabilities()
    assert c["components"] == ["ideal_velocity_source"]
    assert c["outputs"] == ["exterior_pressure"]
    w = request()
    w["solver_options"]["bem_backend"] = "cpu" if adapter._IS_METAL else "metal"
    with pytest.raises(ValueError, match="bem_backend"):
        adapter._plan(w)
    if adapter._IS_METAL:
        w = request()
        w["solver_options"]["precision"] = "float64"
        with pytest.raises(ValueError, match="precision"):
            adapter._plan(w)
    else:
        w = request()
        w["compiled_system"]["regions"][0]["sound_speed_m_per_s"] = 340
        with pytest.raises(ValueError, match="343"):
            adapter._plan(w)


def test_frozen_official_request_is_accepted(monkeypatch, tmp_path):
    import json

    import meshio

    w = json.loads(
        (Path(__file__).parent / "fixtures/beat-official-exterior.json").read_text()
    )
    w["solver_options"]["bem_backend"] = "metal" if adapter._IS_METAL else "cpu"
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    tags = np.full(4, 2)
    meshio.write(
        tmp_path / "exterior.msh",
        meshio.Mesh(
            points,
            [("triangle", faces)],
            cell_data={"gmsh:physical": [tags], "gmsh:geometrical": [tags]},
        ),
        file_format="gmsh22",
        binary=False,
    )

    def solve(loaded, freqs, cfg):
        assert np.allclose(
            meshio.read(loaded.staged_path).points, np.array(points) * 0.001
        )
        assert cfg.velocity_sources == {2: 1.0}
        return SimpleNamespace(
            frequencies_hz=freqs, pressure_complex=np.ones((2, 1, 1), complex)
        )

    monkeypatch.setattr(api, "solve_frequencies", solve)
    result = api.solve_compiled(w, base_dir=tmp_path)
    assert result.pressure["pressure"].shape == (2, 1, 1)
    np.testing.assert_array_equal(result.frequencies_hz, [500, 250])


def test_explicit_uniform_normal_v2_and_extreme_axis():
    w = request()
    w["compiled_system"]["contract_version"] = 2
    for c in w["compiled_system"]["components"]:
        c["parameters"] = {"motion_profile": "uniform_normal"}
    assert adapter._plan(w)["kwargs"]["source_motion"] == "normal"
    for c in w["compiled_system"]["components"]:
        c["parameters"] = {
            "motion_profile": "rigid_translation",
            "motion_axis": [1e300, 0, 0],
        }
    assert adapter._plan(w)["kwargs"]["source_axes"][11] == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "change",
    [
        lambda w: w["outputs"][0].update(quantity="pressure"),
        lambda w: w["outputs"][0].update(options={"points": [[0, 0, 2]]}),
        lambda w: w["solver_options"].update(symmetry="full"),
        lambda w: w["compiled_system"]["components"][0]["parameters"].update(
            motion_profile="normal"
        ),
        lambda w: w["outputs"][0]["options"].update(points_m=[[False, 0.0, 2.0]]),
        lambda w: w["outputs"][0]["options"].update(points_m=[["0", 0, 2]]),
    ],
)
def test_noncanonical_or_coerced_fields_are_refused(change):
    w = request()
    change(w)
    with pytest.raises(ValueError):
        adapter._plan(w)


def test_duplicate_copies_are_backend_neutral():
    # No backend identity is embedded in a fixture beyond solver_options.
    w = request()
    w["solver_options"].pop("bem_backend")
    w["solver_options"].pop("precision")
    assert adapter._plan(w)["frequencies"].tolist() == [200.0, 80.0, 120.0]


def test_legacy_import_does_not_require_beat():
    import subprocess
    import sys

    code = """
import importlib.abc, sys
class NoBeat(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'beat_engine' or fullname.startswith('beat_engine.'):
            raise ImportError('BEAT deliberately absent')
sys.meta_path.insert(0, NoBeat())
import PACKAGE as legacy
legacy.SolveConfig()
legacy.capabilities()
legacy.compiled_capabilities()
assert not any(n == 'beat_engine' or n.startswith('beat_engine.') for n in sys.modules)
try:
    legacy.solve_compiled({})
except RuntimeError as error:
    assert 'requires official beat-engine' in str(error)
else:
    raise AssertionError('new API did not report missing optional validator')
""".replace("PACKAGE", _PACKAGE)
    subprocess.run([sys.executable, "-c", code], check=True)

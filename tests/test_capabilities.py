"""Keep the advertised request contract aligned with the public solver API."""

from dataclasses import fields
from importlib import import_module, metadata
import inspect
import json

import pytest

import hornlab_metal_bem as metal
from hornlab_metal_bem.config import SolveConfig


def test_request_fields_match_solve_config_exactly():
    declared = metal.capabilities()["request_fields"]
    expected = {field.name for field in fields(SolveConfig) if field.init}
    assert set(declared) == expected
    assert len(declared) == len(expected)
    assert set(declared) == set(inspect.signature(SolveConfig).parameters)


def test_versioned_json_report_and_public_exports():
    report = metal.capabilities()
    assert report["schema_version"] == metal.CAPABILITY_SCHEMA_VERSION == 1
    assert report["request_schema_version"] == metal.REQUEST_SCHEMA_VERSION == 1
    assert report["package"] == "hornlab-metal-bem"
    assert report["package_version"] == metadata.version("hornlab-metal-bem")
    assert json.loads(json.dumps(report)) == report
    assert {"capabilities", "CAPABILITY_SCHEMA_VERSION", "REQUEST_SCHEMA_VERSION"} <= set(metal.__all__)
    assert report["conventions"] == {
        "time_convention": "exp(-i*omega*t)",
        "outgoing_wave": "exp(+i*k*r)",
        "neumann_coefficient": "q = +i*rho*omega*v_n",
    }


def test_uninstalled_package_version_is_explicit(monkeypatch):
    module = import_module("hornlab_metal_bem.capabilities")

    def missing_version(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(module.metadata, "version", missing_version)
    assert metal.capabilities()["package_version"] == "unknown"


def test_report_returns_independent_containers():
    first = metal.capabilities()
    original = metal.capabilities()
    first["request_fields"].clear()
    first["features"]["source_motion"].append("invalid")
    first["features"]["ground_plane"]["planes"].clear()
    first["conventions"].clear()
    assert metal.capabilities() == original


def test_wg_optional_keyword_probes_are_declared():
    report = metal.capabilities()
    features = report["features"]
    for name in (
        "source_axes", "complex_k_shift", "frame_override",
        "on_frequency_result", "return_surface_traces",
        "require_closed_mesh", "workers",
    ):
        assert features[name] is (name in report["request_fields"])
    assert features["require_closed_mesh"] is False
    assert features["workers"] is False
    assert {
        "source_motion", "source_axes", "aperture_tag", "formulation",
        "complex_k_shift", "frame_override", "ground_plane",
        "native_symmetry_plane", "on_frequency_result", "return_surface_traces",
    } <= set(report["request_fields"])


@pytest.mark.parametrize("field", ["source_motion", "formulation", "native_symmetry_plane"])
def test_advertised_modes_match_config_validation(field):
    values = metal.capabilities()["features"][field]
    for value in values:
        config = SolveConfig(**{field: value})
        assert getattr(config, field) == value
    with pytest.raises(ValueError):
        SolveConfig(**{field: "invalid"})
    if field == "source_motion":
        assert set(values) == {"normal", "axial"}
    elif field == "formulation":
        assert set(values) == {"standard", "complex_k", "burton_miller"}
    else:
        assert set(values) == {None, "xy", "yz", "xz", "yz+xz"}


@pytest.mark.parametrize("plane", ["xy", "yz", "xz"])
def test_ground_plane_and_composition_match_config(plane):
    feature = metal.capabilities()["features"]["ground_plane"]
    assert feature["supported"] is True
    assert set(feature["planes"]) == {"xy", "yz", "xz"}
    for formulation in feature["formulations"]:
        SolveConfig(ground_plane=plane, formulation=formulation)
    assert set(feature["formulations"]) == {"standard", "complex_k"}
    with pytest.raises(ValueError):
        SolveConfig(ground_plane=plane, formulation="burton_miller")
    assert feature["composes_with_symmetry"] is False
    for symmetry in metal.capabilities()["features"]["native_symmetry_plane"]:
        if symmetry is not None:
            with pytest.raises(ValueError):
                SolveConfig(ground_plane=plane, native_symmetry_plane=symmetry)
    assert feature["composes_with_infinite_baffle"] is False
    with pytest.raises(ValueError):
        SolveConfig(ground_plane=plane, aperture_tag=12)


def test_infinite_baffle_formulations_match_config():
    feature = metal.capabilities()["features"]["infinite_baffle"]
    assert feature["supported"] is True
    assert feature["field"] == "aperture_tag"
    assert set(feature["formulations"]) == {"standard", "complex_k"}
    for formulation in feature["formulations"]:
        SolveConfig(aperture_tag=12, formulation=formulation)
    with pytest.raises(ValueError):
        SolveConfig(aperture_tag=12, formulation="burton_miller")

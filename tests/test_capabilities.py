"""Capability declarations exercised against the real config and solver guards."""

from dataclasses import fields
from importlib import import_module
import json
import subprocess
import sys

import pytest

import hornlab_metal_bem as package
from hornlab_metal_bem.config import SolveConfig

capabilities = package.capabilities


def test_schema_exports_and_request_field_coverage():
    report = capabilities()
    assert report["schema"] == "hornlab-bem-capabilities"
    assert report["schema_version"] == package.CAPABILITY_SCHEMA_VERSION == 1
    assert report["request_schema_version"] == package.REQUEST_SCHEMA_VERSION == 1
    assert report["package"] == "hornlab-metal-bem"
    assert json.loads(json.dumps(report)) == report
    actual = {item.name for item in fields(SolveConfig) if item.init}
    assert set(report["request_fields"]) == actual
    assert len(report["request_fields"]) == len(actual)
    assert {
        "capabilities",
        "CAPABILITY_SCHEMA_VERSION",
        "REQUEST_SCHEMA_VERSION",
    } <= set(package.__all__)
    assert report["conventions"]["time_convention"] == "exp(-i*omega*t)"


def test_distribution_metadata_and_missing_metadata(monkeypatch):
    module = import_module("hornlab_metal_bem.capabilities")
    calls = []

    def version(name):
        calls.append(name)
        return "1.2.3"

    monkeypatch.setattr(module, "version", version)
    assert capabilities()["package_version"] == "1.2.3"
    assert calls == ["hornlab-metal-bem"]

    def missing(name):
        raise module.PackageNotFoundError(name)

    monkeypatch.setattr(module, "version", missing)
    assert capabilities()["package_version"] is None


def test_report_is_a_fresh_snapshot():
    original = capabilities()
    changed = capabilities()
    changed["request_fields"].clear()
    changed["features"]["source_motion"]["values"].clear()
    changed["features"]["ground_plane"]["requires"]["formulation"].clear()
    changed["features"]["infinite_baffle"]["refuses"][0].clear()
    assert capabilities() == original


def test_handshake_does_not_load_numerical_backends():
    script = """
import sys
class ForbidBackends:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'bempp_cl', 'pyopencl'}:
            raise AssertionError('handshake loaded numerical backend')
sys.meta_path.insert(0, ForbidBackends())
from hornlab_metal_bem import capabilities
assert capabilities()['features']['ground_plane']['supported']
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


@pytest.mark.parametrize(
    "feature,field",
    [
        ("source_motion", "source_motion"),
        ("formulation", "formulation"),
        ("native_symmetry", "native_symmetry_plane"),
        ("assembly_backend", "metal_native_assembly_mode"),
    ],
)
def test_enumerated_choices_pass_config_and_guard(feature, field):
    for value in capabilities()["features"][feature]["values"]:
        config = SolveConfig(**{field: value})
        image_guard(config)
    with pytest.raises((ValueError, NotImplementedError)):
        config = SolveConfig(**{field: "invalid"})
        image_guard(config)


def test_ground_compositions_and_formulations_match_actual_guards():
    detail = capabilities()["features"]["ground_plane"]
    compositions = (
        detail["composes_with"][0]["values"] if detail["composes_with"] else []
    )
    for ground in detail["values"]:
        for symmetry in [
            None,
            *capabilities()["features"]["native_symmetry"]["values"],
        ]:
            for formulation in capabilities()["features"]["formulation"]["values"]:
                try:
                    image_guard(
                        SolveConfig(
                            ground_plane=ground,
                            native_symmetry_plane=symmetry,
                            formulation=formulation,
                        )
                    )
                except (ValueError, NotImplementedError):
                    allowed = False
                else:
                    allowed = True
                expected = formulation in detail["requires"]["formulation"] and (
                    symmetry is None
                    or {"native_symmetry_plane": symmetry, "ground_plane": ground}
                    in compositions
                )
                assert allowed == expected


def image_guard(config):
    pass  # Metal composition guards are in SolveConfig.__post_init__.


def trace_guard(config):
    from hornlab_metal_bem.sweep import run_sweep_native_metal

    run_sweep_native_metal(None, [100.0], None, config)


def aperture_guard(config):
    from hornlab_metal_bem.metal.geometry import (
        validate_native_infinite_baffle_aperture,
    )

    validate_native_infinite_baffle_aperture(
        None, config.aperture_tag, symmetry_plane=config.native_symmetry_plane
    )


def case(
    feature, identifier, related, kwargs, guard=image_guard, error=ValueError, match=""
):
    return (feature, identifier, related, kwargs, guard, error, match)


CASES = [
    case(
        "infinite_baffle",
        "ib_burton_miller",
        ["formulation"],
        dict(aperture_tag=3, formulation="burton_miller"),
        match="infinite-baffle",
    ),
    case(
        "infinite_baffle",
        "ib_surface_traces",
        ["return_surface_traces"],
        dict(aperture_tag=3, return_surface_traces=True),
        trace_guard,
        match="return_surface_traces",
    ),
    case(
        "return_surface_traces",
        "traces_infinite_baffle",
        ["aperture_tag"],
        dict(aperture_tag=3, return_surface_traces=True),
        trace_guard,
        match="return_surface_traces",
    ),
    case(
        "infinite_baffle",
        "ib_xy_symmetry",
        ["native_symmetry_plane"],
        dict(aperture_tag=3, native_symmetry_plane="xy"),
        aperture_guard,
        match="does not compose",
    ),
    *[
        case(
            "infinite_baffle",
            "ib_assembly",
            ["metal_native_assembly_mode"],
            dict(aperture_tag=3, metal_native_assembly_mode=mode),
            match="requires.*corrected",
        )
        for mode in ("optimized", "reference", "parity")
    ],
    case(
        "infinite_baffle",
        "ib_ground",
        ["ground_plane"],
        dict(aperture_tag=3, ground_plane="xy"),
        match="ground_plane does not compose",
    ),
    case(
        "formulation",
        "bm_infinite_baffle",
        ["aperture_tag"],
        dict(aperture_tag=3, formulation="burton_miller"),
        match="infinite-baffle",
    ),
    case(
        "formulation",
        "bm_robin",
        ["impedance_sources"],
        dict(formulation="burton_miller", impedance_sources={1: 0.1}),
        match="Robin/impedance",
    ),
    case(
        "formulation",
        "bm_robin_callback",
        ["impedance_source_callback"],
        dict(formulation="burton_miller", impedance_source_callback=lambda f: {1: 0.1}),
        match="Robin/impedance",
    ),
    case(
        "formulation",
        "bm_chief",
        ["chief_points"],
        dict(formulation="burton_miller", chief_points=[[0, 0, 0]]),
        match="CHIEF",
    ),
    *[
        case(
            "formulation",
            "bm_ground",
            ["ground_plane"],
            dict(formulation="burton_miller", ground_plane=plane),
            match="supports exterior",
        )
        for plane in ("xy", "yz", "xz")
    ],
    *[
        case(
            "ground_plane",
            "ground_burton_miller",
            ["formulation"],
            dict(formulation="burton_miller", ground_plane=plane),
            match="supports exterior",
        )
        for plane in ("xy", "yz", "xz")
    ],
    *[
        case(
            "ground_plane",
            "ground_symmetry",
            ["native_symmetry_plane"],
            dict(ground_plane=ground, native_symmetry_plane=symmetry),
            match="does not yet compose",
        )
        for ground in ("xy", "yz", "xz")
        for symmetry in ("xy", "yz", "xz", "yz+xz")
    ],
    *[
        case(
            "ground_plane",
            "ground_infinite_baffle",
            ["aperture_tag"],
            dict(ground_plane=plane, aperture_tag=3),
            match="does not compose",
        )
        for plane in ("xy", "yz", "xz")
    ],
]


def test_infinite_baffle_requires_corrected_assembly_and_lists_transverse_compositions():
    feature = capabilities()["features"]["infinite_baffle"]
    assert feature["requires"]["metal_native_assembly_mode"] == ["corrected"]
    assert feature["requires"]["formulation"] == ["standard", "complex_k"]
    assert feature["composes_with"] == [
        {"feature": "native_symmetry", "values": ["yz", "xz", "yz+xz"]}
    ]
    for formulation in feature["requires"]["formulation"]:
        SolveConfig(
            aperture_tag=3,
            formulation=formulation,
            metal_native_assembly_mode="corrected",
        )


# Each declaration must have a test case and vice versa; removing a refusal or
# adding an untested declaration fails coverage. Multiple cases exercise every
# enum arm where a refusal applies to several modes.
def test_every_declared_refusal_has_guard_cases():
    declared = {
        (name, refusal["id"]): refusal["request_fields"]
        for name, feature in capabilities()["features"].items()
        for refusal in feature["refuses"]
    }
    expected = {(name, identifier): related for name, identifier, related, *_ in CASES}
    assert declared == expected


@pytest.mark.parametrize("feature,identifier,related,kwargs,guard,error,match", CASES)
def test_declared_refusal_reaches_actual_guard(
    feature, identifier, related, kwargs, guard, error, match
):
    declaration = next(
        item
        for item in capabilities()["features"][feature]["refuses"]
        if item["id"] == identifier
    )
    assert declaration["request_fields"] == related
    with pytest.raises(error, match=match):
        config = SolveConfig(**kwargs)
        guard(config)


@pytest.mark.parametrize("plane", [None, "yz", "xz", "yz+xz"])
@pytest.mark.parametrize("formulation", ["standard", "complex_k"])
def test_infinite_baffle_compositions_pass_reduced_geometry_guard(plane, formulation):
    import numpy as np
    from test_metal_geometry import _topology_buffers
    from hornlab_metal_bem.metal.geometry import (
        validate_native_infinite_baffle_aperture,
    )

    feature = capabilities()["features"]["infinite_baffle"]
    if plane is not None:
        assert plane in feature["composes_with"][0]["values"]
    config = SolveConfig(
        aperture_tag=7, native_symmetry_plane=plane, formulation=formulation
    )
    # A tetrahedral cavity, open only on the requested cut plane(s), with its
    # mouth in z=0 and outward aperture normal -z.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1]])
    triangles = [[0, 2, 1], [1, 2, 3]]
    if plane not in ("yz", "yz+xz"):
        triangles.append([2, 0, 3])
    if plane not in ("xz", "yz+xz"):
        triangles.append([0, 1, 3])
    buffers = _topology_buffers(
        vertices, np.array(triangles), np.array([7, *([1] * (len(triangles) - 1))])
    )
    assert (
        validate_native_infinite_baffle_aperture(
            buffers, config.aperture_tag, symmetry_plane=config.native_symmetry_plane
        )
        == 7
    )

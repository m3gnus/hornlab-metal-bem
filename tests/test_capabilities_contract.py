"""Shared contract test; this copy must match the other BEM repository's."""

from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = next(
    name
    for name in ("hornlab_metal_bem", "hornlab_bempp_bem")
    if (ROOT / name).is_dir()
)
capabilities = importlib.import_module(PACKAGE).capabilities
SCHEMA = json.loads((ROOT / "docs" / "capabilities-schema.json").read_text())


def validate(value, schema, root=SCHEMA):
    """Validate the JSON Schema subset used here, without a test dependency.

    Unknown keywords fail closed so extending the schema cannot silently weaken
    validation. This is intentionally not a general-purpose schema validator.
    """
    known = {
        "$schema",
        "$comment",
        "$defs",
        "$ref",
        "type",
        "const",
        "enum",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "uniqueItems",
    }
    assert not set(schema) - known, "unsupported JSON Schema keyword"
    if "$ref" in schema:
        assert schema["$ref"].startswith("#/$defs/")
        return validate(value, root["$defs"][schema["$ref"].split("/")[-1]], root)
    if "const" in schema:
        assert type(value) is type(schema["const"]) and value == schema["const"]
    if "enum" in schema:
        assert value in schema["enum"]
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "null": type(None),
        "boolean": bool,
    }
    if "type" in schema:
        allowed = (
            schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        )
        assert type(value) in [types[name] for name in allowed]
    if isinstance(value, dict):
        props = schema.get("properties", {})
        assert set(schema.get("required", [])) <= set(value)
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in props:
                validate(item, props[key], root)
            else:
                assert extra is not False, f"unexpected key: {key}"
                if isinstance(extra, dict):
                    validate(item, extra, root)
    if isinstance(value, list):
        if schema.get("uniqueItems"):
            assert len({json.dumps(item, sort_keys=True) for item in value}) == len(
                value
            )
        for item in value:
            validate(item, schema.get("items", {}), root)


def test_report_matches_shared_json_schema():
    validate(json.loads(json.dumps(capabilities())), SCHEMA)
    for feature in capabilities()["features"].values():
        assert feature["supported"] == all(
            field in capabilities()["request_fields"]
            for field in feature["request_fields"]
        )
        ids = [refusal["id"] for refusal in feature["refuses"]]
        assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "mutation",
    [
        "list_feature",
        "bool_feature",
        "old_field",
        "missing_feature",
        "beat_identity",
        "wrong_time",
        "int_supported",
    ],
)
def test_schema_rejects_incompatible_reports(mutation):
    report = deepcopy(capabilities())
    if mutation == "list_feature":
        report["features"]["source_motion"] = ["normal", "axial"]
    elif mutation == "bool_feature":
        report["features"]["source_axes"] = True
    elif mutation == "old_field":
        report["features"]["infinite_baffle"]["field"] = "aperture_tag"
    elif mutation == "missing_feature":
        del report["features"]["native_symmetry"]
    elif mutation == "beat_identity":
        report["schema"] = "beat-provider-v2"
    elif mutation == "wrong_time":
        report["conventions"]["time_convention"] = "exp(+i*omega*t)"
    else:
        report["features"]["source_axes"]["supported"] = 1
    with pytest.raises(AssertionError):
        validate(report, SCHEMA)


def test_validator_rejects_unknown_keywords():
    with pytest.raises(AssertionError, match="unsupported JSON Schema keyword"):
        validate("anything", {"minLength": 1})

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_apple_silicon_package_builds_and_ships_release_helper() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    package_text = (
        ROOT / "hornlab_metal_bem" / "metal" / "native_helper" / "Package.swift"
    ).read_text(encoding="utf-8")
    helper_text = (
        ROOT
        / "hornlab_metal_bem"
        / "metal"
        / "native_helper"
        / "Sources"
        / "HornlabMetalBemNative"
        / "main.swift"
    ).read_text(encoding="utf-8")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["hornlab_metal_bem"]

    assert "swift, \"build\", \"-c\", \"release\"" in setup_text
    assert "metal/native_helper/.build/release/HornlabMetalBemNative" in package_data
    assert '.process("Resources")' not in package_text
    assert "return Bundle.module" not in helper_text
    assert "../../Sources/HornlabMetalBemNative/Resources/regular_assembly.metallib" in helper_text

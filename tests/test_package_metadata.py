import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_apple_silicon_package_builds_and_ships_release_helper() -> None:
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["hornlab_metal_bem"]

    assert "swift, \"build\", \"-c\", \"release\"" in setup_text
    assert "metal/native_helper/.build/release/HornlabMetalBemNative" in package_data

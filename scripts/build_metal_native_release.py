"""Build the HornLab Metal BEM native helper and precompiled Metal library."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "hornlab_metal_bem" / "metal" / "native_helper"
MAIN_SWIFT = PACKAGE_DIR / "Sources" / "HornlabMetalBemNative" / "main.swift"
RESOURCE_DIR = PACKAGE_DIR / "Sources" / "HornlabMetalBemNative" / "Resources"
METALLIB = RESOURCE_DIR / "regular_assembly.metallib"
HELPER_BUILD_INPUTS = (MAIN_SWIFT, PACKAGE_DIR / "Package.swift")


def _payload(**values: Any) -> dict[str, Any]:
    return {
        "available": False,
        "built": False,
        "metallibBuilt": False,
        "skipped": False,
        "reason": None,
        "helperPath": None,
        "helperSource": None,
        "helperBuild": None,
        "metallibPath": str(METALLIB),
        **values,
    }


def _print(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if payload.get("built"):
        print(f"Metal native release helper ready: {payload.get('helperPath')}")
        if payload.get("reason"):
            print(f"Metal library not precompiled: {payload.get('reason')}")
        else:
            print(f"Metal library ready: {payload.get('metallibPath')}")
    elif payload.get("available"):
        print(f"Metal native helper already ready: {payload.get('helperPath')}")
        if payload.get("reason"):
            print(f"Metal library not precompiled: {payload.get('reason')}")
        else:
            print(f"Metal library already ready: {payload.get('metallibPath')}")
    elif payload.get("skipped"):
        print(f"Metal native release helper skipped: {payload.get('reason')}")
    else:
        print(f"Metal native release helper unavailable: {payload.get('reason')}")


def _helper_build(path: Path | None) -> str | None:
    if path is None:
        return None
    parts = path.parts
    if "release" in parts:
        return "release"
    if "debug" in parts:
        return "debug"
    return "custom"


def _status_payload(
    status: Any,
    *,
    built: bool = False,
    metallib_built: bool = False,
) -> dict[str, Any]:
    helper_path = getattr(status, "helper_executable_path", None)
    helper = Path(helper_path) if helper_path else None
    return _payload(
        available=bool(getattr(status, "available", False)),
        built=built,
        metallibBuilt=metallib_built,
        helperPath=str(helper) if helper else None,
        helperSource=getattr(status, "helper_source", None),
        helperBuild=_helper_build(helper),
    )


def _embedded_metal_source() -> str:
    text = MAIN_SWIFT.read_text(encoding="utf-8")
    marker = 'let regularAssemblyMetalSource = """\n'
    start = text.index(marker) + len(marker)
    end = text.index('\n"""', start)
    return text[start:end] + "\n"


def _metallib_ready() -> bool:
    return METALLIB.is_file() and METALLIB.stat().st_mtime >= MAIN_SWIFT.stat().st_mtime


def _helper_current(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    helper_mtime = path.stat().st_mtime
    return all(helper_mtime >= input_path.stat().st_mtime for input_path in HELPER_BUILD_INPUTS)


def _metallib_error_message(exc: Exception) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        details = (exc.stderr or exc.stdout or "").strip()
        if details:
            return f"{exc}; {details.splitlines()[-1]}"
    return str(exc)


def _build_metallib() -> None:
    xcrun = shutil.which("xcrun")
    if not xcrun:
        raise RuntimeError("xcrun executable is unavailable.")
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hornlab-metal-bem-metallib-") as tmp:
        tmp_dir = Path(tmp)
        source_path = tmp_dir / "regular_assembly.metal"
        air_path = tmp_dir / "regular_assembly.air"
        source_path.write_text(_embedded_metal_source(), encoding="utf-8")
        subprocess.run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-O",
                "-c",
                str(source_path),
                "-o",
                str(air_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metallib",
                str(air_path),
                "-o",
                str(METALLIB),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def build_release_helper(
    *,
    json_output: bool = False,
    require_metallib: bool = False,
) -> int:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        _print(
            _payload(skipped=True, reason="Metal native helper is only used on Apple Silicon."),
            json_output=json_output,
        )
        return 0

    sys.path.insert(0, str(ROOT))
    try:
        from hornlab_metal_bem.metal.native import discover_native_runtime
    except Exception as exc:
        _print(
            _payload(reason=f"hornlab-metal-bem is unavailable: {exc}"),
            json_output=json_output,
        )
        return 1

    probe_error = None
    current: dict[str, Any] | None = None
    release_helper_current = False
    try:
        status = discover_native_runtime(run_smoke_test=True)
    except Exception as exc:
        probe_error = str(exc)
    else:
        current = _status_payload(status)
        helper_path = Path(current["helperPath"]) if current.get("helperPath") else None
        release_helper_current = current.get("helperBuild") == "release" and _helper_current(
            helper_path
        )
        if release_helper_current and _metallib_ready():
            _print(current, json_output=json_output)
            return 0

    if not (PACKAGE_DIR / "Package.swift").is_file():
        reason = f"native helper Package.swift missing under {PACKAGE_DIR}"
        if probe_error:
            reason = f"{reason}; initial native runtime probe failed: {probe_error}"
        _print(_payload(reason=reason), json_output=json_output)
        return 1

    swift = shutil.which("swift")
    if not swift:
        reason = "swift executable is unavailable."
        if probe_error:
            reason = f"{reason} Initial native runtime probe failed: {probe_error}"
        _print(_payload(reason=reason), json_output=json_output)
        return 1

    metallib_built = False
    metallib_error = None
    try:
        _build_metallib()
        metallib_built = True
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        metallib_error = (
            "Metal library build failed; helper will use source fallback: "
            f"{_metallib_error_message(exc)}"
        )
        if require_metallib:
            _print(
                _payload(reason=metallib_error),
                json_output=json_output,
            )
            return 1
        if release_helper_current and current is not None:
            current["reason"] = metallib_error
            _print(current, json_output=json_output)
            return 0

    try:
        subprocess.run([swift, "build", "-c", "release"], cwd=PACKAGE_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        _print(
            _payload(reason=f"swift release build failed with exit code {exc.returncode}"),
            json_output=json_output,
        )
        return int(exc.returncode) or 1

    try:
        next_status = discover_native_runtime(run_smoke_test=True)
    except Exception as exc:
        _print(
            _payload(reason=f"release build completed but native runtime probe failed: {exc}"),
            json_output=json_output,
        )
        return 1
    payload = _status_payload(
        next_status,
        built=True,
        metallib_built=metallib_built,
    )
    if payload.get("helperBuild") != "release":
        payload["reason"] = "release build completed but runtime did not select the release helper."
        _print(payload, json_output=json_output)
        return 1
    if metallib_error is not None:
        payload["reason"] = metallib_error

    _print(payload, json_output=json_output)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--require-metallib",
        action="store_true",
        help="Fail if xcrun metal/metallib cannot produce the precompiled library.",
    )
    args = parser.parse_args()
    return build_release_helper(
        json_output=bool(args.json),
        require_metallib=bool(args.require_metallib),
    )


if __name__ == "__main__":
    raise SystemExit(main())

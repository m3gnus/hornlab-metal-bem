from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from distutils import log
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).resolve().parent
NATIVE_PACKAGE_DIR = ROOT / "hornlab_metal_bem" / "metal" / "native_helper"


def _load_metallib_builder():
    script = ROOT / "scripts" / "build_metal_native_release.py"
    if not script.is_file():
        raise RuntimeError(f"Metal library build script is missing: {script}")
    spec = importlib.util.spec_from_file_location("_hornlab_metal_bem_release_build", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Metal library build script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_metallib


class build_py(_build_py):
    def run(self) -> None:
        self._build_native_release_helper()
        self._build_optional_metallib()
        super().run()

    def _build_native_release_helper(self) -> None:
        if os.environ.get("HORNLAB_METAL_BEM_SKIP_NATIVE_HELPER") == "1":
            return
        if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
            return
        swift = shutil.which("swift")
        if swift is None:
            raise RuntimeError(
                "Swift is required to build hornlab-metal-bem on Apple Silicon; "
                "install the Xcode command-line tools or set "
                "HORNLAB_METAL_BEM_SKIP_NATIVE_HELPER=1 for a source-only package."
            )
        try:
            subprocess.run(
                [swift, "build", "-c", "release"],
                cwd=NATIVE_PACKAGE_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
            # Wheel extraction gives packaged source files the wheel build
            # timestamp. Keep the verified helper at least as new so runtime
            # discovery does not incorrectly report the packaged binary as
            # stale.
            release_helper = NATIVE_PACKAGE_DIR / ".build" / "release" / "HornlabMetalBemNative"
            release_helper.touch()
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            message = details.splitlines()[-1] if details else str(exc)
            raise RuntimeError(f"failed to build HornLab Metal release helper: {message}") from exc

    def _build_optional_metallib(self) -> None:
        if os.environ.get("HORNLAB_METAL_BEM_SKIP_METALLIB") == "1":
            return
        if platform.system() != "Darwin":
            return
        try:
            _build_metallib = _load_metallib_builder()
        except Exception as exc:
            self.announce(f"Skipping Metal library precompile: {exc}", level=log.WARN)
            return
        try:
            _build_metallib()
        except (RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
            self.announce(
                f"Skipping Metal library precompile; helper will use source fallback: {exc}",
                level=log.WARN,
            )


setup(cmdclass={"build_py": build_py})

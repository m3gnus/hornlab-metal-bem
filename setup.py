from __future__ import annotations

import os
import platform
import subprocess
from distutils import log

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self) -> None:
        self._build_optional_metallib()
        super().run()

    def _build_optional_metallib(self) -> None:
        if os.environ.get("HORNLAB_METAL_BEM_SKIP_METALLIB") == "1":
            return
        if platform.system() != "Darwin":
            return
        try:
            from scripts.build_metal_native_release import _build_metallib
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

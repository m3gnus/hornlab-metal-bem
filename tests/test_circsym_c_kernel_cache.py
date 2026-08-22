from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys

import pytest

from hornlab_metal_bem.circsym import _circsym_c_kernel_cache_dir


ROOT = Path(__file__).resolve().parents[1]


def test_c_kernel_cache_uses_the_current_users_cache(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    expected_base = (
        tmp_path / "home" / "Library" / "Caches"
        if platform.system() == "Darwin"
        else tmp_path / "home" / ".cache"
    )
    assert Path(_circsym_c_kernel_cache_dir()) == (
        expected_base / "hornlab-metal-bem" / "circsym"
    )


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_concurrent_processes_compile_one_private_atomic_kernel(tmp_path: Path):
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        pytest.skip("runtime C compiler is unavailable")

    cache_home = tmp_path / "cache"
    invocation_log = tmp_path / "compiler-invocations"
    compiler_wrapper = tmp_path / "counting-cc"
    compiler_wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        "import time\n"
        f"log = {str(invocation_log)!r}\n"
        "fd = os.open(log, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)\n"
        "try:\n"
        "    os.write(fd, b'compile\\n')\n"
        "finally:\n"
        "    os.close(fd)\n"
        "time.sleep(0.2)\n"
        f"compiler = {compiler!r}\n"
        "os.execv(compiler, [compiler, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    compiler_wrapper.chmod(0o700)

    env = os.environ.copy()
    env.update(
        {
            "CC": str(compiler_wrapper),
            "XDG_CACHE_HOME": str(cache_home),
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(ROOT), env.get("PYTHONPATH", "")))
            ),
        }
    )
    probe = (
        "from hornlab_metal_bem.circsym import "
        "_load_circsym_remainder_c_kernel as load; "
        "assert load() is not None"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(6)
    ]
    failures = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=60.0)
        if process.returncode != 0:
            failures.append((process.returncode, stdout, stderr))
    assert failures == []

    assert invocation_log.read_text(encoding="utf-8").splitlines() == ["compile"]
    cache_dir = cache_home / "hornlab-metal-bem" / "circsym"
    assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
    libraries = [
        path
        for path in cache_dir.iterdir()
        if path.suffix in {".so", ".dylib"}
    ]
    assert len(libraries) == 1
    assert libraries[0].stat().st_size > 0
    assert not [path for path in cache_dir.iterdir() if ".tmp" in path.name]

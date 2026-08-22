from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys

import pytest

import hornlab_metal_bem.circsym as circsym
from hornlab_metal_bem.circsym import (
    _circsym_c_kernel_cache_dir,
    _circsym_c_kernel_cache_key,
    _prepare_circsym_c_kernel_cache,
)


ROOT = Path(__file__).resolve().parents[1]


def _counting_compiler(tmp_path: Path, *, delay_s: float = 0.0):
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        pytest.skip("runtime C compiler is unavailable")

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
        f"time.sleep({delay_s!r})\n"
        f"compiler = {compiler!r}\n"
        "os.execv(compiler, [compiler, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    compiler_wrapper.chmod(0o700)
    return compiler_wrapper, invocation_log


def _kernel_library_path(cache_dir: Path) -> Path:
    """Return the pre-generation canonical path used by old cache versions."""

    extension = ".dylib" if platform.system() == "Darwin" else ".so"
    return cache_dir / f"circsym_remainder_{_circsym_c_kernel_cache_key()}{extension}"


def _kernel_selection_path(cache_dir: Path) -> Path:
    return cache_dir / f"circsym_remainder_{_circsym_c_kernel_cache_key()}.current"


def _generation_library_path(cache_dir: Path, generation: str) -> Path:
    extension = ".dylib" if platform.system() == "Darwin" else ".so"
    return cache_dir / (
        f"circsym_remainder_{_circsym_c_kernel_cache_key()}.{generation}{extension}"
    )


def _selected_library_path(cache_dir: Path) -> Path:
    return cache_dir / _kernel_selection_path(cache_dir).read_text(
        encoding="ascii"
    ).strip()


def _compile_library_without_kernel_exports(output: Path, tmp_path: Path) -> None:
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        pytest.skip("runtime C compiler is unavailable")
    source = tmp_path / f"wrong-{output.stem}.c"
    source.write_text("int unrelated_export(void) { return 7; }\n", encoding="ascii")
    command = [compiler, "-O2", "-fPIC"]
    command.append("-dynamiclib" if platform.system() == "Darwin" else "-shared")
    command.extend([str(source), "-o", str(output)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    output.chmod(0o700)
    _kernel_selection_path(output.parent).write_text(output.name + "\n", encoding="ascii")
    _kernel_selection_path(output.parent).chmod(0o600)


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
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


def test_c_kernel_cache_key_includes_architecture_and_python_abi(monkeypatch):
    native_key = _circsym_c_kernel_cache_key()
    with monkeypatch.context() as architecture_patch:
        architecture_patch.setattr(
            platform,
            "machine",
            lambda: "different-architecture",
        )
        architecture_key = _circsym_c_kernel_cache_key()
    with monkeypatch.context() as abi_patch:
        original_get_config_var = circsym.sysconfig.get_config_var

        def different_soabi(name):
            if name == "SOABI":
                return "different-python-abi"
            return original_get_config_var(name)

        abi_patch.setattr(
            circsym.sysconfig,
            "get_config_var",
            different_soabi,
        )
        abi_key = _circsym_c_kernel_cache_key()

    assert architecture_key != native_key
    assert abi_key != native_key


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
@pytest.mark.parametrize("symlink_component", ["application", "leaf"])
def test_c_kernel_cache_rejects_directory_symlinks(
    tmp_path: Path,
    symlink_component: str,
):
    cache_base = tmp_path / "cache"
    cache_base.mkdir()
    target = tmp_path / "redirected"
    target.mkdir()
    application_dir = cache_base / "hornlab-metal-bem"
    if symlink_component == "application":
        application_dir.symlink_to(target, target_is_directory=True)
    else:
        application_dir.mkdir()
        (application_dir / "circsym").symlink_to(
            target,
            target_is_directory=True,
        )

    with pytest.raises(PermissionError, match="not a real directory"):
        _prepare_circsym_c_kernel_cache(
            str(cache_base / "hornlab-metal-bem" / "circsym")
        )


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
@pytest.mark.parametrize("planted_kind", ["symlink", "permissive-file"])
def test_c_kernel_cache_rebuilds_an_unsafe_planted_library(
    monkeypatch,
    tmp_path: Path,
    planted_kind: str,
):
    compiler_wrapper, invocation_log = _counting_compiler(tmp_path)
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("CC", str(compiler_wrapper))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    cache_dir = Path(_circsym_c_kernel_cache_dir())
    _prepare_circsym_c_kernel_cache(str(cache_dir))

    planted = tmp_path / "planted-library"
    planted.write_bytes(b"not a shared library")
    library_path = _kernel_library_path(cache_dir)
    if planted_kind == "symlink":
        library_path.symlink_to(planted)
    else:
        library_path.write_bytes(planted.read_bytes())
        library_path.chmod(0o777)
    cache_dir.chmod(0o777)

    original_loader = circsym._CircsymRemainderCKernel
    planted_load_attempts = []

    def reject_if_planted(path):
        if Path(path).read_bytes() == planted.read_bytes():
            planted_load_attempts.append(path)
        return original_loader(path)

    monkeypatch.setattr(circsym, "_CircsymRemainderCKernel", reject_if_planted)
    circsym._load_circsym_remainder_c_kernel.cache_clear()
    try:
        assert circsym._load_circsym_remainder_c_kernel() is not None
    finally:
        circsym._load_circsym_remainder_c_kernel.cache_clear()

    assert invocation_log.read_text(encoding="utf-8").splitlines() == [
        "compile"
    ]
    assert planted_load_attempts == []
    assert not library_path.exists() and not library_path.is_symlink()
    selected = _selected_library_path(cache_dir)
    assert stat.S_ISREG(selected.lstat().st_mode)
    assert stat.S_IMODE(selected.stat().st_mode) == 0o700
    assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_c_kernel_cache_recompiles_once_after_load_failure(
    monkeypatch,
    tmp_path: Path,
):
    compiler_wrapper, invocation_log = _counting_compiler(tmp_path)
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    cache_dir = Path(_circsym_c_kernel_cache_dir())
    _prepare_circsym_c_kernel_cache(str(cache_dir))

    rejected_path = _generation_library_path(cache_dir, "missing-exports")
    _compile_library_without_kernel_exports(rejected_path, tmp_path)
    monkeypatch.setenv("CC", str(compiler_wrapper))
    # Reproduce dyld/dlopen's pathname cache: the wrong image is already a
    # valid loaded shared library, but it lacks the two required kernel exports.
    preloaded = ctypes.CDLL(str(rejected_path))
    assert not hasattr(preloaded, "circsym_eval_near_remainder")

    circsym._load_circsym_remainder_c_kernel.cache_clear()
    try:
        kernel = circsym._load_circsym_remainder_c_kernel()
        assert kernel is not None
    finally:
        circsym._load_circsym_remainder_c_kernel.cache_clear()

    # Rebuilding at the rejected pathname would make CDLL return `preloaded`
    # again. Publication selects a new generation pathname instead.
    assert invocation_log.read_text(encoding="utf-8").splitlines() == [
        "compile"
    ]
    selected = _selected_library_path(cache_dir)
    assert selected != rejected_path
    assert Path(kernel.library._name) == selected
    assert hasattr(kernel.library, "circsym_eval_near_remainder")


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_concurrent_processes_heal_one_missing_export_generation(tmp_path: Path):
    cache_home = tmp_path / "cache"
    cache_dir = cache_home / "hornlab-metal-bem" / "circsym"
    _prepare_circsym_c_kernel_cache(str(cache_dir))
    rejected_path = _generation_library_path(cache_dir, "missing-exports")
    _compile_library_without_kernel_exports(rejected_path, tmp_path)
    compiler_wrapper, invocation_log = _counting_compiler(tmp_path, delay_s=0.2)

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
        "kernel = load(); assert kernel is not None; "
        f"assert kernel.library._name != {str(rejected_path)!r}"
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
    assert _selected_library_path(cache_dir) != rejected_path


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_concurrent_processes_compile_one_private_atomic_kernel(tmp_path: Path):
    cache_home = tmp_path / "cache"
    compiler_wrapper, invocation_log = _counting_compiler(tmp_path, delay_s=0.2)

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
    assert stat.S_IMODE(libraries[0].stat().st_mode) == 0o700
    sources = list(cache_dir.glob("*.c"))
    assert len(sources) == 1
    assert stat.S_IMODE(sources[0].stat().st_mode) == 0o600
    assert not [path for path in cache_dir.iterdir() if ".tmp" in path.name]

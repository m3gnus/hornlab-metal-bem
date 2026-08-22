from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import signal
import shutil
import stat
import subprocess
import sys
import time

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
        "is_compile = '-o' in sys.argv and '-E' not in sys.argv\n"
        "if is_compile:\n"
        "    fd = os.open(log, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)\n"
        "    try:\n"
        "        os.write(fd, b'compile\\n')\n"
        "    finally:\n"
        "        os.close(fd)\n"
        f"    time.sleep({delay_s!r})\n"
        f"compiler = {compiler!r}\n"
        "os.execv(compiler, [compiler, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    compiler_wrapper.chmod(0o700)
    return compiler_wrapper, invocation_log, compiler


def _write_fp_mode_compiler_wrapper(path: Path, compiler: str, flag: str) -> None:
    path.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        f"compiler = {compiler!r}\n"
        f"flag = {flag!r}\n"
        "probe = {'--version', '-dumpmachine', '-print-sysroot'}\n"
        "args = sys.argv[1:]\n"
        "if not probe.intersection(args):\n"
        "    args.insert(0, flag)\n"
        "os.execv(compiler, [compiler, *args])\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _kernel_library_path(cache_dir: Path) -> Path:
    """Return the pre-generation canonical path used by old cache versions."""

    extension = ".dylib" if platform.system() == "Darwin" else ".so"
    return cache_dir / f"circsym_remainder_{_circsym_c_kernel_cache_key()}{extension}"


def _kernel_selection_path(cache_dir: Path, cache_key: str | None = None) -> Path:
    key = _circsym_c_kernel_cache_key() if cache_key is None else cache_key
    return cache_dir / f"circsym_remainder_{key}.current"


def _generation_library_path(
    cache_dir: Path,
    generation: str,
    cache_key: str | None = None,
) -> Path:
    extension = ".dylib" if platform.system() == "Darwin" else ".so"
    key = _circsym_c_kernel_cache_key() if cache_key is None else cache_key
    return cache_dir / (
        f"circsym_remainder_{key}.{generation}{extension}"
    )


def _selected_library_path(cache_dir: Path, cache_key: str | None = None) -> Path:
    return cache_dir / _kernel_selection_path(cache_dir, cache_key).read_text(
        encoding="ascii"
    ).strip()


def _compile_library_without_kernel_exports(
    output: Path,
    tmp_path: Path,
    *,
    compiler: str | None = None,
    cache_key: str | None = None,
) -> None:
    compiler = shutil.which(os.environ.get("CC", "cc")) if compiler is None else compiler
    if compiler is None:
        pytest.skip("runtime C compiler is unavailable")
    source = tmp_path / f"wrong-{output.stem}.c"
    source.write_text("int unrelated_export(void) { return 7; }\n", encoding="ascii")
    command = [compiler, "-O2", "-fPIC"]
    command.append("-dynamiclib" if platform.system() == "Darwin" else "-shared")
    command.extend([str(source), "-o", str(output)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    output.chmod(0o700)
    selection_path = _kernel_selection_path(output.parent, cache_key)
    selection_path.write_text(output.name + "\n", encoding="ascii")
    selection_path.chmod(0o600)


def _compile_library_with_stale_fingerprint(
    output: Path,
    tmp_path: Path,
    *,
    compiler: str,
    cache_key: str,
) -> None:
    source = tmp_path / f"stale-{output.stem}.c"
    source.write_text(
        "const char *circsym_c_kernel_build_fingerprint(void) {\n"
        '    return "stale-build-fingerprint";\n'
        "}\n"
        "int circsym_eval_near_remainder(void) { return 0; }\n"
        "int circsym_eval_far_remainder_onthefly(void) { return 0; }\n",
        encoding="ascii",
    )
    command = [compiler, "-O2", "-fPIC"]
    command.append("-dynamiclib" if platform.system() == "Darwin" else "-shared")
    command.extend([str(source), "-o", str(output)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    output.chmod(0o700)
    selection_path = _kernel_selection_path(output.parent, cache_key)
    selection_path.write_text(output.name + "\n", encoding="ascii")
    selection_path.chmod(0o600)


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
    build_fingerprint = "test-build-fingerprint"
    native_key = _circsym_c_kernel_cache_key(
        build_fingerprint=build_fingerprint
    )
    with monkeypatch.context() as architecture_patch:
        architecture_patch.setattr(
            platform,
            "machine",
            lambda: "different-architecture",
        )
        architecture_key = _circsym_c_kernel_cache_key(
            build_fingerprint=build_fingerprint
        )
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
        abi_key = _circsym_c_kernel_cache_key(
            build_fingerprint=build_fingerprint
        )

    assert architecture_key != native_key
    assert abi_key != native_key


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_c_kernel_cache_key_changes_with_compiler_recipe(monkeypatch, tmp_path: Path):
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        pytest.skip("runtime C compiler is unavailable")
    wrapper = tmp_path / "recipe-cc"
    monkeypatch.setenv("CC", str(wrapper))

    _write_fp_mode_compiler_wrapper(wrapper, compiler, "-fno-fast-math")
    strict_key = _circsym_c_kernel_cache_key()
    _write_fp_mode_compiler_wrapper(wrapper, compiler, "-ffast-math")
    fast_key = _circsym_c_kernel_cache_key()

    assert fast_key != strict_key


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_c_kernel_cache_key_changes_with_declared_compile_flags(monkeypatch):
    native_key = _circsym_c_kernel_cache_key()
    monkeypatch.setattr(
        circsym,
        "_CIRCSYM_C_KERNEL_COMPILE_ARGS",
        (*circsym._CIRCSYM_C_KERNEL_COMPILE_ARGS, "-fno-strict-aliasing"),
    )

    assert _circsym_c_kernel_cache_key() != native_key


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
    compiler_wrapper, invocation_log, _compiler = _counting_compiler(tmp_path)
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
    compiler_wrapper, invocation_log, compiler = _counting_compiler(tmp_path)
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    cache_dir = Path(_circsym_c_kernel_cache_dir())
    _prepare_circsym_c_kernel_cache(str(cache_dir))

    monkeypatch.setenv("CC", str(compiler_wrapper))
    rejected_path = _generation_library_path(cache_dir, "missing-exports")
    _compile_library_without_kernel_exports(
        rejected_path,
        tmp_path,
        compiler=compiler,
    )
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
    assert not rejected_path.exists()
    assert Path(kernel.library._name) == selected
    assert hasattr(kernel.library, "circsym_eval_near_remainder")


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_c_kernel_cache_rejects_loadable_stale_build_fingerprint(
    monkeypatch,
    tmp_path: Path,
):
    compiler_wrapper, invocation_log, compiler = _counting_compiler(tmp_path)
    monkeypatch.setenv("CC", str(compiler_wrapper))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    cache_dir = Path(_circsym_c_kernel_cache_dir())
    _prepare_circsym_c_kernel_cache(str(cache_dir))
    cache_key = _circsym_c_kernel_cache_key()
    rejected_path = _generation_library_path(
        cache_dir,
        "stale-fingerprint",
        cache_key,
    )
    _compile_library_with_stale_fingerprint(
        rejected_path,
        tmp_path,
        compiler=compiler,
        cache_key=cache_key,
    )

    circsym._load_circsym_remainder_c_kernel.cache_clear()
    try:
        kernel = circsym._load_circsym_remainder_c_kernel()
        assert kernel is not None
    finally:
        circsym._load_circsym_remainder_c_kernel.cache_clear()

    assert invocation_log.read_text(encoding="utf-8").splitlines() == ["compile"]
    assert not rejected_path.exists()
    assert Path(kernel.library._name) == _selected_library_path(cache_dir, cache_key)


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_concurrent_processes_heal_one_missing_export_generation(tmp_path: Path):
    cache_home = tmp_path / "cache"
    cache_dir = cache_home / "hornlab-metal-bem" / "circsym"
    _prepare_circsym_c_kernel_cache(str(cache_dir))
    compiler_wrapper, invocation_log, compiler = _counting_compiler(
        tmp_path, delay_s=0.2
    )
    cache_key = _circsym_c_kernel_cache_key(compiler=str(compiler_wrapper))
    rejected_path = _generation_library_path(
        cache_dir, "missing-exports", cache_key
    )
    _compile_library_without_kernel_exports(
        rejected_path,
        tmp_path,
        compiler=compiler,
        cache_key=cache_key,
    )
    crash_orphan = _generation_library_path(cache_dir, "crash-orphan", cache_key)
    crash_orphan.write_bytes(b"unpublished crash artifact")
    crash_orphan.chmod(0o700)
    os.utime(crash_orphan, (1, 1))
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
    assert _selected_library_path(cache_dir, cache_key) != rejected_path
    assert not rejected_path.exists()
    assert not crash_orphan.exists()


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_generation_rename_failure_removes_unpublished_library(
    monkeypatch,
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache" / "hornlab-metal-bem" / "circsym"
    _prepare_circsym_c_kernel_cache(str(cache_dir))
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        pytest.skip("runtime C compiler is unavailable")
    extension = ".dylib" if platform.system() == "Darwin" else ".so"
    original_replace = circsym.os.replace

    def fail_after_generation_rename(source, destination):
        original_replace(source, destination)
        if str(source).endswith(f"{extension}.tmp") and str(destination).endswith(
            extension
        ):
            raise OSError("injected failure after generation rename")

    monkeypatch.setattr(circsym.os, "replace", fail_after_generation_rename)
    with pytest.raises(OSError, match="injected failure"):
        circsym._compile_circsym_remainder_c_kernel(
            cache_dir=str(cache_dir),
            cache_key=_circsym_c_kernel_cache_key(),
            platform_name=platform.system().lower(),
            compiler=compiler,
        )

    assert not _kernel_selection_path(cache_dir).exists()
    assert list(
        cache_dir.glob(
            f"circsym_remainder_{_circsym_c_kernel_cache_key()}.*{extension}"
        )
    ) == []


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_orphan_gc_is_bounded_and_preserves_selected_and_recent_generations(
    monkeypatch,
    tmp_path: Path,
):
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))
    cache_dir = Path(_circsym_c_kernel_cache_dir())
    circsym._load_circsym_remainder_c_kernel.cache_clear()
    try:
        assert circsym._load_circsym_remainder_c_kernel() is not None
    finally:
        circsym._load_circsym_remainder_c_kernel.cache_clear()
    selected = _selected_library_path(cache_dir)
    old_orphans = [
        _generation_library_path(cache_dir, f"crash-{index}")
        for index in range(circsym._CIRCSYM_C_KERNEL_GC_MAX_REMOVALS + 2)
    ]
    for orphan in old_orphans:
        orphan.write_bytes(b"unpublished crash artifact")
        orphan.chmod(0o700)
        os.utime(orphan, (1, 1))
    recent = _generation_library_path(cache_dir, "concurrent-healer")
    recent.write_bytes(b"recent unpublished artifact")
    recent.chmod(0o700)

    compile_args = {
        "cache_dir": str(cache_dir),
        "cache_key": _circsym_c_kernel_cache_key(),
        "platform_name": platform.system().lower(),
        "compiler": os.environ.get("CC", "cc"),
    }
    returned, _identity = circsym._compile_circsym_remainder_c_kernel(
        **compile_args
    )
    assert Path(returned) == selected
    assert sum(path.exists() for path in old_orphans) == 2
    assert selected.exists()
    assert recent.exists()

    circsym._compile_circsym_remainder_c_kernel(**compile_args)
    assert not any(path.exists() for path in old_orphans)
    assert selected.exists()
    assert recent.exists()


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_orphan_gc_collects_crash_temporaries_and_obsolete_key_artifacts(
    tmp_path: Path,
):
    cache_home = tmp_path / "cache"
    cache_dir = cache_home / "hornlab-metal-bem" / "circsym"
    _prepare_circsym_c_kernel_cache(str(cache_dir))
    compiler_wrapper, invocation_log, compiler = _counting_compiler(
        tmp_path,
        delay_s=60.0,
    )
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
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from hornlab_metal_bem.circsym import "
            "_load_circsym_remainder_c_kernel as load; load()",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 10.0
    while not invocation_log.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.02)
    if not invocation_log.exists():
        stdout, stderr = process.communicate(timeout=5.0)
        pytest.fail(f"compiler did not start before exit: {stdout!r} {stderr!r}")
    os.killpg(process.pid, signal.SIGKILL)
    process.communicate(timeout=5.0)

    crash_temporaries = [
        path for path in cache_dir.iterdir() if path.name.startswith(".circsym_")
    ]
    assert any(path.name.endswith(".tmp.c") for path in crash_temporaries)
    assert any(path.name.endswith((".so.tmp", ".dylib.tmp")) for path in crash_temporaries)

    obsolete_key = "0" * 24
    extension = ".dylib" if platform.system() == "Darwin" else ".so"
    obsolete_generation = cache_dir / (
        f"circsym_remainder_{obsolete_key}.obsolete{extension}"
    )
    obsolete_source = cache_dir / f"circsym_remainder_{obsolete_key}.c"
    obsolete_selector = cache_dir / f"circsym_remainder_{obsolete_key}.current"
    obsolete_legacy = cache_dir / f"circsym_remainder_{obsolete_key}{extension}"
    selection_temp = cache_dir / (
        f".circsym_remainder_{obsolete_key}.crash.current.tmp"
    )
    for path, mode, content in (
        (obsolete_generation, 0o700, b"old generation"),
        (obsolete_source, 0o600, b"old source"),
        (obsolete_selector, 0o600, obsolete_generation.name.encode()),
        (obsolete_legacy, 0o700, b"old legacy library"),
        (selection_temp, 0o600, obsolete_generation.name.encode()),
    ):
        path.write_bytes(content)
        path.chmod(mode)
    owned_stale = [*crash_temporaries, obsolete_generation, obsolete_source]
    owned_stale.extend((obsolete_selector, obsolete_legacy, selection_temp))
    for path in owned_stale:
        os.utime(path, (1, 1))

    cache_key = _circsym_c_kernel_cache_key(compiler=compiler)
    circsym._compile_circsym_remainder_c_kernel(
        cache_dir=str(cache_dir),
        cache_key=cache_key,
        platform_name=platform.system().lower(),
        compiler=compiler,
    )

    assert not any(path.exists() for path in owned_stale)


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_orphan_gc_reaches_candidates_beyond_the_old_scan_window(
    monkeypatch,
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache" / "hornlab-metal-bem" / "circsym"
    _prepare_circsym_c_kernel_cache(str(cache_dir))
    for index in range(512):
        (cache_dir / f"unrelated-{index:04d}").write_bytes(b"not cache owned")
    candidate = _generation_library_path(cache_dir, "old-after-window")
    candidate.write_bytes(b"old unselected generation")
    candidate.chmod(0o700)
    os.utime(candidate, (1, 1))
    real_scandir = circsym.os.scandir

    class CandidateLastScandir:
        def __init__(self, path):
            with real_scandir(path) as entries:
                self.entries = sorted(
                    entries,
                    key=lambda entry: entry.name == candidate.name,
                )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(self.entries)

    monkeypatch.setattr(circsym.os, "scandir", CandidateLastScandir)
    assert [entry.name for entry in CandidateLastScandir(cache_dir)].index(
        candidate.name
    ) == 512

    removed = circsym._collect_circsym_c_kernel_orphans(
        cache_dir=str(cache_dir),
        cache_key=_circsym_c_kernel_cache_key(),
        platform_name=platform.system().lower(),
        selected_path=None,
    )

    assert removed == 1
    assert not candidate.exists()


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_orphan_gc_does_not_unlink_a_loaded_generation(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    cache_dir = Path(_circsym_c_kernel_cache_dir())
    circsym._load_circsym_remainder_c_kernel.cache_clear()
    try:
        assert circsym._load_circsym_remainder_c_kernel() is not None
    finally:
        circsym._load_circsym_remainder_c_kernel.cache_clear()
    selected = _selected_library_path(cache_dir)
    loaded_path = _generation_library_path(cache_dir, "loaded-by-another-process")
    shutil.copyfile(selected, loaded_path)
    loaded_path.chmod(0o700)
    os.utime(loaded_path, (1, 1))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT), env.get("PYTHONPATH", "")))
    )
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from hornlab_metal_bem.circsym import _CircsymRemainderCKernel; "
            f"kernel = _CircsymRemainderCKernel({str(loaded_path)!r}); "
            "print('ready', flush=True); sys.stdin.readline()",
        ],
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline() == "ready\n"

    circsym._collect_circsym_c_kernel_orphans(
        cache_dir=str(cache_dir),
        cache_key=_circsym_c_kernel_cache_key(),
        platform_name=platform.system().lower(),
        selected_path=str(selected),
    )
    assert loaded_path.exists()

    stdout, stderr = holder.communicate(input="release\n", timeout=5.0)
    assert (holder.returncode, stdout, stderr) == (0, "", "")
    circsym._collect_circsym_c_kernel_orphans(
        cache_dir=str(cache_dir),
        cache_key=_circsym_c_kernel_cache_key(),
        platform_name=platform.system().lower(),
        selected_path=str(selected),
    )
    assert not loaded_path.exists()


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
@pytest.mark.parametrize(
    ("max_artifacts", "max_bytes", "artifact_sizes"),
    [
        (3, 1024, [1] * 6),
        (100, 50, [30] * 3),
    ],
)
def test_orphan_gc_enforces_directory_wide_artifact_ceiling(
    monkeypatch,
    tmp_path: Path,
    max_artifacts: int,
    max_bytes: int,
    artifact_sizes: list[int],
):
    cache_dir = tmp_path / "cache" / "hornlab-metal-bem" / "circsym"
    _prepare_circsym_c_kernel_cache(str(cache_dir))
    monkeypatch.setattr(
        circsym,
        "_CIRCSYM_C_KERNEL_CACHE_MAX_ARTIFACTS",
        max_artifacts,
    )
    monkeypatch.setattr(circsym, "_CIRCSYM_C_KERNEL_CACHE_MAX_BYTES", max_bytes)
    recent_orphans = [
        _generation_library_path(cache_dir, f"recent-overflow-{index}")
        for index in range(len(artifact_sizes))
    ]
    for orphan, size in zip(recent_orphans, artifact_sizes):
        orphan.write_bytes(b"x" * size)
        orphan.chmod(0o700)

    circsym._collect_circsym_c_kernel_orphans(
        cache_dir=str(cache_dir),
        cache_key=_circsym_c_kernel_cache_key(),
        platform_name=platform.system().lower(),
        selected_path=None,
    )

    remaining = [path for path in recent_orphans if path.exists()]
    assert len(remaining) <= max_artifacts
    assert sum(path.stat().st_size for path in remaining) <= max_bytes


@pytest.mark.skipif(os.name != "posix", reason="the runtime C kernel is POSIX-only")
def test_concurrent_processes_compile_one_private_atomic_kernel(tmp_path: Path):
    cache_home = tmp_path / "cache"
    compiler_wrapper, invocation_log, _compiler = _counting_compiler(
        tmp_path, delay_s=0.2
    )

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

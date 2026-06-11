from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_metal_bem.metal import (
    MetalBemBackend,
    MetalBemContext,
    MetalNativeRuntimeConfig,
    MetalNativeStandardSession,
    discover_native_runtime,
    validate_session_with_native_helper,
)
from hornlab_metal_bem.metal.backend import DenseBieSystem
from hornlab_metal_bem.metal import backend as metal_backend
from hornlab_metal_bem.metal import native
from hornlab_metal_bem.metal.geometry import build_metal_geometry_buffers
from hornlab_metal_bem.metal.geometry import MetalGeometryError
from hornlab_metal_bem.metal.geometry import validate_native_symmetry_plane
from hornlab_metal_bem.validation.native_symmetry import orbit_reduce_matrix_rhs


def _write_native_entrypoint(root: Path) -> Path:
    helper = root / "HornlabMetalBemNative.swift"
    helper.write_text("// test helper\n", encoding="utf-8")
    return helper


def _write_native_package_binary(root: Path) -> Path:
    package_dir = root / "native_helper"
    (package_dir / ".build" / "release").mkdir(parents=True)
    (package_dir / "Package.swift").write_text("// test package\n", encoding="utf-8")
    binary = package_dir / ".build" / "release" / "HornlabMetalBemNative"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    return binary


def _arg_after(command: list[str], op: str, offset: int) -> str:
    return command[command.index(op) + offset]


def _tiny_geometry_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([1, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _tiny_robin_geometry_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.3, 0.0, 0.0, -0.2],
                [0.0, 0.0, 0.3, -0.2, 0.0],
                [0.0, 0.0, 0.1, 0.2, 0.25],
            ],
            dtype=np.float64,
        ),
        elements=np.array(
            [
                [0, 0, 0],
                [1, 2, 3],
                [2, 3, 4],
            ],
            dtype=np.int64,
        ),
        number_of_elements=3,
    )
    p1 = SimpleNamespace(
        local2global=np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4]],
            dtype=np.int64,
        ),
        global_dof_count=5,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 8, 9], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=3),
    )


def _tiny_yz_half_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0], [1], [2]], dtype=np.int64),
        number_of_elements=1,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2]], dtype=np.int64),
        global_dof_count=3,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=1),
    )


def _tiny_yz_full_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 3], [1, 5], [2, 4]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [3, 5, 4]], dtype=np.int64),
        global_dof_count=6,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _tiny_yz_xz_quarter_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0], [1], [2]], dtype=np.int64),
        number_of_elements=1,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2]], dtype=np.int64),
        global_dof_count=3,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=1),
    )


def _tiny_yz_xz_full_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array(
            [
                [0, 0, 0, 0],
                [1, 2, 4, 3],
                [2, 3, 1, 4],
            ],
            dtype=np.int64,
        ),
        number_of_elements=4,
    )
    p1 = SimpleNamespace(
        local2global=np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 4, 1],
                [0, 3, 4],
            ],
            dtype=np.int64,
        ),
        global_dof_count=5,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2, 2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=4),
    )


def _tiny_xy_half_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0], [1], [2]], dtype=np.int64),
        number_of_elements=1,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2]], dtype=np.int64),
        global_dof_count=3,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=1),
    )


def _tiny_xy_mirror_full_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 3], [1, 5], [2, 4]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [3, 5, 4]], dtype=np.int64),
        global_dof_count=6,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _tiny_xy_shared_full_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, -1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 3], [2, 1]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int64),
        global_dof_count=4,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _tiny_xy_full_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, -1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _read_complex_assembly(assembly):
    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )
    return matrix, rhs


def test_native_batch_result_count_validation():
    cases = [{"case_id": "case-0000"}]
    assert (
        native._case_results_from_manifest(
            {"cases": cases},
            expected_count=1,
            op="test_batch",
        )
        is cases
    )

    with pytest.raises(RuntimeError, match="result missing cases"):
        native._case_results_from_manifest(
            {},
            expected_count=1,
            op="test_batch",
        )

    with pytest.raises(RuntimeError, match="returned 0 case"):
        native._case_results_from_manifest(
            {"cases": []},
            expected_count=1,
            op="test_batch",
        )


def test_native_diagnostics_helpers_preserve_manifest_metadata():
    batch = native._native_batch_diagnostics(
        {
            "implementation": "batch_impl",
            "session_id": "session",
            "batch_id": "batch",
            "wall_seconds": 1.25,
            "resident_reuse": {"geometry_buffers": True},
            "cases": [],
            "ignored_output_path": "outputs/field.bin",
        }
    )
    case = native._native_case_diagnostics(
        {
            "case_id": "case-0000",
            "assembly_implementation": "assembly_impl",
            "solve_implementation": "solve_impl",
            "field_implementation": "field_impl",
            "lapack_info": 0,
            "duffy_corrections": {"implemented": True},
            "metal_dispatch": {"matrix": {"threads_per_threadgroup": 64}},
            "pressure_real_f32": "outputs/pressure_re.bin",
        },
        batch_diagnostics=batch,
    )

    assert case["assembly_implementation"] == "assembly_impl"
    assert case["duffy_corrections"]["implemented"] is True
    assert case["batch"]["resident_reuse"]["geometry_buffers"] is True
    assert "pressure_real_f32" not in case
    assert "ignored_output_path" not in case["batch"]


def test_native_discovery_reports_missing_helper_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("HORNLAB_METAL_BEM_SWIFT", "/usr/bin/swift")

    status = discover_native_runtime(MetalNativeRuntimeConfig(backend_dir=tmp_path))

    assert status.available is False
    assert status.is_macos is True
    assert status.is_apple_silicon is True
    assert status.swift_path == "/usr/bin/swift"
    assert status.swift_source == "HORNLAB_METAL_BEM_SWIFT"
    assert status.native_entrypoint == tmp_path / "HornlabMetalBemNative.swift"
    assert status.helper_assets_present is False
    assert status.smoke_test_ran is False
    assert any("Swift/Metal helper" in r for r in status.reasons)


def test_native_discovery_finds_swift_on_path(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("HORNLAB_METAL_BEM_SWIFT", raising=False)
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    status = discover_native_runtime(MetalNativeRuntimeConfig(backend_dir=tmp_path))

    assert status.available is True
    assert status.swift_path == "/usr/bin/swift"
    assert status.swift_source == "PATH"
    assert status.helper_assets_present is True
    assert status.smoke_test_ran is False


def test_native_discovery_prefers_compiled_package_helper_without_swift(
    monkeypatch,
    tmp_path,
):
    binary = _write_native_package_binary(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("HORNLAB_METAL_BEM_SWIFT", raising=False)
    monkeypatch.setattr(native.shutil, "which", lambda name: None)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return native.subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    status = discover_native_runtime(
        MetalNativeRuntimeConfig(backend_dir=tmp_path),
        run_smoke_test=True,
    )

    assert status.available is True
    assert status.swift_path is None
    assert status.helper_executable_path == binary
    assert status.helper_source == "swift-package"
    assert status.smoke_test_ran is True
    assert status.smoke_test_ok is True
    assert calls[0] == [str(binary), "--smoke"]


def test_native_discovery_can_run_smoke_test(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return native.subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    status = discover_native_runtime(
        MetalNativeRuntimeConfig(backend_dir=tmp_path),
        run_smoke_test=True,
    )

    assert status.available is True
    assert status.smoke_test_ran is True
    assert status.smoke_test_ok is True
    assert calls[0][-1] == "--smoke"


def test_native_discovery_reports_failed_smoke_test(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    def fake_run(command, **kwargs):
        return native.subprocess.CompletedProcess(
            command,
            1,
            "",
            "Metal device unavailable\n",
        )

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    status = discover_native_runtime(
        MetalNativeRuntimeConfig(backend_dir=tmp_path),
        run_smoke_test=True,
    )

    assert status.available is False
    assert status.smoke_test_ran is True
    assert status.smoke_test_ok is False
    assert status.smoke_test_error == "Metal device unavailable"
    assert any("smoke test failed" in reason for reason in status.reasons)


def test_validate_session_with_native_helper_invokes_swift(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    session_manifest = tmp_path / "session.json"
    session_manifest.write_text("{}", encoding="utf-8")
    result_path = tmp_path / "result.json"

    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        result_path.write_text(
            json.dumps(
                {
                    "schema": "hornlab.metal.standard.v1",
                    "op": "validate_session_result",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    result = validate_session_with_native_helper(
        session_manifest,
        result_path,
        MetalNativeRuntimeConfig(backend_dir=tmp_path),
    )

    assert result["status"] == "ok"
    assert calls[0][-1] == "--smoke"
    assert calls[1][2] == "validate_session"
    assert calls[1][3] == str(session_manifest)
    assert calls[1][4] == str(result_path)


def test_validate_session_with_compiled_helper_does_not_require_swift(
    monkeypatch,
    tmp_path,
):
    binary = _write_native_package_binary(tmp_path)
    session_manifest = tmp_path / "session.json"
    session_manifest.write_text("{}", encoding="utf-8")
    result_path = tmp_path / "result.json"

    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("HORNLAB_METAL_BEM_SWIFT", raising=False)
    monkeypatch.setattr(native.shutil, "which", lambda name: None)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        result_path.write_text(
            json.dumps(
                {
                    "schema": "hornlab.metal.standard.v1",
                    "op": "validate_session_result",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    result = validate_session_with_native_helper(
        session_manifest,
        result_path,
        MetalNativeRuntimeConfig(backend_dir=tmp_path),
    )

    assert result["status"] == "ok"
    assert calls[0] == [str(binary), "--smoke"]
    assert calls[1][0] == str(binary)
    assert calls[1][1] == "validate_session"
    assert calls[1][2] == str(session_manifest)
    assert calls[1][3] == str(result_path)


def test_native_standard_session_writes_manifest_and_validates(
    monkeypatch,
    tmp_path,
):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    def fake_run(command, **kwargs):
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        result_path = Path(_arg_after(command, "validate_session", 2))
        result_path.write_text(
            json.dumps(
                {
                    "schema": "hornlab.metal.standard.v1",
                    "op": "validate_session_result",
                    "session_id": "native-test",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )
    dp0 = SimpleNamespace(global_dof_count=2)
    buffers = build_metal_geometry_buffers(
        grid,
        np.array([1, 2], dtype=np.int32),
        p1,
        dp0,
    )

    session = MetalNativeStandardSession.create_session(
        geometry_buffers=buffers,
        work_dir=tmp_path / "native-session",
        session_id="native-test",
    )
    try:
        result = session.validate_contract()

        assert result["status"] == "ok"
        assert session.info.manifest_path.is_file()
        assert (session.info.work_dir / "native-result.json").is_file()
    finally:
        session.close()


def test_native_symmetry_manifest_and_half_domain_guard(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    def fake_run(command, **kwargs):
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        result_path = Path(_arg_after(command, "validate_session", 2))
        result_path.write_text(
            json.dumps(
                {
                    "schema": "hornlab.metal.standard.v1",
                    "op": "validate_session_result",
                    "session_id": "native-symmetry-test",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    session = MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_half_buffers(),
        work_dir=tmp_path / "native-symmetry-session",
        session_id="native-symmetry-test",
        symmetry_plane="yz",
    )
    try:
        manifest = json.loads(session.info.manifest_path.read_text(encoding="utf-8"))
        assert manifest["assembly_scope"]["symmetry_plane"] == "yz"
    finally:
        session.close()

    quarter_session = MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_quarter_buffers(),
        work_dir=tmp_path / "native-quarter-symmetry-session",
        session_id="native-quarter-symmetry-test",
        symmetry_plane="yz+xz",
    )
    try:
        manifest = json.loads(
            quarter_session.info.manifest_path.read_text(encoding="utf-8")
        )
        assert manifest["assembly_scope"]["symmetry_plane"] == "yz+xz"
    finally:
        quarter_session.close()

    xy_session = MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_half_buffers(),
        work_dir=tmp_path / "native-xy-symmetry-session",
        session_id="native-xy-symmetry-test",
        symmetry_plane="xy",
    )
    try:
        manifest = json.loads(xy_session.info.manifest_path.read_text(encoding="utf-8"))
        assert manifest["assembly_scope"]["symmetry_plane"] == "xy"
    finally:
        xy_session.close()

    assert validate_native_symmetry_plane(_tiny_yz_xz_quarter_buffers(), "xz") == "xz"
    assert validate_native_symmetry_plane(_tiny_xy_half_buffers(), "xy") == "xy"
    assert (
        validate_native_symmetry_plane(_tiny_yz_xz_quarter_buffers(), "yz+xz")
        == "yz+xz"
    )

    with pytest.raises(MetalGeometryError, match="positive-x reduced-domain"):
        validate_native_symmetry_plane(_tiny_yz_full_buffers(), "yz")

    with pytest.raises(MetalGeometryError, match="positive-z reduced-domain"):
        validate_native_symmetry_plane(_tiny_xy_full_buffers(), "xy")

    with pytest.raises(
        MetalGeometryError,
        match="supports 'yz', 'xz', 'xy', and 'yz\\+xz'",
    ):
        validate_native_symmetry_plane(_tiny_yz_half_buffers(), "zx")


def test_native_standard_session_invokes_swift_assembly(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    def fake_run(command, **kwargs):
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "assemble_standard_neumann" in command:
            payload = json.loads(
                Path(_arg_after(command, "assemble_standard_neumann", 2)).read_text(
                    encoding="utf-8"
                )
            )
            root = Path(_arg_after(command, "assemble_standard_neumann", 1)).parent
            for descriptor in payload["outputs"].values():
                if isinstance(descriptor, dict):
                    path = root / descriptor["path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.zeros(descriptor["shape"], dtype=np.float32).tofile(path)
            Path(_arg_after(command, "assemble_standard_neumann", 3)).write_text(
                json.dumps(
                    {
                        "schema": "hornlab.metal.standard.v1",
                        "op": "assemble_standard_neumann_result",
                        "session_id": "native-test",
                        "frequency_hz": 100.0,
                        "matrix_layout": "row_major_c",
                        "matrix_shape": [4, 4],
                        "rhs_shape": [4],
                        "matrix_real_f32": payload["outputs"]["A_real_f32"]["path"],
                        "matrix_imag_f32": payload["outputs"]["A_imag_f32"]["path"],
                        "rhs_real_f32": payload["outputs"]["rhs_real_f32"]["path"],
                        "rhs_imag_f32": payload["outputs"]["rhs_imag_f32"]["path"],
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )
    buffers = build_metal_geometry_buffers(
        grid,
        np.array([1, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )

    session = MetalNativeStandardSession.create_session(
        geometry_buffers=buffers,
        work_dir=tmp_path / "native-session",
        session_id="native-test",
    )
    try:
        result = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-assembly-test",
        )

        assert result.matrix_shape == (4, 4)
        assert result.rhs_shape == (4,)
        assert result.matrix_real_f32.is_file()
        assembly_manifest = json.loads(
            (tmp_path / "native-session" / "native-assembly-test" / "assembly.json")
            .read_text(encoding="utf-8")
        )
        assert assembly_manifest["outputs"]["matrix_layout"] == "row_major_c"
        assert assembly_manifest["neumann_dp0"]["real_f32"]["shape"] == [2]
    finally:
        session.close()


def test_native_standard_session_invokes_swift_batch_assembly(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "assemble_standard_neumann_batch" in command:
            payload = json.loads(
                Path(_arg_after(command, "assemble_standard_neumann_batch", 2))
                .read_text(encoding="utf-8")
            )
            root = Path(
                _arg_after(command, "assemble_standard_neumann_batch", 1)
            ).parent
            case_results = []
            for case in payload["cases"]:
                for descriptor in case["outputs"].values():
                    if isinstance(descriptor, dict):
                        path = root / descriptor["path"]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.zeros(descriptor["shape"], dtype=np.float32).tofile(path)
                case_results.append(
                    {
                        "schema": "hornlab.metal.standard.v1",
                        "op": "assemble_standard_neumann_result",
                        "session_id": "native-test",
                        "frequency_hz": case["frequency_hz"],
                        "matrix_layout": "row_major_c",
                        "matrix_shape": [4, 4],
                        "rhs_shape": [4],
                        "matrix_real_f32": case["outputs"]["A_real_f32"]["path"],
                        "matrix_imag_f32": case["outputs"]["A_imag_f32"]["path"],
                        "rhs_real_f32": case["outputs"]["rhs_real_f32"]["path"],
                        "rhs_imag_f32": case["outputs"]["rhs_imag_f32"]["path"],
                    }
                )
            Path(_arg_after(command, "assemble_standard_neumann_batch", 3)).write_text(
                json.dumps(
                    {
                        "schema": "hornlab.metal.standard.v1",
                        "op": "assemble_standard_neumann_batch_result",
                        "session_id": "native-test",
                        "cases": case_results,
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    session = MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-session",
        session_id="native-test",
    )
    try:
        result = session.assemble_standard_neumann_batch(
            np.array([100.0, 200.0], dtype=np.float64),
            np.array([1.8318326, 3.6636652], dtype=np.float32),
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.5j],
                    [2.0 + 0.0j, 0.0 + 1.0j],
                ],
                dtype=np.complex64,
            ),
            operation_id="native-batch-assembly-test",
        )

        assert len(result) == 2
        assert result[0].matrix_shape == (4, 4)
        assert result[1].frequency_hz == 200.0
        manifest = json.loads(
            (
                tmp_path
                / "native-session"
                / "native-batch-assembly-test"
                / "assembly-batch.json"
            ).read_text(encoding="utf-8")
        )
        assert manifest["op"] == "assemble_standard_neumann_batch"
        assert len(manifest["cases"]) == 2
        assert "assemble_standard_neumann_batch" in calls[-1]
    finally:
        session.close()


def test_native_executable_session_contract_and_tiny_assembly(monkeypatch, tmp_path):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    for env_name in (
        "HORNLAB_METAL_BEM_NATIVE_THREADS_PER_GROUP",
        "HORNLAB_METAL_BEM_NATIVE_MATRIX_THREADS_PER_GROUP",
        "HORNLAB_METAL_BEM_NATIVE_RHS_THREADS_PER_GROUP",
        "HORNLAB_METAL_BEM_NATIVE_DUFFY_THREADS_PER_GROUP",
        "HORNLAB_METAL_BEM_NATIVE_FIELD_THREADS_PER_GROUP",
    ):
        monkeypatch.delenv(env_name, raising=False)

    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )
    buffers = build_metal_geometry_buffers(
        grid,
        np.array([1, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=buffers,
        work_dir=tmp_path / "native-exec-session",
        session_id="native-exec-test",
    ) as session:
        validation = session.validate_contract()
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-exec-assembly",
        )

    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-exec-session"
            / "native-exec-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )

    assert validation["status"] == "ok"
    assert validation["implementation"] == "swift_native_contract_probe"
    assert result["metal_dispatch"]["matrix"]["requested_threads_per_threadgroup"] is None
    assert result["metal_dispatch"]["matrix"]["threads_per_threadgroup"] == 64
    assert result["metal_dispatch"]["rhs"]["requested_threads_per_threadgroup"] is None
    assert result["metal_dispatch"]["rhs"]["threads_per_threadgroup"] == 64
    assert assembly.matrix_shape == (4, 4)
    assert assembly.rhs_shape == (4,)
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(rhs))
    assert np.linalg.norm(matrix) > 0.0
    assert np.linalg.norm(rhs) > 0.0


def test_native_executable_optimized_matches_reference_on_tiny_mesh(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "parity")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_THREADS_PER_GROUP", "32")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "entrywise")
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-parity-session",
        session_id="native-parity-test",
    ) as session:
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-parity-assembly",
        )

    result = json.loads(
        (
            tmp_path
            / "native-parity-session"
            / "native-parity-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["implementation"] == "swift_native_metal_regular_quadrature"
    assert result["assembly_mode"] == "parity"
    assert result["duffy_corrections"]["implemented"] is False
    assert result["duffy_corrections"]["planned_pairs"] == {
        "coincident": 2,
        "edge": 2,
        "total": 4,
        "vertex": 0,
    }
    assert result["duffy_corrections"]["raw_triplets_if_expanded"] == 36
    assert result["reference_parity"]["matrix_relative_l2"] < 1e-4
    assert result["reference_parity"]["rhs_relative_l2"] < 1e-4
    assert result["metal_dispatch"]["matrix"]["requested_threads_per_threadgroup"] == 32
    assert result["metal_dispatch"]["matrix"]["threads_per_threadgroup"] == 32
    assert result["metal_dispatch"]["rhs"]["requested_threads_per_threadgroup"] == 32
    assert result["metal_dispatch"]["rhs"]["threads_per_threadgroup"] == 32
    assert result["regular_assembly_seconds"] > 0.0
    assert assembly.matrix_shape == (4, 4)


def test_native_executable_block_staged_matches_reference_on_tiny_mesh(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "parity")
    monkeypatch.setenv(
        "HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL",
        "block_staged",
    )
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-block-staged-parity-session",
        session_id="native-block-staged-parity-test",
    ) as session:
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-block-staged-parity-assembly",
        )

    result = json.loads(
        (
            tmp_path
            / "native-block-staged-parity-session"
            / "native-block-staged-parity-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["implementation"] == "swift_native_metal_block_staged_regular_quadrature"
    assert result["reference_parity"]["matrix_relative_l2"] < 1e-4
    assert result["reference_parity"]["rhs_relative_l2"] < 1e-4
    assert result["metal_dispatch"]["regular_assembly_implementation"] == "block_staged"
    assert result["metal_dispatch"]["pair_blocks"]["kernel"] == "assemble_pair_blocks_regular"
    assert result["metal_dispatch"]["pair_blocks"]["triangle_pairs"] == 4
    assert assembly.matrix_shape == (4, 4)


def test_native_executable_pair_atomic_matches_reference_on_tiny_mesh(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "parity")
    monkeypatch.setenv(
        "HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL",
        "pair_atomic",
    )
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-pair-atomic-parity-session",
        session_id="native-pair-atomic-parity-test",
    ) as session:
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-pair-atomic-parity-assembly",
        )

    result = json.loads(
        (
            tmp_path
            / "native-pair-atomic-parity-session"
            / "native-pair-atomic-parity-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["implementation"] == "swift_native_metal_pair_atomic_regular_quadrature"
    assert result["reference_parity"]["matrix_relative_l2"] < 1e-4
    assert result["reference_parity"]["rhs_relative_l2"] < 1e-4
    assert result["metal_dispatch"]["regular_assembly_implementation"] == "pair_atomic"
    assert result["metal_dispatch"]["matrix"]["triangle_pairs"] == 4
    assert assembly.matrix_shape == (4, 4)


def test_native_executable_pair_atomic_corrected_yz_xz_matches_full_domain(
    monkeypatch,
    tmp_path,
):
    """pair_atomic must reproduce the entrywise yz+xz half-vs-full parity."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    monkeypatch.setenv(
        "HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL",
        "pair_atomic",
    )
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_full_buffers(),
        work_dir=tmp_path / "native-full-pair-atomic-yz-xz-session",
        session_id="native-full-pair-atomic-yz-xz-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.ones(4, dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_quarter_buffers(),
        work_dir=tmp_path / "native-quarter-pair-atomic-yz-xz-session",
        session_id="native-quarter-pair-atomic-yz-xz-test",
        symmetry_plane="yz+xz",
    ) as quarter_session:
        quarter_assembly = quarter_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="quarter-assembly",
        )

    full_matrix, full_rhs = _read_complex_assembly(full_assembly)
    quarter_matrix, quarter_rhs = _read_complex_assembly(quarter_assembly)
    row_orbits = [
        np.array([0], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        np.array([2, 4], dtype=np.int64),
    ]
    even_full_matrix, even_full_rhs = orbit_reduce_matrix_rhs(
        full_matrix,
        full_rhs,
        row_orbits,
    )

    assert np.linalg.norm(quarter_matrix - even_full_matrix) / np.linalg.norm(
        even_full_matrix
    ) < 1.0e-6
    assert np.linalg.norm(quarter_rhs - even_full_rhs) / np.linalg.norm(
        even_full_rhs
    ) < 1.0e-6

    result = json.loads(
        (
            tmp_path
            / "native-quarter-pair-atomic-yz-xz-session"
            / "quarter-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["implementation"] == (
        "swift_native_metal_pair_atomic_regular_plus_metal_duffy_blocks"
    )
    assert result["duffy_corrections"]["image_singular_correction"] is True
    assert result["duffy_corrections"]["image_adjacent_pairs"] == 15


def test_native_executable_corrected_mode_applies_duffy_on_tiny_mesh(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "entrywise")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_THREADS_PER_GROUP", "256")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_MATRIX_THREADS_PER_GROUP", "32")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_RHS_THREADS_PER_GROUP", "64")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_THREADS_PER_GROUP", "128")
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-corrected-session",
        session_id="native-corrected-test",
    ) as session:
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-corrected-assembly",
        )

    result = json.loads(
        (
            tmp_path
            / "native-corrected-session"
            / "native-corrected-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )

    assert result["implementation"] == "swift_native_metal_regular_plus_metal_duffy_blocks"
    assert result["assembly_mode"] == "corrected"
    assert result["duffy_corrections"]["implemented"] is True
    assert (
        result["duffy_corrections"]["implementation"]
        == "metal_duffy_blocks_cpu_reduction"
    )
    assert result["duffy_corrections"]["block_seconds"] > 0.0
    assert result["duffy_corrections"]["reduction_seconds"] >= 0.0
    assert result["duffy_corrections"]["metal_dispatch"]["kernel"] == "duffy_delta_blocks"
    assert result["metal_dispatch"]["matrix"]["env"] == (
        "HORNLAB_METAL_BEM_NATIVE_MATRIX_THREADS_PER_GROUP"
    )
    assert result["metal_dispatch"]["matrix"]["requested_threads_per_threadgroup"] == 32
    assert result["metal_dispatch"]["rhs"]["env"] == (
        "HORNLAB_METAL_BEM_NATIVE_RHS_THREADS_PER_GROUP"
    )
    assert result["metal_dispatch"]["rhs"]["requested_threads_per_threadgroup"] == 64
    assert result["duffy_corrections"]["metal_dispatch"]["env"] == (
        "HORNLAB_METAL_BEM_NATIVE_DUFFY_THREADS_PER_GROUP"
    )
    assert (
        result["duffy_corrections"]["metal_dispatch"][
            "requested_threads_per_threadgroup"
        ]
        == 128
    )
    assert result["duffy_corrections"]["planned_pairs"] == {
        "coincident": 2,
        "edge": 2,
        "total": 4,
        "vertex": 0,
    }
    assert result["duffy_corrections"]["raw_triplets_if_expanded"] == 36
    assert result["duffy_corrections"]["unique_triplets"] == 16
    assert result["duffy_corrections"]["correction_seconds"] > 0.0
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(rhs))
    assert np.linalg.norm(matrix) > 0.0
    assert np.linalg.norm(rhs) > 0.0


def test_native_executable_yz_symmetry_matches_even_full_domain_solve(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "optimized")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_full_buffers(),
        work_dir=tmp_path / "native-full-yz-session",
        session_id="native-full-yz-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_half_buffers(),
        work_dir=tmp_path / "native-half-yz-session",
        session_id="native-half-yz-test",
        symmetry_plane="yz",
    ) as half_session:
        half_assembly = half_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="half-assembly",
        )

        half_matrix = np.fromfile(
            half_assembly.matrix_real_f32, dtype="<f4",
        ).reshape(half_assembly.matrix_shape) + 1j * np.fromfile(
            half_assembly.matrix_imag_f32, dtype="<f4",
        ).reshape(half_assembly.matrix_shape)
        half_rhs = np.fromfile(half_assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
            half_assembly.rhs_imag_f32,
            dtype="<f4",
        )
        half_pressure = np.linalg.solve(half_matrix, half_rhs).astype(np.complex64)
        half_field = half_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            half_pressure,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            np.array([[0.25, 0.2, 1.0], [-0.25, 0.2, 1.0]], dtype=np.float32),
            batch_id="symmetry-points",
            operation_id="half-field",
        )

    full_matrix = np.fromfile(
        full_assembly.matrix_real_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape) + 1j * np.fromfile(
        full_assembly.matrix_imag_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape)
    full_rhs = np.fromfile(full_assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_assembly.rhs_imag_f32,
        dtype="<f4",
    )
    real_dofs = np.array([0, 1, 2], dtype=np.int64)
    mirror_dofs = np.array([3, 4, 5], dtype=np.int64)
    even_full_matrix = (
        full_matrix[np.ix_(real_dofs, real_dofs)]
        + full_matrix[np.ix_(real_dofs, mirror_dofs)]
    )
    even_full_rhs = full_rhs[real_dofs]
    even_full_pressure = np.linalg.solve(
        even_full_matrix,
        even_full_rhs,
    ).astype(np.complex64)
    full_pressure = np.zeros(6, dtype=np.complex64)
    full_pressure[real_dofs] = even_full_pressure
    full_pressure[mirror_dofs] = even_full_pressure

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_full_buffers(),
        work_dir=tmp_path / "native-full-yz-field-session",
        session_id="native-full-yz-field-test",
    ) as full_field_session:
        full_field = full_field_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            full_pressure,
            np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64),
            np.array([[0.25, 0.2, 1.0], [-0.25, 0.2, 1.0]], dtype=np.float32),
            batch_id="full-points",
            operation_id="full-field",
        )

    assert np.allclose(
        even_full_pressure,
        half_pressure,
        rtol=5.0e-4,
        atol=5.0e-5,
    )

    full_values = np.fromfile(full_field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_field.pressure_imag_f32,
        dtype="<f4",
    )
    half_values = np.fromfile(half_field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        half_field.pressure_imag_f32,
        dtype="<f4",
    )
    assert np.allclose(full_values, half_values, rtol=5.0e-4, atol=5.0e-5)

    result = json.loads(
        (
            tmp_path
            / "native-half-yz-session"
            / "half-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "yz"


def test_native_executable_yz_xz_symmetry_matches_even_full_domain_solve(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "optimized")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_full_buffers(),
        work_dir=tmp_path / "native-full-yz-xz-session",
        session_id="native-full-yz-xz-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.ones(4, dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_quarter_buffers(),
        work_dir=tmp_path / "native-quarter-yz-xz-session",
        session_id="native-quarter-yz-xz-test",
        symmetry_plane="yz+xz",
    ) as quarter_session:
        quarter_assembly = quarter_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="quarter-assembly",
        )

        quarter_matrix = np.fromfile(
            quarter_assembly.matrix_real_f32, dtype="<f4",
        ).reshape(quarter_assembly.matrix_shape) + 1j * np.fromfile(
            quarter_assembly.matrix_imag_f32, dtype="<f4",
        ).reshape(quarter_assembly.matrix_shape)
        quarter_rhs = np.fromfile(
            quarter_assembly.rhs_real_f32, dtype="<f4",
        ) + 1j * np.fromfile(
            quarter_assembly.rhs_imag_f32,
            dtype="<f4",
        )
        quarter_pressure = np.linalg.solve(
            quarter_matrix,
            quarter_rhs,
        ).astype(np.complex64)
        quarter_field = quarter_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            quarter_pressure,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            np.array(
                [[0.25, -0.25, 0.25], [0.2, 0.2, -0.2], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            batch_id="symmetry-points",
            operation_id="quarter-field",
        )

    full_matrix = np.fromfile(
        full_assembly.matrix_real_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape) + 1j * np.fromfile(
        full_assembly.matrix_imag_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape)
    full_rhs = np.fromfile(full_assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_assembly.rhs_imag_f32,
        dtype="<f4",
    )
    row_orbits = [
        np.array([0], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        np.array([2, 4], dtype=np.int64),
    ]
    col_orbits = row_orbits
    even_full_matrix = np.array(
        [
            [
                full_matrix[np.ix_(row_orbits[row], col_orbits[col])].sum()
                for col in range(3)
            ]
            for row in range(3)
        ],
        dtype=np.complex64,
    )
    even_full_rhs = np.array(
        [full_rhs[row_dofs].sum() for row_dofs in row_orbits],
        dtype=np.complex64,
    )
    assert np.allclose(
        even_full_matrix,
        quarter_matrix,
        rtol=5.0e-4,
        atol=5.0e-5,
    )
    assert np.allclose(
        even_full_rhs,
        quarter_rhs,
        rtol=5.0e-4,
        atol=5.0e-5,
    )
    even_full_pressure = np.linalg.solve(
        even_full_matrix,
        even_full_rhs,
    ).astype(np.complex64)
    full_pressure = np.zeros(5, dtype=np.complex64)
    for value, image_dofs in zip(even_full_pressure, col_orbits, strict=True):
        full_pressure[image_dofs] = value

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_full_buffers(),
        work_dir=tmp_path / "native-full-yz-xz-field-session",
        session_id="native-full-yz-xz-field-test",
    ) as full_field_session:
        full_field = full_field_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            full_pressure,
            np.ones(4, dtype=np.complex64),
            np.array(
                [[0.25, -0.25, 0.25], [0.2, 0.2, -0.2], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            batch_id="full-points",
            operation_id="full-field",
        )

    assert np.allclose(
        even_full_pressure,
        quarter_pressure,
        rtol=5.0e-4,
        atol=5.0e-5,
    )

    full_values = np.fromfile(full_field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_field.pressure_imag_f32,
        dtype="<f4",
    )
    quarter_values = np.fromfile(
        quarter_field.pressure_real_f32,
        dtype="<f4",
    ) + 1j * np.fromfile(
        quarter_field.pressure_imag_f32,
        dtype="<f4",
    )
    assert np.allclose(full_values, quarter_values, rtol=5.0e-4, atol=5.0e-5)

    result = json.loads(
        (
            tmp_path
            / "native-quarter-yz-xz-session"
            / "quarter-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "yz+xz"


def test_native_executable_corrected_yz_xz_symmetry_applies_image_duffy(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_full_buffers(),
        work_dir=tmp_path / "native-full-corrected-yz-xz-session",
        session_id="native-full-corrected-yz-xz-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.ones(4, dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_quarter_buffers(),
        work_dir=tmp_path / "native-quarter-corrected-yz-xz-session",
        session_id="native-quarter-corrected-yz-xz-test",
        symmetry_plane="yz+xz",
    ) as quarter_session:
        quarter_assembly = quarter_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="quarter-assembly",
        )

    full_matrix, full_rhs = _read_complex_assembly(full_assembly)
    quarter_matrix, quarter_rhs = _read_complex_assembly(quarter_assembly)
    row_orbits = [
        np.array([0], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        np.array([2, 4], dtype=np.int64),
    ]
    even_full_matrix, even_full_rhs = orbit_reduce_matrix_rhs(
        full_matrix,
        full_rhs,
        row_orbits,
    )

    assert np.linalg.norm(quarter_matrix - even_full_matrix) / np.linalg.norm(
        even_full_matrix
    ) < 1.0e-6
    assert np.linalg.norm(quarter_rhs - even_full_rhs) / np.linalg.norm(
        even_full_rhs
    ) < 1.0e-6

    result = json.loads(
        (
            tmp_path
            / "native-quarter-corrected-yz-xz-session"
            / "quarter-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "yz+xz"
    assert result["duffy_corrections"]["image_singular_correction"] is True
    assert result["duffy_corrections"]["image_adjacent_pairs"] == 15


def test_native_executable_xy_symmetry_matches_even_full_domain_solve(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "optimized")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_mirror_full_buffers(),
        work_dir=tmp_path / "native-full-xy-session",
        session_id="native-full-xy-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_half_buffers(),
        work_dir=tmp_path / "native-half-xy-session",
        session_id="native-half-xy-test",
        symmetry_plane="xy",
    ) as half_session:
        half_assembly = half_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="half-assembly",
        )

        half_matrix = np.fromfile(
            half_assembly.matrix_real_f32, dtype="<f4",
        ).reshape(half_assembly.matrix_shape) + 1j * np.fromfile(
            half_assembly.matrix_imag_f32, dtype="<f4",
        ).reshape(half_assembly.matrix_shape)
        half_rhs = np.fromfile(half_assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
            half_assembly.rhs_imag_f32,
            dtype="<f4",
        )
        half_pressure = np.linalg.solve(half_matrix, half_rhs).astype(np.complex64)
        half_field = half_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            half_pressure,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            np.array([[0.25, 0.2, 1.0], [0.25, 0.2, -1.0]], dtype=np.float32),
            batch_id="symmetry-points",
            operation_id="half-field",
        )

    full_matrix = np.fromfile(
        full_assembly.matrix_real_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape) + 1j * np.fromfile(
        full_assembly.matrix_imag_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape)
    full_rhs = np.fromfile(full_assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_assembly.rhs_imag_f32,
        dtype="<f4",
    )
    real_dofs = np.array([0, 1, 2], dtype=np.int64)
    mirror_dofs = np.array([3, 4, 5], dtype=np.int64)
    even_full_matrix = (
        full_matrix[np.ix_(real_dofs, real_dofs)]
        + full_matrix[np.ix_(real_dofs, mirror_dofs)]
    )
    even_full_rhs = full_rhs[real_dofs]
    even_full_pressure = np.linalg.solve(
        even_full_matrix,
        even_full_rhs,
    ).astype(np.complex64)
    full_pressure = np.zeros(6, dtype=np.complex64)
    full_pressure[real_dofs] = even_full_pressure
    full_pressure[mirror_dofs] = even_full_pressure

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_mirror_full_buffers(),
        work_dir=tmp_path / "native-full-xy-field-session",
        session_id="native-full-xy-field-test",
    ) as full_field_session:
        full_field = full_field_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            full_pressure,
            np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64),
            np.array([[0.25, 0.2, 1.0], [0.25, 0.2, -1.0]], dtype=np.float32),
            batch_id="full-points",
            operation_id="full-field",
        )

    assert np.allclose(
        even_full_pressure,
        half_pressure,
        rtol=5.0e-4,
        atol=5.0e-5,
    )

    full_values = np.fromfile(full_field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_field.pressure_imag_f32,
        dtype="<f4",
    )
    half_values = np.fromfile(half_field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        half_field.pressure_imag_f32,
        dtype="<f4",
    )
    assert np.allclose(full_values, half_values, rtol=5.0e-4, atol=5.0e-5)

    result = json.loads(
        (
            tmp_path
            / "native-half-xy-session"
            / "half-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "xy"


def test_native_executable_corrected_xy_symmetry_applies_image_duffy(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_shared_full_buffers(),
        work_dir=tmp_path / "native-full-corrected-xy-session",
        session_id="native-full-corrected-xy-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.ones(2, dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_half_buffers(),
        work_dir=tmp_path / "native-half-corrected-xy-session",
        session_id="native-half-corrected-xy-test",
        symmetry_plane="xy",
    ) as half_session:
        half_assembly = half_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="half-assembly",
        )

    full_matrix, full_rhs = _read_complex_assembly(full_assembly)
    half_matrix, half_rhs = _read_complex_assembly(half_assembly)
    row_orbits = [
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
    ]
    even_full_matrix, even_full_rhs = orbit_reduce_matrix_rhs(
        full_matrix,
        full_rhs,
        row_orbits,
    )

    assert np.linalg.norm(half_matrix - even_full_matrix) / np.linalg.norm(
        even_full_matrix
    ) < 1.0e-6
    assert np.linalg.norm(half_rhs - even_full_rhs) / np.linalg.norm(
        even_full_rhs
    ) < 1.0e-6

    result = json.loads(
        (
            tmp_path
            / "native-half-corrected-xy-session"
            / "half-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "xy"
    assert result["duffy_corrections"]["image_singular_correction"] is True
    assert result["duffy_corrections"]["image_adjacent_pairs"] >= 1


def test_native_executable_xy_image_duffy_fires_for_near_plane_vertex(
    monkeypatch,
    tmp_path,
):
    """A CAD vertex at z=5e-7 must snap onto the symmetry plane.

    5e-7 sits in the crack between Python plane validation (1e-7) and the
    Swift image-pair coordinate keys (1e-6 quantization): without snapping,
    the vertex neither counts as on-plane nor matches its own mirror, so
    image Duffy pairs silently stop firing.
    """
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 5.0e-7, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0], [1], [2]], dtype=np.int64),
        number_of_elements=1,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2]], dtype=np.int64),
        global_dof_count=3,
    )
    near_plane_buffers = build_metal_geometry_buffers(
        grid,
        np.array([2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=1),
    )
    np.testing.assert_array_equal(
        near_plane_buffers.vertices_3xn_f32,
        _tiny_xy_half_buffers().vertices_3xn_f32,
    )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=near_plane_buffers,
        work_dir=tmp_path / "native-near-plane-xy-session",
        session_id="native-near-plane-xy-test",
        symmetry_plane="xy",
    ) as session:
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="near-plane-assembly",
        )

    result = json.loads(
        (
            tmp_path
            / "native-near-plane-xy-session"
            / "near-plane-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "xy"
    assert result["duffy_corrections"]["image_singular_correction"] is True
    assert result["duffy_corrections"]["image_adjacent_pairs"] >= 1
    # The mirrored triangle shares the full on-plane edge, so the image pair
    # must be edge-kind; without snapping, the 5e-7 vertex fails to match its
    # mirror and the pair silently degrades to a vertex-kind correction.
    assert result["duffy_corrections"]["planned_pairs"]["edge"] >= 1

    half_matrix, half_rhs = _read_complex_assembly(assembly)
    assert np.all(np.isfinite(half_matrix))
    assert np.all(np.isfinite(half_rhs))


def test_native_executable_rejects_payload_without_wavenumber(tmp_path):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available or status.helper_executable_path is None:
        pytest.skip("compiled native helper unavailable")

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-missing-k-session",
        session_id="native-missing-k-test",
    ) as session:
        session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="seed-assembly",
        )
        op_dir = session.info.work_dir / "seed-assembly"
        payload_path = op_dir / "assembly.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        del payload["k_real_f32"]
        doctored_path = op_dir / "assembly-missing-k.json"
        doctored_path.write_text(json.dumps(payload), encoding="utf-8")

        import subprocess

        completed = subprocess.run(
            [
                str(status.helper_executable_path),
                "assemble_standard_neumann",
                str(session.info.manifest_path),
                str(doctored_path),
                str(op_dir / "missing-k-result.json"),
            ],
            capture_output=True,
            text=True,
        )

    assert completed.returncode != 0
    assert "k_real_f32" in completed.stderr + completed.stdout


def test_native_executable_gpu_duffy_matches_cpu_duffy_on_tiny_mesh(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    def run_case(name: str, duffy_mode: str):
        monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
        monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", duffy_mode)
        with MetalNativeStandardSession.create_session(
            geometry_buffers=_tiny_geometry_buffers(),
            work_dir=tmp_path / f"native-{name}-duffy-session",
            session_id=f"native-{name}-duffy-test",
        ) as session:
            assembly = session.assemble_standard_neumann(
                100.0,
                1.8318326,
                np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
                operation_id=f"native-{name}-duffy-assembly",
            )
        matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
            assembly.matrix_shape
        ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
            assembly.matrix_shape
        )
        rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
            assembly.rhs_imag_f32,
            dtype="<f4",
        )
        return matrix, rhs

    gpu_matrix, gpu_rhs = run_case("gpu", "gpu_blocks")
    cpu_matrix, cpu_rhs = run_case("cpu", "cpu")

    assert np.linalg.norm(gpu_matrix - cpu_matrix) / np.linalg.norm(cpu_matrix) < 1e-5
    assert np.linalg.norm(gpu_rhs - cpu_rhs) / np.linalg.norm(cpu_rhs) < 1e-5


def test_native_executable_resident_batch_matches_single_assembly(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    neumann = np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-resident-batch-session",
        session_id="native-resident-batch-test",
    ) as session:
        single = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            neumann,
            operation_id="single-assembly",
        )
        batch = session.assemble_standard_neumann_batch(
            np.array([100.0], dtype=np.float64),
            np.array([1.8318326], dtype=np.float32),
            neumann.reshape(1, -1),
            operation_id="resident-batch-assembly",
        )[0]

    single_matrix = np.fromfile(single.matrix_real_f32, dtype="<f4").reshape(
        single.matrix_shape
    ) + 1j * np.fromfile(single.matrix_imag_f32, dtype="<f4").reshape(
        single.matrix_shape
    )
    single_rhs = np.fromfile(single.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        single.rhs_imag_f32,
        dtype="<f4",
    )
    batch_matrix = np.fromfile(batch.matrix_real_f32, dtype="<f4").reshape(
        batch.matrix_shape
    ) + 1j * np.fromfile(batch.matrix_imag_f32, dtype="<f4").reshape(
        batch.matrix_shape
    )
    batch_rhs = np.fromfile(batch.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        batch.rhs_imag_f32,
        dtype="<f4",
    )

    result = json.loads(
        (
            tmp_path
            / "native-resident-batch-session"
            / "resident-batch-assembly"
            / "assembly-batch-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["op"] == "assemble_standard_neumann_batch_result"
    assert result["resident_reuse"]["geometry_buffers"] is True
    assert result["resident_reuse"]["duffy_rules"] is True
    assert np.linalg.norm(batch_matrix - single_matrix) / np.linalg.norm(single_matrix) < 1e-5
    assert np.linalg.norm(batch_rhs - single_rhs) / np.linalg.norm(single_rhs) < 1e-5


def test_native_executable_resident_assembly_solve_matches_python_solve(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.5j]], dtype=np.complex64)
    frequency_hz = np.array([100.0], dtype=np.float64)
    k_real = np.array([1.8318326], dtype=np.float32)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-resident-assembly-solve-session",
        session_id="native-resident-assembly-solve-test",
    ) as session:
        assembly = session.assemble_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="resident-batch-assembly",
        )[0]
        solved = session.assemble_solve_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="resident-batch-assembly-solve",
        )[0]

    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )
    expected = np.linalg.solve(matrix, rhs).astype(np.complex64)
    pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-resident-assembly-solve-session"
            / "resident-batch-assembly-solve"
            / "assembly-solve-batch-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["op"] == "assemble_solve_standard_neumann_batch_result"
    assert result["implementation"] == (
        "swift_native_resident_metal_assembly_accelerate_solve_batch"
    )
    assert result["resident_reuse"]["duffy_reduction_plan"] is True
    assert solved.lapack_info == 0
    assert solved.assembly_s > 0.0
    assert solved.dense_solve_s > 0.0
    assert np.linalg.norm(pressure - expected) / np.linalg.norm(expected) < 1e-5


def test_native_executable_resident_assembly_solve_lu_factor_variant_matches_python(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv(
        "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_IMPL",
        "cgetrf_cgetrs",
    )
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.5j]], dtype=np.complex64)
    frequency_hz = np.array([100.0], dtype=np.float64)
    k_real = np.array([1.8318326], dtype=np.float32)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-resident-assembly-solve-lu-factor-session",
        session_id="native-resident-assembly-solve-lu-factor-test",
    ) as session:
        assembly = session.assemble_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="resident-batch-assembly",
        )[0]
        solved = session.assemble_solve_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="resident-batch-assembly-solve",
        )[0]

    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )
    expected = np.linalg.solve(matrix, rhs).astype(np.complex64)
    pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-resident-assembly-solve-lu-factor-session"
            / "resident-batch-assembly-solve"
            / "assembly-solve-batch-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["cases"][0]["solve_implementation"] == (
        "accelerate_lapack_cgetrf_cgetrs"
    )
    assert solved.lapack_info == 0
    assert solved.dense_solve_s > 0.0
    assert np.linalg.norm(pressure - expected) / np.linalg.norm(expected) < 1e-5


def test_native_executable_resident_assembly_solve_field_matches_split_path(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.5j]], dtype=np.complex64)
    frequency_hz = np.array([100.0], dtype=np.float64)
    k_real = np.array([1.8318326], dtype=np.float32)
    observation_points = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
        dtype=np.float32,
    )
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-resident-assembly-solve-field-session",
        session_id="native-resident-assembly-solve-field-test",
    ) as session:
        combined = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="resident-batch-assembly-solve-field",
            source_tags=[2],
            impedance_source_tag=2,
        )[0]
        solved = session.assemble_solve_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="resident-batch-assembly-solve",
        )[0]
        pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
            solved.pressure_imag_f32,
            dtype="<f4",
        )
        field = session.evaluate_standard_exterior_batch(
            frequency_hz,
            k_real,
            pressure.reshape(1, -1),
            neumann,
            observation_points.T,
            operation_id="resident-batch-field",
        )[0]
        reduced = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="resident-batch-assembly-solve-field-reduced",
            source_tags=[2],
            impedance_source_tag=2,
            write_surface_pressure=False,
        )[0]
        batched = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="resident-batch-assembly-solve-field-batched-output",
            source_tags=[2],
            impedance_source_tag=2,
            write_surface_pressure=False,
            write_batched_field=True,
        )[0]

    combined_pressure = np.fromfile(combined.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        combined.pressure_imag_f32,
        dtype="<f4",
    )
    split_field = np.fromfile(field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        field.pressure_imag_f32,
        dtype="<f4",
    )
    combined_field = np.fromfile(combined.field_real_f32, dtype="<f4") + 1j * np.fromfile(
        combined.field_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-resident-assembly-solve-field-session"
            / "resident-batch-assembly-solve-field"
            / "assembly-solve-field-batch-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["op"] == "assemble_solve_evaluate_standard_neumann_batch_result"
    assert result["implementation"] == (
        "swift_native_resident_metal_assembly_accelerate_solve_field_batch"
    )
    assert result["resident_reuse"]["field_output_buffers"] is True
    assert result["resident_reuse"]["observation_points_buffer"] is True
    assert combined.lapack_info == 0
    assert combined.assembly_s > 0.0
    assert combined.dense_solve_s > 0.0
    assert combined.field_s > 0.0
    # cgecon condition estimate must ride along in per-case diagnostics so
    # interior-resonance spikes in sweeps are attributable.
    assert 0.0 < combined.diagnostics["dense_solve_rcond"] <= 1.0
    assert combined.diagnostics["dense_solve_condition_1norm"] >= 1.0
    expected_source_avg = (
        combined_pressure[0] + combined_pressure[2] + combined_pressure[3]
    ) / 3.0
    assert combined.impedance == pytest.approx(expected_source_avg, rel=1.0e-6)
    assert combined.surface_pressure_avg is not None
    assert combined.surface_pressure_avg[2] == pytest.approx(
        expected_source_avg,
        rel=1.0e-6,
    )
    assert reduced.pressure_real_f32 is None
    assert reduced.pressure_imag_f32 is None
    assert reduced.impedance == pytest.approx(expected_source_avg, rel=1.0e-6)
    assert reduced.surface_pressure_avg is not None
    assert reduced.surface_pressure_avg[2] == pytest.approx(
        expected_source_avg,
        rel=1.0e-6,
    )
    assert batched.pressure_real_f32 is None
    assert batched.pressure_imag_f32 is None
    assert batched.field_row_index == 0
    assert batched.field_batch_shape == (1, 2)
    batched_field = (
        np.fromfile(batched.field_real_f32, dtype="<f4").reshape(batched.field_batch_shape)
        + 1j
        * np.fromfile(batched.field_imag_f32, dtype="<f4").reshape(
            batched.field_batch_shape
        )
    )
    assert np.linalg.norm(batched_field[0] - split_field) / np.linalg.norm(split_field) < 1e-5
    assert np.linalg.norm(combined_pressure - pressure) / np.linalg.norm(pressure) < 1e-5
    assert np.linalg.norm(combined_field - split_field) / np.linalg.norm(split_field) < 1e-5


def test_native_executable_complex_k_robin_tags_8_9_solve_field(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64)
    frequency_hz = np.array([172.0], dtype=np.float64)
    k_real = np.array([np.float32(2.0 * np.pi * 172.0 / 343.0)], dtype=np.float32)
    k_imag = (k_real * np.float32(0.005)).astype(np.float32)
    observation_points = np.array(
        [[0.0, 0.0, 0.7], [0.2, 0.0, 0.8]],
        dtype=np.float32,
    )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_robin_geometry_buffers(),
        work_dir=tmp_path / "native-complex-robin-session",
        session_id="native-complex-robin-test",
    ) as session:
        solved = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            k_imag_f32=k_imag,
            impedance_sources={8: 0.05 + 0.0j, 9: 0.02 + 0.01j},
            operation_id="resident-complex-robin",
            source_tags=[2],
            impedance_source_tag=2,
        )[0]

    pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )
    field = np.fromfile(solved.field_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.field_imag_f32,
        dtype="<f4",
    )

    assert solved.lapack_info == 0
    assert np.all(np.isfinite(pressure))
    assert np.all(np.isfinite(field))
    assert solved.diagnostics["assembly_mode"] == "reference"
    assert solved.diagnostics["assembly_implementation"] == (
        "swift_native_reference_complex_robin_quadrature"
    )
    assert solved.diagnostics["complex_k"] is True
    assert solved.diagnostics["robin_boundary"] is True
    assert solved.diagnostics["field_uses_total_neumann"] is True
    assert 0.0 < solved.diagnostics["dense_solve_rcond"] <= 1.0
    assert solved.diagnostics["dense_solve_condition_1norm"] >= 1.0


def test_native_executable_streams_per_case_results(monkeypatch, tmp_path):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    neumann = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.5j],
            [0.5 + 0.0j, 0.0 + 0.25j],
            [0.25 + 0.0j, 0.0 + 0.125j],
        ],
        dtype=np.complex64,
    )
    frequency_hz = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    k_real = np.array([1.83, 3.66, 5.49], dtype=np.float32)
    observation_points = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
        dtype=np.float32,
    )
    streamed_calls: list[tuple[int, object]] = []
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-streamed-case-results-session",
        session_id="native-streamed-case-results-test",
    ) as session:
        streamed = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="streamed-batch",
            source_tags=[2],
            impedance_source_tag=2,
            on_case_result=lambda i, solved: streamed_calls.append((i, solved)),
        )
        oneshot = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="oneshot-batch",
            source_tags=[2],
            impedance_source_tag=2,
        )

        assert [index for index, _ in streamed_calls] == [0, 1, 2]
        assert [solved for _, solved in streamed_calls] == streamed
        assert len(streamed) == len(oneshot) == 3
        case_dir = (
            tmp_path
            / "native-streamed-case-results-session"
            / "streamed-batch"
            / "case-results"
        )
        case_files = sorted(path.name for path in case_dir.glob("case-*.json"))
        assert case_files == ["case-0000.json", "case-0001.json", "case-0002.json"]
        for solved, reference in zip(streamed, oneshot):
            assert solved.frequency_hz == reference.frequency_hz
            assert solved.lapack_info == 0
            assert solved.impedance == pytest.approx(reference.impedance, rel=1e-6)
            streamed_field = np.fromfile(solved.field_real_f32, dtype="<f4")
            reference_field = np.fromfile(reference.field_real_f32, dtype="<f4")
            np.testing.assert_allclose(streamed_field, reference_field, rtol=1e-5)
            # Whole-batch diagnostics are only known once the batch ends, so
            # streamed per-case diagnostics must omit them rather than guess.
            assert "batch" not in solved.diagnostics
            assert "batch" in reference.diagnostics
        result = json.loads(
            (
                tmp_path
                / "native-streamed-case-results-session"
                / "streamed-batch"
                / "assembly-solve-field-batch-result.json"
            ).read_text(encoding="utf-8")
        )
        assert result["streamed_case_results"] is True

        early = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="early-stop-batch",
            source_tags=[2],
            impedance_source_tag=2,
            on_case_result=lambda i, solved: False,
        )
        assert len(early) == 1
        assert early[0].frequency_hz == pytest.approx(100.0)

        with pytest.raises(ValueError, match="write_batched_field"):
            session.assemble_solve_evaluate_standard_neumann_batch(
                frequency_hz,
                k_real,
                neumann,
                observation_points,
                operation_id="streamed-batched-field",
                source_tags=[2],
                impedance_source_tag=2,
                write_batched_field=True,
                on_case_result=lambda i, solved: None,
            )


def test_native_executable_field_evaluation_on_tiny_mesh(tmp_path):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-field-session",
        session_id="native-field-test",
    ) as session:
        field = session.evaluate_standard_exterior(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.5 + 0.1j, 0.25 + 0.0j, 0.1 - 0.2j], dtype=np.complex64),
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            batch_id="horizontal",
            operation_id="native-field-eval",
        )

    result = json.loads(
        (
            tmp_path
            / "native-field-session"
            / "native-field-eval"
            / "field-result.json"
        ).read_text(encoding="utf-8")
    )
    pressure = np.fromfile(field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        field.pressure_imag_f32,
        dtype="<f4",
    )

    assert result["implementation"] == "swift_native_reference_regular_field"
    assert result["field_mode"] == "reference"
    assert result["field_seconds"] > 0.0
    assert field.shape == (2,)
    assert np.all(np.isfinite(pressure))
    assert np.linalg.norm(pressure) > 0.0


def test_native_executable_optimized_field_matches_reference_on_tiny_mesh(
    tmp_path,
    monkeypatch,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "parity")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_THREADS_PER_GROUP", "32")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_THREADS_PER_GROUP", "64")
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-field-parity-session",
        session_id="native-field-parity-test",
    ) as session:
        field = session.evaluate_standard_exterior(
            100.0,
            1.8318326,
            np.array(
                [1.0 + 0.0j, 0.5 + 0.1j, 0.25 + 0.0j, 0.1 - 0.2j],
                dtype=np.complex64,
            ),
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            batch_id="horizontal",
            operation_id="native-field-parity",
        )

    result = json.loads(
        (
            tmp_path
            / "native-field-parity-session"
            / "native-field-parity"
            / "field-result.json"
        ).read_text(encoding="utf-8")
    )
    pressure = np.fromfile(field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        field.pressure_imag_f32,
        dtype="<f4",
    )

    assert result["implementation"] == "swift_native_metal_regular_field"
    assert result["field_mode"] == "parity"
    assert result["reference_parity"]["field_relative_l2"] < 1.0e-4
    assert result["reference_parity"]["tolerance"] == 1.0e-4
    assert result["metal_dispatch"]["field"]["env"] == (
        "HORNLAB_METAL_BEM_NATIVE_FIELD_THREADS_PER_GROUP"
    )
    assert result["metal_dispatch"]["field"]["requested_threads_per_threadgroup"] == 64
    assert result["metal_dispatch"]["field"]["threads_per_threadgroup"] == 64
    assert field.shape == (2,)
    assert np.all(np.isfinite(pressure))
    assert np.linalg.norm(pressure) > 0.0


def test_native_executable_resident_batch_matches_single_field(
    tmp_path,
    monkeypatch,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    pressure = np.array(
        [1.0 + 0.0j, 0.5 + 0.1j, 0.25 + 0.0j, 0.1 - 0.2j],
        dtype=np.complex64,
    )
    neumann = np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64)
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-field-batch-session",
        session_id="native-field-batch-test",
    ) as session:
        single = session.evaluate_standard_exterior(
            100.0,
            1.8318326,
            pressure,
            neumann,
            points,
            batch_id="single",
            operation_id="single-field",
        )
        batch = session.evaluate_standard_exterior_batch(
            np.array([100.0], dtype=np.float64),
            np.array([1.8318326], dtype=np.float32),
            pressure.reshape(1, -1),
            neumann.reshape(1, -1),
            points,
            batch_id="batch",
            operation_id="resident-field-batch",
        )[0]

    single_field = np.fromfile(single.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        single.pressure_imag_f32,
        dtype="<f4",
    )
    batch_field = np.fromfile(batch.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        batch.pressure_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-field-batch-session"
            / "resident-field-batch"
            / "field-batch-result.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            tmp_path
            / "native-field-batch-session"
            / "resident-field-batch"
            / "field-batch.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["observation_points"]["shape"] == [3, 2]
    assert "observation_points" not in manifest["cases"][0]
    assert result["op"] == "evaluate_standard_exterior_batch_result"
    assert result["resident_reuse"]["geometry_buffers"] is True
    assert result["resident_reuse"]["field_output_buffers"] is True
    assert np.linalg.norm(batch_field - single_field) / np.linalg.norm(single_field) < 1e-5


@pytest.mark.parametrize(
    ("case_name", "mutate", "message"),
    [
        (
            "bad-path",
            lambda manifest: manifest["mesh"]["vertices_f32"].update(
                {"path": "../outside.bin"}
            ),
            "must be relative",
        ),
        (
            "bad-byte-order",
            lambda manifest: manifest["mesh"]["vertices_f32"].update(
                {"byte_order": "big"}
            ),
            "byte_order must be little",
        ),
        (
            "bad-matrix-layout",
            lambda manifest: manifest.update({"matrix_layout": "column_major"}),
            "expected row_major_c matrix layout",
        ),
        (
            "bad-shape",
            lambda manifest: manifest["mesh"]["physical_tags_i32"].update(
                {"shape": [3]}
            ),
            "mesh.physical_tags_i32.shape",
        ),
    ],
)
def test_native_executable_validator_rejects_contract_violations(
    tmp_path,
    case_name,
    mutate,
    message,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-negative-session",
        session_id="native-negative-test",
    ) as session:
        manifest = json.loads(session.info.manifest_path.read_text(encoding="utf-8"))
        mutate(manifest)
        manifest_path = session.info.work_dir / f"{case_name}.json"
        result_path = session.info.work_dir / f"{case_name}-result.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(RuntimeError, match=message):
            validate_session_with_native_helper(manifest_path, result_path)


def test_native_config_defaults_to_packaged_helper_directory():
    config = MetalNativeRuntimeConfig()

    assert config.resolved_backend_dir == Path(native.__file__).resolve().parent


def test_metal_bem_backend_wraps_native_session_without_routing(monkeypatch):
    class FakeInfo:
        session_id = "adapter-test"

    class FakeResult:
        session_id = "adapter-test"
        frequency_hz = 100.0
        matrix_real_f32 = Path("A_re.bin")
        matrix_imag_f32 = Path("A_im.bin")
        rhs_real_f32 = Path("rhs_re.bin")
        rhs_imag_f32 = Path("rhs_im.bin")
        matrix_shape = (4, 4)
        rhs_shape = (4,)
        matrix_layout = "row_major_c"

    class FakeSession:
        info = FakeInfo()
        closed = False

        def validate_contract(self):
            return {"status": "ok"}

        def assemble_standard_neumann(self, frequency_hz, k_real, neumann, **kwargs):
            assert frequency_hz == 100.0
            assert neumann.shape == (2,)
            return FakeResult()

        def evaluate_standard_exterior(self, *args, **kwargs):
            return "field-result"

        def close(self):
            self.closed = True

    fake_session = FakeSession()

    monkeypatch.setattr(
        metal_backend.MetalNativeStandardSession,
        "create_session",
        lambda **kwargs: fake_session,
    )

    context = MetalBemBackend().create_context(geometry_buffers=object())
    try:
        assert isinstance(context, MetalBemContext)
        assert context.session_id == "adapter-test"
        assert context.validate_contract() == {"status": "ok"}

        system = context.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
        )

        assert isinstance(system, DenseBieSystem)
        assert system.matrix_shape == (4, 4)
        assert system.matrix_layout == "row_major_c"
        assert context.evaluate_field_batch(
            100.0,
            1.8318326,
            np.zeros(4, dtype=np.complex64),
            np.zeros(2, dtype=np.complex64),
            np.zeros((3, 1), dtype=np.float32),
        ) == "field-result"
    finally:
        context.close()

    assert fake_session.closed is True

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_solver.backends import AssemblyBackendUnavailable
from hornlab_solver.metal import session as metal_session
from hornlab_solver.metal.geometry import build_metal_geometry_buffers
from hornlab_solver.metal.runtime import MetalRuntimeStatus
from hornlab_solver.metal.session import (
    AssemblyPayload,
    BinaryArrayDescriptor,
    FieldPayload,
    INDEX_BASE,
    JuliaMetalValidationSession,
    MATRIX_LAYOUT_ROW_MAJOR_C,
    METAL_STANDARD_SCHEMA,
    MetalStandardSession,
    payload_to_manifest,
    read_json_manifest,
    write_binary_array,
    write_json_manifest,
)


def _mock_grid() -> SimpleNamespace:
    triangles_3xm = np.array(
        [
            [0, 0],
            [1, 2],
            [2, 3],
        ],
        dtype=np.int64,
    )
    return SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=triangles_3xm,
        number_of_elements=triangles_3xm.shape[1],
    )


def _mock_p1() -> SimpleNamespace:
    return SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )


def _mock_dp0() -> SimpleNamespace:
    return SimpleNamespace(global_dof_count=2)


def _buffers():
    return build_metal_geometry_buffers(
        _mock_grid(),
        np.array([1, 2], dtype=np.int32),
        _mock_p1(),
        _mock_dp0(),
    )


def _runtime_status(*, available: bool, tmp_path: Path) -> MetalRuntimeStatus:
    reasons = () if available else ("Packaged Julia/Metal backend assets missing.",)
    return MetalRuntimeStatus(
        available=available,
        platform_system="Darwin",
        platform_machine="arm64",
        is_macos=True,
        is_apple_silicon=True,
        julia_path="/usr/local/bin/julia" if available else None,
        julia_source="PATH" if available else None,
        backend_dir=tmp_path / "packaged-backend",
        backend_entrypoint=tmp_path / "packaged-backend" / "HornlabSolverMetal.jl",
        backend_project=tmp_path / "packaged-backend" / "Project.toml",
        backend_assets_present=available,
        smoke_test_ran=available,
        smoke_test_ok=available,
        smoke_test_error=None,
        reasons=reasons,
    )


def test_session_manifest_json_shape_and_relative_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(
        metal_session,
        "discover_runtime",
        lambda config=None, **kwargs: _runtime_status(
            available=True,
            tmp_path=tmp_path,
        ),
    )

    session = MetalStandardSession.create_session(
        geometry_buffers=_buffers(),
        work_dir=tmp_path / "session",
        session_id="metal-test-0001",
    )
    try:
        manifest = read_json_manifest(session.info.manifest_path)

        assert manifest["schema"] == METAL_STANDARD_SCHEMA
        assert manifest["op"] == "create_session"
        assert manifest["session_id"] == "metal-test-0001"
        assert manifest["index_base"] == INDEX_BASE
        assert manifest["matrix_layout"] == MATRIX_LAYOUT_ROW_MAJOR_C
        assert manifest["space"]["p1_dof_count"] == 4
        assert manifest["space"]["dp0_dof_count"] == 2
        assert manifest["assembly_scope"] == {
            "formulation": "standard_neumann",
            "basis_trial": "P1",
            "basis_test": "P1",
            "source_basis": "DP0",
            "symmetry_plane": None,
        }

        mesh = manifest["mesh"]
        assert mesh["vertices_f32"]["path"] == "geometry/vertices_3xn_f32.bin"
        assert mesh["vertices_f32"]["shape"] == [3, 4]
        assert mesh["vertices_f32"]["dtype"] == "float32"
        assert mesh["vertices_f32"]["byte_order"] == "little"
        assert mesh["vertices_f32"]["order"] == "C"
        assert mesh["triangles_i32"]["path"] == "geometry/triangles_3xm_i32.bin"
        assert mesh["triangles_i32"]["shape"] == [3, 2]
        assert mesh["triangles_i32"]["dtype"] == "int32"

        for descriptor in mesh.values():
            assert not Path(descriptor["path"]).is_absolute()
            assert (session.info.work_dir / descriptor["path"]).is_file()

        manifest_text = session.info.manifest_path.read_text(encoding="utf-8")
        assert "runs/scratch" not in manifest_text
        assert "metal-bem-probe" not in manifest_text
    finally:
        session.close()


def test_julia_validation_session_alias_preserves_original_api():
    assert JuliaMetalValidationSession is MetalStandardSession


def test_geometry_binary_buffers_are_little_endian_and_c_contiguous(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        metal_session,
        "discover_runtime",
        lambda config=None, **kwargs: _runtime_status(
            available=True,
            tmp_path=tmp_path,
        ),
    )

    session = MetalStandardSession.create_session(
        geometry_buffers=_buffers(),
        work_dir=tmp_path / "session",
    )
    try:
        manifest = read_json_manifest(session.info.manifest_path)
        vertices_path = session.info.work_dir / manifest["mesh"]["vertices_f32"]["path"]
        triangles_path = session.info.work_dir / manifest["mesh"]["triangles_i32"]["path"]

        vertices = np.fromfile(vertices_path, dtype="<f4").reshape((3, 4))
        triangles = np.fromfile(triangles_path, dtype="<i4").reshape((3, 2))

        assert vertices.flags.c_contiguous
        assert triangles.flags.c_contiguous
        assert vertices.dtype == np.dtype("<f4")
        assert triangles.dtype == np.dtype("<i4")
        np.testing.assert_array_equal(
            triangles,
            np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int32),
        )
        assert triangles_path.read_bytes()[4:8] == b"\x00\x00\x00\x00"
        assert triangles_path.read_bytes()[8:12] == b"\x01\x00\x00\x00"
    finally:
        session.close()


def test_write_binary_array_normalizes_noncontiguous_array(tmp_path):
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=">i4").T
    assert not array.flags.c_contiguous

    descriptor = write_binary_array(
        array,
        tmp_path / "array_i32.bin",
        dtype=np.int32,
        relative_to=tmp_path,
    )

    assert descriptor.to_manifest() == {
        "path": "array_i32.bin",
        "shape": [3, 2],
        "dtype": "int32",
        "byte_order": "little",
        "order": "C",
    }
    loaded = np.fromfile(tmp_path / "array_i32.bin", dtype="<i4").reshape((3, 2))
    np.testing.assert_array_equal(loaded, array)


def test_binary_descriptor_rejects_non_contract_paths_and_layouts():
    with pytest.raises(ValueError, match="relative"):
        BinaryArrayDescriptor("/tmp/array.bin", (1,), "float32")

    with pytest.raises(ValueError, match="little-endian"):
        BinaryArrayDescriptor("array.bin", (1,), "float32", byte_order="big")

    with pytest.raises(ValueError, match="C-contiguous"):
        BinaryArrayDescriptor("array.bin", (1,), "float32", order="F")

    with pytest.raises(ValueError, match="float32 or int32"):
        BinaryArrayDescriptor("array.bin", (1,), "float64")


def test_assembly_and_field_payload_manifests_keep_ipc_contract(tmp_path):
    vector_re = BinaryArrayDescriptor("in/re.bin", (2,), "float32")
    vector_im = BinaryArrayDescriptor("in/im.bin", (2,), "float32")
    matrix_re = BinaryArrayDescriptor("out/A_re.bin", (4, 4), "float32")
    matrix_im = BinaryArrayDescriptor("out/A_im.bin", (4, 4), "float32")
    rhs_re = BinaryArrayDescriptor("out/rhs_re.bin", (4,), "float32")
    rhs_im = BinaryArrayDescriptor("out/rhs_im.bin", (4,), "float32")

    assembly = AssemblyPayload(
        session_id="metal-test",
        frequency_hz=100.0,
        k_real_f32=1.8318326,
        neumann_dp0={"real_f32": vector_re, "imag_f32": vector_im},
        outputs={
            "A_real_f32": matrix_re,
            "A_imag_f32": matrix_im,
            "rhs_real_f32": rhs_re,
            "rhs_imag_f32": rhs_im,
        },
    )
    write_json_manifest(assembly, tmp_path / "assembly.json")

    assembly_manifest = read_json_manifest(tmp_path / "assembly.json")
    assert assembly_manifest["schema"] == METAL_STANDARD_SCHEMA
    assert assembly_manifest["op"] == "assemble_standard_neumann"
    assert assembly_manifest["index_base"] == INDEX_BASE
    assert assembly_manifest["outputs"]["matrix_layout"] == MATRIX_LAYOUT_ROW_MAJOR_C
    assert assembly_manifest["outputs"]["A_real_f32"]["path"] == "out/A_re.bin"

    field = FieldPayload(
        session_id="metal-test",
        batch_id="horizontal",
        frequency_hz=100.0,
        k_real_f32=1.8318326,
        pressure_p1={"real_f32": vector_re, "imag_f32": vector_im},
        neumann_dp0={"real_f32": vector_re, "imag_f32": vector_im},
        observation_points=BinaryArrayDescriptor(
            "in/obs_points_3xn_f32.bin",
            (3, 181),
            "float32",
        ),
        output={
            "pressure_real_f32": BinaryArrayDescriptor(
                "out/pressure_re.bin",
                (181,),
                "float32",
            ),
            "pressure_imag_f32": BinaryArrayDescriptor(
                "out/pressure_im.bin",
                (181,),
                "float32",
            ),
        },
    )
    write_json_manifest(field, tmp_path / "field.json")

    field_manifest = read_json_manifest(tmp_path / "field.json")
    assert field_manifest["schema"] == METAL_STANDARD_SCHEMA
    assert field_manifest["op"] == "evaluate_standard_exterior"
    assert field_manifest["index_base"] == INDEX_BASE
    assert field_manifest["observation_points"]["shape"] == [3, 181]
    assert field_manifest["output"]["pressure_real_f32"]["path"] == (
        "out/pressure_re.bin"
    )


def test_payload_contract_rejects_non_row_major_assembly():
    descriptor = BinaryArrayDescriptor("in/vector.bin", (2,), "float32")
    matrix = BinaryArrayDescriptor("out/A.bin", (4, 4), "float32")
    rhs = BinaryArrayDescriptor("out/rhs.bin", (4,), "float32")
    payload = AssemblyPayload(
        session_id="metal-test",
        frequency_hz=100.0,
        k_real_f32=1.8318326,
        neumann_dp0={"real_f32": descriptor, "imag_f32": descriptor},
        outputs={
            "A_real_f32": matrix,
            "A_imag_f32": matrix,
            "rhs_real_f32": rhs,
            "rhs_imag_f32": rhs,
        },
        matrix_layout="column_major",
    )

    with pytest.raises(ValueError, match="row_major_c"):
        payload_to_manifest(payload)


def test_payload_contract_rejects_geometry_shape_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        metal_session,
        "discover_runtime",
        lambda config=None, **kwargs: _runtime_status(
            available=True,
            tmp_path=tmp_path,
        ),
    )

    session = MetalStandardSession.create_session(
        geometry_buffers=_buffers(),
        work_dir=tmp_path / "session",
    )
    try:
        manifest = read_json_manifest(session.info.manifest_path)
        manifest["mesh"]["physical_tags_i32"]["shape"] = [3]

        with pytest.raises(ValueError, match="physical_tags_i32"):
            payload_to_manifest(manifest)
    finally:
        session.close()


def test_payload_contract_rejects_field_output_shape_mismatch():
    vector = BinaryArrayDescriptor("in/vector.bin", (2,), "float32")
    manifest = FieldPayload(
        session_id="metal-test",
        batch_id="horizontal",
        frequency_hz=100.0,
        k_real_f32=1.8318326,
        pressure_p1={"real_f32": vector, "imag_f32": vector},
        neumann_dp0={"real_f32": vector, "imag_f32": vector},
        observation_points=BinaryArrayDescriptor(
            "in/obs_points_3xn_f32.bin",
            (3, 3),
            "float32",
        ),
        output={
            "pressure_real_f32": BinaryArrayDescriptor(
                "out/pressure_re.bin",
                (2,),
                "float32",
            ),
            "pressure_imag_f32": BinaryArrayDescriptor(
                "out/pressure_im.bin",
                (2,),
                "float32",
            ),
        },
    )

    with pytest.raises(ValueError, match="pressure_real_f32"):
        payload_to_manifest(manifest)


def test_session_fails_cleanly_without_packaged_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(
        metal_session,
        "discover_runtime",
        lambda config=None, **kwargs: _runtime_status(
            available=False,
            tmp_path=tmp_path,
        ),
    )

    with pytest.raises(AssemblyBackendUnavailable, match="Julia/Metal backend"):
        MetalStandardSession.create_session(
            geometry_buffers=_buffers(),
            work_dir=tmp_path / "session",
        )

    assert not (tmp_path / "session").exists()


def test_session_invokes_packaged_backend_for_assembly_and_field(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        metal_session,
        "discover_runtime",
        lambda config=None, **kwargs: _runtime_status(
            available=True,
            tmp_path=tmp_path,
        ),
    )

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        op = command[3]
        result_path = Path(command[6])
        if op == "assemble_standard_neumann":
            payload = read_json_manifest(Path(command[5]))
            for descriptor in payload["outputs"].values():
                if isinstance(descriptor, dict):
                    path = Path(command[4]).parent / descriptor["path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.zeros(descriptor["shape"], dtype=np.float32).tofile(path)
            result_path.write_text(
                """{
  "schema": "hornlab.metal.standard.v1",
  "op": "assemble_standard_neumann_result",
  "session_id": "metal-test",
  "frequency_hz": 100.0,
  "matrix_layout": "row_major_c",
  "matrix_shape": [4, 4],
  "rhs_shape": [4],
  "matrix_real_f32": "assembly-100p000000-test/outputs/A_re_f32.bin",
  "matrix_imag_f32": "assembly-100p000000-test/outputs/A_im_f32.bin",
  "rhs_real_f32": "assembly-100p000000-test/outputs/rhs_re_f32.bin",
  "rhs_imag_f32": "assembly-100p000000-test/outputs/rhs_im_f32.bin"
}
""",
                encoding="utf-8",
            )
        elif op == "evaluate_standard_exterior":
            payload = read_json_manifest(Path(command[5]))
            for descriptor in payload["output"].values():
                path = Path(command[4]).parent / descriptor["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                np.zeros(descriptor["shape"], dtype=np.float32).tofile(path)
            result_path.write_text(
                """{
  "schema": "hornlab.metal.standard.v1",
  "op": "evaluate_standard_exterior_result",
  "session_id": "metal-test",
  "batch_id": "horizontal",
  "frequency_hz": 100.0,
  "shape": [2],
  "pressure_real_f32": "field-100p000000-test/outputs/obs_pressure_re_f32.bin",
  "pressure_imag_f32": "field-100p000000-test/outputs/obs_pressure_im_f32.bin"
}
""",
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(metal_session.subprocess, "run", fake_run)

    session = MetalStandardSession.create_session(
        geometry_buffers=_buffers(),
        work_dir=tmp_path / "session",
        session_id="metal-test",
    )
    try:
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="assembly-100p000000-test",
        )
        field = session.evaluate_standard_exterior(
            100.0,
            1.8318326,
            np.ones(4, dtype=np.complex64),
            np.ones(2, dtype=np.complex64),
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            batch_id="horizontal",
            operation_id="field-100p000000-test",
        )

        assert assembly.matrix_shape == (4, 4)
        assert assembly.rhs_shape == (4,)
        assert assembly.matrix_layout == MATRIX_LAYOUT_ROW_MAJOR_C
        assert field.shape == (2,)
        assert [call[3] for call in calls] == [
            "assemble_standard_neumann",
            "evaluate_standard_exterior",
        ]

        assembly_manifest = read_json_manifest(
            tmp_path / "session" / "assembly-100p000000-test" / "assembly.json"
        )
        assert assembly_manifest["neumann_dp0"]["real_f32"]["shape"] == [2]
        assert assembly_manifest["outputs"]["A_real_f32"]["shape"] == [4, 4]

        field_manifest = read_json_manifest(
            tmp_path / "session" / "field-100p000000-test" / "field.json"
        )
        assert field_manifest["pressure_p1"]["real_f32"]["shape"] == [4]
        assert field_manifest["observation_points"]["shape"] == [3, 2]
    finally:
        session.close()


def test_owned_temp_session_cleanup(monkeypatch, tmp_path):
    monkeypatch.setattr(
        metal_session,
        "discover_runtime",
        lambda config=None, **kwargs: _runtime_status(
            available=True,
            tmp_path=tmp_path,
        ),
    )

    session = MetalStandardSession.create_session(geometry_buffers=_buffers())
    work_dir = session.info.work_dir
    assert work_dir.exists()

    session.close()

    assert not work_dir.exists()


def test_session_module_does_not_reference_scratch_paths():
    source = Path(metal_session.__file__).read_text(encoding="utf-8")
    backend_source = (
        Path(metal_session.__file__).resolve().parent / "HornlabSolverMetal.jl"
    ).read_text(encoding="utf-8")

    assert "runs/scratch" not in source
    assert "metal-bem-probe" not in source
    assert "runs/scratch" not in backend_source
    assert "metal-bem-probe" not in backend_source

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_solver.metal import session as metal_session
from hornlab_solver.metal.geometry import build_metal_geometry_buffers
from hornlab_solver.metal.session import (
    AssemblyPayload,
    BinaryArrayDescriptor,
    FieldPayload,
    GeometryPayload,
    INDEX_BASE,
    MATRIX_LAYOUT_ROW_MAJOR_C,
    METAL_STANDARD_SCHEMA,
    payload_to_manifest,
    read_json_manifest,
    write_binary_array,
    write_geometry_buffers,
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


def _write_geometry_payload(tmp_path: Path) -> tuple[Path, GeometryPayload]:
    work_dir = tmp_path / "session"
    geometry_dir = work_dir / "geometry"
    buffers = _buffers()
    mesh = write_geometry_buffers(buffers, geometry_dir, relative_to=work_dir)
    payload = GeometryPayload(
        session_id="metal-test-0001",
        mesh=mesh,
        p1_dof_count=buffers.p1_dof_count,
        dp0_dof_count=buffers.dp0_dof_count,
    )
    write_json_manifest(payload, work_dir / "session.json")
    return work_dir, payload


def test_session_manifest_json_shape_and_relative_paths(tmp_path):
    work_dir, _ = _write_geometry_payload(tmp_path)
    manifest = read_json_manifest(work_dir / "session.json")

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
        assert (work_dir / descriptor["path"]).is_file()

    manifest_text = (work_dir / "session.json").read_text(encoding="utf-8")
    assert "runs/scratch" not in manifest_text
    assert "metal-bem-probe" not in manifest_text


def test_geometry_binary_buffers_are_little_endian_and_c_contiguous(tmp_path):
    work_dir, _ = _write_geometry_payload(tmp_path)
    manifest = read_json_manifest(work_dir / "session.json")
    vertices_path = work_dir / manifest["mesh"]["vertices_f32"]["path"]
    triangles_path = work_dir / manifest["mesh"]["triangles_i32"]["path"]

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


def test_payload_contract_rejects_geometry_shape_mismatch(tmp_path):
    work_dir, _ = _write_geometry_payload(tmp_path)
    manifest = read_json_manifest(work_dir / "session.json")
    manifest["mesh"]["physical_tags_i32"]["shape"] = [3]

    with pytest.raises(ValueError, match="physical_tags_i32"):
        payload_to_manifest(manifest)


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


def test_session_module_does_not_reference_scratch_paths():
    source = Path(metal_session.__file__).read_text(encoding="utf-8")

    assert "runs/scratch" not in source
    assert "metal-bem-probe" not in source

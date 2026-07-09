from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import hornlab_metal_bem
from hornlab_metal_bem import sweep
from hornlab_metal_bem.config import SolveConfig
from hornlab_metal_bem.mesh import MeshError, load_mesh, make_pure_function_spaces
from hornlab_metal_bem.metal.geometry import (
    build_metal_geometry_buffers,
    validate_native_infinite_baffle_aperture,
)

MESHER_REPO = Path(__file__).resolve().parents[2] / "hornlab-waveguide-mesher"
if MESHER_REPO.exists() and str(MESHER_REPO) not in sys.path:
    sys.path.insert(0, str(MESHER_REPO))

config_builder = pytest.importorskip("hornlab_mesher.config_builder")
build_from_config = config_builder.build_from_config


def _ib_config(quadrants: int) -> dict:
    return {
        "formula": "OSSE",
        "mode": "infinite-baffle",
        "profile": {
            "L": 80,
            "a": 40,
            "a0": 8,
            "r0": 10,
            "k": 1,
            "n": 4,
            "q": 0.995,
        },
        "mesh": {
            "angularSegments": 24,
            "lengthSegments": 8,
            "quadrants": quadrants,
            "throatResolution": 8,
            "mouthResolution": 20,
            "rearResolution": 40,
            "scaleToMetres": False,
        },
        "source": {"sourceShape": 0},
    }


@pytest.mark.parametrize(
    ("quadrants", "native_symmetry_plane"),
    [(1234, None), (1, "yz+xz")],
)
def test_load_mesh_accepts_mesher_coupled_ib_with_validation(
    tmp_path: Path,
    quadrants: int,
    native_symmetry_plane: str | None,
) -> None:
    result = build_from_config(
        _ib_config(quadrants),
        tmp_path / f"coupled-ib-{quadrants}.msh",
    )

    assert result.metadata["apertureTag"] == 12
    assert result.native_symmetry_plane == native_symmetry_plane

    inferred = load_mesh(
        result.mesh_path,
        scale=1.0,
        native_symmetry_plane=result.native_symmetry_plane,
    )
    explicit = load_mesh(
        result.mesh_path,
        scale=1.0,
        native_symmetry_plane=result.native_symmetry_plane,
        aperture_tag=int(result.metadata["apertureTag"]),
    )

    assert 12 in {int(tag) for tag in inferred.physical_tags}
    assert 12 in {int(tag) for tag in explicit.physical_tags}
    assert inferred.coupled_ib_aperture_tag == 12
    assert explicit.coupled_ib_aperture_tag == 12
    assert inferred.info.n_triangles > 0
    assert explicit.info.n_triangles == inferred.info.n_triangles

    p1, dp0 = make_pure_function_spaces(inferred.grid)
    buffers = build_metal_geometry_buffers(
        inferred.grid,
        inferred.physical_tags,
        p1,
        dp0,
    )
    assert (
        validate_native_infinite_baffle_aperture(
            buffers,
            inferred.coupled_ib_aperture_tag,
            velocity_source_tags=[2],
            symmetry_plane=result.native_symmetry_plane,
        )
        == 12
    )


def test_public_solve_propagates_mesher_mouth_aperture_into_native_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = build_from_config(_ib_config(1234), tmp_path / "coupled-ib.msh")
    sentinel = object()
    seen: dict[str, object] = {}

    monkeypatch.setattr(hornlab_metal_bem, "_resolve_frame", lambda mesh, config: object())

    def fake_run(mesh, frequencies, frame, config):
        seen["loaded_tag"] = mesh.coupled_ib_aperture_tag
        seen["config_tag"] = config.aperture_tag
        return sentinel

    monkeypatch.setattr(sweep, "run_sweep_native_metal", fake_run)

    solved = hornlab_metal_bem.solve_frequencies(
        result.mesh_path,
        [1000.0],
        SolveConfig(mesh_scale=1.0),
    )

    assert solved is sentinel
    assert seen == {"loaded_tag": 12, "config_tag": 12}


def test_load_mesh_rejects_explicit_tag_conflicting_with_mouth_aperture(
    tmp_path: Path,
) -> None:
    result = build_from_config(_ib_config(1234), tmp_path / "coupled-ib.msh")

    with pytest.raises(MeshError, match="conflicts with canonical mouth_aperture"):
        load_mesh(result.mesh_path, scale=1.0, aperture_tag=2)


def test_raw_tag_12_without_canonical_name_does_not_enable_coupled_ib(
    tmp_path: Path,
) -> None:
    meshio = pytest.importorskip("meshio")
    path = tmp_path / "ordinary-tag-12.msh"
    meshio.write(
        path,
        meshio.Mesh(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int32))],
            cell_data={
                "gmsh:physical": [np.array([12], dtype=np.int32)],
                "gmsh:geometrical": [np.array([12], dtype=np.int32)],
            },
            field_data={"ordinary_boundary": np.array([12, 2], dtype=np.int32)},
        ),
        file_format="gmsh22",
        binary=False,
    )

    loaded = load_mesh(path, validate=False)

    assert loaded.coupled_ib_aperture_tag is None


def test_load_mesh_rejects_inverse_coupled_ib_winding(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    result = build_from_config(_ib_config(1234), tmp_path / "coupled-ib.msh")
    mesh = meshio.read(result.mesh_path)
    triangles = np.asarray(mesh.cells_dict["triangle"], dtype=np.int32)
    tags = np.asarray(mesh.cell_data_dict["gmsh:physical"]["triangle"], dtype=np.int32)
    inverse_path = tmp_path / "coupled-ib-inverse.msh"
    inverse = meshio.Mesh(
        points=mesh.points,
        cells=[("triangle", triangles[:, [0, 2, 1]])],
        cell_data={
            "gmsh:physical": [tags],
            "gmsh:geometrical": [tags],
        },
        field_data=mesh.field_data,
    )
    meshio.write(inverse_path, inverse, file_format="gmsh22", binary=False)

    with pytest.raises(MeshError, match="Coupled infinite-baffle mesh winding appears inverse"):
        load_mesh(
            inverse_path,
            scale=1.0,
            aperture_tag=int(result.metadata["apertureTag"]),
        )


def test_load_mesh_rejects_legacy_plus_z_coupled_ib_aperture(tmp_path: Path) -> None:
    meshio = pytest.importorskip("meshio")
    result = build_from_config(_ib_config(1234), tmp_path / "coupled-ib.msh")
    mesh = meshio.read(result.mesh_path)
    triangles = np.asarray(mesh.cells_dict["triangle"], dtype=np.int32)
    tags = np.asarray(mesh.cell_data_dict["gmsh:physical"]["triangle"], dtype=np.int32)
    aperture_tag = int(result.metadata["apertureTag"])
    mixed = np.array(triangles, copy=True)
    aperture = tags == aperture_tag
    mixed[aperture] = mixed[aperture][:, [0, 2, 1]]
    mixed_path = tmp_path / "coupled-ib-legacy-plus-z-aperture.msh"
    mixed_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[("triangle", mixed)],
        cell_data={
            "gmsh:physical": [tags],
            "gmsh:geometrical": [tags],
        },
        field_data=mesh.field_data,
    )
    meshio.write(mixed_path, mixed_mesh, file_format="gmsh22", binary=False)

    with pytest.raises(MeshError, match="aperture normals must point -Z"):
        load_mesh(
            mixed_path,
            scale=1.0,
            aperture_tag=aperture_tag,
        )

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from hornlab_metal_bem.mesh import MeshError, load_mesh

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
    assert inferred.info.n_triangles > 0
    assert explicit.info.n_triangles == inferred.info.n_triangles


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

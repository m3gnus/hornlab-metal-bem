from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np

import hornlab_metal_bem
from hornlab_metal_bem import sweep
from hornlab_metal_bem.result import MeshInfo, SolveResult


def test_native_config_defaults_to_strict_metal():
    config = hornlab_metal_bem.native_config(freq_count=3)

    assert config.metal_native_assembly_mode == "corrected"
    assert config.native_symmetry_plane is None
    assert config.freq_count == 3


def test_public_namespace_exports_only_metal_bem_surface():
    assert "solve" in hornlab_metal_bem.__all__
    assert "solve_frequencies" in hornlab_metal_bem.__all__
    assert "native_config" in hornlab_metal_bem.__all__
    assert "SolveConfig" in hornlab_metal_bem.__all__
    assert "SolveResult" in hornlab_metal_bem.__all__

    assert "BIEFormulation" not in hornlab_metal_bem.__all__
    assert "LinearSolver" not in hornlab_metal_bem.__all__
    assert "DenseBieSystem" not in hornlab_metal_bem.__all__


def test_hornlab_metal_bem_solve_defaults_to_pure_native_dispatch(monkeypatch):
    loaded = SimpleNamespace(grid="pure", physical_tags=np.asarray([2], dtype=np.int32))
    sentinel = object()
    calls = {}

    def fake_load_mesh(
        mesh,
        *,
        scale,
        validate=True,
        merge_tol=1e-9,
        repair_normals=False,
        native_symmetry_plane=None,
    ):
        calls["mesh"] = mesh
        calls["scale"] = scale
        calls["native_symmetry_plane"] = native_symmetry_plane
        return loaded

    monkeypatch.setattr("hornlab_metal_bem.load_mesh", fake_load_mesh)
    monkeypatch.setattr("hornlab_metal_bem._resolve_frame", lambda mesh, config: object())
    monkeypatch.setattr(
        sweep,
        "run_sweep_native_metal",
        lambda mesh, frequencies, frame, config: sentinel,
    )

    result = hornlab_metal_bem.solve("waveguide.msh")

    assert result is sentinel
    assert calls == {
        "mesh": "waveguide.msh",
        "scale": 1.0,
        "native_symmetry_plane": None,
    }


def test_importing_public_metal_namespace_does_not_import_bempp():
    script = (
        "import sys, hornlab_metal_bem; "
        "print(any(n == 'bempp_cl' or n.startswith('bempp_cl.') "
        "for n in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "False"


def test_solve_result_directivity_is_primary_normalized_output_name():
    directivity = np.zeros((1, 1, 3), dtype=np.float64)
    result = SolveResult(
        frequencies_hz=np.asarray([1000.0], dtype=np.float64),
        pressure_complex=np.ones((1, 1, 3), dtype=np.complex128),
        directivity_db=directivity,
        impedance=np.asarray([1.0 + 0.0j], dtype=np.complex128),
        observation_angles_deg=np.asarray([0.0, 90.0, 180.0], dtype=np.float64),
        observation_points=np.zeros((1, 3, 3), dtype=np.float64),
        observation_planes=["horizontal"],
        config=hornlab_metal_bem.native_config(freq_count=1),
        mesh_info=MeshInfo(
            n_vertices=3,
            n_triangles=1,
            physical_groups={1: "wall", 2: "source"},
            bounding_box_m=(
                np.zeros(3, dtype=np.float64),
                np.ones(3, dtype=np.float64),
            ),
        ),
    )

    assert result.directivity_db is directivity
    assert result.spl_norm_db is directivity
    assert result.surface_pressure_complex is None
    assert result.native_diagnostics == []
    assert not hasattr(result, "spl_db")

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np

import hornlab_metal_bem
from hornlab_solver import sweep


def test_native_config_defaults_to_strict_metal():
    config = hornlab_metal_bem.native_config(freq_count=3)

    assert config.assembly_backend == "metal"
    assert config.experimental_metal_backend is True
    assert config.metal_backend_fallback == "error"
    assert config.metal_native_assembly_mode == "corrected"
    assert config.freq_count == 3


def test_hornlab_metal_bem_solve_defaults_to_pure_native_dispatch(monkeypatch):
    loaded = SimpleNamespace(grid="pure", physical_tags=np.asarray([2], dtype=np.int32))
    sentinel = object()
    calls = {}

    def fake_load_mesh(mesh, *, scale, grid_backend):
        calls["mesh"] = mesh
        calls["scale"] = scale
        calls["grid_backend"] = grid_backend
        return loaded

    monkeypatch.setattr("hornlab_solver.load_mesh", fake_load_mesh)
    monkeypatch.setattr("hornlab_solver._resolve_frame", lambda mesh, config: object())
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
        "grid_backend": "pure",
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

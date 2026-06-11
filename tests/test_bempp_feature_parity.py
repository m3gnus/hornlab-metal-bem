from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem.config import BIEFormulation as MetalBIEFormulation
from hornlab_metal_bem.mesh import LoadedMesh as MetalLoadedMesh
from hornlab_metal_bem.mesh import make_pure_grid
from hornlab_metal_bem.result import MeshInfo as MetalMeshInfo


_WORKSPACE = Path(__file__).resolve().parents[2]
_BEMPP_REPO = _WORKSPACE / "hornlab-bempp-bem"


def _require_bempp_and_native():
    if str(_BEMPP_REPO) not in sys.path:
        sys.path.insert(0, str(_BEMPP_REPO))
    try:
        import bempp_cl.api as bempp_api
        import hornlab_bempp_bem as bempp_bem
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"bempp parity dependencies unavailable: {exc}")

    from hornlab_metal_bem.metal import discover_native_runtime

    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    return bempp_api, bempp_bem


def _sphere_meshes(tags: np.ndarray):
    bempp_api, bempp_bem = _require_bempp_and_native()
    grid = bempp_api.shapes.regular_sphere(1)
    tags = np.asarray(tags, dtype=np.int32)
    assert tags.shape == (grid.number_of_elements,)

    vertices_nx3 = np.asarray(grid.vertices.T, dtype=np.float64)
    triangles_nx3 = np.asarray(grid.elements.T, dtype=np.int32)
    bbox = (vertices_nx3.min(axis=0), vertices_nx3.max(axis=0))
    metal_mesh = MetalLoadedMesh(
        grid=make_pure_grid(vertices_nx3, triangles_nx3),
        physical_tags=tags,
        info=MetalMeshInfo(
            n_vertices=vertices_nx3.shape[0],
            n_triangles=triangles_nx3.shape[0],
            physical_groups={int(tag): str(int(tag)) for tag in np.unique(tags)},
            bounding_box_m=bbox,
        ),
    )
    bempp_mesh = bempp_bem.LoadedMesh(
        grid=grid,
        physical_tags=tags,
        info=bempp_bem.MeshInfo(
            n_vertices=vertices_nx3.shape[0],
            n_triangles=triangles_nx3.shape[0],
            physical_groups={int(tag): str(int(tag)) for tag in np.unique(tags)},
            bounding_box_m=bbox,
        ),
    )
    return bempp_bem, metal_mesh, bempp_mesh


def _observation_configs(metal_module, bempp_module):
    points = np.array(
        [[0.0, 0.0, 2.2], [0.6, 0.0, 2.1]],
        dtype=np.float64,
    )
    metal_obs = metal_module.ObservationConfig(
        planes=["probe"],
        angle_count=2,
        custom_points={"probe": points},
    )
    bempp_obs = bempp_module.ObservationConfig(
        planes=["probe"],
        angle_count=2,
        custom_points={"probe": points},
    )
    return metal_obs, bempp_obs


@pytest.mark.slow
def test_complex_k_near_unit_sphere_interior_resonance_matches_bempp():
    tags = np.ones(32, dtype=np.int32)
    tags[:4] = 2
    bempp_bem, metal_mesh, bempp_mesh = _sphere_meshes(tags)
    metal_obs, bempp_obs = _observation_configs(metal_bem, bempp_bem)

    frequency_hz = 171.5  # unit sphere k*r ~= pi.
    metal_cfg = metal_bem.native_config(
        formulation=MetalBIEFormulation.COMPLEX_K,
        complex_k_shift=0.005,
        velocity_sources={2: 1.0},
        observation=metal_obs,
        return_surface_pressure=True,
    )
    bempp_cfg = bempp_bem.SolveConfig(
        formulation=bempp_bem.BIEFormulation.COMPLEX_K,
        complex_k_shift=0.005,
        velocity_sources={2: 1.0},
        observation=bempp_obs,
        solver=bempp_bem.LinearSolver.LU,
        precision="single",
        assembly_backend="numba",
        workers=1,
    )

    metal_result = metal_bem.solve_frequencies(metal_mesh, [frequency_hz], metal_cfg)
    bempp_result = bempp_bem.solve_frequencies(bempp_mesh, [frequency_hz], bempp_cfg)

    # Measured gap with CPU Duffy corrections on the reference path: ~1.4e-6
    # relative (f32-level agreement with bempp's numba dense assembly).
    np.testing.assert_allclose(
        metal_result.pressure_complex,
        bempp_result.pressure_complex,
        rtol=1.0e-4,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        metal_result.impedance,
        bempp_result.impedance,
        rtol=1.0e-4,
        atol=1.0e-6,
    )
    diagnostics = metal_result.native_diagnostics[0]
    assert diagnostics["complex_k"] is True
    assert 0.0 < diagnostics["dense_solve_rcond"] <= 1.0


@pytest.mark.slow
def test_robin_impedance_tags_8_9_match_bempp():
    tags = np.ones(32, dtype=np.int32)
    tags[:4] = 2
    tags[8:14] = 8
    tags[14:20] = 9
    bempp_bem, metal_mesh, bempp_mesh = _sphere_meshes(tags)
    metal_obs, bempp_obs = _observation_configs(metal_bem, bempp_bem)

    frequency_hz = 100.0
    impedance_sources = {8: 0.02 + 0.0j, 9: 0.01 + 0.005j}
    metal_cfg = metal_bem.native_config(
        velocity_sources={2: 1.0},
        impedance_sources=impedance_sources,
        observation=metal_obs,
        return_surface_pressure=True,
    )
    bempp_cfg = bempp_bem.SolveConfig(
        formulation=bempp_bem.BIEFormulation.STANDARD,
        velocity_sources={2: 1.0},
        impedance_sources=impedance_sources,
        observation=bempp_obs,
        solver=bempp_bem.LinearSolver.LU,
        precision="single",
        assembly_backend="numba",
        workers=1,
    )

    metal_result = metal_bem.solve_frequencies(metal_mesh, [frequency_hz], metal_cfg)
    bempp_result = bempp_bem.solve_frequencies(bempp_mesh, [frequency_hz], bempp_cfg)

    # Measured gap with CPU Duffy corrections on the reference path: ~4e-7
    # relative (f32-level agreement with bempp's numba dense assembly).
    np.testing.assert_allclose(
        metal_result.pressure_complex,
        bempp_result.pressure_complex,
        rtol=1.0e-4,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        metal_result.impedance,
        bempp_result.impedance,
        rtol=1.0e-4,
        atol=1.0e-6,
    )
    diagnostics = metal_result.native_diagnostics[0]
    assert diagnostics["robin_boundary"] is True
    assert diagnostics["field_uses_total_neumann"] is True
    assert 0.0 < diagnostics["dense_solve_rcond"] <= 1.0

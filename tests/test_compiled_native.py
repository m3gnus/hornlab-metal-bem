"""Numerical bridge equivalence against independently constructed legacy calls."""

import meshio
import numpy as np
import pytest
from test_compiled import api, request


@pytest.mark.slow
@pytest.mark.parametrize("motion", ["normal", "rigid_translation"])
@pytest.mark.parametrize("port_count", [1, 2])
def test_compiled_matches_direct_legacy_call(tmp_path, motion, port_count):
    if api.__name__ == "hornlab_metal_bem":
        from native_helper_guard import require_fresh_native_helper

        require_fresh_native_helper()
    w = request()
    w["excitation_port_ids"] = w["excitation_port_ids"][:port_count]
    w["compiled_system"]["contract_version"] = 2 if motion == "rigid_translation" else 1
    for c in w["compiled_system"]["components"]:
        c["parameters"] = (
            {}
            if motion == "normal"
            else {"motion_profile": motion, "motion_axis": [1.0, 2.0, 3.0]}
        )
    compiled = api.solve_compiled(w)
    # Independently construct the native surface and settings in SI. Do not
    # call the adapter's planner/mesh writer to build the comparison.
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ) * 0.001 + [0.2, 0.3, 0.4]
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    tags = np.array([11, 12, 13, 14])
    file = tmp_path / "direct.msh"
    meshio.write(
        file,
        meshio.Mesh(
            vertices,
            [("triangle", faces)],
            cell_data={"gmsh:physical": [tags], "gmsh:geometrical": [tags]},
        ),
        file_format="gmsh22",
        binary=False,
    )
    obs = api.ObservationConfig(
        planes=["compiled"],
        custom_points={"compiled": np.array([[0, 0, 2], [2, 0, 0], [0, 2, 0]])},
        angle_count=3,
    )
    frame = api.ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.zeros(3),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=np.zeros(3),
        source_center=np.zeros(3),
    )
    axis = tuple(np.array([1.0, 2.0, 3.0]) / np.sqrt(14))
    common = {
        "velocity_mode": api.VelocityMode.VELOCITY,
        "source_motion": "normal" if motion == "normal" else "axial",
        "source_axes": None if motion == "normal" else {11: axis, 12: axis, 13: axis},
        "air_density": 1.21,
        "observation": obs,
        "frame_override": frame,
    }
    if api.__name__ == "hornlab_metal_bem":
        common.update(
            speed_of_sound=343.0,
            dense_solve_dtype="float32",
            metal_native_assembly_mode="corrected",
            mesh_merge_tol=0.0,
        )
        # Several ports share one factorization (solve_multi_source), which
        # matches separate solves to the multi-source parity budget.
        budget = 2e-4
    else:
        common.update(
            precision="double",
            assembly_backend="numba",
            require_closed_mesh=True,
            slp_dlp_quadrature=4,
            slp_dlp_singular_quadrature=4,
            hyp_adlp_quadrature=4,
        )
        budget = 1e-12
    drives = [{11: 0.0, 12: 0.0, 13: 1.0}, {11: 1.0, 12: 1.0, 13: 0.0}]
    for i, drive in enumerate(drives[:port_count]):
        direct = api.solve_frequencies(
            file,
            [200.0, 80.0, 120.0],
            api.SolveConfig(**common, velocity_sources=drive),
        )
        actual = np.concatenate(
            [compiled.pressure["near"][:, i, :], compiled.pressure["far"][:, i, :]],
            axis=1,
        )
        assert np.all(np.abs(actual) > 0)
        np.testing.assert_allclose(
            actual, direct.pressure_complex[:, 0, :], rtol=budget, atol=1e-12
        )
        np.testing.assert_array_equal(compiled.frequencies_hz, direct.frequencies_hz)
        assert compiled.native_results[i].solver_log
        if api.__name__ == "hornlab_metal_bem":
            runtime = compiled.native_results[i].native_runtime
            assert runtime["helper"]["sha256"]
            assert compiled.diagnostics["engine_provenance"]["runtime"]["helper"] == runtime["helper"]


@pytest.mark.slow
def test_frozen_official_example_reaches_native_solver(tmp_path):
    import json
    from pathlib import Path

    if api.__name__ == "hornlab_metal_bem":
        from native_helper_guard import require_fresh_native_helper

        require_fresh_native_helper()
    w = json.loads(
        (Path(__file__).parent / "fixtures/beat-official-exterior.json").read_text()
    )
    w["solver_options"]["bem_backend"] = (
        "metal" if api.__name__ == "hornlab_metal_bem" else "cpu"
    )
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    tags = np.full(4, 2)
    meshio.write(
        tmp_path / "exterior.msh",
        meshio.Mesh(
            points,
            [("triangle", faces)],
            cell_data={"gmsh:physical": [tags], "gmsh:geometrical": [tags]},
        ),
        file_format="gmsh22",
        binary=False,
    )
    result = api.solve_compiled(w, base_dir=tmp_path)
    assert result.pressure["pressure"].shape == (2, 1, 1)
    assert np.isfinite(result.pressure["pressure"]).all()
    assert np.all(np.abs(result.pressure["pressure"]) > 0)
    np.testing.assert_array_equal(result.frequencies_hz, [500, 250])
    assert result.native_results[0].solver_log

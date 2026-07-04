from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from hornlab_metal_bem import sweep
from hornlab_metal_bem.metal import native
from hornlab_metal_bem.metal.native import MetalNativeRuntimeStatus


def test_append_sphere_field_points_concatenates_and_counts():
    arc = sweep._field_points_3xn(np.zeros((2, 3, 3)))  # 2 planes x 3 angles -> (3, 6)
    combined, n_sphere = sweep._append_sphere_field_points(arc, np.ones((4, 3)))
    assert n_sphere == 4
    assert combined.shape == (3, 10)

    unchanged, zero = sweep._append_sphere_field_points(arc, None)
    assert zero == 0
    assert unchanged.shape == (3, 6)


def test_system_field_splits_arc_and_sphere():
    n_planes, n_angles, n_sphere = 2, 3, 4
    flat = np.arange(n_planes * n_angles + n_sphere, dtype=np.complex128)
    system = SimpleNamespace(field_row_index=0)

    arc, sphere = sweep._system_field(system, n_planes, n_angles, n_sphere, np.asarray([flat]))
    np.testing.assert_array_equal(arc, flat[:6].reshape(n_planes, n_angles))
    np.testing.assert_array_equal(sphere, flat[6:])

    arc_only, no_sphere = sweep._system_field(system, n_planes, n_angles, 0, np.asarray([flat[:6]]))
    assert no_sphere is None
    assert arc_only.shape == (n_planes, n_angles)


def test_discover_runtime_smoke_cached_reuses_validated_helper(monkeypatch, tmp_path):
    helper = tmp_path / "HornlabMetalBemNative"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")

    def status_for(*, smoke_test_ran: bool) -> MetalNativeRuntimeStatus:
        return MetalNativeRuntimeStatus(
            available=True,
            platform_system="Darwin",
            platform_machine="arm64",
            is_macos=True,
            is_apple_silicon=True,
            swift_path=None,
            swift_source=None,
            helper_executable_path=helper,
            helper_source="test",
            backend_dir=tmp_path,
            native_entrypoint=tmp_path / "HornlabMetalBemNative.swift",
            native_package_dir=tmp_path / "native_helper",
            helper_assets_present=True,
            smoke_test_ran=smoke_test_ran,
            smoke_test_ok=smoke_test_ran,
            smoke_test_error=None,
            reasons=(),
        )

    no_smoke_status = status_for(smoke_test_ran=False)
    smoke_status = status_for(smoke_test_ran=True)
    calls: list[bool] = []

    def discover_stub(*, run_smoke_test: bool = False):
        calls.append(run_smoke_test)
        return smoke_status if run_smoke_test else no_smoke_status

    monkeypatch.setattr(native, "discover_native_runtime", discover_stub)
    monkeypatch.setattr(sweep, "_SMOKE_VALIDATED_HELPERS", {})

    assert sweep._discover_runtime_smoke_cached() is smoke_status
    assert calls == [False, True]

    assert sweep._discover_runtime_smoke_cached() is no_smoke_status
    assert calls == [False, True, False]

    bumped_mtime = helper.stat().st_mtime + 10.0
    os.utime(helper, (bumped_mtime, bumped_mtime))

    assert sweep._discover_runtime_smoke_cached() is smoke_status
    assert calls == [False, True, False, False, True]


def test_append_system_result_keeps_complex_field_surface_pressure_and_diagnostics(
    tmp_path,
):
    field_real = tmp_path / "field_re.bin"
    field_imag = tmp_path / "field_im.bin"
    pressure_real = tmp_path / "pressure_re.bin"
    pressure_imag = tmp_path / "pressure_im.bin"
    np.asarray([1.0, 2.0], dtype="<f4").tofile(field_real)
    np.asarray([0.5, -0.5], dtype="<f4").tofile(field_imag)
    np.asarray([3.0, 4.0, 5.0], dtype="<f4").tofile(pressure_real)
    np.asarray([1.0, 1.5, 2.0], dtype="<f4").tofile(pressure_imag)

    system = SimpleNamespace(
        impedance=1.0 + 2.0j,
        surface_pressure_avg={2: 3.0 + 4.0j},
        field_real_f32=field_real,
        field_imag_f32=field_imag,
        field_shape=(2,),
        pressure_real_f32=pressure_real,
        pressure_imag_f32=pressure_imag,
        pressure_shape=(3,),
        assembly_s=0.1,
        dense_solve_s=0.2,
        field_s=0.3,
        lapack_info=0,
        diagnostics={"assembly_implementation": "test_assembly"},
    )

    surface_pavg = {2: []}
    pressure_rows = []
    spl_rows = []
    impedance_rows = []
    surface_pressure_rows = []
    native_diagnostics_rows = []
    solver_log = []
    completed_freqs = []

    entry = sweep._append_system_result(
        frequency_hz=1000.0,
        system=system,
        backend="test_backend",
        timing_s=0.6,
        mesh=SimpleNamespace(),
        p1_space=SimpleNamespace(),
        source_tags=[2],
        impedance_source_tag=2,
        n_planes=1,
        n_angles=2,
        on_axis_idx=0,
        field_batch_complex=None,
        surface_pavg=surface_pavg,
        pressure_rows=pressure_rows,
        spl_rows=spl_rows,
        impedance_rows=impedance_rows,
        surface_pressure_rows=surface_pressure_rows,
        native_diagnostics_rows=native_diagnostics_rows,
        solver_log=solver_log,
        completed_freqs=completed_freqs,
    )

    np.testing.assert_allclose(pressure_rows[0], [[1.0 + 0.5j, 2.0 - 0.5j]])
    np.testing.assert_allclose(
        surface_pressure_rows[0],
        [3.0 + 1.0j, 4.0 + 1.5j, 5.0 + 2.0j],
    )
    assert entry["lapack_info"] == 0
    assert entry["native_diagnostics"]["assembly_implementation"] == "test_assembly"
    assert native_diagnostics_rows == [entry["native_diagnostics"]]


def test_dense_solve_policy_marks_low_rcond_suspect():
    diagnostics = {"dense_solve_rcond": 1e-8}

    sweep._apply_dense_solve_policy(diagnostics, threshold=1e-6)

    assert diagnostics["dense_solve_rcond_warning_threshold"] == 1e-6
    assert diagnostics["dense_solve_suspect"] is True
    assert "nudge or densify" in diagnostics["dense_solve_recommendation"]


def test_mesh_resolution_policy_marks_underresolved_frequency():
    diagnostics = {}

    sweep._apply_mesh_resolution_policy(
        diagnostics,
        frequency_hz=1000.0,
        mesh_max_edge_m=0.1,
        elements_per_wavelength_min=6.0,
    )

    np.testing.assert_allclose(diagnostics["mesh_elements_per_wavelength"], 3.43)
    np.testing.assert_allclose(
        diagnostics["mesh_max_valid_frequency_hz"],
        571.6666666666666,
    )
    assert diagnostics["mesh_resolution_suspect"] is True


def test_mesh_max_edge_accepts_bempp_shaped_arrays():
    mesh = SimpleNamespace(
        grid=SimpleNamespace(
            vertices=np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            elements=np.array([[0], [1], [2]], dtype=np.int32),
        )
    )

    np.testing.assert_allclose(sweep._mesh_max_edge_m(mesh), np.sqrt(2.0))


def test_dense_solve_policy_marks_availability():
    checked = {"dense_solve_rcond": 1e-2}
    sweep._apply_dense_solve_policy(checked, threshold=1e-6)
    assert checked["dense_solve_policy_available"] is True
    assert checked["dense_solve_suspect"] is False

    # The CHIEF zgels path returns no rcond: the policy must say so instead
    # of leaving a plain False that reads as "checked and fine".
    unchecked = {}
    sweep._apply_dense_solve_policy(unchecked, threshold=1e-6)
    assert unchecked["dense_solve_policy_available"] is False
    assert unchecked["dense_solve_suspect"] is False


def _single_triangle_mesh():
    # bempp-shaped (3, n) arrays; a unit right triangle in the z=0 plane.
    return SimpleNamespace(
        grid=SimpleNamespace(
            vertices=np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            elements=np.array([[0], [1], [2]], dtype=np.int32),
        )
    )


def test_chief_points_near_boundary_warn(caplog):
    mesh = _single_triangle_mesh()
    with caplog.at_level("WARNING"):
        sweep._warn_near_boundary_chief_points(
            mesh, np.array([[0.0, 1.0, 0.0]]), max_edge_m=float(np.sqrt(2.0))
        )
    assert any("chief_points" in record.message for record in caplog.records)


def test_chief_points_deep_interior_do_not_warn(caplog):
    mesh = _single_triangle_mesh()
    with caplog.at_level("WARNING"):
        sweep._warn_near_boundary_chief_points(
            mesh, np.array([[5.0, 5.0, 5.0]]), max_edge_m=float(np.sqrt(2.0))
        )
    assert not caplog.records

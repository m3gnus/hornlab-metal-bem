from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from hornlab_metal_bem import sweep
from hornlab_metal_bem.metal import native
from hornlab_metal_bem.metal.native import MetalNativeRuntimeStatus


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

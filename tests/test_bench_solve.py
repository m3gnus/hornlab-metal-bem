from __future__ import annotations

import json

import pytest

from scripts import bench_solve


def test_bench_solve_parses_sphere_grid_and_symmetry_ab_controls():
    args = bench_solve._parse_args(
        [
            "--sphere-grid",
            "37,72",
            "--native-symmetry-plane",
            "yz+xz",
            "--disable-sphere-symmetry-dedupe",
            "--native-allow-open-rim",
        ]
    )

    assert args.sphere_grid == (37, 72)
    assert args.native_symmetry_plane == "yz+xz"
    assert args.disable_sphere_symmetry_dedupe is True
    assert args.native_allow_open_rim is True


def test_bench_solve_builtin_fixture_json_smoke(capsys):
    from hornlab_metal_bem.metal import discover_native_runtime

    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    exit_code = bench_solve.main(
        [
            "--frequencies",
            "2",
            "--repeat",
            "1",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["benchmark"] == "native_solve_batch"
    assert payload["skipped"] is False
    assert payload["frequency_count"] == 2
    assert len(payload["frequencies_hz"]) == 2
    assert payload["fixture"]["name"] == "built-in-coupled-box"
    assert payload["fixture"]["system_order_dofs"] > 0
    assert payload["repeat_count"] == 1
    assert payload["runs"] == []
    assert "HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE" in payload["environment"]
    assert "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE" in payload["environment"]
    assert "HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE" in payload["environment"]

    warmup = payload["warmup"]
    assert warmup["warmup"] is True
    assert warmup["wall_seconds"] >= 0.0
    assert warmup["system_order_dofs"] == payload["fixture"]["system_order_dofs"]
    assert set(warmup["stages"]) == {
        "regular_assembly_seconds",
        "duffy_and_near_corrections_seconds",
        "dense_solve_seconds",
        "field_seconds",
    }
    assert "reported_stages_sum_seconds" in warmup
    assert "unaccounted_process_serialization_dispatch_seconds" in warmup
    assert len(warmup["case_diagnostics"]) == 2
    assert warmup["first_result_latency_source"] in {
        "streamed_case_callback",
        "batch_total_proxy",
    }

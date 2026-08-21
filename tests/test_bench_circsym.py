from __future__ import annotations

import json

from scripts import bench_circsym


def test_bench_circsym_cpu_json_smoke(capsys):
    exit_code = bench_circsym.main(
        [
            "--fixture",
            "freestanding",
            "--f1",
            "1000",
            "--f2",
            "1000",
            "--frequencies",
            "1",
            "--angles",
            "3",
            "--repeat",
            "1",
            "--target-edge-mm",
            "30",
            "--backend",
            "cpu",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["benchmark"] == "circsym_sweep"
    assert payload["fixture"] == "freestanding"
    assert payload["frequency_count"] == 1
    assert payload["angle_count"] == 3
    assert payload["meridian_segments"] > 0
    assert payload["environment"]["HORNLAB_CIRCSYM_ASSEMBLY_BACKEND"] == "cpu"
    assert payload["environment"]["HORNLAB_CIRCSYM_FIELD_BACKEND"] == "cpu"
    assert payload["warmup"]["wall_seconds"] >= 0.0
    assert payload["warmup"]["seconds_per_frequency"] >= 0.0
    assert payload["runs"] == []

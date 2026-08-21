from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.bench_circsym import solve_fixture


GOLDEN_PATH = Path(__file__).parent / "fixtures" / "circsym_cross_os_golden.json"


@pytest.mark.parametrize("case_name", ["freestanding", "infinite-baffle"])
def test_portable_cpu_circsym_matches_cross_os_golden(
    case_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin real free-standing and coupled-IB physics across C/Numba hosts."""

    payload = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    expected = payload["cases"][case_name]
    monkeypatch.setenv("HORNLAB_CIRCSYM_ASSEMBLY_BACKEND", "cpu")
    monkeypatch.setenv("HORNLAB_CIRCSYM_FIELD_BACKEND", "cpu")
    monkeypatch.setenv("HORNLAB_CIRCSYM_CPU_REMAINDER_BACKEND", "auto")

    frequencies = np.asarray(expected["frequencies_hz"], dtype=np.float64)
    meridian, result = solve_fixture(
        case_name,
        frequencies,
        target_edge_m=float(expected["target_edge_m"]),
        angle_count=int(expected["angle_count"]),
    )

    assert meridian.segment_count == expected["meridian_segments"]
    np.testing.assert_allclose(result.frequencies_hz, frequencies, atol=0.0)
    np.testing.assert_allclose(
        result.observation_angles_deg,
        expected["angles_deg"],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.directivity_db[:, 0, :],
        expected["directivity_db"],
        rtol=2.0e-5,
        atol=2.0e-4,
    )
    expected_pressure = np.asarray(expected["pressure_on_axis_real"]) + 1j * np.asarray(
        expected["pressure_on_axis_imag"]
    )
    np.testing.assert_allclose(
        result.pressure_complex[:, 0, 0],
        expected_pressure,
        rtol=2.0e-5,
        atol=2.0e-7,
    )
    expected_impedance = np.asarray(expected["impedance_real"]) + 1j * np.asarray(
        expected["impedance_imag"]
    )
    np.testing.assert_allclose(
        result.impedance,
        expected_impedance,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert [
        bool(item.get("coupled_ib", False)) for item in result.native_diagnostics
    ] == expected["coupled_ib"]

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import bench_axisymmetric_vs_quarter as benchmark


def _result(pressure: np.ndarray, directivity: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        pressure_complex=np.asarray(pressure, dtype=np.complex128),
        directivity_db=np.asarray(directivity, dtype=np.float64),
    )


def test_accuracy_reports_identical_results_as_zero_difference():
    pressure = np.array([[1.0 + 2.0j, -0.5 + 0.25j]])
    directivity = np.array([[0.0, -6.0]])

    metrics = benchmark._accuracy(
        _result(pressure, directivity),
        _result(pressure, directivity),
    )

    assert metrics == {
        "pressure_relative_l2": 0.0,
        "pressure_magnitude_relative_l2": 0.0,
        "directivity_max_abs_db": 0.0,
        "directivity_max_abs_db_above_minus_40": 0.0,
        "phase_rms_degrees_above_floor": 0.0,
    }


def test_accuracy_by_frequency_keeps_error_rows_separate():
    reference = _result(
        np.array([[1.0 + 0.0j, 1.0 + 0.0j], [2.0 + 0.0j, 2.0 + 0.0j]]),
        np.array([[0.0, -6.0], [0.0, -6.0]]),
    )
    candidate = _result(
        np.array([[1.0 + 0.0j, 1.0 + 0.0j], [2.2 + 0.0j, 2.2 + 0.0j]]),
        np.array([[0.0, -6.0], [0.0, -5.0]]),
    )

    rows = benchmark._accuracy_by_frequency(candidate, reference, [100.0, 200.0])

    assert rows[0]["frequency_hz"] == 100.0
    assert rows[0]["pressure_relative_l2"] == 0.0
    assert rows[1]["pressure_relative_l2"] == pytest.approx(0.1)
    assert rows[1]["directivity_max_abs_db_above_minus_40"] == 1.0


def test_source_areas_return_full_domain_equivalents():
    meridian = SimpleNamespace(
        physical_tags=np.array([2]),
        segment_geometry=lambda: SimpleNamespace(area_weights=np.array([0.5])),
    )
    quarter = SimpleNamespace(
        physical_tags=np.array([2]),
        grid=SimpleNamespace(
            vertices=np.array(
                [[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]
            ),
            elements=np.array([[0], [1], [2]]),
        ),
    )

    meridian_area, quarter_area = benchmark._source_areas(meridian, quarter)

    assert meridian_area == pytest.approx(0.5)
    assert quarter_area == pytest.approx(0.5)


def test_parse_args_defaults_to_original_qualification_gate(tmp_path):
    args = benchmark._parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--quarter-mesh",
            str(tmp_path / "quarter.msh"),
        ]
    )

    assert args.axisym_backend == "cpu"
    assert args.cpu_field == "numba"
    assert args.azimuth_min == 64
    assert args.qualification_ratio == 0.5


def test_overall_qualification_requires_compact_cpu_parity():
    common = {
        "speed_ratio": True,
        "half_second": True,
        "numerical_gate": True,
    }

    assert benchmark._passes_overall_qualification(
        **common,
        compact_cpu_parity=True,
    )
    assert not benchmark._passes_overall_qualification(
        **common,
        compact_cpu_parity=False,
    )


def test_matched_input_validation_rejects_wrong_quarter_extent():
    meridian = SimpleNamespace(
        physical_tags=np.array([1, 2]),
        nodes=np.array([[0.0, -0.01], [0.16, 0.17]]),
        segment_geometry=lambda: SimpleNamespace(area_weights=np.array([1.0, 1.0])),
    )
    quarter = SimpleNamespace(
        physical_tags=np.array([1, 2]),
        grid=SimpleNamespace(
            vertices=np.array(
                [
                    [0.0, 0.10, 0.0, 0.10],
                    [0.0, 0.0, 0.10, 0.10],
                    [-0.01, -0.01, 0.17, 0.17],
                ]
            ),
            elements=np.array([[0, 1], [1, 2], [2, 3]]),
        ),
    )
    config = {
        "mode": "freestanding",
        "cross_section": {"exponent": 2.0, "aspectRatio": 1.0},
        "morph": {"morphTarget": 0.0},
    }

    with pytest.raises(ValueError, match="radial extent"):
        benchmark._validate_matched_inputs(config, meridian, quarter)

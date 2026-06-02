"""Unit tests for sweep.py — on-axis normalisation, early stopping, callbacks.

These tests mock out bempp-cl internals to test the sweep logic in isolation.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hornlab_solver.config import ObservationConfig, SolveConfig
from hornlab_solver.observation import ObservationFrame
from hornlab_solver.result import MeshInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame() -> ObservationFrame:
    return ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=np.array([0.0, 0.0, 0.0]),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=np.array([0.0, 0.0, 1.0]),
        source_center=np.array([0.0, 0.0, 0.0]),
    )


def _make_mesh():
    mesh = MagicMock()
    mesh.info = MeshInfo(
        n_vertices=100, n_triangles=200,
        physical_groups={1: "body", 2: "source"},
        bounding_box_m=(np.zeros(3), np.ones(3)),
    )
    mesh.grid = MagicMock()
    mesh.physical_tags = np.array([1] * 180 + [2] * 20, dtype=np.int32)
    return mesh


def _fake_frequency_result(freq_hz: float):
    fr = MagicMock()
    fr.frequency_hz = freq_hz
    fr.iterations = 10
    fr.timing_s = 0.5
    fr.impedance = 400.0 + 50j
    fr.pressure_on_surface = MagicMock()
    fr.pressure_on_surface.space = MagicMock()
    fr.neumann_data = MagicMock()
    fr.neumann_data.space = MagicMock()
    return fr


# Patch targets within hornlab_solver.sweep
_SWEEP = "hornlab_solver.sweep"


def _standard_sweep_patches():
    """Return a dict of patches that isolate run_sweep_serial from bempp."""
    return {
        "spaces": patch(f"{_SWEEP}._setup_function_spaces",
                        return_value=(MagicMock(), MagicMock())),
        "solve": patch(f"{_SWEEP}.solve_single_frequency",
                       side_effect=lambda *a, **kw: _fake_frequency_result(
                           a[2] if len(a) > 2 else 1000.0)),
        "pavg": patch(f"{_SWEEP}.compute_surface_pressure_avg",
                      return_value={2: 1.0 + 0.1j}),
        "ff": patch(f"{_SWEEP}._evaluate_far_field",
                    return_value=np.ones(5, dtype=np.complex128)),
        "op_kw": patch(f"{_SWEEP}._operator_kwargs", return_value={}),
        "dir": patch(f"{_SWEEP}._evaluate_directivity",
                     side_effect=lambda fr, obs, ang, cfg: (
                         np.ones((len(fr), obs.shape[0], obs.shape[1]),
                                 dtype=np.complex128),
                         np.zeros((len(fr), obs.shape[0], obs.shape[1]),
                                  dtype=np.float64),
                     )),
    }


# ---------------------------------------------------------------------------
# On-axis index at non-zero angle_min
# ---------------------------------------------------------------------------

class TestOnAxisIndex:

    def test_on_axis_at_nonzero_angle_min(self):
        angles = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
        on_axis_idx = int(np.argmin(np.abs(angles)))
        assert on_axis_idx == 2

    def test_on_axis_offset_from_zero(self):
        angles = np.linspace(5.0, 180.0, 36)
        on_axis_idx = int(np.argmin(np.abs(angles)))
        assert on_axis_idx == 0

    def test_on_axis_negative_start(self):
        angles = np.linspace(-180.0, 180.0, 73)
        on_axis_idx = int(np.argmin(np.abs(angles)))
        assert angles[on_axis_idx] == 0.0

    def test_spl_normalisation_uses_on_axis_idx(self):
        from hornlab_solver._constants import REFERENCE_PRESSURE

        angles = np.linspace(-90.0, 90.0, 37)
        on_axis_idx = int(np.argmin(np.abs(angles)))
        assert on_axis_idx == 18

        amplitudes = np.linspace(0.5, 1.0, 37)
        amplitudes[on_axis_idx] = 2.0
        spl_raw = 20.0 * np.log10(amplitudes / REFERENCE_PRESSURE)
        spl_norm = spl_raw - spl_raw[on_axis_idx]

        assert spl_norm[on_axis_idx] == 0.0
        assert np.all(spl_norm[:on_axis_idx] < 0.0)
        assert np.all(spl_norm[on_axis_idx + 1:] < 0.0)


# ---------------------------------------------------------------------------
# Early stopping via on_frequency_result
# ---------------------------------------------------------------------------

class TestEarlyStopping:

    def test_early_stop_after_two_frequencies(self):
        from hornlab_solver.sweep import run_sweep_serial

        patches = _standard_sweep_patches()
        stop_after = 2
        call_count = [0]

        def stopper(freq_idx, freq_hz, log_entry):
            call_count[0] += 1
            return freq_idx < stop_after - 1

        config = SolveConfig(
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
            on_frequency_result=stopper,
        )

        with patches["spaces"], patches["solve"], patches["pavg"], \
             patches["ff"], patches["op_kw"], patches["dir"]:
            result = run_sweep_serial(
                _make_mesh(), np.array([500.0, 1000.0, 2000.0, 4000.0]),
                _make_frame(), config,
            )

        assert len(result.frequencies_hz) == 2
        assert call_count[0] == 2
        np.testing.assert_allclose(result.frequencies_hz, [500.0, 1000.0])

    def test_no_early_stop_all_frequencies_solved(self):
        from hornlab_solver.sweep import run_sweep_serial

        patches = _standard_sweep_patches()

        config = SolveConfig(
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
            on_frequency_result=lambda i, f, log: True,
        )

        with patches["spaces"], patches["solve"], patches["pavg"], \
             patches["ff"], patches["op_kw"], patches["dir"] as dir_mock:
            result = run_sweep_serial(
                _make_mesh(), np.array([500.0, 1000.0, 2000.0]),
                _make_frame(), config,
            )

        assert len(result.frequencies_hz) == 3
        assert dir_mock.call_count == 0


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:

    def test_progress_callback_called_per_frequency(self):
        from hornlab_solver.sweep import run_sweep_serial

        patches = _standard_sweep_patches()
        progress_calls = []

        config = SolveConfig(
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
            progress_callback=lambda i, n, f: progress_calls.append((i, n, f)),
        )

        with patches["spaces"], patches["solve"], patches["pavg"], \
             patches["ff"], patches["op_kw"], patches["dir"]:
            run_sweep_serial(
                _make_mesh(), np.array([100.0, 200.0, 300.0]),
                _make_frame(), config,
            )

        assert len(progress_calls) == 3
        assert progress_calls[0] == (0, 3, 100.0)
        assert progress_calls[1] == (1, 3, 200.0)
        assert progress_calls[2] == (2, 3, 300.0)


# ---------------------------------------------------------------------------
# Parallel mode rejects callbacks
# ---------------------------------------------------------------------------

class TestParallelRejectsCallbacks:

    def test_progress_callback_rejected(self):
        from hornlab_solver.sweep import run_sweep_parallel

        config = SolveConfig(progress_callback=lambda i, n, f: None)
        with pytest.raises(ValueError, match="not supported in parallel mode"):
            run_sweep_parallel(
                _make_mesh(), np.array([100.0]), _make_frame(), config, 2,
            )

    def test_on_frequency_result_rejected(self):
        from hornlab_solver.sweep import run_sweep_parallel

        config = SolveConfig(on_frequency_result=lambda i, f, l: True)
        with pytest.raises(ValueError, match="not supported in parallel mode"):
            run_sweep_parallel(
                _make_mesh(), np.array([100.0]), _make_frame(), config, 2,
            )


# ---------------------------------------------------------------------------
# surface_pressure_avg populated in result
# ---------------------------------------------------------------------------

class TestSurfacePressureAvg:

    def test_surface_pressure_avg_in_result(self):
        from hornlab_solver.sweep import run_sweep_serial

        patches = _standard_sweep_patches()
        # Override pavg to return a known value
        patches["pavg"] = patch(
            f"{_SWEEP}.compute_surface_pressure_avg",
            return_value={2: 100.0 + 50j},
        )

        config = SolveConfig(
            observation=ObservationConfig(planes=["horizontal"], angle_count=5),
        )

        with patches["spaces"], patches["solve"], patches["pavg"], \
             patches["ff"], patches["op_kw"], patches["dir"]:
            result = run_sweep_serial(
                _make_mesh(), np.array([500.0, 1000.0]),
                _make_frame(), config,
            )

        assert result.surface_pressure_avg is not None
        assert 2 in result.surface_pressure_avg
        assert len(result.surface_pressure_avg[2]) == 2
        np.testing.assert_allclose(
            result.surface_pressure_avg[2],
            [100.0 + 50j, 100.0 + 50j],
        )

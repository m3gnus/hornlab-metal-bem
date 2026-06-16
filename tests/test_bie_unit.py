"""Unit tests for pure acoustic helpers in hornlab_metal_bem.bie."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from hornlab_metal_bem.config import SolveConfig, VelocityMode


# ---------------------------------------------------------------------------
# air_density passed into native Neumann coefficients
# ---------------------------------------------------------------------------


class TestAirDensityInNeumann:

    def test_default_air_density_in_coefficients(self):
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=4)

        tags = np.array([1, 2, 2, 1], dtype=np.int32)
        omega = 2 * np.pi * 1000.0
        config = SolveConfig(velocity_sources={2: 1.0})

        coeffs = _build_driver_neumann_coeffs(
            dp0_space,
            tags,
            omega,
            config,
            np.complex64,
        )

        v_n = 1.0 / (1j * omega)
        expected_coeff = 1j * 1.2041 * omega * v_n

        source_dofs = np.where(tags == 2)[0]
        for dof in source_dofs:
            np.testing.assert_allclose(coeffs[dof], expected_coeff, rtol=1e-6)

    def test_custom_air_density_propagates(self):
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=4)

        tags = np.array([1, 2, 2, 1], dtype=np.int32)
        omega = 2 * np.pi * 500.0
        custom_rho = 1.18
        config = SolveConfig(
            velocity_sources={2: 1.0},
            air_density=custom_rho,
        )

        coeffs = _build_driver_neumann_coeffs(
            dp0_space,
            tags,
            omega,
            config,
            np.complex128,
        )

        v_n = 1.0 / (1j * omega)
        expected_coeff = 1j * custom_rho * omega * v_n

        source_dofs = np.where(tags == 2)[0]
        for dof in source_dofs:
            np.testing.assert_allclose(coeffs[dof], expected_coeff, rtol=1e-6)

    def test_velocity_mode_velocity_uses_weight_directly(self):
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=3)

        tags = np.array([2, 2, 1], dtype=np.int32)
        omega = 2 * np.pi * 2000.0
        config = SolveConfig(
            velocity_sources={2: 0.5},
            velocity_mode=VelocityMode.VELOCITY,
            air_density=1.2041,
        )

        coeffs = _build_driver_neumann_coeffs(
            dp0_space,
            tags,
            omega,
            config,
            np.complex64,
        )

        expected_coeff = 1j * 1.2041 * omega * 0.5

        np.testing.assert_allclose(coeffs[0], expected_coeff, rtol=1e-6)
        np.testing.assert_allclose(coeffs[1], expected_coeff, rtol=1e-6)
        assert coeffs[2] == 0.0

    def test_zero_omega_acceleration_gives_zero(self):
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=2)

        tags = np.array([2, 2], dtype=np.int32)
        config = SolveConfig(velocity_sources={2: 1.0})

        coeffs = _build_driver_neumann_coeffs(
            dp0_space,
            tags,
            0.0,
            config,
            np.complex64,
        )

        assert coeffs[0] == 0.0
        assert coeffs[1] == 0.0


# ---------------------------------------------------------------------------
# impedance-tag skip set (Robin BC suppresses the velocity BC on that tag)
# ---------------------------------------------------------------------------


class TestImpedanceTagSkip:

    def test_static_impedance_tag_skips_velocity(self):
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=2)
        tags = np.array([2, 5], dtype=np.int32)
        # Tag 5 is both a velocity source and a static impedance source; the
        # Robin BC must suppress the prescribed velocity there.
        config = SolveConfig(
            velocity_sources={2: 1.0, 5: 1.0},
            impedance_sources={5: 0.05 + 0.0j},
        )

        coeffs = _build_driver_neumann_coeffs(
            dp0_space, tags, 2 * np.pi * 1000.0, config, np.complex64
        )

        assert coeffs[0] != 0.0  # tag 2 still driven
        assert coeffs[1] == 0.0  # tag 5 skipped (Robin BC carries it)

    def test_callback_only_impedance_tag_skips_velocity(self):
        """A tag driven by impedance_source_callback alone (absent from the
        static impedance_sources dict) must still be skipped by the Neumann
        builder via the tag-set union — otherwise it gets a double BC."""
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=2)
        tags = np.array([2, 5], dtype=np.int32)
        config = SolveConfig(
            velocity_sources={2: 1.0, 5: 1.0},
            impedance_sources={},  # tag 5 NOT declared statically
            impedance_source_callback=lambda f: {5: 0.05 + 0.0j},
        )

        coeffs = _build_driver_neumann_coeffs(
            dp0_space, tags, 2 * np.pi * 1000.0, config, np.complex64
        )

        assert coeffs[0] != 0.0  # tag 2 still driven
        assert coeffs[1] == 0.0  # tag 5 skipped via callback tag union

    def test_no_callback_leaves_velocity_tags_untouched(self):
        """Regression: without a callback the skip set is exactly the static
        impedance tags (canonical behavior unchanged)."""
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=2)
        tags = np.array([2, 5], dtype=np.int32)
        config = SolveConfig(velocity_sources={2: 1.0, 5: 1.0})

        coeffs = _build_driver_neumann_coeffs(
            dp0_space, tags, 2 * np.pi * 1000.0, config, np.complex64
        )

        assert coeffs[0] != 0.0
        assert coeffs[1] != 0.0  # tag 5 driven (no Robin, no callback)

    def test_resolved_impedance_tags_override_callback(self):
        """When the caller passes a resolved impedance-tag set, the Neumann
        builder uses it VERBATIM and never calls the callback. A stateful
        callback that would return a different tag is irrelevant: the skip set is
        exactly the resolved set, so the skipped tag can never diverge from the
        Robin tag the solver applies."""
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=2)
        tags = np.array([2, 5], dtype=np.int32)
        calls = []

        def stateful_cb(f):
            # Would skip tag 2 on this call (and flip on the next) — but it must
            # NOT be consulted when resolved tags are supplied.
            calls.append(f)
            return {2: 0.05 + 0.0j} if len(calls) % 2 else {5: 0.05 + 0.0j}

        config = SolveConfig(
            velocity_sources={2: 1.0, 5: 1.0},
            impedance_sources={},
            impedance_source_callback=stateful_cb,
        )

        coeffs = _build_driver_neumann_coeffs(
            dp0_space,
            tags,
            2 * np.pi * 1000.0,
            config,
            np.complex64,
            impedance_tags={5},
        )

        assert calls == []  # callback NEVER invoked when tags are resolved
        assert coeffs[0] != 0.0  # tag 2 driven
        assert coeffs[1] == 0.0  # tag 5 skipped per the resolved set


class TestSingleCallbackEvaluation:
    """Fix: impedance_source_callback is evaluated EXACTLY ONCE per frequency in
    sweep, and the resolved tags flow into the Neumann builder — so a stateful or
    non-repeatable callback cannot make the Robin payload and the velocity-skip
    set diverge (double BC / undriven tag)."""

    def test_stateful_callback_evaluated_once_per_frequency(self):
        from hornlab_metal_bem.sweep import (
            _build_neumann_rows,
            _impedance_sources_for_frequencies,
        )

        dp0_space = SimpleNamespace(global_dof_count=2)
        physical_tags = np.array([2, 5], dtype=np.int32)
        frequencies = np.array([500.0, 1000.0, 2000.0], dtype=np.float64)

        call_count = {"n": 0}

        def stateful_cb(f):
            # A non-repeatable callback: the tag it reports changes with internal
            # state. If anything evaluated it a SECOND time per frequency, the
            # resolved Robin tag and the Neumann skip tag would disagree.
            call_count["n"] += 1
            return {5: (0.01 * call_count["n"]) + 0.0j}

        config = SolveConfig(
            velocity_sources={2: 1.0, 5: 1.0},
            impedance_sources={},
            impedance_source_callback=stateful_cb,
        )

        resolved = _impedance_sources_for_frequencies(
            physical_tags, frequencies, config
        )
        # Exactly one evaluation per frequency to build the Robin payloads.
        assert call_count["n"] == len(frequencies)
        assert isinstance(resolved, list) and len(resolved) == len(frequencies)
        assert all(set(d.keys()) == {5} for d in resolved)

        rows = _build_neumann_rows(
            dp0_space, physical_tags, frequencies, config, resolved
        )
        # No further callback evaluations happened inside the Neumann builder.
        assert call_count["n"] == len(frequencies)
        # Every frequency: tag 2 driven, tag 5 skipped (carried by the resolved
        # Robin payload) — consistent across all cases.
        assert rows.shape == (len(frequencies), 2)
        assert np.all(rows[:, 0] != 0.0)
        assert np.all(rows[:, 1] == 0.0)

    def test_no_callback_static_dict_threaded_unchanged(self):
        """Back-compat: with no callback, _impedance_sources_for_frequencies
        returns a single static dict and _build_neumann_rows applies its tags to
        every frequency (canonical behavior, no per-case list)."""
        from hornlab_metal_bem.sweep import (
            _build_neumann_rows,
            _impedance_sources_for_frequencies,
        )

        dp0_space = SimpleNamespace(global_dof_count=2)
        physical_tags = np.array([2, 5], dtype=np.int32)
        frequencies = np.array([500.0, 1500.0], dtype=np.float64)
        config = SolveConfig(
            velocity_sources={2: 1.0, 5: 1.0},
            impedance_sources={5: 0.05 + 0.0j},
        )

        resolved = _impedance_sources_for_frequencies(
            physical_tags, frequencies, config
        )
        assert isinstance(resolved, dict) and set(resolved.keys()) == {5}

        rows = _build_neumann_rows(
            dp0_space, physical_tags, frequencies, config, resolved
        )
        assert rows.shape == (len(frequencies), 2)
        assert np.all(rows[:, 0] != 0.0)  # tag 2 driven every case
        assert np.all(rows[:, 1] == 0.0)  # tag 5 carried by static Robin


# ---------------------------------------------------------------------------
# compute_surface_pressure_avg
# ---------------------------------------------------------------------------

class TestComputeSurfacePressureAvg:

    def test_single_tag_uniform_pressure(self):
        from hornlab_metal_bem.bie import compute_surface_pressure_avg

        n_verts = 6

        grid = MagicMock()
        grid.elements = MagicMock()
        grid.elements.T = np.array([
            [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
        ], dtype=np.int32)
        grid.volumes = np.array([0.01, 0.01, 0.01, 0.01])

        pressure_val = 100.0 + 20.0j
        coeffs = np.full(n_verts, pressure_val, dtype=np.complex128)
        p_surface = MagicMock()
        p_surface.coefficients = coeffs

        p1_space = MagicMock()
        p1_space.local2global = np.array([
            [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
        ])

        tags = np.array([2, 2, 2, 2], dtype=np.int32)

        result = compute_surface_pressure_avg(
            grid, p_surface, tags, p1_space, [2],
        )

        np.testing.assert_allclose(result[2], pressure_val, rtol=1e-10)

    def test_area_weighting_matters(self):
        from hornlab_metal_bem.bie import compute_surface_pressure_avg

        grid = MagicMock()
        grid.elements = MagicMock()
        grid.elements.T = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        grid.volumes = np.array([0.03, 0.01])

        coeffs = np.array([100, 100, 100, 200, 200, 200], dtype=np.complex128)
        p_surface = MagicMock()
        p_surface.coefficients = coeffs

        p1_space = MagicMock()
        p1_space.local2global = np.array([[0, 1, 2], [3, 4, 5]])

        tags = np.array([2, 2], dtype=np.int32)

        result = compute_surface_pressure_avg(
            grid, p_surface, tags, p1_space, [2],
        )

        expected = (100.0 * 0.03 + 200.0 * 0.01) / 0.04
        np.testing.assert_allclose(result[2], expected, rtol=1e-10)

    def test_missing_tag_returns_zero(self):
        from hornlab_metal_bem.bie import compute_surface_pressure_avg

        grid = MagicMock()
        grid.elements = MagicMock()
        grid.elements.T = np.array([[0, 1, 2]], dtype=np.int32)
        grid.volumes = np.array([0.01])

        p_surface = MagicMock()
        p_surface.coefficients = np.array([1.0, 1.0, 1.0], dtype=np.complex128)

        p1_space = MagicMock()
        p1_space.local2global = np.array([[0, 1, 2]])

        tags = np.array([1], dtype=np.int32)

        result = compute_surface_pressure_avg(
            grid, p_surface, tags, p1_space, [2, 3],
        )

        assert result[2] == 0.0 + 0.0j
        assert result[3] == 0.0 + 0.0j

    def test_multiple_tags_independent(self):
        from hornlab_metal_bem.bie import compute_surface_pressure_avg

        grid = MagicMock()
        grid.elements = MagicMock()
        grid.elements.T = np.array([
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
        ], dtype=np.int32)
        grid.volumes = np.array([0.01, 0.01, 0.01])

        coeffs = np.array(
            [50, 50, 50, 150, 150, 150, 50, 50, 50], dtype=np.complex128,
        )
        p_surface = MagicMock()
        p_surface.coefficients = coeffs

        p1_space = MagicMock()
        p1_space.local2global = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        tags = np.array([2, 3, 2], dtype=np.int32)

        result = compute_surface_pressure_avg(
            grid, p_surface, tags, p1_space, [2, 3],
        )

        np.testing.assert_allclose(result[2], 50.0, rtol=1e-10)
        np.testing.assert_allclose(result[3], 150.0, rtol=1e-10)

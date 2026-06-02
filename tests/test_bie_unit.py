"""Unit tests for pure acoustic helpers in hornlab_solver.bie."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from hornlab_solver.config import SolveConfig, VelocityMode


# ---------------------------------------------------------------------------
# air_density passed into native Neumann coefficients
# ---------------------------------------------------------------------------


class TestAirDensityInNeumann:

    def test_default_air_density_in_coefficients(self):
        from hornlab_solver.bie import _build_driver_neumann_coeffs

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
        from hornlab_solver.bie import _build_driver_neumann_coeffs

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
        from hornlab_solver.bie import _build_driver_neumann_coeffs

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
        from hornlab_solver.bie import _build_driver_neumann_coeffs

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
# compute_surface_pressure_avg
# ---------------------------------------------------------------------------

class TestComputeSurfacePressureAvg:

    def test_single_tag_uniform_pressure(self):
        from hornlab_solver.bie import compute_surface_pressure_avg

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
        from hornlab_solver.bie import compute_surface_pressure_avg

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
        from hornlab_solver.bie import compute_surface_pressure_avg

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
        from hornlab_solver.bie import compute_surface_pressure_avg

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

"""Unit tests for pure acoustic helpers in hornlab_metal_bem.bie."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from hornlab_metal_bem.config import (
    AnnularProfile,
    CallableProfile,
    PerFaceProfile,
    SolveConfig,
    SourceMotion,
    TaperProfile,
    VelocityMode,
)


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


# ---------------------------------------------------------------------------
# Axial (rigid-piston) source motion: _build_axial_face_scale
# ---------------------------------------------------------------------------


def _two_face_cap_grid(theta_rad: float) -> SimpleNamespace:
    """Two triangles whose outward unit normals sit at +/- ``theta`` from +z,
    area-equal. The area-weighted mean normal (piston axis) is therefore +z and
    each face projects to ``cos(theta)``. ``theta == 0`` is a flat disc (both
    normals +z, projection 1). Vertices are chosen so ``cross(P1-P0, P2-P0)``
    equals the target unit normal exactly (triangle area 0.5)."""
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    verts = np.array(
        [
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-c, 0.0, s],    # face A -> ( s,0,c)
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-c, 0.0, -s],   # face B -> (-s,0,c)
        ],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return SimpleNamespace(vertices=verts.T, elements=elements.T)


def _flat_radial_source_grid(radii: list[float]) -> SimpleNamespace:
    """Flat +z triangles whose centroids sit on the x-axis at ``radii``."""
    eps = 0.01
    verts = []
    elements = []
    for radius in radii:
        base = len(verts)
        r = float(radius)
        verts.extend(
            [
                [r - eps, -eps, 0.0],
                [r + eps, -eps, 0.0],
                [r, 2.0 * eps, 0.0],
            ]
        )
        elements.append([base, base + 1, base + 2])
    return SimpleNamespace(
        vertices=np.asarray(verts, dtype=np.float64).T,
        elements=np.asarray(elements, dtype=np.int32).T,
    )


def _two_sided_diaphragm_grid() -> SimpleNamespace:
    """Two equal-area faces with one normal +z and one normal -z."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -0.01],
            [0.0, 1.0, -0.01],
            [1.0, 0.0, -0.01],
        ],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return SimpleNamespace(vertices=verts.T, elements=elements.T)


class TestAxialFaceScale:

    AXIS = np.array([0.0, 0.0, 1.0])  # forward (throat->mouth) axis for the fixture

    def test_flat_disc_projects_to_unity(self):
        from hornlab_metal_bem.bie import _build_axial_face_scale

        grid = _two_face_cap_grid(0.0)
        tags = np.array([2, 2], dtype=np.int32)
        scale = _build_axial_face_scale(grid, tags, [2], self.AXIS)
        np.testing.assert_allclose(scale, [1.0, 1.0], atol=1e-12)

    def test_tilted_cap_projects_to_cos_theta(self):
        from hornlab_metal_bem.bie import _build_axial_face_scale

        theta = np.deg2rad(35.0)
        grid = _two_face_cap_grid(theta)
        tags = np.array([2, 2], dtype=np.int32)
        scale = _build_axial_face_scale(grid, tags, [2], self.AXIS)
        # Both faces sit at theta off the axis -> both project to cos(theta).
        np.testing.assert_allclose(
            scale, [np.cos(theta), np.cos(theta)], rtol=1e-9
        )

    def test_non_source_faces_stay_zero(self):
        from hornlab_metal_bem.bie import _build_axial_face_scale

        theta = np.deg2rad(40.0)
        grid = _two_face_cap_grid(theta)
        tags = np.array([2, 7], dtype=np.int32)  # only face 0 is the source
        scale = _build_axial_face_scale(grid, tags, [2], self.AXIS)
        np.testing.assert_allclose(scale[0], np.cos(theta), rtol=1e-9)
        assert scale[1] == 0.0  # tag 7 is not a source face

    def test_absent_source_tag_returns_none(self):
        """No source faces -> None, so the caller keeps the bit-identical
        uniform-normal path."""
        from hornlab_metal_bem.bie import _build_axial_face_scale

        grid = _two_face_cap_grid(np.deg2rad(20.0))
        tags = np.array([2, 2], dtype=np.int32)
        assert _build_axial_face_scale(grid, tags, [99], self.AXIS) is None

    def test_degenerate_axis_returns_none(self):
        """A zero-length axis -> None (fall back to uniform normal) rather than
        dividing by zero."""
        from hornlab_metal_bem.bie import _build_axial_face_scale

        grid = _two_face_cap_grid(np.deg2rad(20.0))
        tags = np.array([2, 2], dtype=np.int32)
        assert _build_axial_face_scale(grid, tags, [2], np.zeros(3)) is None

    def test_two_sided_tag_is_axial_dipole_path(self):
        """Dipole path: one source tag covering both diaphragm sides under
        source_motion='axial' preserves opposite front/back drive signs."""
        from hornlab_metal_bem.bie import (
            _build_driver_neumann_coeffs,
            _build_source_face_scale,
        )

        grid = _two_sided_diaphragm_grid()
        tags = np.array([2, 2], dtype=np.int32)
        config = SolveConfig(
            velocity_sources={2: 1.0},
            velocity_mode=VelocityMode.VELOCITY,
            source_motion=SourceMotion.AXIAL,
        )

        scale = _build_source_face_scale(grid, tags, config, self.AXIS, np.zeros(3))
        np.testing.assert_allclose(scale, [1.0, -1.0], atol=1e-12)

        coeffs = _build_driver_neumann_coeffs(
            SimpleNamespace(global_dof_count=2),
            tags,
            2 * np.pi * 1000.0,
            config,
            np.complex128,
            source_face_scale=scale,
        )
        np.testing.assert_allclose(coeffs[0], -coeffs[1], rtol=1e-12)


# ---------------------------------------------------------------------------
# General source velocity profiles: _build_source_face_scale
# ---------------------------------------------------------------------------


class TestSourceFaceScaleProfiles:

    AXIS = np.array([0.0, 0.0, 1.0])
    CENTER = np.array([0.0, 0.0, 0.0])

    def test_default_normal_returns_none_for_scalar_path(self):
        from hornlab_metal_bem.bie import (
            _build_driver_neumann_coeffs,
            _build_source_face_scale,
        )

        dp0_space = SimpleNamespace(global_dof_count=2)
        grid = _two_face_cap_grid(0.0)
        tags = np.array([2, 2], dtype=np.int32)
        omega = 2 * np.pi * 1000.0
        config = SolveConfig(velocity_sources={2: 1.0})

        scale = _build_source_face_scale(grid, tags, config, self.AXIS, self.CENTER)
        assert scale is None

        direct = _build_driver_neumann_coeffs(
            dp0_space, tags, omega, config, np.complex64
        )
        via_scale = _build_driver_neumann_coeffs(
            dp0_space,
            tags,
            omega,
            config,
            np.complex64,
            source_face_scale=scale,
        )
        assert np.array_equal(direct, via_scale)

    def test_axial_source_motion_matches_legacy_helper(self):
        from hornlab_metal_bem.bie import (
            _build_axial_face_scale,
            _build_source_face_scale,
        )

        theta = np.deg2rad(35.0)
        grid = _two_face_cap_grid(theta)
        tags = np.array([2, 2], dtype=np.int32)
        config = SolveConfig(
            velocity_sources={2: 1.0}, source_motion=SourceMotion.AXIAL
        )

        legacy = _build_axial_face_scale(grid, tags, [2], self.AXIS)
        general = _build_source_face_scale(grid, tags, config, self.AXIS, self.CENTER)
        assert np.array_equal(general, legacy)

    def test_degenerate_axis_axial_matches_legacy_none(self):
        from hornlab_metal_bem.bie import _build_source_face_scale

        grid = _two_face_cap_grid(np.deg2rad(20.0))
        tags = np.array([2, 2], dtype=np.int32)
        config = SolveConfig(
            velocity_sources={2: 1.0}, source_motion=SourceMotion.AXIAL
        )

        assert (
            _build_source_face_scale(grid, tags, config, np.zeros(3), self.CENTER)
            is None
        )

    def test_raised_cosine_taper_decreases_with_normalized_radius(self):
        from hornlab_metal_bem.bie import _build_source_face_scale

        grid = _flat_radial_source_grid([0.0, 0.5, 1.0])
        tags = np.array([2, 2, 2], dtype=np.int32)
        config = SolveConfig(
            velocity_sources={2: 1.0},
            source_velocity_profiles={2: TaperProfile(start=0.0)},
        )

        scale = _build_source_face_scale(grid, tags, config, self.AXIS, self.CENTER)
        np.testing.assert_allclose(scale[0], 1.0, atol=1e-12)
        assert scale[0] >= scale[1] >= scale[2]
        np.testing.assert_allclose(scale[1], 0.5, atol=1e-12)
        np.testing.assert_allclose(scale[2], 0.0, atol=1e-12)

    def test_annular_profile_selects_normalized_radius_band(self):
        from hornlab_metal_bem.bie import _build_source_face_scale

        grid = _flat_radial_source_grid([0.1, 0.5, 0.9])
        tags = np.array([2, 2, 2], dtype=np.int32)
        config = SolveConfig(
            velocity_sources={2: 1.0},
            source_velocity_profiles={2: AnnularProfile(0.4, 0.7)},
        )

        scale = _build_source_face_scale(grid, tags, config, self.AXIS, self.CENTER)
        np.testing.assert_allclose(scale, [0.0, 1.0, 0.0], atol=1e-12)

    def test_per_face_profile_applies_in_physical_tag_face_order(self):
        from hornlab_metal_bem.bie import _build_source_face_scale

        grid = _flat_radial_source_grid([0.0, 1.0])
        tags = np.array([2, 2], dtype=np.int32)
        weights = np.array([1.0 + 0.5j, 0.25 - 0.25j], dtype=np.complex128)
        config = SolveConfig(
            velocity_sources={2: 1.0},
            source_velocity_profiles={2: PerFaceProfile(weights)},
        )

        scale = _build_source_face_scale(grid, tags, config, self.AXIS, self.CENTER)
        np.testing.assert_allclose(scale, weights, rtol=1e-12)

    def test_callable_profile_receives_geometry_and_applies_weights(self):
        from hornlab_metal_bem.bie import _build_source_face_scale

        grid = _flat_radial_source_grid([0.0, 1.0])
        tags = np.array([2, 2], dtype=np.int32)
        seen = {}

        def callback(centroids, normals, axis, source_center):
            seen["centroids"] = centroids
            seen["normals"] = normals
            seen["axis"] = axis
            seen["source_center"] = source_center
            return np.array([0.75 + 0.25j, 0.25 + 0.0j], dtype=np.complex128)

        config = SolveConfig(
            velocity_sources={2: 1.0},
            source_velocity_profiles={2: CallableProfile(callback)},
        )

        scale = _build_source_face_scale(grid, tags, config, self.AXIS, self.CENTER)
        np.testing.assert_allclose(scale, [0.75 + 0.25j, 0.25 + 0.0j])
        assert seen["centroids"].shape == (2, 3)
        np.testing.assert_allclose(seen["normals"], [[0.0, 0.0, 1.0]] * 2)
        np.testing.assert_allclose(seen["axis"], self.AXIS)
        np.testing.assert_allclose(seen["source_center"], self.CENTER)


# ---------------------------------------------------------------------------
# Axial source motion applied through _build_driver_neumann_coeffs
# ---------------------------------------------------------------------------


class TestAxialNeumannCoeffs:

    def test_axial_scale_applied_per_face(self):
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=3)
        tags = np.array([2, 2, 1], dtype=np.int32)
        omega = 2 * np.pi * 1500.0
        config = SolveConfig(
            velocity_sources={2: 1.0},
            velocity_mode=VelocityMode.VELOCITY,
            source_motion=SourceMotion.AXIAL,
        )
        scale = np.array([1.0, 0.5, 0.0], dtype=np.float64)

        coeffs = _build_driver_neumann_coeffs(
            dp0_space, tags, omega, config, np.complex64,
            axial_face_scale=scale,
        )

        base = 1j * config.air_density * omega
        # Pole face full, rim face halved, non-source face untouched.
        np.testing.assert_allclose(coeffs[0], base * 1.0, rtol=1e-6)
        np.testing.assert_allclose(coeffs[1], base * 0.5, rtol=1e-6)
        assert coeffs[2] == 0.0

    def test_axial_all_ones_matches_uniform_normal(self):
        """A flat-disc projection (all ones) reproduces the uniform-normal
        coefficients -- the axial<->normal degeneracy for a flat piston."""
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=3)
        tags = np.array([2, 2, 1], dtype=np.int32)
        omega = 2 * np.pi * 1000.0
        config = SolveConfig(velocity_sources={2: 1.0})

        normal = _build_driver_neumann_coeffs(
            dp0_space, tags, omega, config, np.complex128,
        )
        axial = _build_driver_neumann_coeffs(
            dp0_space, tags, omega, config, np.complex128,
            axial_face_scale=np.ones(3, dtype=np.float64),
        )
        np.testing.assert_allclose(axial, normal, rtol=1e-12)

    def test_axial_acceleration_mode_divides_per_face(self):
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=2)
        tags = np.array([2, 2], dtype=np.int32)
        omega = 2 * np.pi * 800.0
        # Acceleration mode (default): v_n = weight*scale/(1j*omega), so the
        # coefficient 1j*rho*omega*v_n collapses to rho*weight*scale (real).
        config = SolveConfig(
            velocity_sources={2: 2.0}, source_motion=SourceMotion.AXIAL,
        )
        scale = np.array([1.0, 0.25], dtype=np.float64)

        coeffs = _build_driver_neumann_coeffs(
            dp0_space, tags, omega, config, np.complex128,
            axial_face_scale=scale,
        )
        expected = config.air_density * 2.0 * scale
        np.testing.assert_allclose(coeffs, expected, rtol=1e-9)

    def test_axial_respects_impedance_skip(self):
        """A Robin (impedance) tag is still skipped under axial motion -- no
        double boundary condition."""
        from hornlab_metal_bem.bie import _build_driver_neumann_coeffs

        dp0_space = SimpleNamespace(global_dof_count=2)
        tags = np.array([2, 5], dtype=np.int32)
        config = SolveConfig(
            velocity_sources={2: 1.0, 5: 1.0},
            impedance_sources={5: 0.05 + 0.0j},
            source_motion=SourceMotion.AXIAL,
        )
        scale = np.array([1.0, 1.0], dtype=np.float64)

        coeffs = _build_driver_neumann_coeffs(
            dp0_space, tags, 2 * np.pi * 1000.0, config, np.complex64,
            axial_face_scale=scale,
        )
        assert coeffs[0] != 0.0  # tag 2 driven
        assert coeffs[1] == 0.0  # tag 5 carried by Robin BC, not velocity


class TestNeumannRowsAxial:
    """The sweep row-stacker forwards the geometry-only axial scale to every
    frequency's Neumann builder call."""

    def test_axial_face_scale_forwarded_to_every_row(self):
        from hornlab_metal_bem.sweep import _build_neumann_rows

        dp0_space = SimpleNamespace(global_dof_count=2)
        physical_tags = np.array([2, 2], dtype=np.int32)
        frequencies = np.array([500.0, 1000.0, 2000.0], dtype=np.float64)
        config = SolveConfig(
            velocity_sources={2: 1.0}, source_motion=SourceMotion.AXIAL,
        )
        scale = np.array([1.0, 0.4], dtype=np.float64)

        rows = _build_neumann_rows(
            dp0_space, physical_tags, frequencies, config, {},
            axial_face_scale=scale,
        )

        assert rows.shape == (len(frequencies), 2)
        # Rim face is 0.4x the pole face in every frequency row.
        for r in range(rows.shape[0]):
            np.testing.assert_allclose(rows[r, 1] / rows[r, 0], 0.4, rtol=1e-6)

    def test_no_scale_is_uniform_across_source_faces(self):
        """Regression: without a scale (default), both source faces share the
        same coefficient -- the uniform-normal breathing cap."""
        from hornlab_metal_bem.sweep import _build_neumann_rows

        dp0_space = SimpleNamespace(global_dof_count=2)
        physical_tags = np.array([2, 2], dtype=np.int32)
        frequencies = np.array([750.0], dtype=np.float64)
        config = SolveConfig(velocity_sources={2: 1.0})

        rows = _build_neumann_rows(
            dp0_space, physical_tags, frequencies, config, {},
        )
        np.testing.assert_allclose(rows[0, 0], rows[0, 1], rtol=1e-12)

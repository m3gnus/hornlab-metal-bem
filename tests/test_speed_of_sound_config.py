"""``SolveConfig.speed_of_sound`` must reach the solve, not just be accepted.

The value was hardcoded in ``_constants.SPEED_OF_SOUND`` until 2026-09-03 while
``air_density`` was already configurable -- an asymmetry that showed up as a
fixed 0.093% bias in every comparison against ABEC3, which defaults to 343.32
m/s (``benchmarks/abec-g8-circsym-ib/``).

A knob that is accepted and ignored is worse than no knob, so the tests below
pin the value where it actually does work: the wavenumber ``k = 2*pi*f/c``, in
both the CircSym and the 3-D native paths, plus the mesh-resolution
diagnostics derived from the wavelength.
"""

from __future__ import annotations

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem._constants import SPEED_OF_SOUND
from hornlab_metal_bem.circsym import MeridianMesh, _complex_wavenumber
from hornlab_metal_bem.config import (
    BIEFormulation,
    ObservationConfig,
    SolveConfig,
    VelocityMode,
)
from hornlab_metal_bem.sweep import (
    _apply_mesh_resolution_policy,
    _k_values_for_native,
)

C_ABEC = 343.32  # ABEC3's default, the value that motivated the knob


def _sphere_meridian(radius: float = 0.1, segments: int = 24) -> MeridianMesh:
    theta = np.linspace(0.0, np.pi, segments + 1)
    points = np.column_stack([radius * np.sin(theta), radius * np.cos(theta)])
    return MeridianMesh.from_polyline(points, tags=2)


def _config(speed_of_sound: float | None = None) -> SolveConfig:
    kwargs = {} if speed_of_sound is None else {"speed_of_sound": speed_of_sound}
    return SolveConfig(
        velocity_sources={2: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        observation=ObservationConfig(
            distance_m=2.0,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=3,
            planes=["horizontal"],
            origin="throat",
        ),
        **kwargs,
    )


def test_default_matches_the_shipped_constant():
    """The default must not move: every result solved before 2026-09-03 used it."""
    assert SolveConfig().speed_of_sound == SPEED_OF_SOUND == 343.0


def test_custom_value_is_stored():
    assert SolveConfig(speed_of_sound=C_ABEC).speed_of_sound == C_ABEC


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_non_physical_values_are_rejected(bad):
    with pytest.raises(ValueError, match="speed_of_sound"):
        SolveConfig(speed_of_sound=bad)


def test_wavenumber_uses_the_configured_speed():
    frequency = 1000.0
    k_default = _complex_wavenumber(frequency, _config())
    k_abec = _complex_wavenumber(frequency, _config(C_ABEC))
    assert k_default.real == pytest.approx(2.0 * np.pi * frequency / SPEED_OF_SOUND)
    assert k_abec.real == pytest.approx(2.0 * np.pi * frequency / C_ABEC)
    assert k_abec.real != k_default.real


def test_wavenumber_keeps_the_complex_k_shift_relative():
    """The imaginary shift is a fraction of k_real, so it must move with c too."""
    cfg = _config(C_ABEC)
    cfg.formulation = BIEFormulation.COMPLEX_K
    cfg.complex_k_shift = 0.01
    k = _complex_wavenumber(1000.0, cfg)
    assert k.imag == pytest.approx(k.real * 0.01)


def test_native_k_values_use_the_configured_speed():
    frequencies = np.array([500.0, 5000.0])
    k_default, _ = _k_values_for_native(frequencies, _config())
    k_abec, _ = _k_values_for_native(frequencies, _config(C_ABEC))
    np.testing.assert_allclose(
        k_abec.astype(np.float64) / k_default.astype(np.float64),
        SPEED_OF_SOUND / C_ABEC,
        rtol=1e-6,
    )


def test_mesh_resolution_diagnostics_use_the_configured_speed():
    slow, fast = {}, {}
    kwargs = dict(
        frequency_hz=10_000.0,
        mesh_max_edge_m=0.005,
        elements_per_wavelength_min=6.0,
    )
    _apply_mesh_resolution_policy(slow, speed_of_sound=SPEED_OF_SOUND, **kwargs)
    _apply_mesh_resolution_policy(fast, speed_of_sound=2.0 * SPEED_OF_SOUND, **kwargs)
    assert fast["mesh_elements_per_wavelength"] == pytest.approx(
        2.0 * slow["mesh_elements_per_wavelength"]
    )
    assert fast["mesh_max_valid_frequency_hz"] == pytest.approx(
        2.0 * slow["mesh_max_valid_frequency_hz"]
    )


def test_solve_scales_with_the_configured_speed_end_to_end():
    """The end-to-end gate: same k must give the same answer, and only c changed.

    ``k = 2*pi*f/c``, so solving at ``(f, c)`` and at ``(f*r, c*r)`` is the same
    physical problem on the same mesh and must agree to solver noise. Holding
    ``f`` and changing only ``c`` must NOT agree -- that second half is what
    catches a value accepted into the dataclass and then ignored.
    """
    mesh = _sphere_meridian()
    ratio = C_ABEC / SPEED_OF_SOUND
    frequency = 2000.0

    base = metal_bem.solve_circsym_frequencies(mesh, [frequency], _config())
    rescaled = metal_bem.solve_circsym_frequencies(
        mesh, [frequency * ratio], _config(C_ABEC)
    )
    changed = metal_bem.solve_circsym_frequencies(mesh, [frequency], _config(C_ABEC))

    # Same wavenumber -> same solve. The pressure carries the rho*c factor of
    # the medium, so compare the rho-independent shape and the normalised
    # impedance rather than raw pascals.
    np.testing.assert_allclose(
        rescaled.directivity_db, base.directivity_db, atol=1e-6
    )
    np.testing.assert_allclose(
        rescaled.impedance / (rescaled.config.air_density * C_ABEC),
        base.impedance / (base.config.air_density * SPEED_OF_SOUND),
        rtol=1e-6,
    )

    # Same frequency, different c -> a different wavenumber and a different
    # answer. Guards against the knob being stored and never read.
    z_base = base.impedance[0] / (base.config.air_density * SPEED_OF_SOUND)
    z_changed = changed.impedance[0] / (changed.config.air_density * C_ABEC)
    assert abs(z_changed - z_base) > 1e-9

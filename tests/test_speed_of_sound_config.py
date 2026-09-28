"""``SolveConfig.speed_of_sound`` must reach the solve, not just be accepted.

The value was hardcoded in ``_constants.SPEED_OF_SOUND`` until 2026-09-03 while
``air_density`` was already configurable -- an asymmetry that showed up as a
fixed 0.093% bias in every comparison against ABEC3, which defaults to 343.32
m/s.

A knob that is accepted and ignored is worse than no knob, so the tests below
pin the value where it actually does work: the wavenumber ``k = 2*pi*f/c``
in the 3-D native path, plus mesh-resolution diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem._constants import SPEED_OF_SOUND
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
from test_native_coupled_ib_validation import _straight_channel_mesh, TAG_APERTURE, TAG_THROAT
from hornlab_metal_bem.metal import discover_native_runtime

C_ABEC = 343.32  # ABEC3's default, the value that motivated the knob


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


def test_native_k_values_use_the_configured_speed():
    frequencies = np.array([500.0, 5000.0])
    k_default, _ = _k_values_for_native(frequencies, _config())
    k_abec, _ = _k_values_for_native(frequencies, _config(C_ABEC))
    np.testing.assert_allclose(
        k_abec.astype(np.float64) / k_default.astype(np.float64),
        SPEED_OF_SOUND / C_ABEC,
        rtol=1e-6,
    )


def test_native_complex_k_shift_tracks_configured_speed():
    config = _config(C_ABEC)
    config.formulation = BIEFormulation.COMPLEX_K
    config.complex_k_shift = 0.01
    real, imag = _k_values_for_native(np.array([1000.0]), config)
    assert real[0] == pytest.approx(2.0 * np.pi * 1000.0 / C_ABEC, rel=1e-6)
    assert imag[0] == pytest.approx(real[0] * 0.01, rel=1e-6)


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
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip("Swift/Metal native helper unavailable: " + "; ".join(status.unavailable_reasons))

    mesh = _straight_channel_mesh(0.04, 0.003, rings=5, sectors=32)
    frequency = 1600.0
    ratio = 1.1
    def config(speed: float) -> SolveConfig:
        return SolveConfig(
            velocity_sources={TAG_THROAT: 1.0},
            velocity_mode=VelocityMode.VELOCITY,
            aperture_tag=TAG_APERTURE,
            speed_of_sound=speed,
            observation=ObservationConfig(
                distance_m=1.5, angle_min_deg=0.0, angle_max_deg=90.0,
                angle_count=5, planes=["horizontal"], origin="mouth",
            ),
            metal_native_assembly_mode="corrected",
            dense_solve_dtype="float64",
        )

    base = metal_bem.solve_frequencies(mesh, [frequency], config(SPEED_OF_SOUND))
    rescaled = metal_bem.solve_frequencies(
        mesh, [frequency * ratio], config(SPEED_OF_SOUND * ratio)
    )
    changed = metal_bem.solve_frequencies(mesh, [frequency], config(SPEED_OF_SOUND * ratio))
    np.testing.assert_allclose(rescaled.directivity_db, base.directivity_db, atol=1e-3)
    np.testing.assert_allclose(
        rescaled.impedance / ratio, base.impedance, rtol=1e-3
    )
    assert abs(changed.impedance[0] / ratio - base.impedance[0]) > 0.01

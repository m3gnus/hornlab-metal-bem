"""Independent reference for a straight circular channel opening into an infinite baffle.

This module is the analytic side of the resonant coupled-IB validation gate in
``test_native_coupled_ib_validation.py``. It uses no BEM code.

Model
-----
A rigid-walled circular pipe of radius ``a`` and length ``L`` is closed at the
throat (z = -L) by a piston of prescribed uniform normal velocity ``v0 = 1 m/s``
and opens at z = 0 into a rigid infinite baffle (half space z > 0).

* Inside the pipe the field is one-dimensional and lossless (plane waves)::

      p(z) = A e^{ikz} + B e^{-ikz},   u(z) = (A e^{ikz} - B e^{-ikz}) / (rho c)

  with time convention exp(-i omega t), the convention of this solver.
* The mouth loads the pipe with the radiation impedance of a rigid baffled
  circular piston of radius ``a`` (King, 1934)::

      Z_r = rho c [ (1 - 2 J1(2ka)/(2ka)) - i (2 H1(2ka)/(2ka)) ]      (H1 = Struve)

  and the boundary condition at z = 0 is ``p(0) = Z_r u(0)``.
* Throat condition: ``u(-L) = v0`` (velocity source, unit amplitude).
* The mouth velocity ``u(0)`` obtained from the two conditions is treated as the
  uniform velocity of a baffled piston, and the pressure at the observation
  point is the Rayleigh half-space result (closed form on axis)::

      p(D) = rho c u(0) [ e^{ikD} - e^{ik sqrt(D^2 + a^2)} ]

Geometry used by the gate: a = 40 mm, L = 100 mm. Observation: on axis (0 deg),
D = 1.5 m from the mouth plane (``ObservationConfig.origin = "mouth"``). The
first quarter-wave resonance of the closed-open pipe with end correction sits near
650 Hz, so the gate frequency band brackets a resonance at its low-frequency
end and rises across the pipe's first half-wave region.

Valid range
-----------
The 1-D plane-wave assumption needs frequencies below the first cross mode of the
pipe, ka = 1.84 (about 2.5 kHz for a = 40 mm). The model additionally assumes a
uniform mouth velocity, which the true mouth field departs from as ka grows, so
the gate stops at 2.2 kHz (ka = 1.6) and its tolerances include that model gap
(the converged solver is not expected to equal the reference exactly; see the
tolerance derivation in the gate's docstring).
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1, struve


def baffled_piston_radiation_impedance(
    k: float, radius: float, rho: float, c: float
) -> complex:
    """King radiation impedance of a rigid baffled circular piston, exp(-i w t)."""
    x = 2.0 * k * radius
    return rho * c * ((1.0 - 2.0 * j1(x) / x) - 1j * (2.0 * struve(1, x) / x))


def baffled_piston_on_axis(
    radius: float, distance: float, k: float, rho: float, c: float
) -> complex:
    """On-axis pressure of a unit-velocity baffled piston (Rayleigh, closed form)."""
    return rho * c * (
        np.exp(1j * k * distance) - np.exp(1j * k * np.hypot(distance, radius))
    )


def pipe_mouth_velocity(
    k: float, radius: float, depth: float, rho: float, c: float
) -> complex:
    """Mouth velocity of the lossless pipe for unit throat velocity."""
    zr = baffled_piston_radiation_impedance(k, radius, rho, c)
    matrix = np.array(
        [
            [np.exp(-1j * k * depth) / (rho * c), -np.exp(1j * k * depth) / (rho * c)],
            [1.0 - zr / (rho * c), 1.0 + zr / (rho * c)],
        ]
    )
    forward, backward = np.linalg.solve(matrix, np.array([1.0, 0.0]))
    return (forward - backward) / (rho * c)


def pipe_on_axis_pressure(
    frequencies_hz: np.ndarray,
    *,
    radius: float,
    depth: float,
    distance: float,
    rho: float,
    c: float,
) -> np.ndarray:
    """Reference on-axis complex pressure for each frequency (unit throat velocity)."""
    out = np.empty(len(frequencies_hz), dtype=np.complex128)
    for i, f in enumerate(np.asarray(frequencies_hz, dtype=np.float64)):
        k = 2.0 * np.pi * f / c
        out[i] = pipe_mouth_velocity(k, radius, depth, rho, c) * baffled_piston_on_axis(
            radius, distance, k, rho, c
        )
    return out

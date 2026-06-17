"""Frequency-dependent wall-admittance models for Robin BCs.

All functions return normalized admittance beta = rho*c/Zs (the convention
``SolveConfig.impedance_sources`` / ``robinBetasByTriangle`` expect). Re(beta)
>= 0 for passive surfaces.

These helpers are intended to be wrapped in an
``impedance_source_callback`` (see ``SolveConfig.impedance_source_callback``)
so that wall admittance can vary with frequency. The first-order visco-thermal
model below is the *wide-duct / low-reduced-frequency* limit; for genuinely
narrow slots/taps (where the viscous boundary-layer thickness ``delta_v`` is a
non-negligible fraction of the channel half-width) the physically correct route
is a Zwikker-Kosten / Kirchhoff duct model giving a complex propagation
constant ``k_tv = k_phase + i*alpha``, converted to an equivalent wall beta
``beta_wall ~ (2A/P)*(alpha - i*(k_phase - k0))``. That is a documented
extension point; this module ships the first-order model plus a small material
``Zs(f)`` / beta-prior table.
"""
from __future__ import annotations

import math

RHO0 = 1.2041          # kg/m^3  (matches SolveConfig.air_density default)
C0 = 343.0             # m/s     (matches _constants.SPEED_OF_SOUND)
MU = 1.825e-5          # Pa*s    dynamic viscosity of air @ 20 C
PR = 0.71              # Prandtl number
GAMMA = 1.4            # ratio of specific heats


def viscothermal_wall_beta(f_hz: float, *, hydraulic_radius_m: float) -> complex:
    """Cremer / low-reduced-frequency visco-thermal boundary-layer wall
    admittance for a narrow duct/slot of given hydraulic radius
    ``r_h = 2*A/P``.

    Boundary-layer thicknesses ``delta_v = sqrt(2*mu/(rho*omega))``,
    ``delta_t = delta_v/sqrt(Pr)``. The equivalent normalized wall admittance
    from the visco-thermal surface loss is, to first order,

        beta ~ (omega/(2*c)) * (delta_v + (gamma-1)*delta_t)

    which is real (purely dissipative) and passive by construction. This is the
    LOW-frequency / wide-duct limit; for very narrow slots use a full
    Zwikker-Kosten model and convert its propagation constant (see module
    docstring). ``hydraulic_radius_m`` is accepted so callers can record the
    geometry the model was derived for; the first-order surface term does not
    depend on it (it cancels into the area/perimeter factor of the wall
    boundary condition).
    """
    if f_hz <= 0:
        return 0.0 + 0.0j
    omega = 2.0 * math.pi * f_hz
    delta_v = math.sqrt(2.0 * MU / (RHO0 * omega))
    delta_t = delta_v / math.sqrt(PR)
    # dissipative (real) admittance; passive by construction
    beta = (omega / (2.0 * C0)) * (delta_v + (GAMMA - 1.0) * delta_t)
    return complex(beta, 0.0)


def beta_from_surface_impedance(Zs: complex) -> complex:
    """Convert a (complex) specific surface impedance Zs [Pa*s/m] to beta.

    Zs -> 0 is a pressure-release/short boundary (infinite admittance), the
    OPPOSITE of a rigid wall, so a near-zero magnitude is rejected rather than
    silently mapped to beta = 0 (rigid).
    """
    if abs(complex(Zs)) < 1e-12:
        raise ValueError(
            "Zs is at/near zero (pressure-release limit); beta = rho*c/Zs diverges"
        )
    return complex(RHO0 * C0) / complex(Zs)


# Material / Zs(f) table - specific surface impedances (Pa*s/m), or beta priors.
# Sources: impedance-tube (ISO 10534-2 / ASTM E1050) normal-incidence data.
MATERIAL_BETA_PRIORS: dict[str, complex] = {
    "sealed_painted_mdf": 0.003 + 0.0j,    # 0.003-0.015
    "raw_mdf_plywood":    0.02 + 0.0j,     # 0.01-0.03 (incl. layer texture)
    "lossy_port_interior": 0.06 + 0.0j,    # 0.03-0.10
    "felt_liner":         0.20 + 0.0j,     # 0.10-0.30
}


def constant_beta_callback(tag_to_beta: dict[int, complex]):
    """Trivial f-independent callback (use the static impedance_sources instead
    unless you want to compose with a frequency-dependent term)."""
    def _cb(_f_hz: float) -> dict[int, complex]:
        return dict(tag_to_beta)
    return _cb


def viscothermal_port_callback(port_tag: int, *, hydraulic_radius_m: float):
    """Ready-to-use callback applying visco-thermal wall loss to one port tag."""
    def _cb(f_hz: float) -> dict[int, complex]:
        return {port_tag: viscothermal_wall_beta(
            f_hz, hydraulic_radius_m=hydraulic_radius_m)}
    return _cb

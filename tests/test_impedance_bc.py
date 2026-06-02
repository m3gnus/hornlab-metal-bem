"""Robin / impedance BC tests for the canonical solver.

Covers the new ``_assemble_and_solve_impedance`` code path added
2026-05-18:

- ``test_rigid_recovery_when_beta_is_zero``: setting
  ``impedance_sources={tag: 0+0j}`` must reproduce the rigid solve
  bit-for-bit. Guards against accidental coupling between the Robin
  branch and the standard branch when β = 0.
- ``test_light_damping_stays_finite_near_eigenvalue``: probe near a
  discrete interior-Dirichlet eigenvalue of a closed sphere mesh and
  assert both the rigid and the damped solves stay finite. The damped
  branch must not NaN/blow up.
- ``test_bm_plus_impedance_raises``: Burton-Miller + impedance BC is
  intentionally not supported (workspace BM=OFF policy); the solver
  must raise ``NotImplementedError`` rather than silently dropping one
  of the two.

Tests skip cleanly when bempp-cl or the OpenCL CPU runtime isn't
available (matches the convention in ``test_reference_asro68.py``).
"""
from __future__ import annotations

import numpy as np
import pytest


def _require_bempp_cpu() -> None:
    """Skip if bempp-cl or the OpenCL CPU runtime aren't importable."""
    try:
        import bempp_cl.api  # noqa: F401  -- import probe
    except Exception as exc:  # pragma: no cover - depends on env
        pytest.skip(f"bempp-cl unavailable: {exc}")

    try:
        from hornlab_solver import configure_opencl

        configure_opencl("cpu")
    except Exception as exc:  # pragma: no cover - depends on env
        pytest.skip(f"OpenCL CPU runtime unavailable: {exc}")


def _build_small_closed_mesh():
    """Tiny closed sphere from bempp shapes — no gmsh CLI dependency.

    Tag 1 = rigid wall (most of the surface). Tag 2 = velocity source
    (first ~24 elements). Returns ``(grid, physical_tags)``.
    """
    import bempp_cl.api as bempp_api

    grid = bempp_api.shapes.regular_sphere(3)  # 512 triangles
    n = grid.number_of_elements
    tags = np.ones(n, dtype=np.int32)
    tags[:24] = 2  # small "throat" patch driving the sphere
    return grid, tags


# ---------------------------------------------------------------------------
# beta = 0 must reproduce rigid solve
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_rigid_recovery_when_beta_is_zero():
    """Robin branch with β = 0 must match the standard branch."""
    _require_bempp_cpu()

    from hornlab_solver.bie import solve_single_frequency
    from hornlab_solver.config import (
        BIEFormulation,
        LinearSolver,
        SolveConfig,
    )

    grid, tags = _build_small_closed_mesh()

    base_kwargs = dict(
        velocity_sources={2: 1.0},
        solver=LinearSolver.LU,
        formulation=BIEFormulation.STANDARD,
        precision="double",
    )

    cfg_rigid = SolveConfig(**base_kwargs)
    cfg_beta0 = SolveConfig(impedance_sources={1: 0.0 + 0.0j}, **base_kwargs)

    freq = 500.0
    r_rigid = solve_single_frequency(grid, tags, freq, cfg_rigid)
    r_beta0 = solve_single_frequency(grid, tags, freq, cfg_beta0)

    p_rigid = np.asarray(r_rigid.pressure_on_surface.coefficients)
    p_beta0 = np.asarray(r_beta0.pressure_on_surface.coefficients)

    # β = 0 collapses the Robin contribution to zero, so the LHS reduces
    # to A_mat exactly. We solve the same system, just via a dense LU
    # instead of the bempp linalg.lu path — that means floating-point
    # rounding can differ slightly. Tighten to machine epsilon scaled by
    # the magnitude of the rigid solution.
    rigid_norm = np.max(np.abs(p_rigid))
    max_abs_diff = np.max(np.abs(p_beta0 - p_rigid))
    np.testing.assert_allclose(
        p_beta0, p_rigid,
        atol=1e-12 * rigid_norm,
        rtol=1e-10,
        err_msg=(
            f"β=0 Robin solve diverged from rigid solve by "
            f"{max_abs_diff:.3e} (rigid norm {rigid_norm:.3e})"
        ),
    )


# ---------------------------------------------------------------------------
# Light damping near interior-Dirichlet eigenvalue: both solves stay finite
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_light_damping_stays_finite_near_eigenvalue():
    """Damped Robin solve must not NaN/blow up near k*r ≈ π.

    For the bempp unit-sphere mesh the first nontrivial interior
    Dirichlet eigenvalue is around k*r = π (≈171.5 Hz with c=343 m/s).
    On the BIGMEH td24 reference this is where unphysical +7.4 dB peaks
    appear in the rigid sweep; β=0.05 light damping cleans them up.
    We don't try to reproduce the 7.4 dB number here — we just assert
    the damped path is numerically well-behaved.
    """
    _require_bempp_cpu()

    from hornlab_solver.bie import solve_single_frequency
    from hornlab_solver.config import (
        BIEFormulation,
        LinearSolver,
        SolveConfig,
    )

    grid, tags = _build_small_closed_mesh()

    # Closest discrete eigenfreq for the faceted regular_sphere(3) mesh
    # is slightly above the analytical k=π. 172 Hz lands in the regime
    # where the rigid system conditioning starts to degrade.
    freq = 172.0

    base_kwargs = dict(
        velocity_sources={2: 1.0},
        solver=LinearSolver.LU,
        formulation=BIEFormulation.STANDARD,
        precision="double",
    )
    cfg_rigid = SolveConfig(**base_kwargs)
    cfg_damped = SolveConfig(
        impedance_sources={1: 0.05 + 0.0j}, **base_kwargs,
    )

    r_rigid = solve_single_frequency(grid, tags, freq, cfg_rigid)
    r_damped = solve_single_frequency(grid, tags, freq, cfg_damped)

    p_rigid = np.asarray(r_rigid.pressure_on_surface.coefficients)
    p_damped = np.asarray(r_damped.pressure_on_surface.coefficients)

    assert np.all(np.isfinite(p_rigid)), "Rigid solve produced non-finite pressure"
    assert np.all(np.isfinite(p_damped)), "Damped solve produced non-finite pressure"

    # Neither solve should blow up to "1000 dB" territory (i.e. > 1e30 Pa).
    # The rigid sphere problem at f=172 Hz is well-conditioned enough that
    # this is a soft sanity check, but it would catch a true Robin-path
    # numerical break (NaN propagation, sign flip, etc.).
    max_pa = max(np.max(np.abs(p_rigid)), np.max(np.abs(p_damped)))
    assert max_pa < 1e10, f"Pressure blew up: max |p| = {max_pa:.3e} Pa"

    # And the damped solve should give a slightly different (damped)
    # answer — not bit-identical. Guard against the impedance branch
    # silently no-op'ing.
    diff_norm = np.max(np.abs(p_damped - p_rigid))
    rigid_norm = np.max(np.abs(p_rigid))
    assert diff_norm > 1e-6 * rigid_norm, (
        "Damped solve identical to rigid — impedance branch may not be "
        "actually applying the Robin BC"
    )


# ---------------------------------------------------------------------------
# Burton-Miller + impedance must raise (intentional gap, BM=OFF policy)
# ---------------------------------------------------------------------------

def test_bm_plus_impedance_raises():
    """BM + Robin BC is intentionally unsupported and must error early.

    Workspace policy is BM=OFF everywhere, so the combination is not on
    the deprecation gate (see
    docs/plans/canonical-solver-migration.md "Out-of-production solver
    code paths..."). We assert the early failure so a future caller
    flipping BM on doesn't silently get a Robin-less solve.
    """
    _require_bempp_cpu()

    from hornlab_solver.bie import solve_single_frequency
    from hornlab_solver.config import (
        BIEFormulation,
        LinearSolver,
        SolveConfig,
    )

    grid, tags = _build_small_closed_mesh()

    cfg = SolveConfig(
        velocity_sources={2: 1.0},
        impedance_sources={1: 0.05 + 0.0j},
        formulation=BIEFormulation.BURTON_MILLER,
        solver=LinearSolver.LU,
        precision="double",
    )

    with pytest.raises(NotImplementedError, match="Robin.*Burton-Miller"):
        solve_single_frequency(grid, tags, 500.0, cfg)

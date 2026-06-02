"""BIE assembly, boundary conditions, and field evaluation.

This is the physics core. One formulation, one solve, one evaluation path.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._constants import REFERENCE_PRESSURE, SPEED_OF_SOUND
from .backends import resolve_assembly_backend
from .config import BIEFormulation, LinearSolver, SolveConfig, VelocityMode

logger = logging.getLogger(__name__)

VELOCITY_PROFILES = {
    "piston": lambda r_norm: np.ones_like(r_norm),
    "dome": lambda r_norm: np.cos(np.pi * r_norm / 2),
    "ring": lambda r_norm: np.where(r_norm > 0.7, 1.0, 0.0),
}


@dataclass
class FrequencyResult:
    """Result from solving a single frequency."""
    frequency_hz: float
    pressure_on_surface: object       # bempp GridFunction (P1)
    neumann_data: object              # bempp GridFunction (DP0)
    impedance: complex
    iterations: int | None
    timing_s: float


def _choose_solver(config: SolveConfig, n_triangles: int) -> LinearSolver:
    if config.solver is not LinearSolver.AUTO:
        return config.solver
    return LinearSolver.LU if n_triangles <= config.lu_threshold else LinearSolver.GMRES


def _operator_kwargs(
    backend: str,
    precision: str,
    opencl_device: str = "cpu",
    quadrature_order: int | None = None,
) -> dict:
    """Build kwargs for bempp operator construction."""
    kwargs: dict = {}
    if backend != "auto":
        kwargs["assembler"] = "dense"
        kwargs["device_interface"] = backend
    if backend == "opencl":
        from .device import configure_opencl

        configure_opencl(opencl_device)
    if precision == "single":
        kwargs["precision"] = "single"
    if quadrature_order is not None:
        from bempp_cl.api.utils.parameters import DefaultParameters

        params = DefaultParameters()
        params.quadrature.regular = int(quadrature_order)
        kwargs["parameters"] = params
    return kwargs


def _setup_function_spaces(grid):
    """Create P1 and DP0 function spaces on the grid."""
    import bempp_cl.api as bempp_api

    p1 = bempp_api.function_space(grid, "P", 1)
    dp0 = bempp_api.function_space(grid, "DP", 0)
    return p1, dp0


def _build_neumann_data(
    dp0_space,
    physical_tags: NDArray[np.int32],
    omega: float,
    config: SolveConfig,
    precision: str = "single",
) -> object:
    """Construct Neumann boundary condition: dp/dn = i*rho*omega*v_n.

    Supports multiple velocity source tags with independent weights,
    and acceleration vs velocity driving modes.
    """
    import bempp_cl.api as bempp_api

    dtype = np.complex64 if precision == "single" else np.complex128
    coeffs = np.zeros(dp0_space.global_dof_count, dtype=dtype)

    air_density = config.air_density

    for tag, weight in config.velocity_sources.items():
        mask = physical_tags == tag
        if not np.any(mask):
            continue

        v_n = weight
        if config.velocity_mode is VelocityMode.ACCELERATION:
            # ABEC convention: acceleration a=1, so v = a/(j*omega)
            v_n = weight / (1j * omega) if omega > 0 else 0.0

        # Neumann data: dp/dn = i * rho * omega * v_n
        g_val = 1j * air_density * omega * v_n
        dofs = np.where(mask)[0]
        coeffs[dofs] = g_val

    return bempp_api.GridFunction(dp0_space, coefficients=coeffs)


def _build_p1_to_dp0_projection(p1_space, dp0_space):
    """Sparse P1→DP0 coefficient projection: vertex pressures → per-element
    averages. Each output DP0 DOF = mean of the 3 P1 vertex DOFs on that
    triangle.
    """
    import scipy.sparse as sp
    n_dp0 = dp0_space.global_dof_count
    n_p1 = p1_space.global_dof_count
    local2global = np.array(p1_space.local2global)  # (n_elements, 3)
    if local2global.shape[0] != n_dp0:
        raise RuntimeError(
            f"local2global has {local2global.shape[0]} rows but DP0 space "
            f"has {n_dp0} DOFs"
        )
    rows = np.repeat(np.arange(n_dp0, dtype=np.int64), 3)
    cols = local2global.flatten().astype(np.int64)
    data = np.full(n_dp0 * 3, 1.0 / 3.0, dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_dp0, n_p1))


def _build_driver_neumann_coeffs(
    dp0_space,
    physical_tags: NDArray[np.int32],
    omega: float,
    config: SolveConfig,
    dtype: type,
) -> NDArray:
    """Build Neumann coefficients with velocity sources only — zero on
    impedance tags (Robin BCs are folded into the LHS instead).
    """
    coeffs = np.zeros(dp0_space.global_dof_count, dtype=dtype)
    air_density = config.air_density
    impedance_tag_set = set(config.impedance_sources.keys())
    for tag, weight in config.velocity_sources.items():
        if tag in impedance_tag_set:
            # Don't double-count: impedance tags get their own BC
            continue
        mask = physical_tags == tag
        if not np.any(mask):
            continue
        v_n = weight
        if config.velocity_mode is VelocityMode.ACCELERATION:
            v_n = weight / (1j * omega) if omega > 0 else 0.0
        g_val = 1j * air_density * omega * v_n
        coeffs[np.where(mask)[0]] = g_val
    return coeffs


def _assemble_and_solve_impedance(
    grid,
    p1_space,
    dp0_space,
    physical_tags: NDArray[np.int32],
    k: complex,
    omega: float,
    config: SolveConfig,
    op_kwargs_low: dict,
):
    """Direct (non-iterative) solve for the BIE with Robin BCs.

    Substitutes ∂p/∂n = i·k·β·p on the impedance tags directly into the
    BIE, producing a single linear system

        [(K − ½I) − i·k · V · diag(β) · M_proj] p = V · g_drv

    where M_proj is the P1→DP0 vertex-averaging projection and g_drv is the
    Neumann data with driver/aperture sources only (zero on impedance tags).
    This avoids the Picard fixed point, which diverges near interior
    Dirichlet eigenvalues of the enclosed cavities (chamber/port modes).
    The −i·k·V·M_β term also acts as a complex shift that pushes those
    fictitious eigenvalues off the real axis, making the system more
    robust than the rigid solve.
    """
    import bempp_cl.api as bempp_api
    import scipy.linalg
    import scipy.sparse as sp

    if config.formulation is BIEFormulation.BURTON_MILLER:
        raise NotImplementedError(
            "Robin/impedance BC + Burton-Miller formulation is not "
            "supported. Use formulation=STANDARD or COMPLEX_K."
        )

    identity = bempp_api.operators.boundary.sparse.identity(
        p1_space, p1_space, p1_space,
    )
    dlp = bempp_api.operators.boundary.helmholtz.double_layer(
        p1_space, p1_space, p1_space, k, **op_kwargs_low,
    )
    slp = bempp_api.operators.boundary.helmholtz.single_layer(
        dp0_space, p1_space, p1_space, k, **op_kwargs_low,
    )

    A_op = dlp - 0.5 * identity
    A_mat = bempp_api.as_matrix(A_op.weak_form())
    V_mat = bempp_api.as_matrix(slp.weak_form())

    dtype_solve = np.complex128
    A_mat = np.asarray(A_mat, dtype=dtype_solve)
    V_mat = np.asarray(V_mat, dtype=dtype_solve)

    # β as DP0 coefficients (zero on non-impedance tags)
    n_dp0 = dp0_space.global_dof_count
    beta_vec = np.zeros(n_dp0, dtype=dtype_solve)
    for tag, beta in config.impedance_sources.items():
        mask = physical_tags == tag
        beta_vec[np.where(mask)[0]] = complex(beta)
    beta_diag = sp.diags(beta_vec)

    # P1 → DP0 projection (sparse)
    M_proj = _build_p1_to_dp0_projection(p1_space, dp0_space)

    # Robin contribution: −i·k · V · diag(β) · M_proj  (n_p1 × n_p1 dense)
    robin = (1j * k) * (V_mat @ (beta_diag @ M_proj).toarray())
    lhs_full = A_mat - robin

    # RHS = V · g_drv, with g_drv zero on impedance tags
    g_drv = _build_driver_neumann_coeffs(
        dp0_space, physical_tags, omega, config, dtype_solve,
    )
    rhs_vec = V_mat @ g_drv

    p_coeffs = scipy.linalg.solve(lhs_full, rhs_vec)
    p_surface = bempp_api.GridFunction(p1_space, coefficients=p_coeffs)

    # Build a Neumann GridFunction for downstream far-field evaluation:
    # the TRUE Neumann data including the Robin contribution.
    # n_total = g_drv + i·k·β·p_dp0
    p_dp0 = M_proj @ p_coeffs                          # (n_dp0,)
    g_robin = (1j * k) * beta_vec * p_dp0              # (n_dp0,)
    neumann_total = (g_drv + g_robin).astype(np.complex64)
    neumann_fun = bempp_api.GridFunction(
        dp0_space, coefficients=neumann_total,
    )

    return p_surface, neumann_fun, None  # iterations = None (direct solve)


def _assemble_and_solve(
    grid,
    p1_space,
    dp0_space,
    neumann_fun,
    k: complex,
    k_real: float,
    config: SolveConfig,
    n_triangles: int,
    op_kwargs_low: dict,
    op_kwargs_high: dict,
):
    """Assemble BIE operators and solve the linear system."""
    import bempp_cl.api as bempp_api

    identity = bempp_api.operators.boundary.sparse.identity(
        p1_space, p1_space, p1_space,
    )
    dlp = bempp_api.operators.boundary.helmholtz.double_layer(
        p1_space, p1_space, p1_space, k, **op_kwargs_low,
    )
    slp = bempp_api.operators.boundary.helmholtz.single_layer(
        dp0_space, p1_space, p1_space, k, **op_kwargs_low,
    )

    if config.formulation is BIEFormulation.BURTON_MILLER:
        hyp = bempp_api.operators.boundary.helmholtz.hypersingular(
            p1_space, p1_space, p1_space, k, **op_kwargs_high,
        )
        adlp = bempp_api.operators.boundary.helmholtz.adjoint_double_layer(
            dp0_space, p1_space, p1_space, k, **op_kwargs_high,
        )
        rhs_identity = bempp_api.operators.boundary.sparse.identity(
            dp0_space, p1_space, p1_space,
        )
        coupling = 1j / k
        lhs = 0.5 * identity - dlp - coupling * (-hyp)
        rhs = (-slp - coupling * (adlp + 0.5 * rhs_identity)) * neumann_fun
    else:
        # STANDARD or COMPLEX_K (complex k is already baked into the
        # wavenumber — the BIE structure is the same as STANDARD)
        lhs = dlp - 0.5 * identity
        rhs = slp * neumann_fun

    solver_choice = _choose_solver(config, n_triangles)
    iterations = None

    if solver_choice is LinearSolver.LU:
        p_surface = bempp_api.linalg.lu(lhs, rhs)
    else:
        result = bempp_api.linalg.gmres(
            lhs, rhs,
            tol=config.gmres_tol,
            maxiter=config.gmres_max_iter,
            use_strong_form=True,
            return_iteration_count=True,
        )
        p_surface, info, iterations = result
        if info != 0:
            logger.warning(
                "GMRES did not converge (info=%d) at k=%.4f", info, k_real,
            )

    return p_surface, iterations


def _evaluate_far_field(
    p1_space,
    dp0_space,
    p_surface,
    neumann_fun,
    k_real: float,
    obs_points: NDArray[np.float64],
    op_kwargs: dict,
) -> NDArray[np.complex128]:
    """Evaluate pressure at observation points via representation formula.

    p(x) = DLP_pot[p_surface](x) - SLP_pot[neumann_fun](x)
    """
    import bempp_cl.api as bempp_api

    # obs_points: (N, 3) → bempp wants (3, N)
    pts = np.ascontiguousarray(obs_points.T, dtype=np.float64)

    dlp_pot = bempp_api.operators.potential.helmholtz.double_layer(
        p1_space, pts, k_real, **op_kwargs,
    )
    slp_pot = bempp_api.operators.potential.helmholtz.single_layer(
        dp0_space, pts, k_real, **op_kwargs,
    )

    pressure = (dlp_pot * p_surface - slp_pot * neumann_fun).flatten()
    return pressure


def _compute_impedance(
    grid,
    p_surface,
    physical_tags: NDArray[np.int32],
    p1_space,
    velocity_weights: NDArray | None = None,
    source_tag: int = 2,
) -> complex:
    """Throat impedance: Z = integral(p dA) / Q_eff."""
    elements = np.array(grid.elements.T, dtype=np.int32)
    source_mask = physical_tags == source_tag
    source_elems = np.where(source_mask)[0]

    if len(source_elems) == 0:
        return 0.0 + 0.0j

    areas = np.array(grid.volumes)[source_elems]
    coeffs = np.asarray(p_surface.coefficients)

    # P1 DOFs per element: average the 3 vertex pressures
    local2global = np.array(p1_space.local2global)
    source_p1_dofs = local2global[source_elems]  # (N, 3)
    p_at_verts = coeffs[source_p1_dofs]           # (N, 3) complex
    p_avg = np.mean(p_at_verts, axis=1)           # (N,)

    total_force = np.sum(p_avg * areas)

    if velocity_weights is not None and len(velocity_weights) == len(source_elems):
        q_eff = np.sum(velocity_weights * areas)
    else:
        q_eff = np.sum(areas)

    if abs(q_eff) < 1e-30:
        return 0.0 + 0.0j

    return complex(total_force / q_eff)


def compute_surface_pressure_avg(
    grid,
    p_surface,
    physical_tags: NDArray[np.int32],
    p1_space,
    tags: list[int],
) -> dict[int, complex]:
    """Area-weighted average surface pressure on each tag.

    Returns a dict mapping tag -> complex average pressure.
    """
    elements = np.array(grid.elements.T, dtype=np.int32)
    areas = np.array(grid.volumes)
    coeffs = np.asarray(p_surface.coefficients)
    local2global = np.array(p1_space.local2global)

    result = {}
    for tag in tags:
        mask = physical_tags == tag
        elem_indices = np.where(mask)[0]
        if len(elem_indices) == 0:
            result[tag] = 0.0 + 0.0j
            continue

        elem_areas = areas[elem_indices]
        p1_dofs = local2global[elem_indices]  # (N, 3)
        p_at_verts = coeffs[p1_dofs]           # (N, 3) complex
        p_avg_per_elem = np.mean(p_at_verts, axis=1)  # (N,)

        total_area = np.sum(elem_areas)
        if total_area < 1e-30:
            result[tag] = 0.0 + 0.0j
        else:
            result[tag] = complex(np.sum(p_avg_per_elem * elem_areas) / total_area)

    return result


def solve_single_frequency(
    grid,
    physical_tags: NDArray[np.int32],
    frequency_hz: float,
    config: SolveConfig,
    p1_space=None,
    dp0_space=None,
) -> FrequencyResult:
    """Solve the BEM problem at a single frequency.

    When ``config.impedance_sources`` is non-empty, dispatches to
    ``_assemble_and_solve_impedance``, which folds the Robin BC
    ∂p/∂n = i·k·β·p directly into the BIE for a single LU solve.
    """
    t0 = time.time()

    if p1_space is None or dp0_space is None:
        p1_space, dp0_space = _setup_function_spaces(grid)

    omega = 2.0 * np.pi * frequency_hz
    k_real = omega / SPEED_OF_SOUND

    # Wavenumber: real or complex-shifted
    if config.formulation is BIEFormulation.COMPLEX_K:
        k = k_real * (1 + 1j * config.complex_k_shift)
    else:
        k = k_real

    backend = resolve_assembly_backend(config).effective_backend
    op_kwargs_low = _operator_kwargs(
        backend, config.precision, config.opencl_device, config.slp_dlp_quadrature,
    )
    op_kwargs_high = _operator_kwargs(
        backend, config.precision, config.opencl_device, config.hyp_adlp_quadrature,
    )

    n_tris = grid.number_of_elements

    # Impedance / Robin BC: filter to tags actually present in the mesh.
    has_impedance = bool(config.impedance_sources)
    active_impedance: dict[int, complex] = {}
    if has_impedance:
        for tag, beta in config.impedance_sources.items():
            mask = physical_tags == tag
            if not np.any(mask):
                logger.warning(
                    "impedance_sources references tag %d but mesh has no "
                    "elements with that tag — skipping", tag,
                )
                continue
            active_impedance[tag] = beta
    has_impedance = bool(active_impedance)

    if has_impedance:
        impedance_config = SolveConfig(**{
            **{k_: getattr(config, k_) for k_ in config.__dataclass_fields__},
            "impedance_sources": active_impedance,
        })
        p_surface, neumann_fun, iterations = _assemble_and_solve_impedance(
            grid, p1_space, dp0_space, physical_tags,
            k, omega, impedance_config, op_kwargs_low,
        )
    else:
        neumann_fun = _build_neumann_data(
            dp0_space, physical_tags, omega, config, config.precision,
        )
        p_surface, iterations = _assemble_and_solve(
            grid, p1_space, dp0_space, neumann_fun,
            k, k_real, config, n_tris, op_kwargs_low, op_kwargs_high,
        )

    impedance = _compute_impedance(
        grid, p_surface, physical_tags, p1_space,
        source_tag=min(config.velocity_sources.keys(), default=2),
    )

    elapsed = time.time() - t0
    if has_impedance:
        logger.info(
            "%.1f Hz: solved in %.2fs (direct Robin BC)",
            frequency_hz, elapsed,
        )
    else:
        logger.info(
            "%.1f Hz: solved in %.2fs%s",
            frequency_hz, elapsed,
            f" ({iterations} iters)" if iterations is not None else " (LU)",
        )

    return FrequencyResult(
        frequency_hz=frequency_hz,
        pressure_on_surface=p_surface,
        neumann_data=neumann_fun,
        impedance=impedance,
        iterations=iterations,
        timing_s=elapsed,
    )

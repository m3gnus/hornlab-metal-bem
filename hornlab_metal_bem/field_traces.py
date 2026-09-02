"""Post-solve exterior-field evaluation from retained surface traces."""

from __future__ import annotations

import math
import os
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from .config import NATIVE_GROUND_PLANES, NATIVE_SYMMETRY_PLANES
from .mesh import LoadedMesh, make_pure_function_spaces


def _native_field_env_overrides() -> dict[str, str]:
    """Return the sweep-compatible default field-kernel override."""
    if os.environ.get("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE") is not None:
        return {}
    return {"HORNLAB_METAL_BEM_NATIVE_FIELD_MODE": "optimized"}


def _total_neumann_from_surface_pressure(
    driver_neumann: NDArray[Any],
    pressure_p1: NDArray[Any],
    p1_local2global: NDArray[Any],
    physical_tags: NDArray[Any],
    k_real: NDArray[Any],
    k_imag: NDArray[Any],
    impedance_sources: list[dict[int, complex]] | dict[int, complex],
) -> NDArray[np.complex128]:
    """Reproduce the helper's total DP0 Neumann trace in float32 arithmetic."""
    driver = np.asarray(driver_neumann)
    pressure = np.asarray(pressure_p1)
    local2global = np.asarray(p1_local2global, dtype=np.intp)
    tags = np.asarray(physical_tags, dtype=np.int32)
    k_real_values = np.asarray(k_real, dtype=np.float32)
    k_imag_values = np.asarray(k_imag, dtype=np.float32)

    if driver.ndim != 2:
        raise ValueError("driver_neumann must have shape (F, n_dp0_dofs)")
    if pressure.ndim != 2:
        raise ValueError("pressure_p1 must have shape (F, n_p1_dofs)")
    if pressure.shape[0] != driver.shape[0]:
        raise ValueError("driver_neumann and pressure_p1 frequency counts differ")
    if local2global.shape != (driver.shape[1], 3):
        raise ValueError("p1_local2global must have shape (n_dp0_dofs, 3)")
    if tags.shape != (driver.shape[1],):
        raise ValueError("physical_tags must have shape (n_dp0_dofs,)")
    if k_real_values.shape != (driver.shape[0],) or k_imag_values.shape != (
        driver.shape[0],
    ):
        raise ValueError("k_real and k_imag must have shape (F,)")
    if isinstance(impedance_sources, list) and len(impedance_sources) < driver.shape[0]:
        raise ValueError("impedance_sources has fewer entries than frequencies")

    total = np.array(driver, dtype=np.complex64, order="C", copy=True)
    pressure_f32 = np.asarray(pressure, dtype=np.complex64)
    for frequency_index in range(driver.shape[0]):
        case_sources = (
            impedance_sources[frequency_index]
            if isinstance(impedance_sources, list)
            else impedance_sources
        )
        if not case_sources:
            continue
        beta = np.zeros(driver.shape[1], dtype=np.complex64)
        for tag, value in case_sources.items():
            beta[tags == int(tag)] = np.complex64(value)
        robin_faces = beta != np.complex64(0.0)
        if not np.any(robin_faces):
            continue

        robin_dofs = local2global[robin_faces]
        p_values = pressure_f32[frequency_index]
        p_average_real = (
            p_values[robin_dofs[:, 0]].real + p_values[robin_dofs[:, 1]].real
        )
        p_average_real = p_average_real + p_values[robin_dofs[:, 2]].real
        p_average_real = p_average_real / np.float32(3.0)
        p_average_imag = (
            p_values[robin_dofs[:, 0]].imag + p_values[robin_dofs[:, 1]].imag
        )
        p_average_imag = p_average_imag + p_values[robin_dofs[:, 2]].imag
        p_average_imag = p_average_imag / np.float32(3.0)
        i_k = np.complex64(
            complex(
                -float(k_imag_values[frequency_index]),
                float(k_real_values[frequency_index]),
            )
        )
        robin_beta = beta[robin_faces]
        coupling_real = i_k.real * robin_beta.real - i_k.imag * robin_beta.imag
        coupling_imag = i_k.real * robin_beta.imag + i_k.imag * robin_beta.real
        correction_real = (
            coupling_real * p_average_real - coupling_imag * p_average_imag
        )
        correction_imag = (
            coupling_real * p_average_imag + coupling_imag * p_average_real
        )
        total_row = total[frequency_index]
        total_row.real[robin_faces] = (
            total_row.real[robin_faces] + correction_real
        )
        total_row.imag[robin_faces] = (
            total_row.imag[robin_faces] + correction_imag
        )
    return total.astype(np.complex128)


def evaluate_exterior_from_traces(
    mesh: LoadedMesh,
    frequency_hz: float,
    k_real: float,
    pressure_p1: NDArray[Any],
    neumann_dp0: NDArray[Any],
    points_xyz: NDArray[Any],
    *,
    symmetry_plane: str | None = None,
    ground_plane: str | None = None,
    ground_plane_min_clearance_m: float = 0.0,
    check_open_edges: bool = True,
) -> NDArray[np.complex128]:
    r"""Evaluate complex exterior pressure from one frequency's surface traces.

    Parameters
    ----------
    mesh:
        The same :class:`~hornlab_metal_bem.mesh.LoadedMesh` used for the solve.
        Coupled infinite-baffle meshes are not supported because their field
        also needs solved aperture Neumann unknowns.
    frequency_hz, k_real:
        Frequency metadata and the real acoustic wavenumber in radians/metre.
    pressure_p1:
        Complex P1 surface pressure with shape ``(n_p1_dofs,)``.
    neumann_dp0:
        Complex *total* DP0 trace ``dp/dn`` with shape ``(n_dp0_dofs,)``.
    points_xyz:
        Exterior observation coordinates in metres with shape ``(N, 3)``.
    symmetry_plane:
        The solve's native symmetry plane: ``None``, ``"yz"``, ``"xz"``,
        ``"xy"``, or ``"yz+xz"``.
    ground_plane:
        The solve's rigid half-space plane: ``None``, ``"xy"``, ``"yz"``, or
        ``"xz"``. Pass whichever of these two the solve used -- they are
        mutually exclusive, and passing NEITHER for a solve that had one
        re-evaluates the free-field trace, which is a wrong answer that looks
        entirely plausible. ``SolveResult.config`` carries both fields, so the
        safe idiom is
        ``ground_plane=result.config.ground_plane,
        symmetry_plane=result.config.native_symmetry_plane``.
    check_open_edges:
        Apply the same reduced-mesh open-edge validation used by solves.
        Ignored under ``ground_plane``, whose mesh is a complete body with no
        reduced-domain rim to check.

    Returns
    -------
    numpy.ndarray
        Complex pressure with shape ``(N,)`` in the :math:`e^{-i\omega t}`
        phase convention.
    """
    if not isinstance(mesh, LoadedMesh):
        raise TypeError("mesh must be a LoadedMesh")
    if getattr(mesh, "coupled_ib_aperture_tag", None) is not None:
        raise ValueError(
            "coupled infinite-baffle field evaluation requires aperture traces"
        )
    if symmetry_plane is not None and symmetry_plane not in NATIVE_SYMMETRY_PLANES:
        raise ValueError("symmetry_plane must be None, 'yz', 'xz', 'xy', or 'yz+xz'")
    if ground_plane is not None and ground_plane not in NATIVE_GROUND_PLANES:
        raise ValueError("ground_plane must be None, 'xy', 'yz', or 'xz'")
    if symmetry_plane is not None and ground_plane is not None:
        raise ValueError(
            "symmetry_plane and ground_plane are mutually exclusive; pass the "
            "one the solve used"
        )
    if not (math.isfinite(float(frequency_hz)) and float(frequency_hz) > 0.0):
        raise ValueError("frequency_hz must be finite and positive")
    if not (math.isfinite(float(k_real)) and float(k_real) > 0.0):
        raise ValueError("k_real must be finite and positive")

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    if points.shape[0] == 0:
        raise ValueError("points_xyz must be non-empty")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_xyz must contain only finite values")

    from .metal.geometry import _build_metal_geometry_buffers_with_max_edge
    from .metal.native import MetalNativeStandardSession
    from .sweep import _discover_runtime_smoke_cached, _read_complex_f32

    runtime = _discover_runtime_smoke_cached()
    if not runtime.available:
        raise RuntimeError(
            "Swift/Metal native helper is unavailable: "
            + "; ".join(runtime.unavailable_reasons)
        )
    p1_space, dp0_space = make_pure_function_spaces(mesh.grid)
    geometry_buffers, _ = _build_metal_geometry_buffers_with_max_edge(
        mesh.grid,
        mesh.physical_tags,
        p1_space,
        dp0_space,
    )
    operation_id = f"field-traces-{uuid4().hex}"
    with MetalNativeStandardSession.create_session(
        geometry_buffers=geometry_buffers,
        symmetry_plane=symmetry_plane,
        ground_plane=ground_plane,
        ground_plane_min_clearance_m=ground_plane_min_clearance_m,
        check_open_edges=check_open_edges,
        runtime_status=runtime,
        extra_env=_native_field_env_overrides(),
    ) as session:
        field = session.evaluate_standard_exterior(
            float(frequency_hz),
            float(k_real),
            np.asarray(pressure_p1),
            np.asarray(neumann_dp0),
            np.ascontiguousarray(points.T, dtype=np.float32),
            batch_id="retained-surface-traces",
            operation_id=operation_id,
        )
        return _read_complex_f32(
            field.pressure_real_f32,
            field.pressure_imag_f32,
            tuple(field.shape),
            dtype=np.complex128,
        )

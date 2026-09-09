"""Exact discrete rotational reduction for full-3D P1 validation meshes.

This module is deliberately a qualification tool, not a production solve
route.  It constructs a cyclic full surface from a meridian and restricts an
already assembled full P1 system to the discrete, rotation-invariant (m=0)
subspace.  The restriction is exact for a cyclic mesh, an axisymmetric source,
and an assembled operator that commutes with the mesh rotation.

Keeping this code at the validation boundary is intentional: the native dense
assembler currently materializes every full-3D row.  A useful upper-band
reference must eventually assemble only one representative row per vertex
orbit while accumulating contributions from every rotated column orbit.  The
functions below define and verify that contract without silently routing
product solves through an unqualified discretization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CyclicP1Mesh:
    """A full surface of revolution with exact discrete rotation orbits."""

    vertices_nx3: NDArray[np.float64]
    triangles_nx3: NDArray[np.int64]
    physical_tags: NDArray[np.int32]
    vertex_orbits: tuple[NDArray[np.int64], ...]
    triangle_orbits: tuple[NDArray[np.int64], ...]
    triangle_orbit_segments: NDArray[np.int64]
    sectors: int


@dataclass(frozen=True)
class CyclicM0System:
    """Representative-row restriction of a full cyclic P1 system."""

    matrix: NDArray[np.complex128]
    rhs: NDArray[np.complex128]
    representative_rows: NDArray[np.int64]
    vertex_orbits: tuple[NDArray[np.int64], ...]


def revolve_meridian_p1(
    meridian_rz: NDArray[np.floating],
    *,
    sectors: int,
    segment_tags: NDArray[np.integer] | None = None,
    axis_tolerance: float = 1.0e-12,
) -> CyclicP1Mesh:
    """Revolve a counter-clockwise closed ``(rho, z)`` polygon.

    One P1 vertex is emitted per non-axis meridian point and azimuthal sector.
    Points on the rotation axis are shared singleton vertices.  A segment with
    two axis endpoints has zero swept area and is intentionally omitted.

    The meridian must describe the material boundary counter-clockwise in the
    rho-z half-plane.  That convention makes the generated triangle normals
    point outward.  A repeated closing point is accepted and removed.
    ``segment_tags[i]`` labels the surface swept by segment ``i -> i + 1``.
    """
    points = np.asarray(meridian_rz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("meridian_rz must have shape (n_points, 2)")
    if points.shape[0] < 3 or not np.all(np.isfinite(points)):
        raise ValueError("meridian_rz must contain at least three finite points")
    if int(sectors) != sectors or sectors < 3:
        raise ValueError("sectors must be an integer >= 3")
    if not np.isfinite(axis_tolerance) or axis_tolerance < 0.0:
        raise ValueError("axis_tolerance must be finite and non-negative")
    if np.any(points[:, 0] < -axis_tolerance):
        raise ValueError("meridian rho coordinates must be non-negative")

    geometry_scale = max(
        float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0e-12
    )
    closure_tolerance = max(axis_tolerance, 1.0e-8 * geometry_scale)
    repeated_close = np.linalg.norm(points[0] - points[-1]) <= closure_tolerance
    original_segment_count = points.shape[0] - (1 if repeated_close else 0)
    if repeated_close:
        points = points[:-1]
    if points.shape[0] < 3:
        raise ValueError("meridian must retain at least three unique points")
    points = points.copy()
    points[np.abs(points[:, 0]) <= axis_tolerance, 0] = 0.0
    segment_lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    if np.any(segment_lengths <= closure_tolerance):
        raise ValueError("meridian contains a zero-length or near-zero-length segment")

    tags: NDArray[np.int32]
    if segment_tags is None:
        tags = np.ones(points.shape[0], dtype=np.int32)
    else:
        raw_tags = np.asarray(segment_tags)
        if raw_tags.ndim != 1 or raw_tags.shape[0] != original_segment_count:
            raise ValueError("segment_tags must have one value per meridian segment")
        if repeated_close:
            # There are still n-1 explicit segments plus the implicit closing
            # segment after dropping the duplicate endpoint.
            raw_tags = raw_tags.copy()
        if not np.issubdtype(raw_tags.dtype, np.integer):
            raise ValueError("segment_tags must contain integers")
        tags = np.asarray(raw_tags, dtype=np.int32)

    # Signed polygon area in the (rho, z) plane.  Counter-clockwise is the
    # orientation used by the triangle templates below.
    next_points = np.roll(points, -1, axis=0)
    twice_area = float(
        np.sum(points[:, 0] * next_points[:, 1] - next_points[:, 0] * points[:, 1])
    )
    scale = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    if twice_area <= np.finfo(np.float64).eps * scale * scale:
        raise ValueError("meridian must be a non-degenerate counter-clockwise polygon")

    vertices: list[tuple[float, float, float]] = []
    vertex_orbits: list[NDArray[np.int64]] = []
    point_vertices: list[NDArray[np.int64]] = []
    angles = 2.0 * np.pi * np.arange(sectors, dtype=np.float64) / sectors
    for rho, z in points:
        if rho == 0.0:
            indices = np.asarray([len(vertices)], dtype=np.int64)
            vertices.append((0.0, 0.0, float(z)))
        else:
            start = len(vertices)
            indices = np.arange(start, start + sectors, dtype=np.int64)
            vertices.extend(
                (float(rho * np.cos(theta)), float(rho * np.sin(theta)), float(z))
                for theta in angles
            )
        point_vertices.append(indices)
        vertex_orbits.append(indices)

    triangles: list[tuple[int, int, int]] = []
    triangle_tags: list[int] = []
    triangle_orbits: list[NDArray[np.int64]] = []
    triangle_orbit_segments: list[int] = []
    for segment in range(points.shape[0]):
        a = point_vertices[segment]
        b = point_vertices[(segment + 1) % points.shape[0]]
        if len(a) == 1 and len(b) == 1:
            continue
        orbit_starts: list[list[int]] = [[]] if len(a) == 1 or len(b) == 1 else [[], []]
        for sector in range(sectors):
            nxt = (sector + 1) % sectors
            if len(a) == 1:
                tri = (int(a[0]), int(b[nxt]), int(b[sector]))
                orbit_kind = 0
                emitted = (tri,)
            elif len(b) == 1:
                tri = (int(a[sector]), int(a[nxt]), int(b[0]))
                orbit_kind = 0
                emitted = (tri,)
            else:
                emitted = (
                    (int(a[sector]), int(a[nxt]), int(b[nxt])),
                    (int(a[sector]), int(b[nxt]), int(b[sector])),
                )
            for kind, tri in enumerate(emitted):
                orbit_starts[kind if len(orbit_starts) == 2 else orbit_kind].append(
                    len(triangles)
                )
                triangles.append(tri)
                triangle_tags.append(int(tags[segment]))
        for orbit in orbit_starts:
            triangle_orbits.append(np.asarray(orbit, dtype=np.int64))
            triangle_orbit_segments.append(segment)

    if not triangles:
        raise ValueError("meridian produces no non-degenerate surface triangles")
    return CyclicP1Mesh(
        vertices_nx3=np.asarray(vertices, dtype=np.float64),
        triangles_nx3=np.asarray(triangles, dtype=np.int64),
        physical_tags=np.asarray(triangle_tags, dtype=np.int32),
        vertex_orbits=tuple(vertex_orbits),
        triangle_orbits=tuple(triangle_orbits),
        triangle_orbit_segments=np.asarray(triangle_orbit_segments, dtype=np.int64),
        sectors=int(sectors),
    )


def expand_triangle_orbit_values(
    orbit_values: NDArray[np.generic],
    triangle_count: int,
    triangle_orbits: Iterable[NDArray[np.integer]],
) -> NDArray[np.generic]:
    """Expand one axisymmetric source value per triangle orbit."""
    values = np.asarray(orbit_values)
    orbits = _validated_partition(triangle_orbits, int(triangle_count), "triangle")
    if values.ndim != 1 or values.shape[0] != len(orbits):
        raise ValueError("orbit_values must have one value per triangle orbit")
    expanded = np.empty(int(triangle_count), dtype=values.dtype)
    for value, orbit in zip(values, orbits, strict=True):
        expanded[orbit] = value
    return expanded


def reduce_cyclic_m0_representative_rows(
    full_matrix: NDArray[np.complexfloating],
    full_rhs: NDArray[np.complexfloating],
    vertex_orbits: Iterable[NDArray[np.integer]],
    *,
    verify_invariance: bool = True,
    invariance_rtol: float = 2.0e-5,
    invariance_atol: float = 2.0e-6,
) -> CyclicM0System:
    """Restrict ``A p = b`` to the discrete rotation-invariant P1 subspace.

    For each representative row ``i`` and pressure orbit ``O_j`` the reduced
    entry is ``sum(A[i, O_j])``.  This is the system a scalable native reference
    should assemble directly, without ever materializing ``A``.  The optional
    check verifies that every other row in the same orbit gives the same
    reduced row and right-hand side to the requested tolerance.
    """
    matrix = np.asarray(full_matrix)
    rhs = np.asarray(full_rhs)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("full_matrix must be square")
    if rhs.ndim != 1 or rhs.shape[0] != matrix.shape[0]:
        raise ValueError("full_rhs must have one value per matrix row")
    if not np.iscomplexobj(matrix) or not np.iscomplexobj(rhs):
        raise ValueError("full_matrix and full_rhs must be complex")
    orbits = _validated_partition(vertex_orbits, matrix.shape[0], "vertex")
    representatives = np.asarray([int(orbit[0]) for orbit in orbits], dtype=np.int64)

    reduced = _representative_rows(matrix, representatives, orbits)
    reduced_rhs = np.asarray(rhs[representatives], dtype=np.complex128)
    if verify_invariance:
        for orbit_index, orbit in enumerate(orbits):
            candidate_rows = _representative_rows(matrix, orbit, orbits)
            expected_rows = np.broadcast_to(reduced[orbit_index], candidate_rows.shape)
            if not np.allclose(
                candidate_rows,
                expected_rows,
                rtol=invariance_rtol,
                atol=invariance_atol,
            ):
                error = float(np.max(np.abs(candidate_rows - expected_rows)))
                raise ValueError(
                    "full_matrix is not rotation-invariant on vertex orbit "
                    f"{orbit_index}; max reduced-row difference={error:.6g}"
                )
            if not np.allclose(
                rhs[orbit],
                reduced_rhs[orbit_index],
                rtol=invariance_rtol,
                atol=invariance_atol,
            ):
                error = float(np.max(np.abs(rhs[orbit] - reduced_rhs[orbit_index])))
                raise ValueError(
                    "full_rhs is not rotation-invariant on vertex orbit "
                    f"{orbit_index}; max difference={error:.6g}"
                )

    return CyclicM0System(
        matrix=reduced,
        rhs=reduced_rhs,
        representative_rows=representatives,
        vertex_orbits=orbits,
    )


def load_and_reduce_native_cyclic_m0(
    assembly: Any,
    vertex_orbits: Iterable[NDArray[np.integer]],
    *,
    verify_invariance: bool = True,
    invariance_rtol: float = 2.0e-5,
    invariance_atol: float = 2.0e-6,
) -> CyclicM0System:
    """Load one native dense assembly and apply the qualification reduction.

    ``assembly`` is intentionally accepted by its small structural contract so
    this validation module does not make the native backend a required import.
    It must expose ``matrix_shape`` plus the four matrix/RHS binary paths from
    :class:`~hornlab_metal_bem.metal.session.DenseAssemblyResult`.

    This still pays the full dense assembly and storage cost.  It exists to
    prove numerical equivalence and to provide a stable oracle for a future
    orbit-native assembler, not as an upper-band performance implementation.
    """
    shape = tuple(int(value) for value in assembly.matrix_shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("native assembly matrix_shape must be square")
    matrix_real = np.fromfile(assembly.matrix_real_f32, dtype="<f4")
    matrix_imag = np.fromfile(assembly.matrix_imag_f32, dtype="<f4")
    rhs_real = np.fromfile(assembly.rhs_real_f32, dtype="<f4")
    rhs_imag = np.fromfile(assembly.rhs_imag_f32, dtype="<f4")
    expected_matrix = shape[0] * shape[1]
    if matrix_real.size != expected_matrix or matrix_imag.size != expected_matrix:
        raise ValueError("native assembly matrix files do not match matrix_shape")
    if rhs_real.size != shape[0] or rhs_imag.size != shape[0]:
        raise ValueError("native assembly RHS files do not match matrix_shape")
    matrix = matrix_real.reshape(shape) + 1j * matrix_imag.reshape(shape)
    rhs = rhs_real + 1j * rhs_imag
    return reduce_cyclic_m0_representative_rows(
        matrix,
        rhs,
        vertex_orbits,
        verify_invariance=verify_invariance,
        invariance_rtol=invariance_rtol,
        invariance_atol=invariance_atol,
    )


def expand_cyclic_m0_pressure(
    reduced_pressure: NDArray[np.complexfloating],
    full_dof_count: int,
    vertex_orbits: Iterable[NDArray[np.integer]],
) -> NDArray[np.complex128]:
    """Reconstruct the full cyclic P1 pressure for native field evaluation."""
    pressure = np.asarray(reduced_pressure)
    if pressure.ndim != 1 or not np.iscomplexobj(pressure):
        raise ValueError("reduced_pressure must be a complex vector")
    orbits = _validated_partition(vertex_orbits, int(full_dof_count), "vertex")
    if pressure.shape[0] != len(orbits):
        raise ValueError("reduced_pressure must have one value per vertex orbit")
    expanded = np.empty(int(full_dof_count), dtype=np.complex128)
    for value, orbit in zip(pressure, orbits, strict=True):
        expanded[orbit] = value
    return expanded


def _representative_rows(
    matrix: NDArray[np.complexfloating],
    rows: NDArray[np.integer],
    col_orbits: tuple[NDArray[np.int64], ...],
) -> NDArray[np.complex128]:
    selected = np.asarray(matrix[np.asarray(rows, dtype=np.int64)], dtype=np.complex128)
    return np.column_stack([np.sum(selected[:, orbit], axis=1) for orbit in col_orbits])


def _validated_partition(
    raw_orbits: Iterable[NDArray[np.integer]],
    item_count: int,
    name: str,
) -> tuple[NDArray[np.int64], ...]:
    if item_count <= 0:
        raise ValueError(f"{name} item count must be positive")
    orbits = tuple(np.asarray(orbit, dtype=np.int64).reshape(-1) for orbit in raw_orbits)
    if not orbits or any(orbit.size == 0 for orbit in orbits):
        raise ValueError(f"{name}_orbits must contain non-empty orbits")
    members = np.concatenate(orbits)
    if np.any(members < 0) or np.any(members >= item_count):
        raise ValueError(f"{name}_orbits contain out-of-range indices")
    counts = np.bincount(members, minlength=item_count)
    if counts.shape[0] != item_count or np.any(counts != 1):
        raise ValueError(f"{name}_orbits must partition every item exactly once")
    return orbits

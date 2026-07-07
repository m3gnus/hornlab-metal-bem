from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_metal_bem.metal import (
    MetalBemBackend,
    MetalBemContext,
    MetalNativeRuntimeConfig,
    MetalNativeStandardSession,
    discover_native_runtime,
    validate_session_with_native_helper,
)
from hornlab_metal_bem.metal.backend import DenseBieSystem
from hornlab_metal_bem.metal import backend as metal_backend
from hornlab_metal_bem.metal import native
from hornlab_metal_bem.metal.geometry import build_metal_geometry_buffers
from hornlab_metal_bem.metal.geometry import MetalGeometryError
from hornlab_metal_bem.metal.geometry import validate_native_symmetry_plane
from hornlab_metal_bem.validation.native_symmetry import orbit_reduce_matrix_rhs


def _write_native_entrypoint(root: Path) -> Path:
    helper = root / "HornlabMetalBemNative.swift"
    helper.write_text("// test helper\n", encoding="utf-8")
    return helper


def _write_native_package_binary(root: Path) -> Path:
    package_dir = root / "native_helper"
    (package_dir / ".build" / "release").mkdir(parents=True)
    (package_dir / "Package.swift").write_text("// test package\n", encoding="utf-8")
    binary = package_dir / ".build" / "release" / "HornlabMetalBemNative"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    return binary


def _arg_after(command: list[str], op: str, offset: int) -> str:
    return command[command.index(op) + offset]


def _tiny_geometry_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([1, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _near_quadrature_geometry_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.05, 0.05, 0.05],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 3], [1, 4], [2, 5]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        global_dof_count=6,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([1, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _ib_box_geometry_buffers():
    vertices = np.array(
        [
            [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    triangles_nx3 = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 5, 4],
            [0, 1, 5],
            [1, 6, 5],
            [1, 2, 6],
            [2, 7, 6],
            [2, 3, 7],
            [3, 4, 7],
            [3, 0, 4],
        ],
        dtype=np.int64,
    )
    grid = SimpleNamespace(
        vertices=vertices,
        elements=triangles_nx3.T,
        number_of_elements=triangles_nx3.shape[0],
    )
    p1 = SimpleNamespace(
        local2global=triangles_nx3.astype(np.int64),
        global_dof_count=vertices.shape[1],
    )
    tags = np.array([7, 7, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32)
    return build_metal_geometry_buffers(
        grid,
        tags,
        p1,
        SimpleNamespace(global_dof_count=triangles_nx3.shape[0]),
    )


def _ib_quarter_box_mesh(scale: float = 0.1, depth: float = 0.05):
    vertices = np.array(
        [
            [0.0, scale, scale, 0.0, 0.0, scale, scale, 0.0],
            [0.0, 0.0, scale, scale, 0.0, 0.0, scale, scale],
            [0.0, 0.0, 0.0, 0.0, -depth, -depth, -depth, -depth],
        ],
        dtype=np.float64,
    )
    triangles_nx3 = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [1, 5, 6],
            [1, 6, 2],
            [3, 2, 6],
            [3, 6, 7],
        ],
        dtype=np.int64,
    )
    tags = np.array([7, 7, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    return vertices, triangles_nx3, tags


def _mirror_xy_mesh(vertices_3xn, triangles_nx3, tags):
    vertices_out: list[tuple[float, float, float]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}
    triangles_out: list[list[int]] = []
    tags_out: list[int] = []

    def mirror(point, mask: int) -> tuple[float, float, float]:
        x, y, z = (float(point[0]), float(point[1]), float(point[2]))
        if mask & 1:
            x = -x
        if mask & 2:
            y = -y
        return (x, y, z)

    def vertex_id(point: tuple[float, float, float]) -> int:
        key = tuple(round(value, 10) for value in point)
        if key not in vertex_index:
            vertex_index[key] = len(vertices_out)
            vertices_out.append(point)
        return vertex_index[key]

    for mask in (0, 1, 2, 3):
        reverse_winding = bin(mask).count("1") % 2 == 1
        for triangle, tag in zip(triangles_nx3, tags, strict=True):
            ids = [vertex_id(mirror(vertices_3xn[:, idx], mask)) for idx in triangle]
            if reverse_winding:
                ids = [ids[0], ids[2], ids[1]]
            triangles_out.append(ids)
            tags_out.append(int(tag))

    return (
        np.asarray(vertices_out, dtype=np.float64).T,
        np.asarray(triangles_out, dtype=np.int64),
        np.asarray(tags_out, dtype=np.int32),
    )


def _geometry_buffers_from_mesh(vertices_3xn, triangles_nx3, tags):
    grid = SimpleNamespace(
        vertices=vertices_3xn,
        elements=triangles_nx3.T,
        number_of_elements=triangles_nx3.shape[0],
    )
    p1 = SimpleNamespace(
        local2global=triangles_nx3.astype(np.int64),
        global_dof_count=vertices_3xn.shape[1],
    )
    return build_metal_geometry_buffers(
        grid,
        tags,
        p1,
        SimpleNamespace(global_dof_count=triangles_nx3.shape[0]),
    )


def _ib_quarter_box_geometry_buffers():
    return _geometry_buffers_from_mesh(*_ib_quarter_box_mesh())


def _ib_quarter_box_mirrored_full_geometry_buffers():
    return _geometry_buffers_from_mesh(*_mirror_xy_mesh(*_ib_quarter_box_mesh()))


_TRIANGLE_QX = np.array(
    [
        0.4459484909159651,
        0.0915762135097710,
        0.1081030181680700,
        0.4459484909159651,
        0.8168475729804590,
        0.0915762135097710,
    ],
    dtype=np.float64,
)
_TRIANGLE_QY = np.array(
    [
        0.4459484909159651,
        0.0915762135097700,
        0.4459484909159651,
        0.1081030181680700,
        0.0915762135097700,
        0.8168475729804580,
    ],
    dtype=np.float64,
)
_TRIANGLE_QW = 0.5 * np.array(
    [
        0.2233815896780110,
        0.1099517436553220,
        0.2233815896780110,
        0.2233815896780110,
        0.1099517436553220,
        0.1099517436553220,
    ],
    dtype=np.float64,
)


def _reference_subtriangles(level: int) -> np.ndarray:
    triangles = np.array(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=np.float64,
    )
    for _ in range(level):
        a = triangles[:, 0, :]
        b = triangles[:, 1, :]
        c = triangles[:, 2, :]
        ab = 0.5 * (a + b)
        bc = 0.5 * (b + c)
        ca = 0.5 * (c + a)
        triangles = np.concatenate(
            [
                np.stack([a, ab, ca], axis=1),
                np.stack([ab, b, bc], axis=1),
                np.stack([ca, bc, c], axis=1),
                np.stack([ab, bc, ca], axis=1),
            ],
            axis=0,
        )
    return triangles


def _subtriangle_quadrature(level: int) -> tuple[np.ndarray, np.ndarray]:
    subtriangles = _reference_subtriangles(level)
    q = np.stack([_TRIANGLE_QX, _TRIANGLE_QY], axis=1)
    a = subtriangles[:, None, 0, :]
    b = subtriangles[:, None, 1, :]
    c = subtriangles[:, None, 2, :]
    points = a + q[None, :, 0, None] * (b - a) + q[None, :, 1, None] * (c - a)
    det = np.abs(
        (subtriangles[:, 1, 0] - subtriangles[:, 0, 0])
        * (subtriangles[:, 2, 1] - subtriangles[:, 0, 1])
        - (subtriangles[:, 1, 1] - subtriangles[:, 0, 1])
        * (subtriangles[:, 2, 0] - subtriangles[:, 0, 0])
    )
    weights = det[:, None] * _TRIANGLE_QW[None, :]
    return points.reshape(-1, 2), weights.reshape(-1)


def _parent_basis(ref_points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [1.0 - ref_points[:, 0] - ref_points[:, 1], ref_points[:, 0], ref_points[:, 1]]
    )


def _reference_pair_blocks(buffers, test: int, trial: int, k: float, level: int):
    triangles = buffers.triangles_3xm_i32.T
    vertices = np.asarray(buffers.vertices_3xn_f32, dtype=np.float64)
    test_vertices = vertices[:, triangles[test]].T
    trial_vertices = vertices[:, triangles[trial]].T
    test_ref, test_weights = _subtriangle_quadrature(level)
    trial_ref, trial_weights = _subtriangle_quadrature(level)
    test_basis = _parent_basis(test_ref)
    trial_basis = _parent_basis(trial_ref)
    test_points = test_basis @ test_vertices
    trial_points = trial_basis @ trial_vertices
    dx = trial_points[None, :, 0] - test_points[:, None, 0]
    dy = trial_points[None, :, 1] - test_points[:, None, 1]
    dz = trial_points[None, :, 2] - test_points[:, None, 2]
    r = np.sqrt(dx * dx + dy * dy + dz * dz)
    g = np.exp(1j * k * r) / (4.0 * np.pi * r)
    normal = np.asarray(buffers.triangle_normals_3xm_f32[:, trial], dtype=np.float64)
    projection = (dx * normal[0] + dy * normal[1] + dz * normal[2]) / r
    dlp_kernel = g * (-1.0 / r + 1j * k) * projection
    jac = (
        2.0
        * float(buffers.triangle_areas_f32[test])
        * 2.0
        * float(buffers.triangle_areas_f32[trial])
    )
    pair_weights = test_weights[:, None] * trial_weights[None, :] * jac
    weighted_g = g * pair_weights
    weighted_dlp = dlp_kernel * pair_weights
    slp = np.array(
        [np.sum(weighted_g * test_basis[:, i, None]) for i in range(3)],
        dtype=np.complex128,
    )
    dlp = np.empty((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            dlp[i, j] = np.sum(
                weighted_dlp * test_basis[:, i, None] * trial_basis[None, :, j]
            )
    return SimpleNamespace(slp=slp, dlp=dlp)


def _tiny_robin_geometry_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.3, 0.0, 0.0, -0.2],
                [0.0, 0.0, 0.3, -0.2, 0.0],
                [0.0, 0.0, 0.1, 0.2, 0.25],
            ],
            dtype=np.float64,
        ),
        elements=np.array(
            [
                [0, 0, 0],
                [1, 2, 3],
                [2, 3, 4],
            ],
            dtype=np.int64,
        ),
        number_of_elements=3,
    )
    p1 = SimpleNamespace(
        local2global=np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 4]],
            dtype=np.int64,
        ),
        global_dof_count=5,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 8, 9], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=3),
    )


def _tiny_yz_half_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.3],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0, 1], [1, 3, 2], [3, 2, 3]], dtype=np.int64),
        number_of_elements=3,
    )
    p1 = SimpleNamespace(
        local2global=np.array(
            [[0, 1, 3], [0, 3, 2], [1, 2, 3]],
            dtype=np.int64,
        ),
        global_dof_count=4,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=3),
    )


def _tiny_yz_parity_half_buffers():
    # Single-triangle YZ half mesh whose mirror image across X=0 reconstructs
    # the real side of _tiny_yz_full_buffers exactly. The even-mode parity
    # SOLVE tests depend on this exact 1:1 mirror reduction (one DP0 element,
    # three P1 dofs), so it is kept separate from the multi-triangle
    # _tiny_yz_half_buffers fixture, which exists to exercise the open-rim
    # symmetry-cut validation guard.
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0], [1], [2]], dtype=np.int64),
        number_of_elements=1,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2]], dtype=np.int64),
        global_dof_count=3,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=1),
    )


def _tiny_xz_half_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.3],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0, 1], [1, 3, 2], [3, 2, 3]], dtype=np.int64),
        number_of_elements=3,
    )
    p1 = SimpleNamespace(
        local2global=np.array(
            [[0, 1, 3], [0, 3, 2], [1, 2, 3]],
            dtype=np.int64,
        ),
        global_dof_count=4,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=3),
    )


def _tiny_yz_full_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 3], [1, 5], [2, 4]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [3, 5, 4]], dtype=np.int64),
        global_dof_count=6,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _tiny_yz_xz_quarter_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 1], [1, 3], [2, 2]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
        global_dof_count=4,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _tiny_yz_xz_parity_quarter_buffers():
    # Single-triangle YZ+XZ quarter mesh whose three mirror images reconstruct
    # _tiny_yz_xz_full_buffers exactly. The even-mode parity SOLVE tests depend
    # on this exact 1:1 quadrant reduction (one DP0 element, three P1 dofs), so
    # it is kept separate from the multi-triangle _tiny_yz_xz_quarter_buffers
    # fixture used by the symmetry-cut validation guard.
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0], [1], [2]], dtype=np.int64),
        number_of_elements=1,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2]], dtype=np.int64),
        global_dof_count=3,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=1),
    )


def _tiny_yz_xz_full_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array(
            [
                [0, 0, 0, 0],
                [1, 2, 4, 3],
                [2, 3, 1, 4],
            ],
            dtype=np.int64,
        ),
        number_of_elements=4,
    )
    p1 = SimpleNamespace(
        local2global=np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 4, 1],
                [0, 3, 4],
            ],
            dtype=np.int64,
        ),
        global_dof_count=5,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2, 2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=4),
    )


def _tiny_xy_half_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0, 1], [1, 3, 2], [3, 2, 3]], dtype=np.int64),
        number_of_elements=3,
    )
    p1 = SimpleNamespace(
        local2global=np.array(
            [[0, 1, 3], [0, 3, 2], [1, 2, 3]],
            dtype=np.int64,
        ),
        global_dof_count=4,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=3),
    )


def _tiny_xy_parity_half_buffers():
    # Single-triangle XY half mesh whose mirror image across Z=0 reconstructs
    # the real side of _tiny_xy_mirror_full_buffers and _tiny_xy_shared_full_buffers.
    # It also doubles as the snapped-geometry reference for the near-plane-vertex
    # test (the z=5e-7 vertex must snap onto this clean z=0 mesh). Kept separate
    # from the multi-triangle _tiny_xy_half_buffers fixture used by the
    # symmetry-cut validation guard.
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0], [1], [2]], dtype=np.int64),
        number_of_elements=1,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2]], dtype=np.int64),
        global_dof_count=3,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=1),
    )


def _tiny_xy_mirror_full_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 3], [1, 5], [2, 4]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [3, 5, 4]], dtype=np.int64),
        global_dof_count=6,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _tiny_xy_shared_full_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, -1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 3], [2, 1]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int64),
        global_dof_count=4,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _tiny_xy_full_buffers():
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, -1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )
    return build_metal_geometry_buffers(
        grid,
        np.array([2, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )


def _read_complex_assembly(assembly):
    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )
    return matrix, rhs


def test_native_batch_result_count_validation():
    cases = [{"case_id": "case-0000"}]
    assert (
        native._case_results_from_manifest(
            {"cases": cases},
            expected_count=1,
            op="test_batch",
        )
        is cases
    )

    with pytest.raises(RuntimeError, match="result missing cases"):
        native._case_results_from_manifest(
            {},
            expected_count=1,
            op="test_batch",
        )

    with pytest.raises(RuntimeError, match="returned 0 case"):
        native._case_results_from_manifest(
            {"cases": []},
            expected_count=1,
            op="test_batch",
        )


def test_native_diagnostics_helpers_preserve_manifest_metadata():
    batch = native._native_batch_diagnostics(
        {
            "implementation": "batch_impl",
            "session_id": "session",
            "batch_id": "batch",
            "wall_seconds": 1.25,
            "resident_context_library_seconds": 0.05,
            "resident_context_metal_library_source": "metallib",
            "resident_reuse": {"geometry_buffers": True},
            "cases": [],
            "ignored_output_path": "outputs/field.bin",
        }
    )
    case = native._native_case_diagnostics(
        {
            "case_id": "case-0000",
            "assembly_implementation": "assembly_impl",
            "solve_implementation": "solve_impl",
            "field_implementation": "field_impl",
            "lapack_info": 0,
            "duffy_corrections": {"implemented": True},
            "metal_dispatch": {"matrix": {"threads_per_threadgroup": 64}},
            "pressure_real_f32": "outputs/pressure_re.bin",
        },
        batch_diagnostics=batch,
    )

    assert case["assembly_implementation"] == "assembly_impl"
    assert case["duffy_corrections"]["implemented"] is True
    assert case["batch"]["resident_context_library_seconds"] == 0.05
    assert case["batch"]["resident_context_metal_library_source"] == "metallib"
    assert case["batch"]["resident_reuse"]["geometry_buffers"] is True
    assert "pressure_real_f32" not in case
    assert "ignored_output_path" not in case["batch"]


def _minimal_dense_solve_field_case() -> dict[str, object]:
    return {
        "session_id": "session",
        "batch_id": "batch",
        "frequency_hz": 100.0,
        "pressure_shape": [1],
        "observation_pressure_real_f32": "outputs/field_re.bin",
        "observation_pressure_imag_f32": "outputs/field_im.bin",
        "field_shape": [1],
        "assembly_seconds": 0.01,
        "dense_solve_seconds": 0.02,
        "field_seconds": 0.03,
        "lapack_info": 0,
    }


def test_dense_solve_field_result_requires_complex_k_ack(tmp_path):
    fake_self = SimpleNamespace(info=SimpleNamespace(work_dir=tmp_path))

    with pytest.raises(RuntimeError, match="helper.*complex-k"):
        MetalNativeStandardSession._dense_solve_field_result(
            fake_self,
            _minimal_dense_solve_field_case(),
            {},
            expect_complex_k=True,
        )


def test_dense_solve_field_result_requires_robin_ack(tmp_path):
    fake_self = SimpleNamespace(info=SimpleNamespace(work_dir=tmp_path))

    with pytest.raises(RuntimeError, match="helper.*Robin"):
        MetalNativeStandardSession._dense_solve_field_result(
            fake_self,
            _minimal_dense_solve_field_case(),
            {},
            expect_robin=True,
        )


def test_native_discovery_reports_missing_helper_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("HORNLAB_METAL_BEM_SWIFT", "/usr/bin/swift")

    status = discover_native_runtime(MetalNativeRuntimeConfig(backend_dir=tmp_path))

    assert status.available is False
    assert status.is_macos is True
    assert status.is_apple_silicon is True
    assert status.swift_path == "/usr/bin/swift"
    assert status.swift_source == "HORNLAB_METAL_BEM_SWIFT"
    assert status.native_entrypoint == tmp_path / "HornlabMetalBemNative.swift"
    assert status.helper_assets_present is False
    assert status.smoke_test_ran is False
    assert any("Swift/Metal helper" in r for r in status.reasons)


def test_native_discovery_finds_swift_on_path(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("HORNLAB_METAL_BEM_SWIFT", raising=False)
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    status = discover_native_runtime(MetalNativeRuntimeConfig(backend_dir=tmp_path))

    assert status.available is True
    assert status.swift_path == "/usr/bin/swift"
    assert status.swift_source == "PATH"
    assert status.helper_assets_present is True
    assert status.smoke_test_ran is False


def test_native_discovery_prefers_compiled_package_helper_without_swift(
    monkeypatch,
    tmp_path,
):
    binary = _write_native_package_binary(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("HORNLAB_METAL_BEM_SWIFT", raising=False)
    monkeypatch.setattr(native.shutil, "which", lambda name: None)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return native.subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    status = discover_native_runtime(
        MetalNativeRuntimeConfig(backend_dir=tmp_path),
        run_smoke_test=True,
    )

    assert status.available is True
    assert status.swift_path is None
    assert status.helper_executable_path == binary
    assert status.helper_source == "swift-package"
    assert status.smoke_test_ran is True
    assert status.smoke_test_ok is True
    assert calls[0] == [str(binary), "--smoke"]


def test_native_discovery_can_run_smoke_test(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return native.subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    status = discover_native_runtime(
        MetalNativeRuntimeConfig(backend_dir=tmp_path),
        run_smoke_test=True,
    )

    assert status.available is True
    assert status.smoke_test_ran is True
    assert status.smoke_test_ok is True
    assert calls[0][-1] == "--smoke"


def test_native_discovery_reports_failed_smoke_test(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    def fake_run(command, **kwargs):
        return native.subprocess.CompletedProcess(
            command,
            1,
            "",
            "Metal device unavailable\n",
        )

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    status = discover_native_runtime(
        MetalNativeRuntimeConfig(backend_dir=tmp_path),
        run_smoke_test=True,
    )

    assert status.available is False
    assert status.smoke_test_ran is True
    assert status.smoke_test_ok is False
    assert status.smoke_test_error == "Metal device unavailable"
    assert any("smoke test failed" in reason for reason in status.reasons)


def test_validate_session_with_native_helper_invokes_swift(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    session_manifest = tmp_path / "session.json"
    session_manifest.write_text("{}", encoding="utf-8")
    result_path = tmp_path / "result.json"

    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        result_path.write_text(
            json.dumps(
                {
                    "schema": "hornlab.metal.standard.v1",
                    "op": "validate_session_result",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    result = validate_session_with_native_helper(
        session_manifest,
        result_path,
        MetalNativeRuntimeConfig(backend_dir=tmp_path),
    )

    assert result["status"] == "ok"
    assert calls[0][-1] == "--smoke"
    assert calls[1][2] == "validate_session"
    assert calls[1][3] == str(session_manifest)
    assert calls[1][4] == str(result_path)


def test_validate_session_with_compiled_helper_does_not_require_swift(
    monkeypatch,
    tmp_path,
):
    binary = _write_native_package_binary(tmp_path)
    session_manifest = tmp_path / "session.json"
    session_manifest.write_text("{}", encoding="utf-8")
    result_path = tmp_path / "result.json"

    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("HORNLAB_METAL_BEM_SWIFT", raising=False)
    monkeypatch.setattr(native.shutil, "which", lambda name: None)

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        result_path.write_text(
            json.dumps(
                {
                    "schema": "hornlab.metal.standard.v1",
                    "op": "validate_session_result",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    result = validate_session_with_native_helper(
        session_manifest,
        result_path,
        MetalNativeRuntimeConfig(backend_dir=tmp_path),
    )

    assert result["status"] == "ok"
    assert calls[0] == [str(binary), "--smoke"]
    assert calls[1][0] == str(binary)
    assert calls[1][1] == "validate_session"
    assert calls[1][2] == str(session_manifest)
    assert calls[1][3] == str(result_path)


def test_native_standard_session_writes_manifest_and_validates(
    monkeypatch,
    tmp_path,
):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    def fake_run(command, **kwargs):
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        result_path = Path(_arg_after(command, "validate_session", 2))
        result_path.write_text(
            json.dumps(
                {
                    "schema": "hornlab.metal.standard.v1",
                    "op": "validate_session_result",
                    "session_id": "native-test",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )
    dp0 = SimpleNamespace(global_dof_count=2)
    buffers = build_metal_geometry_buffers(
        grid,
        np.array([1, 2], dtype=np.int32),
        p1,
        dp0,
    )

    session = MetalNativeStandardSession.create_session(
        geometry_buffers=buffers,
        work_dir=tmp_path / "native-session",
        session_id="native-test",
    )
    try:
        result = session.validate_contract()

        assert result["status"] == "ok"
        assert session.info.manifest_path.is_file()
        assert (session.info.work_dir / "native-result.json").is_file()
    finally:
        session.close()


def test_native_symmetry_manifest_and_half_domain_guard(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    def fake_run(command, **kwargs):
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        result_path = Path(_arg_after(command, "validate_session", 2))
        result_path.write_text(
            json.dumps(
                {
                    "schema": "hornlab.metal.standard.v1",
                    "op": "validate_session_result",
                    "session_id": "native-symmetry-test",
                    "status": "ok",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    session = MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_half_buffers(),
        work_dir=tmp_path / "native-symmetry-session",
        session_id="native-symmetry-test",
        symmetry_plane="yz",
    )
    try:
        manifest = json.loads(session.info.manifest_path.read_text(encoding="utf-8"))
        assert manifest["assembly_scope"]["symmetry_plane"] == "yz"
    finally:
        session.close()

    quarter_session = MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_quarter_buffers(),
        work_dir=tmp_path / "native-quarter-symmetry-session",
        session_id="native-quarter-symmetry-test",
        symmetry_plane="yz+xz",
    )
    try:
        manifest = json.loads(
            quarter_session.info.manifest_path.read_text(encoding="utf-8")
        )
        assert manifest["assembly_scope"]["symmetry_plane"] == "yz+xz"
    finally:
        quarter_session.close()

    xy_session = MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_half_buffers(),
        work_dir=tmp_path / "native-xy-symmetry-session",
        session_id="native-xy-symmetry-test",
        symmetry_plane="xy",
    )
    try:
        manifest = json.loads(xy_session.info.manifest_path.read_text(encoding="utf-8"))
        assert manifest["assembly_scope"]["symmetry_plane"] == "xy"
    finally:
        xy_session.close()

    assert validate_native_symmetry_plane(_tiny_xz_half_buffers(), "xz") == "xz"
    assert validate_native_symmetry_plane(_tiny_xy_half_buffers(), "xy") == "xy"
    assert (
        validate_native_symmetry_plane(_tiny_yz_xz_quarter_buffers(), "yz+xz")
        == "yz+xz"
    )

    with pytest.raises(MetalGeometryError, match="every open boundary edge"):
        validate_native_symmetry_plane(_tiny_yz_xz_quarter_buffers(), "yz")

    assert (
        validate_native_symmetry_plane(
            _tiny_yz_xz_quarter_buffers(),
            "yz",
            check_open_edges=False,
        )
        == "yz"
    )

    with pytest.raises(MetalGeometryError, match="positive-x reduced-domain"):
        validate_native_symmetry_plane(_tiny_yz_full_buffers(), "yz")

    with pytest.raises(MetalGeometryError, match="positive-z reduced-domain"):
        validate_native_symmetry_plane(_tiny_xy_full_buffers(), "xy")

    with pytest.raises(
        MetalGeometryError,
        match="supports 'yz', 'xz', 'xy', and 'yz\\+xz'",
    ):
        validate_native_symmetry_plane(_tiny_yz_half_buffers(), "zx")


def test_open_mouth_reduced_mesh_needs_check_open_edges_disabled():
    # A bare horn radiating from an open mouth is a mirror-reduced OPEN shell:
    # its mouth rim is a real free edge that does not lie on any symmetry plane,
    # so the default strict guard rejects it even when every cut plane is
    # requested. The single-triangle quarter fixture reproduces this (its
    # hypotenuse is off both X=0 and Y=0). Production opts these out via
    # SolveConfig.native_check_open_edges=False; the strict default still
    # protects closed meshes cut along an unrequested plane.
    buffers = _tiny_yz_xz_parity_quarter_buffers()
    with pytest.raises(MetalGeometryError, match="every open boundary edge"):
        validate_native_symmetry_plane(buffers, "yz+xz")
    assert (
        validate_native_symmetry_plane(buffers, "yz+xz", check_open_edges=False)
        == "yz+xz"
    )


def test_native_standard_session_invokes_swift_assembly(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    def fake_run(command, **kwargs):
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "assemble_standard_neumann" in command:
            payload = json.loads(
                Path(_arg_after(command, "assemble_standard_neumann", 2)).read_text(
                    encoding="utf-8"
                )
            )
            root = Path(_arg_after(command, "assemble_standard_neumann", 1)).parent
            for descriptor in payload["outputs"].values():
                if isinstance(descriptor, dict):
                    path = root / descriptor["path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.zeros(descriptor["shape"], dtype=np.float32).tofile(path)
            Path(_arg_after(command, "assemble_standard_neumann", 3)).write_text(
                json.dumps(
                    {
                        "schema": "hornlab.metal.standard.v1",
                        "op": "assemble_standard_neumann_result",
                        "session_id": "native-test",
                        "frequency_hz": 100.0,
                        "matrix_layout": "row_major_c",
                        "matrix_shape": [4, 4],
                        "rhs_shape": [4],
                        "matrix_real_f32": payload["outputs"]["A_real_f32"]["path"],
                        "matrix_imag_f32": payload["outputs"]["A_imag_f32"]["path"],
                        "rhs_real_f32": payload["outputs"]["rhs_real_f32"]["path"],
                        "rhs_imag_f32": payload["outputs"]["rhs_imag_f32"]["path"],
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )
    buffers = build_metal_geometry_buffers(
        grid,
        np.array([1, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )

    session = MetalNativeStandardSession.create_session(
        geometry_buffers=buffers,
        work_dir=tmp_path / "native-session",
        session_id="native-test",
    )
    try:
        result = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-assembly-test",
        )

        assert result.matrix_shape == (4, 4)
        assert result.rhs_shape == (4,)
        assert result.matrix_real_f32.is_file()
        assembly_manifest = json.loads(
            (tmp_path / "native-session" / "native-assembly-test" / "assembly.json")
            .read_text(encoding="utf-8")
        )
        assert assembly_manifest["outputs"]["matrix_layout"] == "row_major_c"
        assert assembly_manifest["neumann_dp0"]["real_f32"]["shape"] == [2]
    finally:
        session.close()


def test_native_standard_session_invokes_swift_batch_assembly(monkeypatch, tmp_path):
    _write_native_entrypoint(tmp_path)
    monkeypatch.setattr(native.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(native.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/swift")

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "--smoke":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "assemble_standard_neumann_batch" in command:
            payload = json.loads(
                Path(_arg_after(command, "assemble_standard_neumann_batch", 2))
                .read_text(encoding="utf-8")
            )
            root = Path(
                _arg_after(command, "assemble_standard_neumann_batch", 1)
            ).parent
            case_results = []
            for case in payload["cases"]:
                for descriptor in case["outputs"].values():
                    if isinstance(descriptor, dict):
                        path = root / descriptor["path"]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.zeros(descriptor["shape"], dtype=np.float32).tofile(path)
                case_results.append(
                    {
                        "schema": "hornlab.metal.standard.v1",
                        "op": "assemble_standard_neumann_result",
                        "session_id": "native-test",
                        "frequency_hz": case["frequency_hz"],
                        "matrix_layout": "row_major_c",
                        "matrix_shape": [4, 4],
                        "rhs_shape": [4],
                        "matrix_real_f32": case["outputs"]["A_real_f32"]["path"],
                        "matrix_imag_f32": case["outputs"]["A_imag_f32"]["path"],
                        "rhs_real_f32": case["outputs"]["rhs_real_f32"]["path"],
                        "rhs_imag_f32": case["outputs"]["rhs_imag_f32"]["path"],
                    }
                )
            Path(_arg_after(command, "assemble_standard_neumann_batch", 3)).write_text(
                json.dumps(
                    {
                        "schema": "hornlab.metal.standard.v1",
                        "op": "assemble_standard_neumann_batch_result",
                        "session_id": "native-test",
                        "cases": case_results,
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(native.subprocess, "run", fake_run)

    session = MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-session",
        session_id="native-test",
    )
    try:
        result = session.assemble_standard_neumann_batch(
            np.array([100.0, 200.0], dtype=np.float64),
            np.array([1.8318326, 3.6636652], dtype=np.float32),
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.5j],
                    [2.0 + 0.0j, 0.0 + 1.0j],
                ],
                dtype=np.complex64,
            ),
            operation_id="native-batch-assembly-test",
        )

        assert len(result) == 2
        assert result[0].matrix_shape == (4, 4)
        assert result[1].frequency_hz == 200.0
        manifest = json.loads(
            (
                tmp_path
                / "native-session"
                / "native-batch-assembly-test"
                / "assembly-batch.json"
            ).read_text(encoding="utf-8")
        )
        assert manifest["op"] == "assemble_standard_neumann_batch"
        assert len(manifest["cases"]) == 2
        assert "assemble_standard_neumann_batch" in calls[-1]
    finally:
        session.close()


def test_native_executable_session_contract_and_tiny_assembly(monkeypatch, tmp_path):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    for env_name in (
        "HORNLAB_METAL_BEM_NATIVE_THREADS_PER_GROUP",
        "HORNLAB_METAL_BEM_NATIVE_MATRIX_THREADS_PER_GROUP",
        "HORNLAB_METAL_BEM_NATIVE_RHS_THREADS_PER_GROUP",
        "HORNLAB_METAL_BEM_NATIVE_DUFFY_THREADS_PER_GROUP",
        "HORNLAB_METAL_BEM_NATIVE_FIELD_THREADS_PER_GROUP",
    ):
        monkeypatch.delenv(env_name, raising=False)

    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64),
        number_of_elements=2,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        global_dof_count=4,
    )
    buffers = build_metal_geometry_buffers(
        grid,
        np.array([1, 2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=2),
    )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=buffers,
        work_dir=tmp_path / "native-exec-session",
        session_id="native-exec-test",
    ) as session:
        validation = session.validate_contract()
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-exec-assembly",
        )

    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-exec-session"
            / "native-exec-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )

    assert validation["status"] == "ok"
    assert validation["implementation"] == "swift_native_contract_probe"
    assert result["metal_dispatch"]["matrix"]["requested_threads_per_threadgroup"] is None
    assert result["metal_dispatch"]["matrix"]["threads_per_threadgroup"] == 64
    assert result["metal_dispatch"]["rhs"]["requested_threads_per_threadgroup"] is None
    assert result["metal_dispatch"]["rhs"]["threads_per_threadgroup"] == 64
    assert assembly.matrix_shape == (4, 4)
    assert assembly.rhs_shape == (4,)
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(rhs))
    assert np.linalg.norm(matrix) > 0.0
    assert np.linalg.norm(rhs) > 0.0


def test_native_executable_optimized_matches_reference_on_tiny_mesh(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "parity")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_THREADS_PER_GROUP", "32")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "entrywise")
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-parity-session",
        session_id="native-parity-test",
    ) as session:
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-parity-assembly",
        )

    result = json.loads(
        (
            tmp_path
            / "native-parity-session"
            / "native-parity-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["implementation"] == "swift_native_metal_regular_quadrature"
    assert result["assembly_mode"] == "parity"
    assert result["duffy_corrections"]["implemented"] is False
    assert result["duffy_corrections"]["planned_pairs"] == {
        "coincident": 2,
        "edge": 2,
        "total": 4,
        "vertex": 0,
    }
    assert result["duffy_corrections"]["raw_triplets_if_expanded"] == 36
    assert result["reference_parity"]["matrix_relative_l2"] < 1e-4
    assert result["reference_parity"]["rhs_relative_l2"] < 1e-4
    assert result["metal_dispatch"]["matrix"]["requested_threads_per_threadgroup"] == 32
    assert result["metal_dispatch"]["matrix"]["threads_per_threadgroup"] == 32
    assert result["metal_dispatch"]["rhs"]["requested_threads_per_threadgroup"] == 32
    assert result["metal_dispatch"]["rhs"]["threads_per_threadgroup"] == 32
    assert result["regular_assembly_seconds"] > 0.0
    assert assembly.matrix_shape == (4, 4)


def test_native_executable_block_staged_matches_reference_on_tiny_mesh(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "parity")
    monkeypatch.setenv(
        "HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL",
        "block_staged",
    )
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-block-staged-parity-session",
        session_id="native-block-staged-parity-test",
    ) as session:
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-block-staged-parity-assembly",
        )

    result = json.loads(
        (
            tmp_path
            / "native-block-staged-parity-session"
            / "native-block-staged-parity-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["implementation"] == "swift_native_metal_block_staged_regular_quadrature"
    assert result["reference_parity"]["matrix_relative_l2"] < 1e-4
    assert result["reference_parity"]["rhs_relative_l2"] < 1e-4
    assert result["metal_dispatch"]["regular_assembly_implementation"] == "block_staged"
    assert result["metal_dispatch"]["pair_blocks"]["kernel"] == "assemble_pair_blocks_regular"
    assert result["metal_dispatch"]["pair_blocks"]["triangle_pairs"] == 4
    assert assembly.matrix_shape == (4, 4)


def test_native_executable_pair_atomic_matches_reference_on_tiny_mesh(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "parity")
    monkeypatch.setenv(
        "HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL",
        "pair_atomic",
    )
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-pair-atomic-parity-session",
        session_id="native-pair-atomic-parity-test",
    ) as session:
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-pair-atomic-parity-assembly",
        )

    result = json.loads(
        (
            tmp_path
            / "native-pair-atomic-parity-session"
            / "native-pair-atomic-parity-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["implementation"] == "swift_native_metal_pair_atomic_regular_quadrature"
    assert result["reference_parity"]["matrix_relative_l2"] < 1e-4
    assert result["reference_parity"]["rhs_relative_l2"] < 1e-4
    assert result["metal_dispatch"]["regular_assembly_implementation"] == "pair_atomic"
    assert result["metal_dispatch"]["matrix"]["triangle_pairs"] == 4
    assert assembly.matrix_shape == (4, 4)


def test_native_executable_pair_atomic_corrected_yz_xz_matches_full_domain(
    monkeypatch,
    tmp_path,
):
    """pair_atomic must reproduce the entrywise yz+xz half-vs-full parity."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    monkeypatch.setenv(
        "HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL",
        "pair_atomic",
    )
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_full_buffers(),
        work_dir=tmp_path / "native-full-pair-atomic-yz-xz-session",
        session_id="native-full-pair-atomic-yz-xz-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.ones(4, dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_parity_quarter_buffers(),
        work_dir=tmp_path / "native-quarter-pair-atomic-yz-xz-session",
        session_id="native-quarter-pair-atomic-yz-xz-test",
        symmetry_plane="yz+xz",
        check_open_edges=False,
    ) as quarter_session:
        quarter_assembly = quarter_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="quarter-assembly",
        )

    full_matrix, full_rhs = _read_complex_assembly(full_assembly)
    quarter_matrix, quarter_rhs = _read_complex_assembly(quarter_assembly)
    row_orbits = [
        np.array([0], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        np.array([2, 4], dtype=np.int64),
    ]
    even_full_matrix, even_full_rhs = orbit_reduce_matrix_rhs(
        full_matrix,
        full_rhs,
        row_orbits,
    )

    assert np.linalg.norm(quarter_matrix - even_full_matrix) / np.linalg.norm(
        even_full_matrix
    ) < 1.0e-6
    assert np.linalg.norm(quarter_rhs - even_full_rhs) / np.linalg.norm(
        even_full_rhs
    ) < 1.0e-6

    result = json.loads(
        (
            tmp_path
            / "native-quarter-pair-atomic-yz-xz-session"
            / "quarter-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["implementation"] == (
        "swift_native_metal_pair_atomic_regular_plus_metal_duffy_blocks"
    )
    assert result["duffy_corrections"]["image_singular_correction"] is True
    assert result["duffy_corrections"]["image_adjacent_pairs"] == 15


def test_native_executable_corrected_mode_applies_duffy_on_tiny_mesh(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "entrywise")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_THREADS_PER_GROUP", "256")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_MATRIX_THREADS_PER_GROUP", "32")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_RHS_THREADS_PER_GROUP", "64")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_THREADS_PER_GROUP", "128")
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-corrected-session",
        session_id="native-corrected-test",
    ) as session:
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="native-corrected-assembly",
        )

    result = json.loads(
        (
            tmp_path
            / "native-corrected-session"
            / "native-corrected-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )

    assert result["implementation"] == "swift_native_metal_regular_plus_metal_duffy_blocks"
    assert result["assembly_mode"] == "corrected"
    assert result["duffy_corrections"]["implemented"] is True
    assert (
        result["duffy_corrections"]["implementation"]
        == "metal_duffy_blocks_cpu_reduction"
    )
    assert result["duffy_corrections"]["block_seconds"] > 0.0
    assert result["duffy_corrections"]["reduction_seconds"] >= 0.0
    assert result["duffy_corrections"]["metal_dispatch"]["kernel"] == "duffy_delta_blocks"
    assert result["metal_dispatch"]["matrix"]["env"] == (
        "HORNLAB_METAL_BEM_NATIVE_MATRIX_THREADS_PER_GROUP"
    )
    assert result["metal_dispatch"]["matrix"]["requested_threads_per_threadgroup"] == 32
    assert result["metal_dispatch"]["rhs"]["env"] == (
        "HORNLAB_METAL_BEM_NATIVE_RHS_THREADS_PER_GROUP"
    )
    assert result["metal_dispatch"]["rhs"]["requested_threads_per_threadgroup"] == 64
    assert result["duffy_corrections"]["metal_dispatch"]["env"] == (
        "HORNLAB_METAL_BEM_NATIVE_DUFFY_THREADS_PER_GROUP"
    )
    assert (
        result["duffy_corrections"]["metal_dispatch"][
            "requested_threads_per_threadgroup"
        ]
        == 128
    )
    assert result["duffy_corrections"]["planned_pairs"] == {
        "coincident": 2,
        "edge": 2,
        "total": 4,
        "vertex": 0,
    }
    assert result["duffy_corrections"]["raw_triplets_if_expanded"] == 36
    assert result["duffy_corrections"]["unique_triplets"] == 16
    assert result["duffy_corrections"]["correction_seconds"] > 0.0
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(rhs))
    assert np.linalg.norm(matrix) > 0.0
    assert np.linalg.norm(rhs) > 0.0


def test_native_near_quadrature_default_off_matches_zero_env(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", "cpu")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "entrywise")
    neumann = np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64)

    def run_case(name: str, env_value: str | None):
        if env_value is None:
            monkeypatch.delenv(
                "HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE",
                raising=False,
            )
        else:
            monkeypatch.setenv(
                "HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE",
                env_value,
            )
        with MetalNativeStandardSession.create_session(
            geometry_buffers=_tiny_geometry_buffers(),
            work_dir=tmp_path / f"native-near-default-{name}-session",
            session_id=f"native-near-default-{name}-test",
        ) as session:
            assembly = session.assemble_standard_neumann(
                100.0,
                1.8318326,
                neumann,
                operation_id=f"native-near-default-{name}-assembly",
            )
        result = json.loads(
            (
                tmp_path
                / f"native-near-default-{name}-session"
                / f"native-near-default-{name}-assembly"
                / "assembly-result.json"
            ).read_text(encoding="utf-8")
        )
        matrix, rhs = _read_complex_assembly(assembly)
        return result, matrix, rhs

    unset_result, unset_matrix, unset_rhs = run_case("unset", None)
    zero_result, zero_matrix, zero_rhs = run_case("zero", "0")

    assert "near_quadrature" not in unset_result
    assert "near_quadrature" not in zero_result
    assert np.array_equal(unset_matrix, zero_matrix)
    assert np.array_equal(unset_rhs, zero_rhs)


def test_native_near_quadrature_corrects_close_non_touching_pair(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", "cpu")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "entrywise")
    buffers = _near_quadrature_geometry_buffers()
    neumann = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex64)
    k_real = 1.8318326

    def run_case(name: str, env_value: str | None):
        if env_value is None:
            monkeypatch.delenv(
                "HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE",
                raising=False,
            )
        else:
            monkeypatch.setenv(
                "HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE",
                env_value,
            )
        with MetalNativeStandardSession.create_session(
            geometry_buffers=buffers,
            work_dir=tmp_path / f"native-near-{name}-session",
            session_id=f"native-near-{name}-test",
        ) as session:
            assembly = session.assemble_standard_neumann(
                100.0,
                k_real,
                neumann,
                operation_id=f"native-near-{name}-assembly",
            )
        result = json.loads(
            (
                tmp_path
                / f"native-near-{name}-session"
                / f"native-near-{name}-assembly"
                / "assembly-result.json"
            ).read_text(encoding="utf-8")
        )
        matrix, _ = _read_complex_assembly(assembly)
        return result, matrix

    base_result, base_matrix = run_case("base", None)
    near_result, near_matrix = run_case("enabled", "2:1.5")

    assert "near_quadrature" not in base_result
    assert near_result["near_quadrature"]["level"] == 2
    assert near_result["near_quadrature"]["threshold"] == pytest.approx(1.5)
    assert near_result["near_quadrature"]["pair_count"] >= 1
    assert near_result["near_quadrature"]["seconds"] >= 0.0

    base_block = base_matrix[np.ix_([0, 1, 2], [3, 4, 5])]
    near_block = near_matrix[np.ix_([0, 1, 2], [3, 4, 5])]
    high_reference = _reference_pair_blocks(buffers, 0, 1, k_real, level=4).dlp

    assert np.linalg.norm(near_block - base_block) > 0.0
    base_error = np.linalg.norm(base_block - high_reference)
    near_error = np.linalg.norm(near_block - high_reference)
    assert near_error < base_error


def test_native_near_quadrature_junk_env_fails(monkeypatch, tmp_path):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", "cpu")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "entrywise")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE", "junk")

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-near-junk-session",
        session_id="native-near-junk-test",
    ) as session:
        with pytest.raises(
            RuntimeError,
            match="HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE",
        ):
            session.assemble_standard_neumann(
                100.0,
                1.8318326,
                np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
                operation_id="native-near-junk-assembly",
            )


def test_native_executable_yz_symmetry_matches_even_full_domain_solve(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "optimized")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_full_buffers(),
        work_dir=tmp_path / "native-full-yz-session",
        session_id="native-full-yz-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_parity_half_buffers(),
        work_dir=tmp_path / "native-half-yz-session",
        session_id="native-half-yz-test",
        symmetry_plane="yz",
        check_open_edges=False,
    ) as half_session:
        half_assembly = half_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="half-assembly",
        )

        half_matrix = np.fromfile(
            half_assembly.matrix_real_f32, dtype="<f4",
        ).reshape(half_assembly.matrix_shape) + 1j * np.fromfile(
            half_assembly.matrix_imag_f32, dtype="<f4",
        ).reshape(half_assembly.matrix_shape)
        half_rhs = np.fromfile(half_assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
            half_assembly.rhs_imag_f32,
            dtype="<f4",
        )
        half_pressure = np.linalg.solve(half_matrix, half_rhs).astype(np.complex64)
        half_field = half_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            half_pressure,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            np.array([[0.25, 0.2, 1.0], [-0.25, 0.2, 1.0]], dtype=np.float32),
            batch_id="symmetry-points",
            operation_id="half-field",
        )

    full_matrix = np.fromfile(
        full_assembly.matrix_real_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape) + 1j * np.fromfile(
        full_assembly.matrix_imag_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape)
    full_rhs = np.fromfile(full_assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_assembly.rhs_imag_f32,
        dtype="<f4",
    )
    real_dofs = np.array([0, 1, 2], dtype=np.int64)
    mirror_dofs = np.array([3, 4, 5], dtype=np.int64)
    even_full_matrix = (
        full_matrix[np.ix_(real_dofs, real_dofs)]
        + full_matrix[np.ix_(real_dofs, mirror_dofs)]
    )
    even_full_rhs = full_rhs[real_dofs]
    even_full_pressure = np.linalg.solve(
        even_full_matrix,
        even_full_rhs,
    ).astype(np.complex64)
    full_pressure = np.zeros(6, dtype=np.complex64)
    full_pressure[real_dofs] = even_full_pressure
    full_pressure[mirror_dofs] = even_full_pressure

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_full_buffers(),
        work_dir=tmp_path / "native-full-yz-field-session",
        session_id="native-full-yz-field-test",
    ) as full_field_session:
        full_field = full_field_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            full_pressure,
            np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64),
            np.array([[0.25, 0.2, 1.0], [-0.25, 0.2, 1.0]], dtype=np.float32),
            batch_id="full-points",
            operation_id="full-field",
        )

    assert np.allclose(
        even_full_pressure,
        half_pressure,
        rtol=5.0e-4,
        atol=5.0e-5,
    )

    full_values = np.fromfile(full_field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_field.pressure_imag_f32,
        dtype="<f4",
    )
    half_values = np.fromfile(half_field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        half_field.pressure_imag_f32,
        dtype="<f4",
    )
    assert np.allclose(full_values, half_values, rtol=5.0e-4, atol=5.0e-5)

    result = json.loads(
        (
            tmp_path
            / "native-half-yz-session"
            / "half-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "yz"


def test_native_executable_yz_xz_symmetry_matches_even_full_domain_solve(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "optimized")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_full_buffers(),
        work_dir=tmp_path / "native-full-yz-xz-session",
        session_id="native-full-yz-xz-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.ones(4, dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_parity_quarter_buffers(),
        work_dir=tmp_path / "native-quarter-yz-xz-session",
        session_id="native-quarter-yz-xz-test",
        symmetry_plane="yz+xz",
        check_open_edges=False,
    ) as quarter_session:
        quarter_assembly = quarter_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="quarter-assembly",
        )

        quarter_matrix = np.fromfile(
            quarter_assembly.matrix_real_f32, dtype="<f4",
        ).reshape(quarter_assembly.matrix_shape) + 1j * np.fromfile(
            quarter_assembly.matrix_imag_f32, dtype="<f4",
        ).reshape(quarter_assembly.matrix_shape)
        quarter_rhs = np.fromfile(
            quarter_assembly.rhs_real_f32, dtype="<f4",
        ) + 1j * np.fromfile(
            quarter_assembly.rhs_imag_f32,
            dtype="<f4",
        )
        quarter_pressure = np.linalg.solve(
            quarter_matrix,
            quarter_rhs,
        ).astype(np.complex64)
        quarter_field = quarter_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            quarter_pressure,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            np.array(
                [[0.25, -0.25, 0.25], [0.2, 0.2, -0.2], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            batch_id="symmetry-points",
            operation_id="quarter-field",
        )

    full_matrix = np.fromfile(
        full_assembly.matrix_real_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape) + 1j * np.fromfile(
        full_assembly.matrix_imag_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape)
    full_rhs = np.fromfile(full_assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_assembly.rhs_imag_f32,
        dtype="<f4",
    )
    row_orbits = [
        np.array([0], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        np.array([2, 4], dtype=np.int64),
    ]
    col_orbits = row_orbits
    even_full_matrix = np.array(
        [
            [
                full_matrix[np.ix_(row_orbits[row], col_orbits[col])].sum()
                for col in range(3)
            ]
            for row in range(3)
        ],
        dtype=np.complex64,
    )
    even_full_rhs = np.array(
        [full_rhs[row_dofs].sum() for row_dofs in row_orbits],
        dtype=np.complex64,
    )
    assert np.allclose(
        even_full_matrix,
        quarter_matrix,
        rtol=5.0e-4,
        atol=5.0e-5,
    )
    assert np.allclose(
        even_full_rhs,
        quarter_rhs,
        rtol=5.0e-4,
        atol=5.0e-5,
    )
    even_full_pressure = np.linalg.solve(
        even_full_matrix,
        even_full_rhs,
    ).astype(np.complex64)
    full_pressure = np.zeros(5, dtype=np.complex64)
    for value, image_dofs in zip(even_full_pressure, col_orbits, strict=True):
        full_pressure[image_dofs] = value

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_full_buffers(),
        work_dir=tmp_path / "native-full-yz-xz-field-session",
        session_id="native-full-yz-xz-field-test",
    ) as full_field_session:
        full_field = full_field_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            full_pressure,
            np.ones(4, dtype=np.complex64),
            np.array(
                [[0.25, -0.25, 0.25], [0.2, 0.2, -0.2], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            batch_id="full-points",
            operation_id="full-field",
        )

    assert np.allclose(
        even_full_pressure,
        quarter_pressure,
        rtol=5.0e-4,
        atol=5.0e-5,
    )

    full_values = np.fromfile(full_field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_field.pressure_imag_f32,
        dtype="<f4",
    )
    quarter_values = np.fromfile(
        quarter_field.pressure_real_f32,
        dtype="<f4",
    ) + 1j * np.fromfile(
        quarter_field.pressure_imag_f32,
        dtype="<f4",
    )
    assert np.allclose(full_values, quarter_values, rtol=5.0e-4, atol=5.0e-5)

    result = json.loads(
        (
            tmp_path
            / "native-quarter-yz-xz-session"
            / "quarter-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "yz+xz"


def test_native_executable_corrected_yz_xz_symmetry_applies_image_duffy(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_full_buffers(),
        work_dir=tmp_path / "native-full-corrected-yz-xz-session",
        session_id="native-full-corrected-yz-xz-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.ones(4, dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_yz_xz_parity_quarter_buffers(),
        work_dir=tmp_path / "native-quarter-corrected-yz-xz-session",
        session_id="native-quarter-corrected-yz-xz-test",
        symmetry_plane="yz+xz",
        check_open_edges=False,
    ) as quarter_session:
        quarter_assembly = quarter_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="quarter-assembly",
        )

    full_matrix, full_rhs = _read_complex_assembly(full_assembly)
    quarter_matrix, quarter_rhs = _read_complex_assembly(quarter_assembly)
    row_orbits = [
        np.array([0], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        np.array([2, 4], dtype=np.int64),
    ]
    even_full_matrix, even_full_rhs = orbit_reduce_matrix_rhs(
        full_matrix,
        full_rhs,
        row_orbits,
    )

    assert np.linalg.norm(quarter_matrix - even_full_matrix) / np.linalg.norm(
        even_full_matrix
    ) < 1.0e-6
    assert np.linalg.norm(quarter_rhs - even_full_rhs) / np.linalg.norm(
        even_full_rhs
    ) < 1.0e-6

    result = json.loads(
        (
            tmp_path
            / "native-quarter-corrected-yz-xz-session"
            / "quarter-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "yz+xz"
    assert result["duffy_corrections"]["image_singular_correction"] is True
    assert result["duffy_corrections"]["image_adjacent_pairs"] == 15


def test_native_executable_xy_symmetry_matches_even_full_domain_solve(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "optimized")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_mirror_full_buffers(),
        work_dir=tmp_path / "native-full-xy-session",
        session_id="native-full-xy-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_parity_half_buffers(),
        work_dir=tmp_path / "native-half-xy-session",
        session_id="native-half-xy-test",
        symmetry_plane="xy",
        check_open_edges=False,
    ) as half_session:
        half_assembly = half_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="half-assembly",
        )

        half_matrix = np.fromfile(
            half_assembly.matrix_real_f32, dtype="<f4",
        ).reshape(half_assembly.matrix_shape) + 1j * np.fromfile(
            half_assembly.matrix_imag_f32, dtype="<f4",
        ).reshape(half_assembly.matrix_shape)
        half_rhs = np.fromfile(half_assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
            half_assembly.rhs_imag_f32,
            dtype="<f4",
        )
        half_pressure = np.linalg.solve(half_matrix, half_rhs).astype(np.complex64)
        half_field = half_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            half_pressure,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            np.array([[0.25, 0.2, 1.0], [0.25, 0.2, -1.0]], dtype=np.float32),
            batch_id="symmetry-points",
            operation_id="half-field",
        )

    full_matrix = np.fromfile(
        full_assembly.matrix_real_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape) + 1j * np.fromfile(
        full_assembly.matrix_imag_f32, dtype="<f4",
    ).reshape(full_assembly.matrix_shape)
    full_rhs = np.fromfile(full_assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_assembly.rhs_imag_f32,
        dtype="<f4",
    )
    real_dofs = np.array([0, 1, 2], dtype=np.int64)
    mirror_dofs = np.array([3, 4, 5], dtype=np.int64)
    even_full_matrix = (
        full_matrix[np.ix_(real_dofs, real_dofs)]
        + full_matrix[np.ix_(real_dofs, mirror_dofs)]
    )
    even_full_rhs = full_rhs[real_dofs]
    even_full_pressure = np.linalg.solve(
        even_full_matrix,
        even_full_rhs,
    ).astype(np.complex64)
    full_pressure = np.zeros(6, dtype=np.complex64)
    full_pressure[real_dofs] = even_full_pressure
    full_pressure[mirror_dofs] = even_full_pressure

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_mirror_full_buffers(),
        work_dir=tmp_path / "native-full-xy-field-session",
        session_id="native-full-xy-field-test",
    ) as full_field_session:
        full_field = full_field_session.evaluate_standard_exterior(
            frequency_hz,
            k_real,
            full_pressure,
            np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64),
            np.array([[0.25, 0.2, 1.0], [0.25, 0.2, -1.0]], dtype=np.float32),
            batch_id="full-points",
            operation_id="full-field",
        )

    assert np.allclose(
        even_full_pressure,
        half_pressure,
        rtol=5.0e-4,
        atol=5.0e-5,
    )

    full_values = np.fromfile(full_field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        full_field.pressure_imag_f32,
        dtype="<f4",
    )
    half_values = np.fromfile(half_field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        half_field.pressure_imag_f32,
        dtype="<f4",
    )
    assert np.allclose(full_values, half_values, rtol=5.0e-4, atol=5.0e-5)

    result = json.loads(
        (
            tmp_path
            / "native-half-xy-session"
            / "half-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "xy"


def test_native_executable_corrected_xy_symmetry_applies_image_duffy(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    frequency_hz = 100.0
    k_real = 1.8318326

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_shared_full_buffers(),
        work_dir=tmp_path / "native-full-corrected-xy-session",
        session_id="native-full-corrected-xy-test",
    ) as full_session:
        full_assembly = full_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.ones(2, dtype=np.complex64),
            operation_id="full-assembly",
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_xy_parity_half_buffers(),
        work_dir=tmp_path / "native-half-corrected-xy-session",
        session_id="native-half-corrected-xy-test",
        symmetry_plane="xy",
        check_open_edges=False,
    ) as half_session:
        half_assembly = half_session.assemble_standard_neumann(
            frequency_hz,
            k_real,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="half-assembly",
        )

    full_matrix, full_rhs = _read_complex_assembly(full_assembly)
    half_matrix, half_rhs = _read_complex_assembly(half_assembly)
    row_orbits = [
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
    ]
    even_full_matrix, even_full_rhs = orbit_reduce_matrix_rhs(
        full_matrix,
        full_rhs,
        row_orbits,
    )

    assert np.linalg.norm(half_matrix - even_full_matrix) / np.linalg.norm(
        even_full_matrix
    ) < 1.0e-6
    assert np.linalg.norm(half_rhs - even_full_rhs) / np.linalg.norm(
        even_full_rhs
    ) < 1.0e-6

    result = json.loads(
        (
            tmp_path
            / "native-half-corrected-xy-session"
            / "half-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "xy"
    assert result["duffy_corrections"]["image_singular_correction"] is True
    assert result["duffy_corrections"]["image_adjacent_pairs"] >= 1


def test_native_executable_xy_image_duffy_fires_for_near_plane_vertex(
    monkeypatch,
    tmp_path,
):
    """A CAD vertex at z=5e-7 must snap onto the symmetry plane.

    5e-7 sits in the crack between Python plane validation (1e-7) and the
    Swift image-pair coordinate keys (1e-6 quantization): without snapping,
    the vertex neither counts as on-plane nor matches its own mirror, so
    image Duffy pairs silently stop firing.
    """
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    grid = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 5.0e-7, 1.0],
            ],
            dtype=np.float64,
        ),
        elements=np.array([[0], [1], [2]], dtype=np.int64),
        number_of_elements=1,
    )
    p1 = SimpleNamespace(
        local2global=np.array([[0, 1, 2]], dtype=np.int64),
        global_dof_count=3,
    )
    near_plane_buffers = build_metal_geometry_buffers(
        grid,
        np.array([2], dtype=np.int32),
        p1,
        SimpleNamespace(global_dof_count=1),
    )
    np.testing.assert_array_equal(
        near_plane_buffers.vertices_3xn_f32,
        _tiny_xy_parity_half_buffers().vertices_3xn_f32,
    )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=near_plane_buffers,
        work_dir=tmp_path / "native-near-plane-xy-session",
        session_id="native-near-plane-xy-test",
        symmetry_plane="xy",
        check_open_edges=False,
    ) as session:
        assembly = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j], dtype=np.complex64),
            operation_id="near-plane-assembly",
        )

    result = json.loads(
        (
            tmp_path
            / "native-near-plane-xy-session"
            / "near-plane-assembly"
            / "assembly-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["symmetry_plane"] == "xy"
    assert result["duffy_corrections"]["image_singular_correction"] is True
    assert result["duffy_corrections"]["image_adjacent_pairs"] >= 1
    # The mirrored triangle shares the full on-plane edge, so the image pair
    # must be edge-kind; without snapping, the 5e-7 vertex fails to match its
    # mirror and the pair silently degrades to a vertex-kind correction.
    assert result["duffy_corrections"]["planned_pairs"]["edge"] >= 1

    half_matrix, half_rhs = _read_complex_assembly(assembly)
    assert np.all(np.isfinite(half_matrix))
    assert np.all(np.isfinite(half_rhs))


def test_native_executable_rejects_payload_without_wavenumber(tmp_path):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available or status.helper_executable_path is None:
        pytest.skip("compiled native helper unavailable")

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-missing-k-session",
        session_id="native-missing-k-test",
    ) as session:
        session.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            operation_id="seed-assembly",
        )
        op_dir = session.info.work_dir / "seed-assembly"
        payload_path = op_dir / "assembly.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        del payload["k_real_f32"]
        doctored_path = op_dir / "assembly-missing-k.json"
        doctored_path.write_text(json.dumps(payload), encoding="utf-8")

        import subprocess

        completed = subprocess.run(
            [
                str(status.helper_executable_path),
                "assemble_standard_neumann",
                str(session.info.manifest_path),
                str(doctored_path),
                str(op_dir / "missing-k-result.json"),
            ],
            capture_output=True,
            text=True,
        )

    assert completed.returncode != 0
    assert "k_real_f32" in completed.stderr + completed.stdout


def test_native_executable_gpu_duffy_matches_cpu_duffy_on_tiny_mesh(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    def run_case(name: str, duffy_mode: str):
        monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
        monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", duffy_mode)
        with MetalNativeStandardSession.create_session(
            geometry_buffers=_tiny_geometry_buffers(),
            work_dir=tmp_path / f"native-{name}-duffy-session",
            session_id=f"native-{name}-duffy-test",
        ) as session:
            assembly = session.assemble_standard_neumann(
                100.0,
                1.8318326,
                np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
                operation_id=f"native-{name}-duffy-assembly",
            )
        matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
            assembly.matrix_shape
        ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
            assembly.matrix_shape
        )
        rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
            assembly.rhs_imag_f32,
            dtype="<f4",
        )
        return matrix, rhs

    gpu_matrix, gpu_rhs = run_case("gpu", "gpu_blocks")
    cpu_matrix, cpu_rhs = run_case("cpu", "cpu")

    assert np.linalg.norm(gpu_matrix - cpu_matrix) / np.linalg.norm(cpu_matrix) < 1e-5
    assert np.linalg.norm(gpu_rhs - cpu_rhs) / np.linalg.norm(cpu_rhs) < 1e-5


def test_native_executable_resident_batch_matches_single_assembly(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    neumann = np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-resident-batch-session",
        session_id="native-resident-batch-test",
    ) as session:
        single = session.assemble_standard_neumann(
            100.0,
            1.8318326,
            neumann,
            operation_id="single-assembly",
        )
        batch = session.assemble_standard_neumann_batch(
            np.array([100.0], dtype=np.float64),
            np.array([1.8318326], dtype=np.float32),
            neumann.reshape(1, -1),
            operation_id="resident-batch-assembly",
        )[0]

    single_matrix = np.fromfile(single.matrix_real_f32, dtype="<f4").reshape(
        single.matrix_shape
    ) + 1j * np.fromfile(single.matrix_imag_f32, dtype="<f4").reshape(
        single.matrix_shape
    )
    single_rhs = np.fromfile(single.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        single.rhs_imag_f32,
        dtype="<f4",
    )
    batch_matrix = np.fromfile(batch.matrix_real_f32, dtype="<f4").reshape(
        batch.matrix_shape
    ) + 1j * np.fromfile(batch.matrix_imag_f32, dtype="<f4").reshape(
        batch.matrix_shape
    )
    batch_rhs = np.fromfile(batch.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        batch.rhs_imag_f32,
        dtype="<f4",
    )

    result = json.loads(
        (
            tmp_path
            / "native-resident-batch-session"
            / "resident-batch-assembly"
            / "assembly-batch-result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["op"] == "assemble_standard_neumann_batch_result"
    assert result["resident_reuse"]["geometry_buffers"] is True
    assert result["resident_reuse"]["duffy_rules"] is True
    assert np.linalg.norm(batch_matrix - single_matrix) / np.linalg.norm(single_matrix) < 1e-5
    assert np.linalg.norm(batch_rhs - single_rhs) / np.linalg.norm(single_rhs) < 1e-5


def test_native_executable_resident_assembly_solve_matches_python_solve(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.5j]], dtype=np.complex64)
    frequency_hz = np.array([100.0], dtype=np.float64)
    k_real = np.array([1.8318326], dtype=np.float32)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-resident-assembly-solve-session",
        session_id="native-resident-assembly-solve-test",
    ) as session:
        assembly = session.assemble_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="resident-batch-assembly",
        )[0]
        solved = session.assemble_solve_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="resident-batch-assembly-solve",
        )[0]

    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )
    expected = np.linalg.solve(matrix, rhs).astype(np.complex64)
    pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-resident-assembly-solve-session"
            / "resident-batch-assembly-solve"
            / "assembly-solve-batch-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["op"] == "assemble_solve_standard_neumann_batch_result"
    assert result["implementation"] == (
        "swift_native_resident_metal_assembly_accelerate_solve_batch"
    )
    assert result["resident_reuse"]["duffy_reduction_plan"] is True
    assert solved.lapack_info == 0
    assert solved.assembly_s > 0.0
    assert solved.dense_solve_s > 0.0
    assert np.linalg.norm(pressure - expected) / np.linalg.norm(expected) < 1e-5


def test_native_executable_dense_solve_iterative_refinement(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_REFINE", "3")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.5j]], dtype=np.complex64)
    frequency_hz = np.array([100.0], dtype=np.float64)
    k_real = np.array([1.8318326], dtype=np.float32)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-refine-session",
        session_id="native-refine-test",
    ) as session:
        assembly = session.assemble_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="refine-batch-assembly",
        )[0]
        solved = session.assemble_solve_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="refine-batch-assembly-solve",
        )[0]

    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )
    pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-refine-session"
            / "refine-batch-assembly-solve"
            / "assembly-solve-batch-result.json"
        ).read_text(encoding="utf-8")
    )
    case_result = result["cases"][0]
    # Refinement bookkeeping must ride along in per-case diagnostics, and the
    # refined solution's float64 residual against the float32 operator must
    # sit at the single-precision floor or better.
    assert solved.lapack_info == 0
    assert case_result["dense_solve_refine_iterations"] >= 0
    residual_rel = case_result["dense_solve_refine_residual_rel"]
    assert residual_rel <= 1.0e-5
    exact = np.linalg.solve(matrix.astype(np.complex128), rhs.astype(np.complex128))
    measured = np.abs(
        rhs.astype(np.complex128)
        - matrix.astype(np.complex128) @ pressure.astype(np.complex128)
    ).max() / np.abs(rhs).max()
    assert measured <= max(5.0e-6, 10.0 * residual_rel)
    assert np.linalg.norm(pressure - exact) / np.linalg.norm(exact) < 1e-4


def test_native_executable_float64_dense_solve_matches_numpy(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    # float64 (zgesv) factor/solve of the float32-assembled system, narrowed
    # back to f32. Must match np.linalg.solve in complex128 to a tolerance much
    # tighter than the float32 path (bounded only by the f32-narrowed output),
    # and the per-case diagnostics must report dense_solve_dtype == "float64".
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE", "float64")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_REFINE", raising=False)
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.5j]], dtype=np.complex64)
    frequency_hz = np.array([100.0], dtype=np.float64)
    k_real = np.array([1.8318326], dtype=np.float32)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-float64-session",
        session_id="native-float64-test",
    ) as session:
        assembly = session.assemble_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="float64-batch-assembly",
        )[0]
        solved = session.assemble_solve_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="float64-batch-assembly-solve",
        )[0]

    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )
    pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-float64-session"
            / "float64-batch-assembly-solve"
            / "assembly-solve-batch-result.json"
        ).read_text(encoding="utf-8")
    )
    case_result = result["cases"][0]
    assert solved.lapack_info == 0
    assert case_result["solve_implementation"] == "accelerate_lapack_zgesv"
    assert case_result["dense_solve_dtype"] == "float64"
    # No iterative refinement runs on the float64 path; its bookkeeping keys must
    # be absent (the plan explicitly skips refinement for float64).
    assert "dense_solve_refine_iterations" not in case_result
    assert "dense_solve_refine_residual_rel" not in case_result
    exact = np.linalg.solve(matrix.astype(np.complex128), rhs.astype(np.complex128))
    assert np.linalg.norm(pressure - exact) / np.linalg.norm(exact) < 1e-5


def test_native_executable_float64_and_float32_dense_solve_agree(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    # The float32 and float64 paths solve the SAME assembled operator; on a
    # well-conditioned case they must agree to ~f32 tolerance (the float64
    # output is narrowed back to f32, so the gap is bounded by f32 rounding).
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_REFINE", raising=False)
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.5j]], dtype=np.complex64)
    frequency_hz = np.array([100.0], dtype=np.float64)
    k_real = np.array([1.8318326], dtype=np.float32)

    def _solve(dtype: str, label: str) -> np.ndarray:
        monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE", dtype)
        with MetalNativeStandardSession.create_session(
            geometry_buffers=_tiny_geometry_buffers(),
            work_dir=tmp_path / f"native-agree-{label}-session",
            session_id=f"native-agree-{label}-test",
        ) as session:
            solved = session.assemble_solve_standard_neumann_batch(
                frequency_hz,
                k_real,
                neumann,
                operation_id=f"agree-{label}-batch-assembly-solve",
            )[0]
        return np.fromfile(
            solved.pressure_real_f32, dtype="<f4"
        ) + 1j * np.fromfile(solved.pressure_imag_f32, dtype="<f4")

    pressure_f32 = _solve("float32", "f32")
    pressure_f64 = _solve("float64", "f64")
    assert np.all(np.isfinite(pressure_f32))
    assert np.all(np.isfinite(pressure_f64))
    rel = np.linalg.norm(pressure_f64 - pressure_f32) / np.linalg.norm(pressure_f32)
    assert rel < 1e-4


def test_native_executable_chief_points_diagnostics(monkeypatch, tmp_path):
    """CHIEF interior points route through the evaluate batch: the helper solves
    the overdetermined system by zgels and emits the chief diagnostics. The
    solved pressure stays finite and the chief residual is reported."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64)
    frequency_hz = np.array([172.0], dtype=np.float64)
    k_real = np.array([np.float32(2.0 * np.pi * 172.0 / 343.0)], dtype=np.float32)
    observation_points = np.array(
        [[0.0, 0.0, 0.7], [0.2, 0.0, 0.8]],
        dtype=np.float32,
    )
    # Interior overdetermination points (m, 3) in the mesh frame; marshalled to
    # the (3, m) f32 layout the helper expects, exactly as sweep.run does.
    chief_points = np.array(
        [[0.02, 0.03, 0.05], [-0.03, 0.02, 0.04], [0.01, -0.02, 0.06]],
        dtype=np.float64,
    )
    chief_3xm = np.ascontiguousarray(chief_points.T, dtype=np.float32)

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_robin_geometry_buffers(),
        work_dir=tmp_path / "native-chief-session",
        session_id="native-chief-test",
    ) as session:
        solved = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="resident-chief",
            source_tags=[2],
            impedance_source_tag=2,
            chief_points=chief_3xm,
        )[0]

    pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )
    field = np.fromfile(solved.field_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.field_imag_f32,
        dtype="<f4",
    )

    assert solved.lapack_info == 0
    assert np.all(np.isfinite(pressure))
    assert np.all(np.isfinite(field))
    assert solved.diagnostics["chief_points"] is True
    assert solved.diagnostics["chief_points_count"] == 3
    assert solved.diagnostics["chief_solver"] == "accelerate_lapack_zgels"
    assert solved.diagnostics["solve_implementation"] == "accelerate_lapack_zgels"
    # The least-squares path runs in float64 regardless of dense_solve_dtype.
    assert solved.diagnostics["dense_solve_dtype"] == "float64"
    assert np.isfinite(solved.diagnostics["chief_residual_rel"])
    assert solved.diagnostics["chief_residual_rel"] >= 0.0


def test_native_executable_chief_off_matches_plain_solve(monkeypatch, tmp_path):
    """chief_points=None (the default) must leave the solve bit-for-bit identical
    to the plain square-LU path: same pressure, same square solver, and no chief
    diagnostic keys."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE", "float32")
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64)
    frequency_hz = np.array([172.0], dtype=np.float64)
    k_real = np.array([np.float32(2.0 * np.pi * 172.0 / 343.0)], dtype=np.float32)
    observation_points = np.array(
        [[0.0, 0.0, 0.7], [0.2, 0.0, 0.8]],
        dtype=np.float32,
    )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_robin_geometry_buffers(),
        work_dir=tmp_path / "native-chief-off-session",
        session_id="native-chief-off-test",
    ) as session:
        solved = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="resident-chief-off",
            source_tags=[2],
            impedance_source_tag=2,
        )[0]

    pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )
    assert solved.lapack_info == 0
    assert np.all(np.isfinite(pressure))
    # The default square-LU path (cgesv), not the CHIEF least-squares path.
    assert solved.diagnostics["solve_implementation"] == "accelerate_lapack_cgesv"
    assert "chief_points" not in solved.diagnostics
    assert "chief_residual_rel" not in solved.diagnostics
    assert "chief_solver" not in solved.diagnostics


def test_native_executable_resident_assembly_solve_lu_factor_variant_matches_python(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv(
        "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_IMPL",
        "cgetrf_cgetrs",
    )
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.5j]], dtype=np.complex64)
    frequency_hz = np.array([100.0], dtype=np.float64)
    k_real = np.array([1.8318326], dtype=np.float32)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-resident-assembly-solve-lu-factor-session",
        session_id="native-resident-assembly-solve-lu-factor-test",
    ) as session:
        assembly = session.assemble_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="resident-batch-assembly",
        )[0]
        solved = session.assemble_solve_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="resident-batch-assembly-solve",
        )[0]

    matrix = np.fromfile(assembly.matrix_real_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    ) + 1j * np.fromfile(assembly.matrix_imag_f32, dtype="<f4").reshape(
        assembly.matrix_shape
    )
    rhs = np.fromfile(assembly.rhs_real_f32, dtype="<f4") + 1j * np.fromfile(
        assembly.rhs_imag_f32,
        dtype="<f4",
    )
    expected = np.linalg.solve(matrix, rhs).astype(np.complex64)
    pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-resident-assembly-solve-lu-factor-session"
            / "resident-batch-assembly-solve"
            / "assembly-solve-batch-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["cases"][0]["solve_implementation"] == (
        "accelerate_lapack_cgetrf_cgetrs"
    )
    assert solved.lapack_info == 0
    assert solved.dense_solve_s > 0.0
    assert np.linalg.norm(pressure - expected) / np.linalg.norm(expected) < 1e-5


def test_native_executable_resident_assembly_solve_field_matches_split_path(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", "gpu_blocks")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "pair_atomic")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.5j]], dtype=np.complex64)
    frequency_hz = np.array([100.0], dtype=np.float64)
    k_real = np.array([1.8318326], dtype=np.float32)
    observation_points = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
        dtype=np.float32,
    )
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-resident-assembly-solve-field-session",
        session_id="native-resident-assembly-solve-field-test",
    ) as session:
        combined = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="resident-batch-assembly-solve-field",
            source_tags=[2],
            impedance_source_tag=2,
        )[0]
        solved = session.assemble_solve_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            operation_id="resident-batch-assembly-solve",
        )[0]
        pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
            solved.pressure_imag_f32,
            dtype="<f4",
        )
        field = session.evaluate_standard_exterior_batch(
            frequency_hz,
            k_real,
            pressure.reshape(1, -1),
            neumann,
            observation_points.T,
            operation_id="resident-batch-field",
        )[0]
        reduced = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="resident-batch-assembly-solve-field-reduced",
            source_tags=[2],
            impedance_source_tag=2,
            write_surface_pressure=False,
        )[0]
        batched = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="resident-batch-assembly-solve-field-batched-output",
            source_tags=[2],
            impedance_source_tag=2,
            write_surface_pressure=False,
            write_batched_field=True,
        )[0]

    combined_pressure = np.fromfile(combined.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        combined.pressure_imag_f32,
        dtype="<f4",
    )
    split_field = np.fromfile(field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        field.pressure_imag_f32,
        dtype="<f4",
    )
    combined_field = np.fromfile(combined.field_real_f32, dtype="<f4") + 1j * np.fromfile(
        combined.field_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-resident-assembly-solve-field-session"
            / "resident-batch-assembly-solve-field"
            / "assembly-solve-field-batch-result.json"
        ).read_text(encoding="utf-8")
    )

    assert result["op"] == "assemble_solve_evaluate_standard_neumann_batch_result"
    assert result["implementation"] == (
        "swift_native_resident_metal_assembly_accelerate_solve_field_batch"
    )
    assert result["resident_reuse"]["field_output_buffers"] is True
    assert result["resident_reuse"]["observation_points_buffer"] is True
    assert combined.lapack_info == 0
    assert combined.assembly_s > 0.0
    assert combined.dense_solve_s > 0.0
    assert combined.field_s > 0.0
    # cgecon condition estimate must ride along in per-case diagnostics so
    # interior-resonance spikes in sweeps are attributable.
    assert 0.0 < combined.diagnostics["dense_solve_rcond"] <= 1.0
    assert combined.diagnostics["dense_solve_condition_1norm"] >= 1.0
    expected_source_avg = (
        combined_pressure[0] + combined_pressure[2] + combined_pressure[3]
    ) / 3.0
    assert combined.impedance == pytest.approx(expected_source_avg, rel=1.0e-6)
    assert combined.surface_pressure_avg is not None
    assert combined.surface_pressure_avg[2] == pytest.approx(
        expected_source_avg,
        rel=1.0e-6,
    )
    assert reduced.pressure_real_f32 is None
    assert reduced.pressure_imag_f32 is None
    assert reduced.impedance == pytest.approx(expected_source_avg, rel=1.0e-6)
    assert reduced.surface_pressure_avg is not None
    assert reduced.surface_pressure_avg[2] == pytest.approx(
        expected_source_avg,
        rel=1.0e-6,
    )
    assert batched.pressure_real_f32 is None
    assert batched.pressure_imag_f32 is None
    assert batched.field_row_index == 0
    assert batched.field_batch_shape == (1, 2)
    batched_field = (
        np.fromfile(batched.field_real_f32, dtype="<f4").reshape(batched.field_batch_shape)
        + 1j
        * np.fromfile(batched.field_imag_f32, dtype="<f4").reshape(
            batched.field_batch_shape
        )
    )
    assert np.linalg.norm(batched_field[0] - split_field) / np.linalg.norm(split_field) < 1e-5
    assert np.linalg.norm(combined_pressure - pressure) / np.linalg.norm(pressure) < 1e-5
    assert np.linalg.norm(combined_field - split_field) / np.linalg.norm(split_field) < 1e-5


def test_native_executable_complex_k_robin_tags_8_9_solve_field(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64)
    frequency_hz = np.array([172.0], dtype=np.float64)
    k_real = np.array([np.float32(2.0 * np.pi * 172.0 / 343.0)], dtype=np.float32)
    k_imag = (k_real * np.float32(0.005)).astype(np.float32)
    observation_points = np.array(
        [[0.0, 0.0, 0.7], [0.2, 0.0, 0.8]],
        dtype=np.float32,
    )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_robin_geometry_buffers(),
        work_dir=tmp_path / "native-complex-robin-session",
        session_id="native-complex-robin-test",
    ) as session:
        solved = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            k_imag_f32=k_imag,
            impedance_sources={8: 0.05 + 0.0j, 9: 0.02 + 0.01j},
            operation_id="resident-complex-robin",
            source_tags=[2],
            impedance_source_tag=2,
        )[0]

    pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )
    field = np.fromfile(solved.field_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.field_imag_f32,
        dtype="<f4",
    )

    assert solved.lapack_info == 0
    assert np.all(np.isfinite(pressure))
    assert np.all(np.isfinite(field))
    assert solved.diagnostics["assembly_mode"] == "corrected"
    assert solved.diagnostics["assembly_implementation"] == (
        "swift_native_metal_pair_atomic_regular_plus_metal_duffy_blocks"
    )
    assert solved.diagnostics["complex_k"] is True
    assert solved.diagnostics["robin_boundary"] is True
    assert solved.diagnostics["field_uses_total_neumann"] is True
    assert 0.0 < solved.diagnostics["dense_solve_rcond"] <= 1.0
    assert solved.diagnostics["dense_solve_condition_1norm"] >= 1.0


def test_native_executable_complex_k_robin_reference_mode(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "reference")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "pair_atomic")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    neumann = np.array([[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64)
    frequency_hz = np.array([172.0], dtype=np.float64)
    k_real = np.array([np.float32(2.0 * np.pi * 172.0 / 343.0)], dtype=np.float32)
    k_imag = (k_real * np.float32(0.005)).astype(np.float32)
    observation_points = np.array(
        [[0.0, 0.0, 0.7], [0.2, 0.0, 0.8]],
        dtype=np.float32,
    )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_robin_geometry_buffers(),
        work_dir=tmp_path / "native-complex-robin-reference-session",
        session_id="native-complex-robin-reference-test",
    ) as session:
        solved = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            k_imag_f32=k_imag,
            impedance_sources={8: 0.05 + 0.0j, 9: 0.02 + 0.01j},
            operation_id="resident-complex-robin-reference",
            source_tags=[2],
            impedance_source_tag=2,
        )[0]

    assert solved.lapack_info == 0
    assert solved.diagnostics["assembly_mode"] == "reference"
    assert solved.diagnostics["assembly_implementation"] == (
        "swift_native_reference_complex_robin_quadrature_plus_cpu_duffy"
    )
    assert solved.diagnostics["complex_k"] is True
    assert solved.diagnostics["robin_boundary"] is True
    assert solved.diagnostics["field_uses_total_neumann"] is True
    assert 0.0 < solved.diagnostics["dense_solve_rcond"] <= 1.0


def test_native_executable_streams_per_case_results(monkeypatch, tmp_path):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    neumann = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.5j],
            [0.5 + 0.0j, 0.0 + 0.25j],
            [0.25 + 0.0j, 0.0 + 0.125j],
        ],
        dtype=np.complex64,
    )
    frequency_hz = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    k_real = np.array([1.83, 3.66, 5.49], dtype=np.float32)
    observation_points = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
        dtype=np.float32,
    )
    streamed_calls: list[tuple[int, object]] = []
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-streamed-case-results-session",
        session_id="native-streamed-case-results-test",
    ) as session:
        streamed = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="streamed-batch",
            source_tags=[2],
            impedance_source_tag=2,
            on_case_result=lambda i, solved: streamed_calls.append((i, solved)),
        )
        oneshot = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="oneshot-batch",
            source_tags=[2],
            impedance_source_tag=2,
        )

        assert [index for index, _ in streamed_calls] == [0, 1, 2]
        assert [solved for _, solved in streamed_calls] == streamed
        assert len(streamed) == len(oneshot) == 3
        case_dir = (
            tmp_path
            / "native-streamed-case-results-session"
            / "streamed-batch"
            / "case-results"
        )
        case_files = sorted(path.name for path in case_dir.glob("case-*.json"))
        assert case_files == ["case-0000.json", "case-0001.json", "case-0002.json"]
        for solved, reference in zip(streamed, oneshot):
            assert solved.frequency_hz == reference.frequency_hz
            assert solved.lapack_info == 0
            assert solved.impedance == pytest.approx(reference.impedance, rel=1e-6)
            streamed_field = np.fromfile(solved.field_real_f32, dtype="<f4")
            reference_field = np.fromfile(reference.field_real_f32, dtype="<f4")
            np.testing.assert_allclose(streamed_field, reference_field, rtol=1e-5)
            # Whole-batch diagnostics are only known once the batch ends, so
            # streamed per-case diagnostics must omit them rather than guess.
            assert "batch" not in solved.diagnostics
            assert "batch" in reference.diagnostics
        result = json.loads(
            (
                tmp_path
                / "native-streamed-case-results-session"
                / "streamed-batch"
                / "assembly-solve-field-batch-result.json"
            ).read_text(encoding="utf-8")
        )
        assert result["streamed_case_results"] is True

        early = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="early-stop-batch",
            source_tags=[2],
            impedance_source_tag=2,
            on_case_result=lambda i, solved: False,
        )
        assert len(early) == 1
        assert early[0].frequency_hz == pytest.approx(100.0)

        with pytest.raises(ValueError, match="write_batched_field"):
            session.assemble_solve_evaluate_standard_neumann_batch(
                frequency_hz,
                k_real,
                neumann,
                observation_points,
                operation_id="streamed-batched-field",
                source_tags=[2],
                impedance_source_tag=2,
                write_batched_field=True,
                on_case_result=lambda i, solved: None,
            )


def test_run_native_helper_streaming_spools_output_without_deadlock(tmp_path):
    helper = tmp_path / "fake_native_helper.py"
    helper.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "sys.stdout.write('o' * (2 * 1024 * 1024))\n"
        "sys.stdout.flush()\n"
        "sys.stderr.write('e' * (2 * 1024 * 1024))\n"
        "sys.stderr.flush()\n"
        "Path(sys.argv[-1]).write_text('{}', encoding='utf-8')\n",
        encoding="utf-8",
    )
    status = native.MetalNativeRuntimeStatus(
        available=True,
        platform_system="Darwin",
        platform_machine="arm64",
        is_macos=True,
        is_apple_silicon=True,
        swift_path=sys.executable,
        swift_source="test",
        helper_executable_path=None,
        helper_source=None,
        backend_dir=tmp_path,
        native_entrypoint=helper,
        native_package_dir=tmp_path / "native_helper",
        helper_assets_present=True,
        smoke_test_ran=False,
        smoke_test_ok=True,
        smoke_test_error=None,
        reasons=(),
    )
    session = native.MetalNativeStandardSession(
        native.MetalNativeSessionInfo(
            session_id="streaming-output-test",
            work_dir=tmp_path,
            manifest_path=tmp_path / "session.json",
            geometry_dir=tmp_path / "geometry",
            runtime_status=status,
        ),
        geometry_payload=None,
        owns_work_dir=False,
        runtime_config=native.MetalNativeRuntimeConfig(operation_timeout_s=3.0),
    )

    result_path = tmp_path / "result.json"
    completed = session._run_native_helper_streaming(
        "assemble_solve_evaluate_standard_neumann_batch",
        payload_path=tmp_path / "payload.json",
        result_path=result_path,
        poll=lambda: False,
    )

    assert completed is True
    assert result_path.is_file()


def test_native_executable_coupled_ib_uses_rayleigh_aperture_field(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    buffers = _ib_box_geometry_buffers()
    neumann = np.zeros((1, buffers.n_triangles), dtype=np.complex64)
    neumann[0, buffers.physical_tags_i32 == 1] = 1.0 + 0.0j
    frequency_hz = np.array([220.0], dtype=np.float64)
    k_real = np.array([np.float32(2.0 * np.pi * 220.0 / 343.0)], dtype=np.float32)
    observation_points = np.array(
        [
            [0.0, 0.0, 0.7],
            [0.3, 0.0, 0.8],
            [0.0, 0.0, -0.2],
        ],
        dtype=np.float32,
    )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=buffers,
        work_dir=tmp_path / "native-coupled-ib-session",
        session_id="native-coupled-ib-test",
        aperture_tag=7,
        velocity_source_tags=[1],
    ) as session:
        solved = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            operation_id="resident-coupled-ib",
            source_tags=[1],
        )[0]

    pressure = np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )
    field = np.fromfile(solved.field_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.field_imag_f32,
        dtype="<f4",
    )

    assert solved.lapack_info == 0
    assert np.all(np.isfinite(pressure))
    assert np.all(np.isfinite(field))
    assert solved.diagnostics["coupled_ib"] is True
    assert solved.diagnostics["aperture_tag"] == 7
    assert solved.diagnostics["aperture_triangles"] == 2
    assert solved.diagnostics["aperture_velocity_basis"] == "DP0"
    assert solved.diagnostics["aperture_velocity_dofs"] == 2
    assert solved.diagnostics["ib_field"] == "rayleigh_aperture_only"
    assert "field_uses_total_neumann" not in solved.diagnostics
    assert abs(field[2]) == pytest.approx(0.0, abs=0.0)
    assert np.linalg.norm(field[:2]) > 0.0


def test_native_executable_coupled_ib_gpu_schur_matches_cpu_augmented(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    buffers = _ib_box_geometry_buffers()
    neumann = np.zeros((1, buffers.n_triangles), dtype=np.complex64)
    neumann[0, buffers.physical_tags_i32 == 1] = 1.0 + 0.0j
    frequency_hz = np.array([220.0], dtype=np.float64)
    k_real = np.array([np.float32(2.0 * np.pi * 220.0 / 343.0)], dtype=np.float32)
    observation_points = np.array(
        [
            [0.0, 0.0, 0.7],
            [0.3, 0.0, 0.8],
            [0.0, 0.2, 0.8],
        ],
        dtype=np.float32,
    )

    def run_case(label: str, aperture_assembly: str | None, solve_mode: str | None):
        monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
        monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
        monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "entrywise")
        monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", raising=False)
        monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE", raising=False)
        monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_IMPL", raising=False)
        monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_REFINE", raising=False)
        if aperture_assembly is None:
            monkeypatch.delenv(
                "HORNLAB_METAL_BEM_NATIVE_COUPLED_IB_APERTURE_ASSEMBLY",
                raising=False,
            )
        else:
            monkeypatch.setenv(
                "HORNLAB_METAL_BEM_NATIVE_COUPLED_IB_APERTURE_ASSEMBLY",
                aperture_assembly,
            )
        if solve_mode is None:
            monkeypatch.delenv("HORNLAB_METAL_BEM_NATIVE_COUPLED_IB_SOLVE", raising=False)
        else:
            monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_COUPLED_IB_SOLVE", solve_mode)

        with MetalNativeStandardSession.create_session(
            geometry_buffers=buffers,
            work_dir=tmp_path / f"native-coupled-ib-{label}-session",
            session_id=f"native-coupled-ib-{label}-test",
            aperture_tag=7,
            velocity_source_tags=[1],
        ) as session:
            solved = session.assemble_solve_evaluate_standard_neumann_batch(
                frequency_hz,
                k_real,
                neumann,
                observation_points,
                operation_id=f"resident-coupled-ib-{label}",
                source_tags=[1],
            )[0]

        pressure = np.fromfile(
            solved.pressure_real_f32,
            dtype="<f4",
        ) + 1j * np.fromfile(solved.pressure_imag_f32, dtype="<f4")
        field = np.fromfile(
            solved.field_real_f32,
            dtype="<f4",
        ) + 1j * np.fromfile(solved.field_imag_f32, dtype="<f4")
        return solved, pressure, field

    gpu_schur, gpu_pressure, gpu_field = run_case(
        "gpu-schur",
        aperture_assembly=None,
        solve_mode=None,
    )
    cpu_augmented, cpu_pressure, cpu_field = run_case(
        "cpu-augmented",
        aperture_assembly="cpu",
        solve_mode="augmented",
    )

    assert gpu_schur.lapack_info == 0
    assert cpu_augmented.lapack_info == 0
    assert gpu_schur.diagnostics["ib_aperture_assembly_implementation"] == (
        "swift_native_metal_aperture_slp_blocks"
    )
    assert gpu_schur.diagnostics["ib_coupled_solve"] == "schur"
    assert (
        gpu_schur.diagnostics["solve_implementation"]
        == "accelerate_lapack_cgesv_coupled_ib_schur"
    )
    assert "ib_aperture_metal_dispatch" in gpu_schur.diagnostics
    assert cpu_augmented.diagnostics["ib_aperture_assembly_implementation"] == (
        "swift_native_cpu_aperture_slp_blocks"
    )
    assert cpu_augmented.diagnostics["ib_coupled_solve"] == "augmented"
    assert (
        cpu_augmented.diagnostics["solve_implementation"]
        == "accelerate_lapack_cgesv_coupled_ib_augmented"
    )
    assert "ib_aperture_metal_dispatch" not in cpu_augmented.diagnostics
    assert np.all(np.isfinite(gpu_pressure))
    assert np.all(np.isfinite(gpu_field))
    assert np.all(np.isfinite(cpu_pressure))
    assert np.all(np.isfinite(cpu_field))

    pressure_rel = np.linalg.norm(gpu_pressure - cpu_pressure) / np.linalg.norm(
        cpu_pressure
    )
    field_rel = np.linalg.norm(gpu_field - cpu_field) / np.linalg.norm(cpu_field)
    assert pressure_rel < 5.0e-4
    assert field_rel < 5.0e-4


def test_native_executable_coupled_ib_yz_xz_quadrant_matches_full(
    monkeypatch,
    tmp_path,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE", "float64")

    full_buffers = _ib_quarter_box_mirrored_full_geometry_buffers()
    quarter_buffers = _ib_quarter_box_geometry_buffers()
    frequency_hz = np.array([100.0], dtype=np.float64)
    k_real = np.array([np.float32(2.0 * np.pi * 100.0 / 343.0)], dtype=np.float32)
    k_imag = np.array([np.float32(k_real[0] * 0.02)], dtype=np.float32)
    observation_points = np.array(
        [
            [0.02, 0.01, 0.10],
            [-0.02, 0.01, 0.10],
            [0.02, -0.01, 0.10],
            [0.0, 0.0, 0.12],
        ],
        dtype=np.float32,
    )
    full_neumann = np.zeros((1, full_buffers.n_triangles), dtype=np.complex64)
    full_neumann[0, full_buffers.physical_tags_i32 == 1] = 1.0 + 0.0j
    quarter_neumann = np.zeros((1, quarter_buffers.n_triangles), dtype=np.complex64)
    quarter_neumann[0, quarter_buffers.physical_tags_i32 == 1] = 1.0 + 0.0j

    with MetalNativeStandardSession.create_session(
        geometry_buffers=full_buffers,
        work_dir=tmp_path / "native-coupled-ib-full-session",
        session_id="native-coupled-ib-full-test",
        aperture_tag=7,
        velocity_source_tags=[1],
    ) as full_session:
        full_solved = full_session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            full_neumann,
            observation_points,
            k_imag_f32=k_imag,
            operation_id="resident-coupled-ib-full",
            source_tags=[1],
            dense_solve_dtype="float64",
        )[0]

    with MetalNativeStandardSession.create_session(
        geometry_buffers=quarter_buffers,
        work_dir=tmp_path / "native-coupled-ib-quarter-session",
        session_id="native-coupled-ib-quarter-test",
        symmetry_plane="yz+xz",
        aperture_tag=7,
        velocity_source_tags=[1],
        check_open_edges=False,
    ) as quarter_session:
        quarter_solved = quarter_session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            quarter_neumann,
            observation_points,
            k_imag_f32=k_imag,
            operation_id="resident-coupled-ib-quarter",
            source_tags=[1],
            dense_solve_dtype="float64",
        )[0]

    full_field = np.fromfile(
        full_solved.field_real_f32,
        dtype="<f4",
    ) + 1j * np.fromfile(full_solved.field_imag_f32, dtype="<f4")
    quarter_field = np.fromfile(
        quarter_solved.field_real_f32,
        dtype="<f4",
    ) + 1j * np.fromfile(quarter_solved.field_imag_f32, dtype="<f4")
    relative_error = np.max(np.abs(full_field - quarter_field)) / np.max(
        np.abs(full_field)
    )

    assert full_solved.lapack_info == 0
    assert quarter_solved.lapack_info == 0
    assert quarter_solved.diagnostics["symmetry_plane"] == "yz+xz"
    assert quarter_solved.diagnostics["coupled_ib"] is True
    assert relative_error < 5.0e-4


def test_native_executable_field_evaluation_on_tiny_mesh(tmp_path):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-field-session",
        session_id="native-field-test",
    ) as session:
        field = session.evaluate_standard_exterior(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.5 + 0.1j, 0.25 + 0.0j, 0.1 - 0.2j], dtype=np.complex64),
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            batch_id="horizontal",
            operation_id="native-field-eval",
        )

    result = json.loads(
        (
            tmp_path
            / "native-field-session"
            / "native-field-eval"
            / "field-result.json"
        ).read_text(encoding="utf-8")
    )
    pressure = np.fromfile(field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        field.pressure_imag_f32,
        dtype="<f4",
    )

    assert result["implementation"] == "swift_native_reference_regular_field"
    assert result["field_mode"] == "reference"
    assert result["field_seconds"] > 0.0
    assert field.shape == (2,)
    assert np.all(np.isfinite(pressure))
    assert np.linalg.norm(pressure) > 0.0


def test_native_executable_optimized_field_matches_reference_on_tiny_mesh(
    tmp_path,
    monkeypatch,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "parity")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_THREADS_PER_GROUP", "32")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_THREADS_PER_GROUP", "64")
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-field-parity-session",
        session_id="native-field-parity-test",
    ) as session:
        field = session.evaluate_standard_exterior(
            100.0,
            1.8318326,
            np.array(
                [1.0 + 0.0j, 0.5 + 0.1j, 0.25 + 0.0j, 0.1 - 0.2j],
                dtype=np.complex64,
            ),
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
            np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            batch_id="horizontal",
            operation_id="native-field-parity",
        )

    result = json.loads(
        (
            tmp_path
            / "native-field-parity-session"
            / "native-field-parity"
            / "field-result.json"
        ).read_text(encoding="utf-8")
    )
    pressure = np.fromfile(field.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        field.pressure_imag_f32,
        dtype="<f4",
    )

    assert result["implementation"] == "swift_native_metal_regular_field"
    assert result["field_mode"] == "parity"
    assert result["reference_parity"]["field_relative_l2"] < 1.0e-4
    assert result["reference_parity"]["tolerance"] == 1.0e-4
    assert result["metal_dispatch"]["field"]["env"] == (
        "HORNLAB_METAL_BEM_NATIVE_FIELD_THREADS_PER_GROUP"
    )
    assert result["metal_dispatch"]["field"]["requested_threads_per_threadgroup"] == 64
    assert result["metal_dispatch"]["field"]["threads_per_threadgroup"] == 64
    assert field.shape == (2,)
    assert np.all(np.isfinite(pressure))
    assert np.linalg.norm(pressure) > 0.0


def test_native_executable_resident_batch_matches_single_field(
    tmp_path,
    monkeypatch,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    pressure = np.array(
        [1.0 + 0.0j, 0.5 + 0.1j, 0.25 + 0.0j, 0.1 - 0.2j],
        dtype=np.complex64,
    )
    neumann = np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64)
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32)
    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-field-batch-session",
        session_id="native-field-batch-test",
    ) as session:
        single = session.evaluate_standard_exterior(
            100.0,
            1.8318326,
            pressure,
            neumann,
            points,
            batch_id="single",
            operation_id="single-field",
        )
        batch = session.evaluate_standard_exterior_batch(
            np.array([100.0], dtype=np.float64),
            np.array([1.8318326], dtype=np.float32),
            pressure.reshape(1, -1),
            neumann.reshape(1, -1),
            points,
            batch_id="batch",
            operation_id="resident-field-batch",
        )[0]

    single_field = np.fromfile(single.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        single.pressure_imag_f32,
        dtype="<f4",
    )
    batch_field = np.fromfile(batch.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        batch.pressure_imag_f32,
        dtype="<f4",
    )
    result = json.loads(
        (
            tmp_path
            / "native-field-batch-session"
            / "resident-field-batch"
            / "field-batch-result.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (
            tmp_path
            / "native-field-batch-session"
            / "resident-field-batch"
            / "field-batch.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["observation_points"]["shape"] == [3, 2]
    assert "observation_points" not in manifest["cases"][0]
    assert result["op"] == "evaluate_standard_exterior_batch_result"
    assert result["resident_reuse"]["geometry_buffers"] is True
    assert result["resident_reuse"]["field_output_buffers"] is True
    assert np.linalg.norm(batch_field - single_field) / np.linalg.norm(single_field) < 1e-5


@pytest.mark.parametrize(
    ("case_name", "mutate", "message"),
    [
        (
            "bad-path",
            lambda manifest: manifest["mesh"]["vertices_f32"].update(
                {"path": "../outside.bin"}
            ),
            "must be relative",
        ),
        (
            "bad-byte-order",
            lambda manifest: manifest["mesh"]["vertices_f32"].update(
                {"byte_order": "big"}
            ),
            "byte_order must be little",
        ),
        (
            "bad-matrix-layout",
            lambda manifest: manifest.update({"matrix_layout": "column_major"}),
            "expected row_major_c matrix layout",
        ),
        (
            "bad-shape",
            lambda manifest: manifest["mesh"]["physical_tags_i32"].update(
                {"shape": [3]}
            ),
            "mesh.physical_tags_i32.shape",
        ),
    ],
)
def test_native_executable_validator_rejects_contract_violations(
    tmp_path,
    case_name,
    mutate,
    message,
):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_geometry_buffers(),
        work_dir=tmp_path / "native-negative-session",
        session_id="native-negative-test",
    ) as session:
        manifest = json.loads(session.info.manifest_path.read_text(encoding="utf-8"))
        mutate(manifest)
        manifest_path = session.info.work_dir / f"{case_name}.json"
        result_path = session.info.work_dir / f"{case_name}-result.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(RuntimeError, match=message):
            validate_session_with_native_helper(manifest_path, result_path)


def test_native_config_defaults_to_packaged_helper_directory():
    config = MetalNativeRuntimeConfig()

    assert config.resolved_backend_dir == Path(native.__file__).resolve().parent


def test_metal_bem_backend_wraps_native_session_without_routing(monkeypatch):
    class FakeInfo:
        session_id = "adapter-test"

    class FakeResult:
        session_id = "adapter-test"
        frequency_hz = 100.0
        matrix_real_f32 = Path("A_re.bin")
        matrix_imag_f32 = Path("A_im.bin")
        rhs_real_f32 = Path("rhs_re.bin")
        rhs_imag_f32 = Path("rhs_im.bin")
        matrix_shape = (4, 4)
        rhs_shape = (4,)
        matrix_layout = "row_major_c"

    class FakeSession:
        info = FakeInfo()
        closed = False

        def validate_contract(self):
            return {"status": "ok"}

        def assemble_standard_neumann(self, frequency_hz, k_real, neumann, **kwargs):
            assert frequency_hz == 100.0
            assert neumann.shape == (2,)
            return FakeResult()

        def evaluate_standard_exterior(self, *args, **kwargs):
            return "field-result"

        def close(self):
            self.closed = True

    fake_session = FakeSession()

    monkeypatch.setattr(
        metal_backend.MetalNativeStandardSession,
        "create_session",
        lambda **kwargs: fake_session,
    )

    context = MetalBemBackend().create_context(geometry_buffers=object())
    try:
        assert isinstance(context, MetalBemContext)
        assert context.session_id == "adapter-test"
        assert context.validate_contract() == {"status": "ok"}

        system = context.assemble_standard_neumann(
            100.0,
            1.8318326,
            np.array([1.0 + 0.0j, 0.0 + 0.5j], dtype=np.complex64),
        )

        assert isinstance(system, DenseBieSystem)
        assert system.matrix_shape == (4, 4)
        assert system.matrix_layout == "row_major_c"
        assert context.evaluate_field_batch(
            100.0,
            1.8318326,
            np.zeros(4, dtype=np.complex64),
            np.zeros(2, dtype=np.complex64),
            np.zeros((3, 1), dtype=np.float32),
        ) == "field-result"
    finally:
        context.close()

    assert fake_session.closed is True


def _solve_robin_surface_pressure(
    session,
    *,
    impedance_sources,
    frequency_hz,
    k_real,
    k_imag,
    operation_id,
):
    neumann = np.array(
        [[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]] * len(frequency_hz),
        dtype=np.complex64,
    )
    observation_points = np.array(
        [[0.0, 0.0, 0.7], [0.2, 0.0, 0.8]],
        dtype=np.float32,
    )
    return session.assemble_solve_evaluate_standard_neumann_batch(
        frequency_hz,
        k_real,
        neumann,
        observation_points,
        k_imag_f32=k_imag,
        impedance_sources=impedance_sources,
        operation_id=operation_id,
        source_tags=[2],
        impedance_source_tag=2,
        write_surface_pressure=True,
    )


def _read_surface_pressure(solved):
    return np.fromfile(solved.pressure_real_f32, dtype="<f4") + 1j * np.fromfile(
        solved.pressure_imag_f32,
        dtype="<f4",
    )


def test_impedance_source_callback_per_frequency_beta(monkeypatch, tmp_path):
    """A per-frequency list of Robin betas must reach the helper as distinct
    per-case payloads: two frequencies with different beta on tag 8 produce
    different surface pressure, and both cases are flagged robin_boundary."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    # Two frequencies. The Robin beta on tag 8 differs between them; the only
    # difference between the two cases is the per-case impedance_sources dict.
    frequency_hz = np.array([172.0, 172.0], dtype=np.float64)
    k_real = np.array(
        [np.float32(2.0 * np.pi * 172.0 / 343.0)] * 2,
        dtype=np.float32,
    )
    k_imag = (k_real * np.float32(0.005)).astype(np.float32)
    per_case_beta = [
        {8: 0.05 + 0.0j},
        {8: 0.20 + 0.0j},
    ]

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_robin_geometry_buffers(),
        work_dir=tmp_path / "native-callback-beta-session",
        session_id="native-callback-beta-test",
    ) as session:
        solved = _solve_robin_surface_pressure(
            session,
            impedance_sources=per_case_beta,
            frequency_hz=frequency_hz,
            k_real=k_real,
            k_imag=k_imag,
            operation_id="resident-callback-beta",
        )

    assert len(solved) == 2
    p0 = _read_surface_pressure(solved[0])
    p1 = _read_surface_pressure(solved[1])
    assert np.all(np.isfinite(p0))
    assert np.all(np.isfinite(p1))
    # Both cases carry a Robin boundary.
    assert solved[0].diagnostics["robin_boundary"] is True
    assert solved[1].diagnostics["robin_boundary"] is True
    # Distinct per-frequency beta -> distinct surface pressure (proves the
    # per-case dict reached the helper rather than a single shared payload).
    assert not np.allclose(p0, p1)


def test_impedance_source_callback_equals_static_dict(monkeypatch, tmp_path):
    """A per-case list of identical dicts must produce bit-for-bit the same
    surface pressure as passing that single dict statically (callback ==
    static-dict equivalence)."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    frequency_hz = np.array([172.0, 250.0], dtype=np.float64)
    k_real = (2.0 * np.pi * frequency_hz / 343.0).astype(np.float32)
    k_imag = (k_real * np.float32(0.005)).astype(np.float32)
    static_beta = {8: 0.05 + 0.0j, 9: 0.02 + 0.01j}

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_robin_geometry_buffers(),
        work_dir=tmp_path / "native-callback-static-session",
        session_id="native-callback-static-test",
    ) as session:
        solved_static = _solve_robin_surface_pressure(
            session,
            impedance_sources=static_beta,
            frequency_hz=frequency_hz,
            k_real=k_real,
            k_imag=k_imag,
            operation_id="resident-static-beta",
        )
        solved_list = _solve_robin_surface_pressure(
            session,
            impedance_sources=[dict(static_beta), dict(static_beta)],
            frequency_hz=frequency_hz,
            k_real=k_real,
            k_imag=k_imag,
            operation_id="resident-list-beta",
        )

    assert len(solved_static) == len(solved_list) == 2
    for case_static, case_list in zip(solved_static, solved_list):
        assert case_static.diagnostics["robin_boundary"] is True
        assert case_list.diagnostics["robin_boundary"] is True
        p_static = _read_surface_pressure(case_static)
        p_list = _read_surface_pressure(case_list)
        np.testing.assert_array_equal(p_static, p_list)


def test_impedance_sources_list_length_must_match_frequencies(tmp_path):
    """A per-case impedance_sources list of the wrong length is rejected
    Python-side in the session method (before the helper subprocess runs)."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    frequency_hz = np.array([172.0, 250.0], dtype=np.float64)
    k_real = (2.0 * np.pi * frequency_hz / 343.0).astype(np.float32)
    neumann = np.array(
        [[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]] * 2,
        dtype=np.complex64,
    )
    observation_points = np.array([[0.0, 0.0, 0.7]], dtype=np.float32)

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_robin_geometry_buffers(),
        work_dir=tmp_path / "native-list-length-session",
        session_id="native-list-length-test",
    ) as session:
        with pytest.raises(ValueError, match="one dict per frequency"):
            session.assemble_solve_evaluate_standard_neumann_batch(
                frequency_hz,
                k_real,
                neumann,
                observation_points,
                impedance_sources=[{8: 0.05 + 0.0j}],  # only 1, need 2
                operation_id="resident-list-length",
                source_tags=[2],
                impedance_source_tag=2,
            )


def test_impedance_sources_mixed_active_inactive_per_case(tmp_path):
    """beta(f) active on some frequencies and absent on others must NOT trip the
    per-case Robin capability handshake (regression: it was batch-global, so a
    no-Robin case falsely raised 'helper predates Robin')."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    frequency_hz = np.array([172.0, 250.0], dtype=np.float64)
    k_real = (2.0 * np.pi * frequency_hz / 343.0).astype(np.float32)
    neumann = np.array(
        [[1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]] * 2,
        dtype=np.complex64,
    )
    observation_points = np.array([[0.0, 0.0, 0.7]], dtype=np.float32)

    with MetalNativeStandardSession.create_session(
        geometry_buffers=_tiny_robin_geometry_buffers(),
        work_dir=tmp_path / "native-mixed-robin-session",
        session_id="native-mixed-robin-test",
    ) as session:
        # case 0: no Robin (empty dict); case 1: Robin on tag 8. Must not raise.
        solved = session.assemble_solve_evaluate_standard_neumann_batch(
            frequency_hz,
            k_real,
            neumann,
            observation_points,
            impedance_sources=[{}, {8: 0.05 + 0.0j}],
            operation_id="resident-mixed-robin",
            source_tags=[2],
            impedance_source_tag=2,
        )

    assert len(solved) == 2
    assert solved[0].diagnostics.get("robin_boundary") is not True
    assert solved[1].diagnostics.get("robin_boundary") is True

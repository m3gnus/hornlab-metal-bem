"""Small, portable Galerkin Burton--Miller reference for prescribed Neumann data.

This is deliberately dense and double precision. It is a numerical oracle for
native work, not a production route. Mesh normals must point out of the body.
P1 pressure and facewise DP0 q follow the package's exp(-i*omega*t) convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BurtonMillerBlocks:
    single_layer: NDArray[np.complex128]  # P1 x DP0
    double_layer: NDArray[np.complex128]  # P1 x P1
    adjoint_double_layer: NDArray[np.complex128]  # P1 x DP0
    hypersingular: NDArray[np.complex128]  # P1 x P1
    mass: NDArray[np.float64]  # P1 x P1
    mass_10: NDArray[np.float64]  # P1 x DP0
    lhs: NDArray[np.complex128]
    rhs_operator: NDArray[np.complex128]
    rhs: NDArray[np.complex128]


def _rule(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not 2 <= order <= 12:
        raise ValueError("quadrature_order must be in [2, 12]")
    z, w = np.polynomial.legendre.leggauss(order)
    z, w = (z + 1) / 2, w / 2
    bary, weights = [], []
    for i, u in enumerate(z):
        for j, v in enumerate(z):
            bary.append([1 - u, u * (1 - v), u * v])
            weights.append(w[i] * w[j] * u)
    return np.asarray(bary), np.asarray(weights)


def _images(symmetry_plane: str | None) -> tuple[NDArray[np.int64], ...]:
    axes = {None: (), "yz": (0,), "xz": (1,), "xy": (2,), "yz+xz": (0, 1)}
    if symmetry_plane not in axes:
        raise ValueError("symmetry_plane must be None, 'yz', 'xz', 'xy', or 'yz+xz'")
    result = [np.ones(3, dtype=np.int64)]
    for axis in axes[symmetry_plane]:
        result += [np.array([*s[:axis], -s[axis], *s[axis + 1:]]) for s in result]
    return tuple(result)


def _geometry(vertices: NDArray[np.float64], triangles: NDArray[np.int64]):
    face = vertices[triangles]
    cross = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
    jac = np.linalg.norm(cross, axis=1)
    if np.any(jac <= 0):
        raise ValueError("mesh contains degenerate triangles")
    normal = cross / jac[:, None]
    grad = np.stack((
        np.cross(normal, face[:, 2] - face[:, 1]),
        np.cross(normal, face[:, 0] - face[:, 2]),
        np.cross(normal, face[:, 1] - face[:, 0]),
    ), axis=1) / jac[:, None, None]
    return face, normal, np.cross(normal[:, None, :], grad), jac


def _green(x: NDArray[np.float64], y: NDArray[np.float64], k: float):
    delta = y - x
    radius = np.linalg.norm(delta, axis=-1)
    if np.any(radius <= 0):
        raise ValueError("quadrature reached a coincident point")
    green = np.exp(1j * k * radius) / (4 * np.pi * radius)
    return delta, radius, green, green * (1j * k - 1 / radius) / radius


def _source_rule_at_x(x, face, bary, weights):
    """Fan a projected interior point into three triangles to resolve 1/r.

    The radial Gauss variable supplies the integrable Duffy factor. This is
    used for coincident and near image pairs as well as edge neighbors.
    """
    a, b, c = face
    n = np.cross(b - a, c - a)
    source_jac = np.linalg.norm(n)
    n /= source_jac
    projected = x - np.dot(x - a, n) * n
    v0, v1, v2 = b - a, c - a, projected - a
    dot00, dot01, dot11 = v0 @ v0, v0 @ v1, v1 @ v1
    dot20, dot21 = v2 @ v0, v2 @ v1
    den = dot00 * dot11 - dot01 * dot01
    u = (dot11 * dot20 - dot01 * dot21) / den
    v = (dot00 * dot21 - dot01 * dot20) / den
    if min(u, v, 1 - u - v) < -1e-10:
        return None
    center_basis = np.array([1 - u - v, u, v])
    nodes, node_weights = [], []
    # Reuse the product Gauss rule on each fan triangle; its first coordinate
    # is radial from the projected point, hence cancels the Green singularity.
    for i, j in ((0, 1), (1, 2), (2, 0)):
        e1, e2 = face[i] - projected, face[j] - projected
        fan_jac = np.linalg.norm(np.cross(e1, e2))
        if fan_jac <= source_jac * 1e-14:
            continue
        for beta, weight in zip(bary, weights):
            source_basis = center_basis * beta[0]
            source_basis = source_basis.copy()
            source_basis[i] += beta[1]
            source_basis[j] += beta[2]
            nodes.append(source_basis)
            node_weights.append(weight * fan_jac)
    return np.asarray(nodes), np.asarray(node_weights)


def _pair_integrals(test_face, source_face, test_normal, source_normal,
                    test_curls, source_curls, test_jac, source_jac,
                    k, bary, weights, near):
    s = np.zeros(3, complex)
    d = np.zeros((3, 3), complex)
    kp = np.zeros(3, complex)
    h = np.zeros((3, 3), complex)
    x_nodes = bary @ test_face
    y_regular = bary @ source_face
    for x_basis, x, wx in zip(bary, x_nodes, weights * test_jac):
        source = _source_rule_at_x(x, source_face, bary, weights) if near else None
        if source is None:
            y_basis, wy = bary, weights * source_jac
            y = y_regular
        else:
            y_basis, wy = source
            y = y_basis @ source_face
        delta, radius, green, grad = _green(x, y, k)
        scalar = wx * wy
        sg = green * scalar
        sd = grad * (delta @ source_normal) * scalar
        sk = -grad * (delta @ test_normal) * scalar
        s += x_basis * np.sum(sg)
        d += np.outer(x_basis, np.sum(sd[:, None] * y_basis, axis=0))
        kp += x_basis * np.sum(sk)
        normal_term = -k * k * np.dot(test_normal, source_normal)
        h += np.outer(
            x_basis, np.sum((normal_term * sg)[:, None] * y_basis, axis=0)
        )
        h += (test_curls @ source_curls.T) * np.sum(sg)
    return s, d, kp, h


def assemble_burton_miller_reference(
    vertices: ArrayLike, triangles: ArrayLike, neumann_dp0: ArrayLike,
    k: float, *, symmetry_plane: str | None = None, quadrature_order: int = 4,
) -> BurtonMillerBlocks:
    """Assemble (M/2-D+iH/k)p = (-S-i(K'+M10/2)/k)q.

    Vertices and face indices describe an outward-oriented triangle surface.
    ``neumann_dp0`` is one complex value per face, or faces by drives. The
    quadrature is intended for small, reasonably shaped meshes only.
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    q = np.asarray(neumann_dp0, dtype=complex)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must be finite (n, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("triangles must be valid (m, 3) indices")
    if q.ndim not in (1, 2) or q.shape[0] != len(triangles) or not np.all(np.isfinite(q)):
        raise ValueError("neumann_dp0 must have one finite row per triangle")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive")
    images = _images(symmetry_plane)
    bary, weights = _rule(quadrature_order)
    face, normal, curls, jac = _geometry(vertices, triangles)
    n, m = len(vertices), len(triangles)
    s = np.zeros((n, m), complex)
    d = np.zeros((n, n), complex)
    kp = np.zeros((n, m), complex)
    h = np.zeros((n, n), complex)
    mass = np.zeros((n, n), float)
    m10 = np.zeros((n, m), float)
    for ti, ids in enumerate(triangles):
        area = jac[ti] / 2
        mass[np.ix_(ids, ids)] += area / 12 * (np.ones((3, 3)) + np.eye(3))
        m10[ids, ti] += area / 3
    for signs in images:
        image_face = face * signs
        image_normal = normal * signs
        image_curls = curls * (np.prod(signs) * signs)
        for ti, test_ids in enumerate(triangles):
            for sj, source_ids in enumerate(triangles):
                separation = np.linalg.norm(face[ti].mean(axis=0) - image_face[sj].mean(axis=0))
                scale = max(np.sqrt(jac[ti]), np.sqrt(jac[sj]))
                pair = _pair_integrals(
                    face[ti], image_face[sj], normal[ti], image_normal[sj],
                    curls[ti], image_curls[sj], jac[ti], jac[sj], k,
                    bary, weights, separation < 1.5 * scale,
                )
                s[test_ids, sj] += pair[0]
                d[np.ix_(test_ids, source_ids)] += pair[1]
                kp[test_ids, sj] += pair[2]
                h[np.ix_(test_ids, source_ids)] += pair[3]
    eta = 1j / k
    lhs = mass / 2 - d + eta * h
    rhs_operator = -s - eta * (kp + m10 / 2)
    return BurtonMillerBlocks(s, d, kp, h, mass, m10, lhs, rhs_operator, rhs_operator @ q)


def evaluate_burton_miller_exterior(vertices: ArrayLike, triangles: ArrayLike,
                                    pressure_p1: ArrayLike, neumann_dp0: ArrayLike,
                                    points_xyz: ArrayLike, k: float, *,
                                    symmetry_plane: str | None = None,
                                    quadrature_order: int = 6) -> NDArray[np.complex128]:
    """Evaluate real-k exterior p = D p - S q, including rigid images."""
    vertices = np.asarray(vertices, float)
    triangles = np.asarray(triangles, np.int64)
    pressure = np.asarray(pressure_p1, complex)
    q = np.asarray(neumann_dp0, complex)
    points = np.asarray(points_xyz, float)
    if pressure.shape != (len(vertices),) or q.shape != (len(triangles),):
        raise ValueError("pressure_p1 and neumann_dp0 trace shapes mismatch")
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_xyz must be finite (n, 3)")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive")
    bary, weights = _rule(quadrature_order)
    face, normal, _, jac = _geometry(vertices, triangles)
    out = np.zeros(len(points), complex)
    for signs in _images(symmetry_plane):
        for j, ids in enumerate(triangles):
            y = bary @ (face[j] * signs)
            ny = normal[j] * signs
            p = bary @ pressure[ids]
            for i, x in enumerate(points):
                delta, radius, green, grad = _green(x, y, k)
                out[i] += jac[j] * np.sum(weights * (grad * (delta @ ny) * p - green * q[j]))
    return out

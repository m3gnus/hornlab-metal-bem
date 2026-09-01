"""Numerical validation of the rigid half-space ground plane.

Three independent checks, in increasing order of independence:

1. Image equivalence. A rigid plane IS the image method, so the ground solve of
   one body must equal a free-field solve of that body plus a mirrored copy of
   it, meshed explicitly as two disjoint closed bodies. This is exact, not
   asymptotic, and it also exercises multi-body exterior coupling.
2. Power bookkeeping. The doubled mesh drives two bodies, so the ground solve
   must report HALF its driven-surface power -- the image is fictitious and
   must not be counted, unlike a mirror-reduced symmetry mesh.
3. Analytic interference. A compact source at height h over a rigid plane obeys
   ``p_ground / p_free = 1 + (R1/R2) exp(i k (R2 - R1))``. The COMPLEX ratio is
   compared, because |1 + e^{iD}| = |1 + e^{-iD}| hides a conjugated kernel.
   Controls assert that a pressure-release image and a conjugated kernel both
   fail the same comparison badly, so a pass is not vacuous.
"""
from __future__ import annotations

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem.mesh import LoadedMesh, make_pure_grid
from hornlab_metal_bem.result import MeshInfo

SPEED_OF_SOUND = 343.0
RADIUS = 0.10
HEIGHT = 0.35
FREQUENCIES = [250.0, 700.0, 1800.0]


def _require_native():
    from hornlab_metal_bem.metal import discover_native_runtime

    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )


def _octasphere(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    vertices = [
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ]
    triangles = [
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
    ]
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            cached = cache.get(key)
            if cached is not None:
                return cached
            va, vb = vertices[a], vertices[b]
            mid = ((va[0] + vb[0]) / 2, (va[1] + vb[1]) / 2, (va[2] + vb[2]) / 2)
            norm = (mid[0] ** 2 + mid[1] ** 2 + mid[2] ** 2) ** 0.5
            vertices.append((mid[0] / norm, mid[1] / norm, mid[2] / norm))
            cache[key] = len(vertices) - 1
            return cache[key]

        nxt = []
        for a, b, c in triangles:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nxt.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        triangles = nxt
    return np.asarray(vertices, dtype=np.float64), np.asarray(
        triangles, dtype=np.int32
    )


def _capped_sphere(radius, centre, subdivisions=2):
    """Sphere with a driven cap (tag 2) toward +X, rigid elsewhere (tag 1)."""
    vertices, triangles = _octasphere(subdivisions)
    vertices = vertices * radius + np.asarray(centre, dtype=np.float64)
    local = (vertices - np.asarray(centre, dtype=np.float64)) / radius
    centroids = local[triangles].mean(axis=1)
    tags = np.ones(triangles.shape[0], dtype=np.int32)
    tags[centroids[:, 0] > 0.55] = 2
    assert np.count_nonzero(tags == 2) > 0
    return vertices, triangles, tags


def _mirror_z(vertices, triangles):
    """Reflect through Z=0; reflection reverses orientation, so unwind it."""
    mirrored = vertices.copy()
    mirrored[:, 2] *= -1.0
    return mirrored, triangles[:, [0, 2, 1]].copy()


def _loaded(vertices, triangles, tags, groups) -> LoadedMesh:
    return LoadedMesh(
        grid=make_pure_grid(vertices, triangles),
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles.shape[0],
            physical_groups=groups,
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
    )


def _frame(origin_z: float) -> metal_bem.ObservationFrame:
    return metal_bem.ObservationFrame(
        axis=np.array([1.0, 0.0, 0.0]),
        origin=np.array([0.0, 0.0, origin_z]),
        u=np.array([0.0, 1.0, 0.0]),
        v=np.array([0.0, 0.0, 1.0]),
        mouth_center=np.array([RADIUS, 0.0, origin_z]),
        source_center=np.array([RADIUS, 0.0, origin_z]),
    )


def _single_and_doubled():
    vertices, triangles, tags = _capped_sphere(RADIUS, (0.0, 0.0, HEIGHT))
    single = _loaded(vertices, triangles, tags, {1: "rigid", 2: "cap"})
    mirrored, mirrored_tris = _mirror_z(vertices, triangles)
    doubled = _loaded(
        np.vstack([vertices, mirrored]),
        np.vstack([triangles, mirrored_tris + vertices.shape[0]]),
        np.concatenate([tags, tags]),
        {1: "rigid", 2: "cap"},
    )
    return single, doubled


_PROBE_POINTS = np.array(
    [
        [2.0, 0.0, HEIGHT], [1.8, 0.0, 1.0], [1.2, 0.9, 0.30],
        [0.0, 2.0, 0.60], [-1.5, 0.0, 1.4], [0.4, 0.0, 2.0],
        [3.0, 0.0, 0.05], [0.6, -1.1, 0.9],
    ],
    dtype=np.float64,
)


def _probe_config(**overrides) -> metal_bem.SolveConfig:
    observation = metal_bem.ObservationConfig(
        planes=["probe"],
        angle_count=_PROBE_POINTS.shape[0],
        custom_points={"probe": _PROBE_POINTS},
    )
    return metal_bem.native_config(
        observation=observation, frame_override=_frame(HEIGHT), **overrides
    )


@pytest.mark.slow
def test_ground_plane_equals_an_explicitly_mirrored_two_body_solve():
    _require_native()
    single, doubled = _single_and_doubled()

    ground = metal_bem.solve_frequencies(
        single, FREQUENCIES, _probe_config(ground_plane="xy")
    )
    image = metal_bem.solve_frequencies(doubled, FREQUENCIES, _probe_config())

    np.testing.assert_allclose(
        ground.pressure_complex,
        image.pressure_complex,
        rtol=2.0e-4,
        atol=1.0e-12,
    )


@pytest.mark.slow
def test_ground_plane_actually_changes_the_field():
    """Guard against a silently inert image: a no-op would read 0 dB here."""
    _require_native()
    single, _doubled = _single_and_doubled()

    ground = metal_bem.solve_frequencies(
        single, FREQUENCIES, _probe_config(ground_plane="xy")
    )
    free = metal_bem.solve_frequencies(single, FREQUENCIES, _probe_config())

    delta_db = 20.0 * np.log10(
        np.abs(ground.pressure_complex) / np.abs(free.pressure_complex)
    )
    # The probe set straddles a full interference pattern, so it must show both
    # reinforcement and cancellation, not a uniform offset.
    assert delta_db.max() > 4.0
    assert delta_db.min() < -2.0


@pytest.mark.slow
def test_ground_plane_counts_the_driven_surface_once():
    """A mirror-reduced mesh is N copies of a body; a grounded mesh is one."""
    _require_native()
    single, doubled = _single_and_doubled()

    ground = metal_bem.solve_frequencies(
        single, FREQUENCIES, _probe_config(ground_plane="xy")
    )
    image = metal_bem.solve_frequencies(doubled, FREQUENCIES, _probe_config())

    assert ground.radiated_power_surface_w is not None
    assert image.radiated_power_surface_w is not None
    # The doubled mesh drives two bodies. Applying the symmetry path's x2 here
    # would make these equal instead of halved.
    np.testing.assert_allclose(
        ground.radiated_power_surface_w,
        image.radiated_power_surface_w / 2.0,
        rtol=1.0e-4,
    )


@pytest.mark.slow
def test_ground_plane_matches_the_analytic_rigid_image_of_a_monopole():
    _require_native()
    radius, height = 0.02, 0.60          # ka <= 0.3, h/a = 30
    frequencies = [200.0, 400.0, 800.0]

    vertices, triangles = _octasphere(3)
    vertices = vertices * radius + np.array([0.0, 0.0, height])
    tags = np.full(triangles.shape[0], 2, dtype=np.int32)
    mesh = _loaded(vertices, triangles, tags, {2: "pulsating"})

    theta = np.deg2rad(np.linspace(2.0, 88.0, 25))
    distance = 6.0
    points = np.stack(
        [
            distance * np.cos(theta),
            np.zeros_like(theta),
            height + distance * np.sin(theta),
        ],
        axis=1,
    )
    observation = metal_bem.ObservationConfig(
        planes=["arc"], angle_count=points.shape[0],
        custom_points={"arc": points},
    )
    config = dict(observation=observation, frame_override=_frame(height))

    ground = metal_bem.solve_frequencies(
        mesh, frequencies, metal_bem.native_config(ground_plane="xy", **config)
    )
    free = metal_bem.solve_frequencies(
        mesh, frequencies, metal_bem.native_config(**config)
    )

    r_source = np.linalg.norm(points - np.array([0.0, 0.0, height]), axis=1)
    r_image = np.linalg.norm(points - np.array([0.0, 0.0, -height]), axis=1)

    for index, frequency in enumerate(frequencies):
        k = 2.0 * np.pi * frequency / SPEED_OF_SOUND
        phase = np.exp(1j * k * (r_image - r_source))
        analytic = 1.0 + (r_source / r_image) * phase
        solved = (
            ground.pressure_complex[index, 0, :]
            / free.pressure_complex[index, 0, :]
        )
        assert np.max(np.abs(solved - analytic) / np.abs(analytic)) < 1.0e-2

        # Controls: the same data must NOT match a pressure-release image or a
        # conjugated kernel, or the test above would pass on anything.
        release = 1.0 - (r_source / r_image) * phase
        conjugated = 1.0 + (r_source / r_image) * np.conj(phase)
        assert np.max(np.abs(solved - release) / np.abs(release)) > 0.5
        assert np.max(np.abs(solved - conjugated) / np.abs(conjugated)) > 0.5


@pytest.mark.slow
def test_ground_plane_wall_reflects_about_its_own_axis():
    """The 'yz' wall at X=0 must behave like 'xy' applied to a rotated body."""
    _require_native()
    vertices, triangles, tags = _capped_sphere(RADIUS, (0.0, 0.0, HEIGHT))
    upright = _loaded(vertices, triangles, tags, {1: "rigid", 2: "cap"})

    # Rotate the whole problem by mapping (x, y, z) -> (z, y, x): the floor at
    # Z=0 becomes a wall at X=0, and observation points move the same way.
    swap = [2, 1, 0]
    rotated = _loaded(
        vertices[:, swap].copy(),
        triangles[:, [0, 2, 1]].copy(),      # the swap is a reflection
        tags,
        {1: "rigid", 2: "cap"},
    )

    floor_points = _PROBE_POINTS
    wall_points = floor_points[:, swap].copy()

    def run(mesh, points, plane, axis, origin):
        observation = metal_bem.ObservationConfig(
            planes=["probe"], angle_count=points.shape[0],
            custom_points={"probe": points},
        )
        frame = metal_bem.ObservationFrame(
            axis=axis, origin=origin,
            u=np.array([0.0, 1.0, 0.0]), v=np.cross(axis, [0.0, 1.0, 0.0]),
            mouth_center=origin, source_center=origin,
        )
        return metal_bem.solve_frequencies(
            mesh, FREQUENCIES,
            metal_bem.native_config(
                ground_plane=plane, observation=observation,
                frame_override=frame,
            ),
        )

    floor = run(
        upright, floor_points, "xy",
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, HEIGHT]),
    )
    wall = run(
        rotated, wall_points, "yz",
        np.array([0.0, 0.0, 1.0]), np.array([HEIGHT, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        floor.pressure_complex, wall.pressure_complex, rtol=2.0e-4, atol=1.0e-12
    )

"""Several bodies in one exterior domain, placed by combine_bodies().

The solver needed no change for this -- a boundary integral operator pairs
elements, not connected components -- so what these tests establish is that the
placement bookkeeping is right and that the bodies genuinely couple:

* placement algebra: tag collisions refuse, improper rotations refuse, and a
  rigid motion of the WHOLE scene moves the field with it and nothing else;
* coupling: two cabinets close together do not radiate the sum of what they
  radiate alone, and the difference decays as they separate. A solver that
  ignored inter-body scattering would show zero difference at every spacing.
"""
from __future__ import annotations

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem.mesh import LoadedMesh, MeshError, make_pure_grid
from hornlab_metal_bem.result import MeshInfo

from test_ground_plane import _capped_sphere, _loaded, _require_native

RADIUS = 0.10
FREQUENCIES = [400.0, 1200.0]


def _body(cap_tag: int = 2) -> LoadedMesh:
    vertices, triangles, tags = _capped_sphere(RADIUS, (0.0, 0.0, 0.0))
    return _loaded(vertices, triangles, tags, {1: "rigid", cap_tag: "cap"})


def _is_closed_two_manifold(triangles: np.ndarray) -> bool:
    edges = np.sort(
        np.concatenate(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
        ),
        axis=1,
    )
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(counts.size and np.all(counts == 2))


def test_combined_bodies_stay_separate_closed_surfaces():
    combined = metal_bem.combine_bodies(
        [
            metal_bem.BodyPlacement(_body(), name="left"),
            metal_bem.BodyPlacement(
                _body(), translation_m=(0.5, 0.0, 0.0),
                tag_map={1: 11, 2: 12}, name="right",
            ),
        ]
    )
    triangles = np.asarray(combined.mesh.grid.elements).T
    single = np.asarray(_body().grid.elements).T
    assert triangles.shape[0] == 2 * single.shape[0]
    assert _is_closed_two_manifold(triangles)
    assert combined.tags_for(0) == (1, 2)
    assert combined.tags_for(1) == (11, 12)
    assert set(np.unique(combined.body_ids)) == {0, 1}
    assert combined.body_names == ("left", "right")
    assert combined.mesh.info.physical_groups[12] == "right:cap"


def test_colliding_tags_refuse_rather_than_drive_the_wrong_body():
    with pytest.raises(MeshError, match="already taken"):
        metal_bem.combine_bodies(
            [
                metal_bem.BodyPlacement(_body(), name="left"),
                metal_bem.BodyPlacement(
                    _body(), translation_m=(0.5, 0.0, 0.0), name="right"
                ),
            ]
        )


def test_reflections_refuse_because_they_invert_winding():
    flip = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(MeshError, match="reflection"):
        metal_bem.combine_bodies(
            [metal_bem.BodyPlacement(_body(), rotation=flip, name="mirrored")]
        )


def test_non_orthonormal_rotation_refuses():
    with pytest.raises(MeshError, match="orthonormal"):
        metal_bem.combine_bodies(
            [metal_bem.BodyPlacement(_body(), rotation=np.eye(3) * 1.5)]
        )


def test_tag_map_must_name_tags_the_body_actually_uses():
    with pytest.raises(MeshError, match="does not use"):
        metal_bem.combine_bodies(
            [metal_bem.BodyPlacement(_body(), tag_map={7: 9})]
        )


def test_rotation_matrix_is_a_proper_rotation():
    matrix = metal_bem.rotation_matrix(np.array([0.0, 0.0, 1.0]), 90.0)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)
    np.testing.assert_allclose(
        matrix @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12
    )


def _two_body_scene(separation_m: float):
    return metal_bem.combine_bodies(
        [
            metal_bem.BodyPlacement(
                _body(), translation_m=(0.0, -separation_m / 2.0, 0.0),
                name="left",
            ),
            metal_bem.BodyPlacement(
                _body(), translation_m=(0.0, separation_m / 2.0, 0.0),
                tag_map={1: 11, 2: 12}, name="right",
            ),
        ]
    )


def _probe(points: np.ndarray) -> metal_bem.SolveConfig:
    observation = metal_bem.ObservationConfig(
        planes=["probe"], angle_count=points.shape[0],
        custom_points={"probe": points},
    )
    frame = metal_bem.ObservationFrame(
        axis=np.array([1.0, 0.0, 0.0]), origin=np.zeros(3),
        u=np.array([0.0, 1.0, 0.0]), v=np.array([0.0, 0.0, 1.0]),
        mouth_center=np.zeros(3), source_center=np.zeros(3),
    )
    return metal_bem.native_config(observation=observation, frame_override=frame)


_POINTS = np.array(
    [[3.0, 0.0, 0.0], [2.5, 1.5, 0.0], [2.0, 0.0, 2.0], [0.0, 3.0, 0.5]]
)


@pytest.mark.slow
def test_bodies_couple_and_the_coupling_decays_with_separation():
    _require_native()
    deviations = []
    for separation in (0.28, 3.0):
        scene = _two_body_scene(separation)
        config = _probe(_POINTS)
        both = metal_bem.solve_frequencies(
            scene.mesh, FREQUENCIES,
            metal_bem.native_config(
                velocity_sources={2: 1.0, 12: 1.0, 1: 0.0, 11: 0.0},
                observation=config.observation,
                frame_override=config.frame_override,
            ),
        ).pressure_complex

        alone = []
        for driven, quiet in ((2, 12), (12, 2)):
            alone.append(
                metal_bem.solve_frequencies(
                    scene.mesh, FREQUENCIES,
                    metal_bem.native_config(
                        velocity_sources={driven: 1.0, quiet: 0.0,
                                          1: 0.0, 11: 0.0},
                        observation=config.observation,
                        frame_override=config.frame_override,
                    ),
                ).pressure_complex
            )
        # Superposition of the two drives on the SAME two-body mesh is exact
        # (the BEM is linear in the drive), so it is not the control we want.
        # The control is each body solved with its neighbour absent.
        single = metal_bem.combine_bodies(
            [metal_bem.BodyPlacement(
                _body(), translation_m=(0.0, -separation / 2.0, 0.0))]
        )
        isolated = metal_bem.solve_frequencies(
            single.mesh, FREQUENCIES,
            metal_bem.native_config(
                velocity_sources={2: 1.0, 1: 0.0},
                observation=config.observation,
                frame_override=config.frame_override,
            ),
        ).pressure_complex
        deviations.append(
            float(np.max(np.abs(alone[0] - isolated) / np.abs(isolated)))
        )
        # Linearity of the drive is worth pinning while the data is here.
        np.testing.assert_allclose(
            both, alone[0] + alone[1], rtol=3.0e-4, atol=1.0e-12
        )

    close, far = deviations
    assert close > 0.02, (
        "neighbouring bodies must scatter off each other; "
        f"saw only {close:.3e} relative change"
    )
    assert far < close / 3.0, (
        f"coupling must weaken with separation; {far:.3e} vs {close:.3e}"
    )


@pytest.mark.slow
def test_rigidly_moving_the_whole_scene_moves_the_field_with_it():
    """Rotation and translation are placement, not physics."""
    _require_native()
    scene = _two_body_scene(0.4)
    rotation = metal_bem.rotation_matrix(np.array([0.0, 0.0, 1.0]), 37.0)
    shift = np.array([0.13, -0.27, 0.41])
    moved = metal_bem.combine_bodies(
        [
            metal_bem.BodyPlacement(
                _body(),
                rotation=rotation,
                translation_m=tuple(rotation @ np.array([0.0, -0.2, 0.0]) + shift),
                name="left",
            ),
            metal_bem.BodyPlacement(
                _body(),
                rotation=rotation,
                translation_m=tuple(rotation @ np.array([0.0, 0.2, 0.0]) + shift),
                tag_map={1: 11, 2: 12},
                name="right",
            ),
        ]
    )

    sources = {2: 1.0, 12: 1.0, 1: 0.0, 11: 0.0}
    base_points = _POINTS
    moved_points = base_points @ rotation.T + shift

    def run(mesh, points, axis, origin):
        observation = metal_bem.ObservationConfig(
            planes=["probe"], angle_count=points.shape[0],
            custom_points={"probe": points},
        )
        frame = metal_bem.ObservationFrame(
            axis=axis, origin=origin,
            u=np.array([0.0, 1.0, 0.0]), v=np.array([0.0, 0.0, 1.0]),
            mouth_center=origin, source_center=origin,
        )
        return metal_bem.solve_frequencies(
            mesh, FREQUENCIES,
            metal_bem.native_config(
                velocity_sources=sources, observation=observation,
                frame_override=frame,
            ),
        ).pressure_complex

    at_origin = run(scene.mesh, base_points, np.array([1.0, 0.0, 0.0]), np.zeros(3))
    displaced = run(moved.mesh, moved_points, rotation @ np.array([1.0, 0.0, 0.0]), shift)
    np.testing.assert_allclose(at_origin, displaced, rtol=3.0e-4, atol=1.0e-12)

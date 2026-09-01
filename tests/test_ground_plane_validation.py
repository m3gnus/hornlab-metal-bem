"""Config and geometry contracts for the rigid half-space ground plane.

These are the checks that distinguish a ground plane from a mirror-reduced
symmetry plane: the mesh is the complete body, so it may float clear of the
plane, but it must stay on one side of it and must not put a face flat on it.
No native helper is needed for any of this.
"""
from __future__ import annotations

import numpy as np
import pytest

from hornlab_metal_bem.config import (
    GROUND_PLANE_NORMAL_AXIS,
    NATIVE_GROUND_PLANES,
    SolveConfig,
)
from hornlab_metal_bem.metal.geometry import (
    MetalGeometryError,
    build_metal_geometry_buffers,
    validate_native_ground_plane,
)
from hornlab_metal_bem.mesh import make_pure_function_spaces, make_pure_grid


def _buffers(vertices: np.ndarray, triangles: np.ndarray):
    grid = make_pure_grid(vertices, triangles)
    p1_space, dp0_space = make_pure_function_spaces(grid)
    tags = np.ones(triangles.shape[0], dtype=np.int32)
    buffers = build_metal_geometry_buffers(
        grid, tags, p1_space, dp0_space
    )
    return buffers


def _tetra(offset_z: float) -> tuple[np.ndarray, np.ndarray]:
    """A closed tetrahedron lifted to sit with its lowest vertex at offset_z."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    vertices[:, 2] += offset_z
    triangles = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32
    )
    return vertices, triangles


def _tetra_apex_down(offset_z: float) -> tuple[np.ndarray, np.ndarray]:
    """The same tetrahedron inverted, so only its apex reaches the low Z."""
    vertices, triangles = _tetra(0.0)
    vertices[:, 2] = 1.0 - vertices[:, 2] + offset_z
    return vertices, triangles[:, [0, 2, 1]].copy()


def test_ground_plane_defaults_off_and_changes_nothing():
    config = SolveConfig()
    assert config.ground_plane is None
    assert config.ground_plane_min_clearance_m == 0.0


@pytest.mark.parametrize("plane", NATIVE_GROUND_PLANES)
def test_every_ground_plane_names_a_distinct_normal_axis(plane):
    assert plane in GROUND_PLANE_NORMAL_AXIS
    assert len(set(GROUND_PLANE_NORMAL_AXIS.values())) == 3


def test_ground_plane_rejects_unknown_plane():
    with pytest.raises(ValueError, match="ground_plane must be None"):
        SolveConfig(ground_plane="xyz")


def test_ground_plane_refuses_to_compose_with_native_symmetry():
    with pytest.raises(ValueError, match="does not yet compose"):
        SolveConfig(ground_plane="xy", native_symmetry_plane="yz")


def test_ground_plane_refuses_to_compose_with_coupled_ib_aperture():
    with pytest.raises(ValueError, match="does not compose"):
        SolveConfig(ground_plane="xy", aperture_tag=3)


def test_ground_plane_clearance_must_be_non_negative():
    with pytest.raises(ValueError, match="min_clearance_m"):
        SolveConfig(ground_plane="xy", ground_plane_min_clearance_m=-1.0)


def test_floating_body_is_accepted_though_it_never_touches_the_plane():
    """The contract a symmetry plane cannot express: no vertex on the plane."""
    vertices, triangles = _tetra(offset_z=0.5)
    assert np.min(vertices[:, 2]) > 0.0
    assert validate_native_ground_plane(_buffers(vertices, triangles), "xy") == "xy"


def test_body_touching_the_plane_at_one_vertex_is_accepted():
    """Contact is legal; only a face lying IN the plane is not."""
    vertices, triangles = _tetra_apex_down(offset_z=0.0)
    on_plane = np.count_nonzero(np.abs(vertices[:, 2]) < 1e-12)
    assert on_plane == 1
    assert validate_native_ground_plane(_buffers(vertices, triangles), "xy") == "xy"


def test_body_crossing_the_plane_is_rejected():
    vertices, triangles = _tetra(offset_z=-0.25)
    with pytest.raises(MetalGeometryError, match="must lie at Z >= 0"):
        validate_native_ground_plane(_buffers(vertices, triangles), "xy")


def test_face_lying_flat_on_the_plane_is_rejected():
    """A flat contact face coincides with its own image; that is singular."""
    vertices, triangles = _tetra(offset_z=0.0)
    # Vertices 0, 1, 2 are all at Z=0, so triangle [0, 2, 1] lies on the plane.
    with pytest.raises(MetalGeometryError, match="lies flat on it"):
        validate_native_ground_plane(_buffers(vertices, triangles), "xy")


def test_requested_clearance_is_enforced():
    vertices, triangles = _tetra(offset_z=0.05)
    buffers = _buffers(vertices, triangles)
    assert validate_native_ground_plane(buffers, "xy", min_clearance_m=0.01)
    with pytest.raises(MetalGeometryError, match="clearance"):
        validate_native_ground_plane(buffers, "xy", min_clearance_m=0.10)


def test_wall_planes_check_their_own_axis():
    """A body legal above Z=0 can still be illegal beside a wall at X=0."""
    vertices, triangles = _tetra(offset_z=0.5)
    vertices[:, 0] -= 0.5
    buffers = _buffers(vertices, triangles)
    assert validate_native_ground_plane(buffers, "xy") == "xy"
    with pytest.raises(MetalGeometryError, match="must lie at X >= 0"):
        validate_native_ground_plane(buffers, "yz")


def test_disabled_ground_plane_validates_nothing():
    vertices, triangles = _tetra(offset_z=-10.0)
    assert validate_native_ground_plane(_buffers(vertices, triangles), None) is None

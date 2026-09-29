"""Explicit per-source piston axes (``SolveConfig.source_axes``).

Contract: each axial source tag moves along its own axis, per-face scale
``n_hat . axis`` with no sign vote, no symmetry projection and no dependence on
the observation frame. ``source_axes=None`` is the unchanged legacy path.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem.bie import _build_source_face_scale
from hornlab_metal_bem.config import AxialProfile, SolveConfig, SourceMotion
from hornlab_metal_bem.observation import ObservationFrame

Z = np.array([0.0, 0.0, 1.0])


def _grid_from_normals(normals) -> SimpleNamespace:
    """One unit-area-half triangle per normal with cross(P1-P0, P2-P0) == n."""
    verts = []
    elems = []
    for i, n in enumerate(normals):
        n = np.asarray(n, dtype=np.float64)
        n = n / np.linalg.norm(n)
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)  # e1 x e2 == n
        base = 3 * i
        verts += [np.zeros(3), e1, e2]
        elems.append([base, base + 1, base + 2])
    return SimpleNamespace(
        vertices=np.array(verts).T, elements=np.array(elems, dtype=np.int32).T
    )


def _scale(normals, tags, config, frame_axis=Z):
    grid = _grid_from_normals(normals)
    return _build_source_face_scale(
        grid, np.asarray(tags, dtype=np.int32), config, frame_axis, np.zeros(3)
    )


def _cfg(sources, axes=None, **kw):
    return SolveConfig(
        velocity_sources=dict(sources),
        source_motion=SourceMotion.AXIAL,
        source_axes=axes,
        **kw,
    )


# 1 -----------------------------------------------------------------------
def test_flat_disc_sign_follows_axis_and_legacy_votes():
    normals = [Z, Z]
    up = _scale(normals, [2, 2], _cfg({2: 1.0}, {2: (0, 0, 1)}))
    down = _scale(normals, [2, 2], _cfg({2: 1.0}, {2: (0, 0, -1)}))
    np.testing.assert_allclose(up, [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(down, [-1.0, -1.0], atol=1e-12)
    legacy = _scale(normals, [2, 2], _cfg({2: 1.0}), frame_axis=-Z)
    np.testing.assert_allclose(legacy, [1.0, 1.0], atol=1e-12)  # sign vote


# 2 -----------------------------------------------------------------------
def test_opposed_faces_keep_signs_and_grouping_is_irrelevant():
    normals = [Z, -Z]
    one = _scale(normals, [2, 2], _cfg({2: 1.0}, {2: (0, 0, 1)}))
    np.testing.assert_allclose(one, [1.0, -1.0], atol=1e-12)
    split_cfg = _cfg({2: 1.0, 3: 1.0}, {2: (0, 0, 1), 3: (0, 0, 1)})
    split = _scale(normals, [2, 3], split_cfg)
    np.testing.assert_allclose(split, [1.0, -1.0], atol=1e-12)
    legacy = _scale(normals, [2, 3], _cfg({2: 1.0, 3: 1.0}))
    np.testing.assert_allclose(legacy, [1.0, 1.0], atol=1e-12)  # per-tag vote


# 3 -----------------------------------------------------------------------
def test_projection_cosines():
    n60 = [np.sin(np.deg2rad(60)), 0.0, np.cos(np.deg2rad(60))]
    scale = _scale([n60, [1.0, 0.0, 0.0]], [2, 2], _cfg({2: 1.0}, {2: (0, 0, 1)}))
    np.testing.assert_allclose(scale[0], 0.5, atol=1e-12)
    np.testing.assert_allclose(scale[1], 0.0, atol=1e-12)


# 4 -----------------------------------------------------------------------
def test_axis_is_normalized():
    n = [np.sin(0.4), 0.0, np.cos(0.4)]
    a = _scale([n], [2], _cfg({2: 1.0}, {2: (0, 0, 2)}))
    b = _scale([n], [2], _cfg({2: 1.0}, {2: (0, 0, 1)}))
    np.testing.assert_array_equal(a, b)


# 5 -----------------------------------------------------------------------
@pytest.mark.parametrize("bad", [(0, 0, 0), (0, 0, float("nan"))])
def test_degenerate_axis_raises_explicit(bad):
    with pytest.raises(ValueError, match="tag 2"):
        _cfg({2: 1.0}, {2: bad})


@pytest.mark.parametrize("bad", [np.zeros(3), np.array([np.nan, 0.0, 1.0])])
def test_degenerate_frame_axis_raises_legacy(bad):
    with pytest.raises(ValueError, match="axis"):
        _scale([Z], [2], _cfg({2: 1.0}), frame_axis=bad)
    # profile route (AxialProfile on a NORMAL fallback config)
    profile_cfg = SolveConfig(
        velocity_sources={2: 1.0}, source_velocity_profiles={2: AxialProfile()}
    )
    with pytest.raises(ValueError, match="axis"):
        _scale([Z], [2], profile_cfg, frame_axis=bad)


def test_degenerate_axis_with_no_faces_keeps_old_behaviour():
    assert _scale([Z], [5], _cfg({2: 1.0}), frame_axis=np.zeros(3)) is None


# 6 -----------------------------------------------------------------------
def test_config_validation():
    ok = (0, 0, 1)
    with pytest.raises(ValueError, match="axial"):
        SolveConfig(velocity_sources={2: 1.0}, source_axes={2: ok})
    with pytest.raises(ValueError, match="tag 7"):
        _cfg({2: 1.0}, {2: ok, 7: ok})
    with pytest.raises(ValueError, match=r"\[3\]"):
        _cfg({2: 1.0, 3: 1.0}, {2: ok})
    with pytest.raises(ValueError, match="tag 2"):
        _cfg({2: 1.0}, {2: (0, 1)})
    with pytest.raises(ValueError, match="tag 2"):
        _cfg({2: 1.0}, {2: (0, 0, float("inf"))})
    with pytest.raises(ValueError, match="tag 2"):
        _cfg({2: 1.0}, {2: (0, 0, 1e-13)})
    # AxialProfile on one tag makes source_axes legal without source_motion.
    cfg = SolveConfig(
        velocity_sources={2: 1.0},
        source_velocity_profiles={2: AxialProfile()},
        source_axes={2: ok},
    )
    assert cfg.source_axes == {2: ok}
    # A non-axial tag may not carry an axis (never silently ignored).
    with pytest.raises(ValueError, match="tag 3"):
        SolveConfig(
            velocity_sources={2: 1.0, 3: 1.0},
            source_velocity_profiles={2: AxialProfile()},
            source_axes={2: ok, 3: ok},
        )


# 7 -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "plane,bad,good",
    [
        ("yz", (1, 0, 1), (0, 0, 1)),
        ("xz", (0, 1, 1), (1, 0, 1)),
        ("xy", (1, 0, 1), (1, 1, 0)),
        ("yz+xz", (0, 1, 1), (0, 0, 1)),
    ],
)
def test_symmetry_subspace(plane, bad, good):
    with pytest.raises(ValueError, match="symmetry subspace"):
        _cfg({2: 1.0}, {2: bad}, native_symmetry_plane=plane)
    _cfg({2: 1.0}, {2: good}, native_symmetry_plane=plane)


def test_symmetry_check_also_in_builder():
    cfg = _cfg({2: 1.0}, {2: (0, 0, 1)}, native_symmetry_plane="yz")
    bad = replace(cfg, native_symmetry_plane=None)
    object.__setattr__(bad, "source_axes", {2: (1, 0, 1)})
    object.__setattr__(bad, "native_symmetry_plane", "yz")
    with pytest.raises(ValueError, match="symmetry subspace"):
        _scale([Z], [2], bad)


def test_multi_source_axes_filter_and_missing():
    from hornlab_metal_bem.sweep import _source_axes_for_tags

    cfg = _cfg({2: 1.0, 3: 1.0}, {2: (0, 0, 1), 3: (0, 0, -1)})
    assert _source_axes_for_tags(cfg, {2: 1.0}) == {2: (0, 0, 1)}
    assert _source_axes_for_tags(replace(cfg, source_axes=None), [2]) is None
    with pytest.raises(ValueError, match="no axis"):
        _source_axes_for_tags(cfg, [9])


# Solve-level -------------------------------------------------------------
def _require_native():
    from hornlab_metal_bem.metal import discover_native_runtime

    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip("Swift/Metal native helper unavailable")


_FREQS = [180.0, 260.0]


# GPU float32 solves are not bit-reproducible run to run (about 1e-5 relative),
# so solve-level comparisons use the repository's usual 2e-4 tolerance; the
# deterministic scale arrays are compared exactly where the contract says so.
_TOL = dict(rtol=2.0e-4, atol=1.0e-3)


def _z_frame():
    zero = np.zeros(3)
    return ObservationFrame(
        axis=Z.copy(), origin=zero, u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]), mouth_center=zero, source_center=zero,
    )


def _solve(mesh, **overrides):
    from test_multi_source_parity import _observation_config

    config = metal_bem.native_config(
        observation=_observation_config(),
        return_surface_pressure=True,
        velocity_mode=metal_bem.VelocityMode.VELOCITY,
        **overrides,
    )
    return metal_bem.solve_frequencies(mesh, _FREQS, config)


@pytest.mark.slow
def test_polarity_flip_negates_the_field():
    _require_native()
    from test_multi_source_parity import _two_cap_sphere_mesh

    mesh = _two_cap_sphere_mesh()
    kw = dict(velocity_sources={2: 1.0}, source_motion="axial")
    up = _solve(mesh, source_axes={2: (0, 0, 1)}, **kw)
    down = _solve(mesh, source_axes={2: (0, 0, -1)}, **kw)
    np.testing.assert_allclose(
        down.pressure_complex, -up.pressure_complex, **_TOL
    )
    np.testing.assert_allclose(
        down.surface_pressure_complex, -up.surface_pressure_complex, **_TOL
    )
    assert np.max(np.abs(up.pressure_complex)) > 0


@pytest.mark.slow
def test_explicit_plus_z_equals_legacy_on_plus_z_cap():
    _require_native()
    from test_multi_source_parity import _two_cap_sphere_mesh

    mesh = _two_cap_sphere_mesh()
    kw = dict(
        velocity_sources={2: 1.0}, source_motion="axial", frame_override=_z_frame()
    )
    # The legacy scale (frame axis +z, sign vote positive) and the explicit
    # +z scale are the same array, exactly.
    legacy_cfg = metal_bem.native_config(**kw)
    explicit_cfg = metal_bem.native_config(source_axes={2: (0, 0, 1)}, **kw)
    frame = _z_frame()
    args = (mesh.grid, mesh.physical_tags)
    np.testing.assert_array_equal(
        _build_source_face_scale(*args, legacy_cfg, frame.axis, frame.source_center),
        _build_source_face_scale(*args, explicit_cfg, frame.axis, frame.source_center),
    )
    legacy = _solve(mesh, **kw)
    explicit = _solve(mesh, source_axes={2: (0, 0, 1)}, **kw)
    np.testing.assert_allclose(
        explicit.surface_pressure_complex, legacy.surface_pressure_complex, **_TOL
    )
    np.testing.assert_allclose(explicit.pressure_complex, legacy.pressure_complex, **_TOL)


@pytest.mark.slow
def test_observation_frame_does_not_change_the_drive():
    _require_native()
    from test_multi_source_parity import _two_cap_sphere_mesh

    mesh = _two_cap_sphere_mesh()
    origin = np.zeros(3)

    def frame(axis, u, v):
        return ObservationFrame(
            axis=np.asarray(axis, float), origin=origin,
            u=np.asarray(u, float), v=np.asarray(v, float),
            mouth_center=origin, source_center=origin,
        )

    frames = [
        frame((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        frame((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        frame((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    ]
    kw = dict(
        velocity_sources={2: 1.0}, source_motion="axial",
        source_axes={2: (0, 0, 1)},
    )
    results = [_solve(mesh, frame_override=f, **kw) for f in frames]
    for other in results[1:]:
        np.testing.assert_allclose(
            other.surface_pressure_complex, results[0].surface_pressure_complex, **_TOL
        )
        np.testing.assert_allclose(other.impedance, results[0].impedance, **_TOL)


@pytest.mark.slow
def test_multi_source_honours_explicit_axes():
    _require_native()
    from test_multi_source_parity import _observation_config, _two_cap_sphere_mesh

    mesh = _two_cap_sphere_mesh()
    axes = {2: (0, 0, 1), 3: (0, 0, 1)}  # tag 3 faces -z: driven against its normal
    sources = [{2: 1.0, 3: 0.0}, {2: 0.0, 3: 1.0}]
    common = dict(
        observation=_observation_config(), return_surface_pressure=True,
        velocity_mode=metal_bem.VelocityMode.VELOCITY, source_motion="axial",
        frame_override=_z_frame(),
    )
    config = metal_bem.native_config(
        velocity_sources={2: 1.0, 3: 1.0}, source_axes=axes, **common
    )
    multi = metal_bem.solve_multi_source(mesh, sources, config, frequencies_hz=_FREQS)
    for source, result in zip(sources, multi):
        seq = metal_bem.solve_frequencies(
            mesh, _FREQS,
            metal_bem.native_config(
                velocity_sources=dict(source), source_axes=axes, **common
            ),
        )
        np.testing.assert_allclose(result.pressure_complex, seq.pressure_complex, **_TOL)
    # Missing axis for a driven axial tag is refused, not run on the legacy axis.
    with pytest.raises(ValueError, match="axis"):
        bad = metal_bem.native_config(
            velocity_sources={2: 1.0, 3: 1.0}, source_axes={2: (0, 0, 1)}, **common
        )


@pytest.mark.slow
def test_multi_source_two_axial_one_normal_profiles():
    _require_native()
    from test_multi_source_parity import _observation_config, _two_cap_sphere_mesh

    mesh = _two_cap_sphere_mesh()
    tags = np.asarray(mesh.physical_tags).copy()
    # Third driven region: the equatorial band next to the caps (tag 4, normal).
    verts = np.asarray(mesh.grid.vertices).T
    elems = np.asarray(mesh.grid.elements).T
    cz = verts[elems].mean(axis=1)[:, 2]
    tags[(tags == 1) & (cz > 0.2) & (cz <= 0.55)] = 4
    mesh = replace(mesh, physical_tags=tags)
    assert np.count_nonzero(tags == 4) > 0
    profiles = {2: AxialProfile(), 3: AxialProfile(), 4: metal_bem.NormalProfile()}
    axes = {2: (0, 0, 1), 3: (0, 0, -1)}
    sources = [{2: 1.0}, {3: 1.0}, {4: 1.0}]
    common = dict(
        observation=_observation_config(), return_surface_pressure=True,
        velocity_mode=metal_bem.VelocityMode.VELOCITY,
        source_velocity_profiles=profiles, frame_override=_z_frame(),
    )
    config = metal_bem.native_config(
        velocity_sources={2: 1.0, 3: 1.0, 4: 1.0}, source_axes=axes, **common
    )
    multi = metal_bem.solve_multi_source(mesh, sources, config, frequencies_hz=_FREQS)
    singles = [
        metal_bem.solve_frequencies(
            mesh, _FREQS,
            metal_bem.native_config(
                velocity_sources={2: 1.0, 3: 1.0, 4: 1.0}
                | {t: 0.0 for t in (2, 3, 4) if t not in src},
                source_axes=axes, **common,
            ),
        )
        for src in sources
    ]
    for m, s in zip(multi, singles):
        np.testing.assert_allclose(m.pressure_complex, s.pressure_complex, **_TOL)
    total = sum(m.pressure_complex for m in multi)
    joint = metal_bem.solve_frequencies(mesh, _FREQS, config)
    np.testing.assert_allclose(total, joint.pressure_complex, **_TOL)


@pytest.mark.slow
def test_multi_source_two_axial_sources_without_profiles_or_frame_override():
    """Front/back hemisphere channels: axes +z and -z, source_motion axial,
    each source drives one tag, frame inferred (no frame_override)."""
    _require_native()
    from test_multi_source_parity import _observation_config, _two_cap_sphere_mesh

    mesh = _two_cap_sphere_mesh()
    axes = {2: (0, 0, 1), 3: (0, 0, -1)}
    sources = [{2: 1.0}, {3: 1.0}]
    common = dict(
        observation=_observation_config(), return_surface_pressure=True,
        velocity_mode=metal_bem.VelocityMode.VELOCITY, source_motion="axial",
    )
    config = metal_bem.native_config(
        velocity_sources={2: 1.0, 3: 1.0}, source_axes=axes, **common
    )
    multi = metal_bem.solve_multi_source(mesh, sources, config, frequencies_hz=_FREQS)
    assert len(multi) == 2
    joint = metal_bem.solve_frequencies(mesh, _FREQS, config)
    np.testing.assert_allclose(
        sum(m.pressure_complex for m in multi), joint.pressure_complex, **_TOL
    )


# Follow-ups: pure-Python coverage of the multi-source config narrowing ---
def test_per_source_axes_keep_other_axial_profile_tags():
    from hornlab_metal_bem.sweep import _source_axes_for_tags

    axial = AxialProfile()
    cfg = SolveConfig(
        velocity_sources={2: 1.0, 3: 1.0, 4: 1.0},
        source_velocity_profiles={2: axial, 3: axial},
        source_axes={2: (0, 0, 1), 3: (0, 0, -1)},
    )
    subset = _source_axes_for_tags(cfg, [2])
    # Tag 3 stays axial in tag 2's per-source config, so its axis must remain.
    assert subset == {2: (0, 0, 1), 3: (0, 0, -1)}
    # The per-source config built by the multi-source path must be accepted.
    replace(cfg, velocity_sources={2: 1.0}, source_axes=subset)
    # A normal-only source keeps just the axial-profile axes.
    assert set(_source_axes_for_tags(cfg, [4])) == {2, 3}


def test_frame_config_drops_axes_and_is_accepted():
    from hornlab_metal_bem import _multi_source_frame_config

    cfg = _cfg({2: 1.0, 3: 1.0}, {2: (0, 0, 1), 3: (0, 0, 1)})
    frame_cfg = _multi_source_frame_config(cfg, {2: 1.0})
    assert frame_cfg.source_axes is None
    assert frame_cfg.velocity_sources == {2: 1.0}
    # Without dropping the axes, narrowing to the first source is rejected.
    with pytest.raises(ValueError, match="source_axes"):
        replace(cfg, velocity_sources={2: 1.0})


def test_boundary_lab_channel_axes_cover_every_channel():
    from hornlab_metal_bem.boundary_lab import (
        _multi_source_overrides,
        solve_config_from_boundary_lab,
    )

    channels = [{2: 1.0 + 0j}, {3: 1.0 + 0j}]
    axes = {2: (0, 0, 1), 3: (0, 0, -1)}
    overrides = {"source_motion": SourceMotion.AXIAL, "source_axes": axes}
    overrides.update(_multi_source_overrides(overrides, channels))
    config, _ = solve_config_from_boundary_lab({}, **overrides)
    assert config.source_axes == axes
    # velocity_sources is ignored downstream; the second channel's tag is only
    # listed so its axis validates.
    assert set(config.velocity_sources) == {2, 3}
    # Legacy path (no axes): first channel only, unchanged.
    assert _multi_source_overrides({}, channels) == {
        "velocity_sources": {2: 1.0 + 0j},
        "velocity_source_callback": None,
    }

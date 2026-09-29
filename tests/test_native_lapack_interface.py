"""Guards for the native helper's Accelerate LAPACK interface.

Accelerate's legacy CLAPACK entry points (``cgesv_``, ``zgesv_`` ...) return NaN when
they run concurrently with other Accelerate calls, which the coupled infinite-baffle
solve pipeline does with a large aperture block. The helper therefore links only the
``$NEWLAPACK`` entry points (Package.swift defines ``ACCELERATE_NEW_LAPACK``).

* ``test_helper_links_no_legacy_lapack_symbols`` is the deterministic guard.
* ``test_coupled_ib_large_aperture_concurrent_solve_matches_serial`` is the
  behavioural check. The legacy failure is probabilistic (roughly 5-10 % of solves
  under load), so it can pass on a legacy build; it is not a substitute for the
  symbol check.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem.config import ObservationConfig, SolveConfig, VelocityMode
from hornlab_metal_bem.mesh import LoadedMesh, make_pure_grid
from hornlab_metal_bem.result import MeshInfo
from native_helper_guard import require_fresh_native_helper
from test_native_coupled_ib_validation import (
    TAG_APERTURE,
    TAG_THROAT,
    TAG_WALL,
    _straight_channel_mesh,
    _triangulated_disc,
    _z_axis_frame,
)

# LAPACK/BLAS routines the helper calls (main.swift, GmresSolve.swift, LapackNew.swift).
_LAPACK_ROUTINES = (
    "cgesv", "zgesv", "cgetrf", "cgetrs", "zgels", "cgecon", "zgecon",
    "clange", "zlange", "cblas_cgemm", "cblas_sgemm",
)


def _undefined_symbols(binary) -> list[str]:
    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("nm is unavailable")
    out = subprocess.run(
        [nm, "-u", str(binary)], capture_output=True, text=True, check=True
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def test_helper_links_no_legacy_lapack_symbols():
    status = require_fresh_native_helper()
    if status.helper_executable_path is None:
        pytest.skip("no compiled helper executable")
    symbols = _undefined_symbols(status.helper_executable_path)
    names = "|".join(_LAPACK_ROUTINES)
    lapack = [
        s for s in symbols
        if re.fullmatch(rf"_(?:{names})_?(?:\$NEWLAPACK)?(?:\$ILP64)?", s)
        or re.fullmatch(r"_[a-z]{3,6}_", s)  # any other Fortran-style legacy entry
    ]
    legacy = [s for s in lapack if "$NEWLAPACK" not in s]
    assert not legacy, f"helper links legacy Accelerate LAPACK symbols: {legacy}"
    # The check must have seen the new entry points, or it proves nothing.
    assert any(s == "_cgesv$NEWLAPACK" for s in symbols), symbols
    assert any(s == "_zgesv$NEWLAPACK" for s in symbols), symbols


def _large_aperture_channel(
    radius: float, depth: float, *, aperture_rings: int, throat_rings: int,
    sectors: int, wall_layers: int,
) -> LoadedMesh:
    """Straight channel with a fine aperture disc, a coarse throat and a banded wall.

    Aperture triangles = sectors * (2 * aperture_rings - 1). The coupled-IB Schur
    elimination then factors an aperture block far larger than the remaining
    unknowns, the shape of the WG batch that hit the legacy LAPACK failure.
    """
    top_v, top_t = _triangulated_disc(
        radius, rings=aperture_rings, sectors=sectors, z=0.0, normal_sign=1
    )
    bot_v, bot_t = _triangulated_disc(
        radius, rings=throat_rings, sectors=sectors, z=-depth, normal_sign=-1
    )
    theta = 2.0 * np.pi * np.arange(sectors) / sectors
    middle = [
        np.column_stack([
            radius * np.cos(theta), radius * np.sin(theta),
            np.full(sectors, -depth + depth * layer / wall_layers),
        ])
        for layer in range(1, wall_layers)
    ]
    vertices = np.vstack([top_v, bot_v, *middle])
    bot_offset = top_v.shape[0]
    triangles = [*top_t.tolist(), *(bot_t + bot_offset).tolist()]
    tags = [TAG_APERTURE] * top_t.shape[0] + [TAG_THROAT] * bot_t.shape[0]
    top_outer = 1 + (aperture_rings - 1) * sectors
    bot_outer = bot_offset + 1 + (throat_rings - 1) * sectors
    mid_start = bot_offset + bot_v.shape[0]
    rows = [bot_outer + np.arange(sectors)]
    rows += [mid_start + k * sectors + np.arange(sectors) for k in range(wall_layers - 1)]
    rows.append(top_outer + np.arange(sectors))
    for layer in range(wall_layers):
        lo, hi = rows[layer], rows[layer + 1]
        for j in range(sectors):
            nxt = (j + 1) % sectors
            triangles += [[lo[j], lo[nxt], hi[nxt]], [lo[j], hi[nxt], hi[j]]]
            tags += [TAG_WALL, TAG_WALL]
    triangles_arr = np.asarray(triangles, dtype=np.int32)[:, [0, 2, 1]]
    return LoadedMesh(
        grid=make_pure_grid(vertices, triangles_arr),
        physical_tags=np.asarray(tags, dtype=np.int32),
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles_arr.shape[0],
            physical_groups={
                TAG_THROAT: "throat", TAG_WALL: "wall", TAG_APERTURE: "aperture",
            },
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
    )


def _solve(monkeypatch, concurrency: int):
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_SOLVE_CONCURRENCY", str(concurrency))
    depth = 0.05
    radius = 0.1
    # 1,500 aperture triangles (60 sectors x 13 rings) against roughly 1,000 other
    # unknowns: the proportions at which the legacy cgesv_ returned NaN.
    mesh = _large_aperture_channel(
        radius, depth, aperture_rings=13, throat_rings=3, sectors=60, wall_layers=8
    )
    config = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        aperture_tag=TAG_APERTURE,
        observation=ObservationConfig(
            distance_m=1.0,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=4,
            planes=["horizontal"],
            origin="mouth",
        ),
        frame_override=_z_axis_frame(depth),
        metal_native_assembly_mode="corrected",
    )
    frequencies_hz = np.array(
        [400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0]
    )
    return metal_bem.solve_frequencies(mesh, frequencies_hz, config)


@pytest.mark.slow
def test_coupled_ib_large_aperture_concurrent_solve_matches_serial(monkeypatch):
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    # entrywise assembly is bit-reproducible, so serial and concurrent solves must
    # agree exactly; the default pair_atomic has a ~5e-7 relative noise floor.
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "entrywise")
    require_fresh_native_helper()

    serial = _solve(monkeypatch, 1)
    concurrent = _solve(monkeypatch, 6)

    for result in (serial, concurrent):
        assert np.all(np.isfinite(result.pressure_complex))
        assert np.all(np.isfinite(result.impedance))
    # Bitwise equality is expected on this machine, but Accelerate's results may
    # depend on the thread count on small CI runners: allow float32 rounding.
    np.testing.assert_allclose(
        concurrent.pressure_complex, serial.pressure_complex, rtol=1e-6, atol=0.0
    )
    np.testing.assert_allclose(concurrent.impedance, serial.impedance, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("quantity", ["surface", "field"])
def test_non_finite_result_fails_with_case_and_frequency(monkeypatch, quantity):
    """The helper names the case, frequency and quantity instead of aborting.

    HORNLAB_METAL_BEM_NATIVE_TEST_INJECT_NAN (test-only, off by default) plants a NaN
    in one case's surface pressure or field output before the finiteness scan.
    """
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE", "corrected")
    require_fresh_native_helper()
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_TEST_INJECT_NAN", f"1:{quantity}")
    depth = 0.003
    config = SolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        aperture_tag=TAG_APERTURE,
        observation=ObservationConfig(
            distance_m=1.0, angle_min_deg=0.0, angle_max_deg=90.0,
            angle_count=3, planes=["horizontal"], origin="mouth",
        ),
        frame_override=_z_axis_frame(depth),
        metal_native_assembly_mode="corrected",
    )
    mesh = _straight_channel_mesh(0.04, depth, rings=3, sectors=16)
    with pytest.raises(RuntimeError) as excinfo:
        metal_bem.solve_frequencies(mesh, np.array([800.0, 1600.0, 2400.0]), config)
    message = str(excinfo.value)
    assert "non-finite" in message and quantity in message, message
    assert "case 1" in message and "frequency_hz 1600.0" in message, message

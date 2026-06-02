from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hornlab_solver import (
    BIEFormulation,
    LinearSolver,
    ObservationConfig,
    SolveConfig,
    configure_opencl,
    load_mesh,
    solve_frequencies,
)


ASRO68_MESH = Path(
    "/Users/magnus/IM Dropbox/Magnus Andersen/DOCS/code/misc/"
    "ATH results 0 degree norm/250917asro68/ABEC_FreeStanding/250917asro68.msh"
)

VALIDATION_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "260517-abec-vs-wg-validation-artifacts"
)


def _require_asro68_mesh() -> Path:
    if not ASRO68_MESH.exists():
        pytest.skip(f"ASRO68 ABEC reference mesh not found: {ASRO68_MESH}")
    return ASRO68_MESH


def test_default_solve_config_uses_canonical_profile():
    config = SolveConfig()

    assert config.formulation is BIEFormulation.STANDARD
    assert config.solver is LinearSolver.GMRES
    assert config.assembly_backend == "opencl"
    assert config.opencl_device == "cpu"
    assert config.precision == "single"
    # slp_dlp_quadrature=2 caused a 142% pressure regression vs the May 3
    # bempp-direct reference on the slot130 BIGMEH cabinet. q=4 (bempp's
    # own default for regular integrals) matches bit-exact.
    assert config.slp_dlp_quadrature == 4
    assert config.hyp_adlp_quadrature == 4
    assert config.workers == 1


def test_asro68_mesh_loads_full_stitched_abec_surface():
    mesh = load_mesh(_require_asro68_mesh(), scale=0.001)

    assert mesh.info.n_vertices == 4554
    assert mesh.info.n_triangles == 9104
    assert mesh.info.physical_groups == {1: "SD1G0", 2: "SD1D1001"}
    assert len(mesh.physical_tags) == 9104
    assert np.count_nonzero(mesh.physical_tags == 1) == 9040
    assert np.count_nonzero(mesh.physical_tags == 2) == 64


def test_asro68_validation_artifacts_match_abec_key_angles_all_planes():
    if not VALIDATION_DIR.exists():
        pytest.skip(f"Validation artifacts dir missing: {VALIDATION_DIR}")

    hl_npz = VALIDATION_DIR / "hornlab_postfix_asro68.npz"
    abec_npz = VALIDATION_DIR / "abec_baseline_asro68.npz"
    for npz in (hl_npz, abec_npz):
        if not npz.exists():
            pytest.skip(f"Validation NPZ missing: {npz.name}")

    with np.load(hl_npz) as hl, np.load(abec_npz) as abec:
        assert np.allclose(hl["freq_hz"], abec["freq_hz"])
        assert np.array_equal(hl["polar_angle_deg"], abec["polar_angle_deg"])

        key_angles = np.array([0, 6, 18, 36])  # 0, 30, 90, 180 deg
        f100_idx = int(np.argmin(np.abs(abec["freq_hz"] - 100.0)))
        for plane_key in ("h_spl_db_norm0", "v_spl_db_norm0", "d_spl_db_norm0"):
            diff = (
                abec[plane_key][f100_idx, key_angles]
                - hl[plane_key][f100_idx, key_angles]
            )
            maxabs = float(np.max(np.abs(diff)))
            assert maxabs < 0.05, (
                f"{plane_key} ABEC-vs-hornlab key-angle disagreement at 100 Hz: "
                f"max|d|={maxabs:.4f} dB"
            )


def test_loaded_mesh_is_not_reloaded_when_mesh_scale_is_set(monkeypatch):
    mesh = load_mesh(_require_asro68_mesh(), scale=0.001)

    def fail_load_mesh(*_args, **_kwargs):
        raise AssertionError("LoadedMesh should not be reloaded")

    monkeypatch.setattr("hornlab_solver.load_mesh", fail_load_mesh)
    from hornlab_solver import _resolve_mesh

    assert _resolve_mesh(mesh, scale=0.001) is mesh


@pytest.mark.slow
@pytest.mark.parametrize(
    ("origin", "expected_db"),
    [
        (
            "throat",
            np.array([0.0, -0.24731088, -1.49123359, -2.26645803]),
        ),
        (
            "mouth",
            np.array([0.0, -0.10727048, -0.63626397, -0.81154060]),
        ),
    ],
)
def test_asro68_100hz_hplane_matches_pinned_origin_smoke(origin, expected_db):
    _require_asro68_mesh()
    try:
        configure_opencl("cpu")
    except Exception as exc:
        pytest.skip(f"OpenCL CPU runtime unavailable: {exc}")

    config = SolveConfig(
        observation=ObservationConfig(
            planes=["horizontal"],
            distance_m=2.0,
            angle_min_deg=0.0,
            angle_max_deg=180.0,
            angle_count=37,
            origin=origin,
        ),
        mesh_scale=0.001,
    )

    result = solve_frequencies(ASRO68_MESH, [100.0], config)

    assert result.mesh_info.n_vertices == 4554
    assert result.mesh_info.n_triangles == 9104
    assert result.solver_log[0]["iterations"] < config.gmres_max_iter

    key_angles = result.spl_db[0, 0, [0, 6, 18, 36]]
    # Rebased 2026-05-20 after fixing slp_dlp_quadrature default
    # 2 -> 4 (q=2 caused 142% pressure regression on BIGMEH slot130
    # vs the May 3 bempp-direct reference). The previous expected_db
    # was generated with q=2 and is no longer the right anchor;
    # new values come from q=4 (bempp default) which matches
    # bempp-direct results bit-exact.
    np.testing.assert_allclose(key_angles, expected_db, atol=0.03, rtol=0.0)

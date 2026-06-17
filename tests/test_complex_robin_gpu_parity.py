from __future__ import annotations

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
from hornlab_metal_bem.metal import discover_native_runtime
from tests.test_complex_k_resonance import _unit_sphere_mesh


def _complex_rel_l2(lhs: np.ndarray, rhs: np.ndarray) -> float:
    denom = float(np.linalg.norm(rhs.ravel()))
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm((lhs - rhs).ravel()) / denom)


def _complex_robin_sphere_mesh():
    mesh = _unit_sphere_mesh()
    vertices = np.asarray(mesh.grid.vertices, dtype=np.float64)
    elements = np.asarray(mesh.grid.elements, dtype=np.int32)
    centroids = vertices[:, elements].mean(axis=1)

    tags = np.ones(elements.shape[1], dtype=np.int32)
    tags[centroids[2] > 0.75] = 2
    tags[(centroids[2] <= 0.35) & (centroids[2] > -0.25)] = 8
    tags[centroids[2] <= -0.25] = 9
    mesh.physical_tags = tags
    mesh.info.physical_groups = {1: "rigid", 2: "source", 8: "band_8", 9: "band_9"}
    return mesh


@pytest.mark.slow
def test_complex_k_robin_corrected_gpu_matches_reference(monkeypatch):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE", "gpu_blocks")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_FIELD_MODE", "optimized")
    monkeypatch.setenv("HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL", "pair_atomic")

    mesh = _complex_robin_sphere_mesh()
    observation = metal_bem.ObservationConfig(
        planes=["probe"],
        angle_count=2,
        custom_points={
            "probe": np.array(
                [
                    [0.0, 0.0, 2.2],
                    [0.6, 0.0, 2.1],
                ],
                dtype=np.float64,
            )
        },
    )
    common_config = dict(
        formulation="complex_k",
        complex_k_shift=0.005,
        velocity_sources={2: 1.0},
        impedance_sources={8: 0.05 + 0.0j, 9: 0.02 + 0.01j},
        observation=observation,
    )

    gpu = metal_bem.solve_frequencies(
        mesh,
        [171.5],
        metal_bem.native_config(
            **common_config,
            metal_native_assembly_mode="corrected",
        ),
    )
    reference = metal_bem.solve_frequencies(
        mesh,
        [171.5],
        metal_bem.native_config(
            **common_config,
            metal_native_assembly_mode="reference",
        ),
    )

    assert gpu.native_diagnostics[0]["assembly_mode"] == "corrected"
    assert reference.native_diagnostics[0]["assembly_mode"] == "reference"
    assert _complex_rel_l2(gpu.pressure_complex, reference.pressure_complex) < 2.0e-4
    assert _complex_rel_l2(gpu.impedance, reference.impedance) < 2.0e-4


@pytest.mark.slow
def test_impedance_source_callback_end_to_end_distinct_beta(monkeypatch):
    """Through the full solve pipeline: a beta(f) callback produces different
    results at two frequencies whose ONLY difference is the wall admittance on
    tag 8, and both frequencies are flagged robin_boundary."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )

    mesh = _complex_robin_sphere_mesh()
    observation = metal_bem.ObservationConfig(
        planes=["probe"],
        angle_count=2,
        custom_points={
            "probe": np.array(
                [[0.0, 0.0, 2.2], [0.6, 0.0, 2.1]],
                dtype=np.float64,
            )
        },
    )
    # Static value held fixed for both frequencies; the callback flips tag 8
    # between two betas. Declare tag 8 in impedance_sources so the Neumann
    # builder skip rule is satisfied even without the callback union.
    config = metal_bem.native_config(
        formulation="complex_k",
        complex_k_shift=0.005,
        velocity_sources={2: 1.0},
        impedance_sources={8: 0.0 + 0.0j},
        impedance_source_callback=lambda f: {8: (0.05 if f < 200 else 0.30) + 0j},
        observation=observation,
    )

    # Two frequencies, same k except for the float32 rounding; the meaningful
    # difference is the callback beta (0.05 below 200 Hz, 0.30 at/above).
    result = metal_bem.solve_frequencies(mesh, [180.0, 220.0], config)

    assert all(d["robin_boundary"] is True for d in result.native_diagnostics)
    # Distinct beta -> distinct on-/off-axis pressure between the two cases.
    assert not np.allclose(
        result.pressure_complex[0], result.pressure_complex[1]
    )


def test_impedance_source_callback_passivity_guard_through_solve():
    """A non-passive callback (Re(beta) < 0) must raise ValueError through the
    full solve_frequencies path, before any heavy native work."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    mesh = _complex_robin_sphere_mesh()
    config = metal_bem.native_config(
        formulation="complex_k",
        velocity_sources={2: 1.0},
        impedance_sources={8: 0.0 + 0.0j},
        impedance_source_callback=lambda f: {8: -0.1 + 0.0j},
    )
    with pytest.raises(ValueError, match="passive"):
        metal_bem.solve_frequencies(mesh, [180.0], config)


# CHIEF interior overdetermination points for the unit-sphere closed cavity:
# off-axis, off the symmetry planes, in the interior bulk (well away from the
# r=1 wall). Eight points to avoid all of them landing on a nodal surface of the
# interior eigenmode.
_UNIT_SPHERE_CHIEF_POINTS = np.array(
    [
        [0.20, 0.15, 0.10],
        [-0.15, 0.20, -0.10],
        [0.10, -0.20, 0.15],
        [-0.10, -0.15, -0.20],
        [0.25, -0.10, 0.05],
        [-0.05, 0.10, 0.25],
        [0.00, 0.30, 0.05],
        [0.30, 0.00, -0.05],
    ],
    dtype=np.float64,
)


def _unit_sphere_fictitious_resonance_hz() -> tuple[float, float, float]:
    """Locate the strongest exterior-BIE fictitious-eigenvalue dip of the
    128-triangle unit sphere by the dense-solve rcond minimum over a band, and
    return (resonance_hz, rcond_min, rcond_median)."""
    from tests.test_complex_k_resonance import _unit_sphere_mesh

    mesh = _unit_sphere_mesh()
    observation = metal_bem.ObservationConfig(
        planes=["probe"],
        angle_count=2,
        custom_points={
            "probe": np.array([[0.0, 0.0, 2.2], [0.6, 0.0, 2.1]], dtype=np.float64)
        },
    )
    freqs = np.linspace(240.0, 265.0, 26)
    result = metal_bem.solve_frequencies(
        mesh,
        freqs,
        metal_bem.native_config(velocity_sources={2: 1.0}, observation=observation),
    )
    rcond = np.array(
        [d["dense_solve_rcond"] for d in result.native_diagnostics], dtype=np.float64
    )
    idx = int(np.argmin(rcond))
    return float(freqs[idx]), float(rcond[idx]), float(np.median(rcond))


@pytest.mark.slow
def test_chief_matches_complex_k_at_fictitious_resonance():
    """At the unit sphere's exterior-BIE fictitious eigenfrequency the plain
    standard solve is contaminated by the spurious interior mode. CHIEF (interior
    overdetermination -> least squares) and complex_k (the independent
    wavenumber-shift cure) must both remove that contamination and converge to
    the SAME physical exterior pressure. Agreement of two unrelated cures is the
    strong check that the CHIEF row math (Robin fold sign, +G*g_drv RHS, mirror
    images, real-k kernels) is correct."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    from tests.test_complex_k_resonance import _unit_sphere_mesh

    fres, rcond_min, rcond_median = _unit_sphere_fictitious_resonance_hz()
    # Confirm there really is a fictitious dip to cure at the located frequency.
    assert rcond_min < 0.25 * rcond_median

    mesh = _unit_sphere_mesh()
    observation = metal_bem.ObservationConfig(
        planes=["probe"],
        angle_count=2,
        custom_points={
            "probe": np.array([[0.0, 0.0, 2.2], [0.6, 0.0, 2.1]], dtype=np.float64)
        },
    )
    base = dict(velocity_sources={2: 1.0}, observation=observation)

    standard = metal_bem.solve_frequencies(
        mesh, [fres], metal_bem.native_config(**base)
    )
    chief = metal_bem.solve_frequencies(
        mesh,
        [fres],
        metal_bem.native_config(**base, chief_points=_UNIT_SPHERE_CHIEF_POINTS),
    )
    complex_k = metal_bem.solve_frequencies(
        mesh,
        [fres],
        metal_bem.native_config(
            **base, formulation="complex_k", complex_k_shift=0.02
        ),
    )

    chief_diag = chief.native_diagnostics[0]
    assert chief_diag["chief_points"] is True
    assert chief_diag["chief_points_count"] == _UNIT_SPHERE_CHIEF_POINTS.shape[0]
    assert chief_diag["chief_solver"] == "accelerate_lapack_zgels"
    assert np.isfinite(chief_diag["chief_residual_rel"])
    assert np.all(np.isfinite(chief.pressure_complex))

    p_std = complex(standard.pressure_complex[0, 0, 0])
    p_chief = complex(chief.pressure_complex[0, 0, 0])
    p_ck = complex(complex_k.pressure_complex[0, 0, 0])

    # The two cures agree closely; the contaminated standard solve is further off.
    rel_chief_ck = abs(p_chief - p_ck) / abs(p_ck)
    rel_std_ck = abs(p_std - p_ck) / abs(p_ck)
    assert rel_chief_ck < 0.05
    assert rel_chief_ck < rel_std_ck


@pytest.mark.slow
def test_chief_monotonic_in_beta_at_interior_mode():
    """Cross-feature regression for the documented LF blow-up: on a closed cavity
    at an exterior-BIE fictitious-eigenvalue frequency, with CHIEF points placed
    inside the cavity, the on-resonance on-axis pressure magnitude must be
    NON-INCREASING as a passive wall admittance beta increases. Without the CHIEF
    cure, increasing beta moves the spurious eigenvalue and produces a
    non-monotone bump; with CHIEF, more passive admittance can only ever reduce
    the resonant peak."""
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    from tests.test_complex_k_resonance import _unit_sphere_mesh

    fres, rcond_min, rcond_median = _unit_sphere_fictitious_resonance_hz()
    assert rcond_min < 0.25 * rcond_median

    mesh = _unit_sphere_mesh()
    observation = metal_bem.ObservationConfig(
        planes=["probe"],
        angle_count=2,
        custom_points={
            "probe": np.array([[0.0, 0.0, 2.2], [0.6, 0.0, 2.1]], dtype=np.float64)
        },
    )

    betas = [0.0, 0.02, 0.05, 0.10, 0.20]
    amplitudes: list[float] = []
    for beta in betas:
        result = metal_bem.solve_frequencies(
            mesh,
            [fres],
            metal_bem.native_config(
                velocity_sources={2: 1.0},
                observation=observation,
                # Tag 1 is the rigid bulk of the sphere; make it the passive
                # lossy wall whose admittance we ramp.
                impedance_sources={1: beta + 0.0j},
                chief_points=_UNIT_SPHERE_CHIEF_POINTS,
            ),
        )
        assert result.native_diagnostics[0]["robin_boundary"] is True
        assert result.native_diagnostics[0]["chief_points"] is True
        amplitudes.append(float(np.abs(result.pressure_complex[0, 0, 0])))

    # Strictly non-increasing (a tiny positive tolerance absorbs f32 narrowing).
    diffs = np.diff(np.asarray(amplitudes))
    assert np.all(diffs <= 1e-6), (
        f"on-resonance on-axis |p| not non-increasing in beta at {fres:.1f} Hz: "
        f"betas={betas} -> |p|={amplitudes}"
    )
    # And the constraint must actually bite: max admittance reduces the peak.
    assert amplitudes[-1] < amplitudes[0]

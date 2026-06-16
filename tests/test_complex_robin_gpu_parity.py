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

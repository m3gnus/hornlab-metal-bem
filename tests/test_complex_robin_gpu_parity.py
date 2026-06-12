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

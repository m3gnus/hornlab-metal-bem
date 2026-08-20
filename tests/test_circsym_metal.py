from __future__ import annotations

import numpy as np
import pytest

import hornlab_metal_bem as metal_bem
import hornlab_metal_bem.circsym as circsym
from hornlab_metal_bem import MeridianMesh, ObservationConfig, SolveConfig, VelocityMode
from hornlab_metal_bem.metal import (
    discover_native_runtime,
    evaluate_circsym_ring_field_kernels,
)


def _psi_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    psi = 0.5 * np.pi * (nodes + 1.0)
    return np.cos(psi), 0.5 * np.pi * weights


def _reference_ring_field(
    *,
    target_rho: np.ndarray,
    target_z: np.ndarray,
    source_rho: np.ndarray,
    source_z: np.ndarray,
    measure: np.ndarray,
    normal_rho: np.ndarray,
    normal_z: np.ndarray,
    cos_psi: np.ndarray,
    psi_weights: np.ndarray,
    k: complex,
    baffle_z: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    target_count = target_rho.size
    source_count, line_count = source_rho.shape
    slp = np.zeros((target_count, source_count), dtype=np.complex128)
    dlp = np.zeros_like(slp)

    def accumulate(
        source_z_values: np.ndarray,
        normal_z_values: np.ndarray,
    ) -> None:
        for line in range(line_count):
            rt = target_rho[:, None, None]
            zt = target_z[:, None, None]
            rs = source_rho[:, line][None, :, None]
            zs = source_z_values[:, line][None, :, None]
            nr = normal_rho[None, :, None]
            nz = normal_z_values[None, :, None]
            cp = cos_psi[None, None, :]
            dz = zs - zt
            radius = np.sqrt(
                rt * rt + rs * rs - 2.0 * rt * rs * cp + dz * dz
            )
            phase = np.exp(1j * k * radius)
            green = phase / (4.0 * np.pi * radius)
            numerator = (rs - rt * cp) * nr + dz * nz
            derivative = (
                phase
                * (1j * k * radius - 1.0)
                * numerator
                / (4.0 * np.pi * radius**3)
            )
            azimuth_weight = 2.0 * psi_weights[None, None, :]
            line_measure = measure[:, line][None, :]
            slp[:] += np.sum(green * azimuth_weight, axis=2) * line_measure
            dlp[:] += np.sum(derivative * azimuth_weight, axis=2) * line_measure

    accumulate(source_z, normal_z)
    if baffle_z is not None:
        accumulate(2.0 * baffle_z - source_z, -normal_z)
    return slp, dlp


def _field_fixture() -> dict[str, object]:
    cos_psi, psi_weights = _psi_rule(48)
    source_rho = np.array(
        [
            [0.035, 0.042, 0.050, 0.058],
            [0.105, 0.113, 0.121, 0.129],
            [0.185, 0.193, 0.201, 0.209],
        ]
    )
    source_z = np.array(
        [
            [-0.04, -0.03, -0.02, -0.01],
            [0.01, 0.02, 0.03, 0.04],
            [0.08, 0.09, 0.10, 0.11],
        ]
    )
    return {
        "target_rho": np.array([0.0, 0.18, 0.42, 0.63]),
        "target_z": np.array([1.2, 1.1, 0.95, 0.72]),
        "source_rho": source_rho,
        "source_z": source_z,
        "measure": source_rho
        * np.array(
            [
                [0.006, 0.011, 0.011, 0.006],
                [0.009, 0.017, 0.017, 0.009],
                [0.012, 0.022, 0.022, 0.012],
            ]
        ),
        "normal_rho": np.array([0.96, 0.58, -0.22]),
        "normal_z": np.array([0.28, 0.815, 0.975]),
        "cos_psi": cos_psi,
        "psi_weights": psi_weights,
    }


def test_circsym_metal_field_validates_shapes_before_runtime_discovery():
    values = _field_fixture()
    values["target_z"] = np.array([1.0])

    with pytest.raises(ValueError, match="target_z must have the same shape"):
        evaluate_circsym_ring_field_kernels(**values, k=12.0)


def test_circsym_field_auto_backend_uses_work_threshold(monkeypatch):
    status = type("Status", (), {"available": True})()
    monkeypatch.delenv("HORNLAB_CIRCSYM_FIELD_BACKEND", raising=False)
    monkeypatch.setattr(
        circsym,
        "_circsym_metal_field_runtime_status",
        lambda: status,
    )
    monkeypatch.setattr(circsym, "_circsym_metal_field_auto_failure", None)

    assert circsym._select_circsym_field_backend(2, 3, 64) == ("cpu", None)
    assert circsym._select_circsym_field_backend(37, 463, 96) == (
        "metal",
        status,
    )


@pytest.mark.parametrize("baffle_z", [None, -0.075])
def test_circsym_metal_ring_field_matches_complex128_cpu_reference(baffle_z):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    values = _field_fixture()
    k = 211.0 + 0.35j
    reference_slp, reference_dlp = _reference_ring_field(
        **values,
        k=k,
        baffle_z=baffle_z,
    )
    result = evaluate_circsym_ring_field_kernels(
        **values,
        k=k,
        baffle_z=baffle_z,
        runtime_status=status,
    )

    slp_relative = np.linalg.norm(result.slp - reference_slp) / np.linalg.norm(
        reference_slp
    )
    dlp_relative = np.linalg.norm(result.dlp - reference_dlp) / np.linalg.norm(
        reference_dlp
    )
    # The native contract is float32; coordinates are narrowed before phase
    # evaluation, so a complex128 reference cannot require float64 rounding.
    # This gate is still over 200x tighter than the 0.01 dB field-level gate.
    assert slp_relative < 5.0e-5
    assert dlp_relative < 5.0e-5
    assert result.diagnostics["implementation"] == (
        "swift_native_metal_circsym_ring_field"
    )
    assert result.diagnostics["baffle_image"] is (baffle_z is not None)

    pressure = np.array([0.7 + 0.2j, -0.1 + 0.4j, 0.5 - 0.3j])
    neumann = np.array([0.2 - 0.1j, -0.4 + 0.05j, 0.15 + 0.3j])
    reference_field = reference_dlp @ pressure - reference_slp @ neumann
    metal_field = result.dlp @ pressure - result.slp @ neumann
    reference_db = 20.0 * np.log10(
        np.abs(reference_field) / np.max(np.abs(reference_field))
    )
    metal_db = 20.0 * np.log10(np.abs(metal_field) / np.max(np.abs(metal_field)))
    assert float(np.max(np.abs(metal_db - reference_db))) < 0.01


def test_circsym_full_solve_adaptive_metal_field_matches_cpu(monkeypatch):
    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    theta = np.linspace(0.0, np.pi, 57)
    radius = 0.1
    meridian = MeridianMesh.from_polyline(
        np.column_stack([radius * np.sin(theta), radius * np.cos(theta)]),
        tags=2,
    )
    config = SolveConfig(
        velocity_sources={2: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        formulation="standard",
        observation=ObservationConfig(
            distance_m=3.0,
            angle_count=37,
            planes=["horizontal"],
            origin="throat",
        ),
    )

    monkeypatch.setenv("HORNLAB_CIRCSYM_FIELD_BACKEND", "cpu")
    cpu = metal_bem.solve_circsym_frequencies(meridian, [4_000.0], config)
    monkeypatch.delenv("HORNLAB_CIRCSYM_FIELD_BACKEND")
    circsym._circsym_metal_field_runtime_status.cache_clear()
    monkeypatch.setattr(circsym, "_circsym_metal_field_auto_failure", None)
    metal = metal_bem.solve_circsym_frequencies(meridian, [4_000.0], config)

    pressure_relative = np.linalg.norm(
        metal.pressure_complex - cpu.pressure_complex
    ) / np.linalg.norm(cpu.pressure_complex)
    assert pressure_relative < 5.0e-5
    assert float(np.max(np.abs(metal.directivity_db - cpu.directivity_db))) < 0.01
    assert metal.native_diagnostics[0]["field_backend"] == "metal"
    assert metal.native_diagnostics[0]["field_backend_policy"] == "auto"

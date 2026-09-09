from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_metal_bem import (
    MeridianMesh,
    ObservationConfig,
    SolveConfig,
    solve_circsym_frequencies,
)
from scripts import bench_axisymmetric_vs_quarter as benchmark


def _result(pressure: np.ndarray, directivity: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        pressure_complex=np.asarray(pressure, dtype=np.complex128),
        directivity_db=np.asarray(directivity, dtype=np.float64),
    )


def test_accuracy_reports_identical_results_as_zero_difference():
    pressure = np.array([[1.0 + 2.0j, -0.5 + 0.25j]])
    directivity = np.array([[0.0, -6.0]])

    metrics = benchmark._accuracy(
        _result(pressure, directivity),
        _result(pressure, directivity),
    )

    assert metrics == {
        "pressure_relative_l2": 0.0,
        "pressure_magnitude_relative_l2": 0.0,
        "directivity_max_abs_db": 0.0,
        "directivity_max_abs_db_above_minus_40": 0.0,
        "phase_rms_degrees_above_floor": 0.0,
    }


def test_accuracy_by_frequency_keeps_error_rows_separate():
    reference = _result(
        np.array([[1.0 + 0.0j, 1.0 + 0.0j], [2.0 + 0.0j, 2.0 + 0.0j]]),
        np.array([[0.0, -6.0], [0.0, -6.0]]),
    )
    candidate = _result(
        np.array([[1.0 + 0.0j, 1.0 + 0.0j], [2.2 + 0.0j, 2.2 + 0.0j]]),
        np.array([[0.0, -6.0], [0.0, -5.0]]),
    )

    rows = benchmark._accuracy_by_frequency(candidate, reference, [100.0, 200.0])

    assert rows[0]["frequency_hz"] == 100.0
    assert rows[0]["pressure_relative_l2"] == 0.0
    assert rows[1]["pressure_relative_l2"] == pytest.approx(0.1)
    assert rows[1]["directivity_max_abs_db_above_minus_40"] == 1.0


def test_source_areas_return_full_domain_equivalents():
    meridian = SimpleNamespace(
        physical_tags=np.array([2]),
        segment_geometry=lambda: SimpleNamespace(area_weights=np.array([0.5])),
    )
    quarter = SimpleNamespace(
        physical_tags=np.array([2]),
        grid=SimpleNamespace(
            vertices=np.array(
                [[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]
            ),
            elements=np.array([[0], [1], [2]]),
        ),
    )

    meridian_area, quarter_area = benchmark._source_areas(meridian, quarter)

    assert meridian_area == pytest.approx(0.5)
    assert quarter_area == pytest.approx(0.5)


def test_parse_args_defaults_to_original_qualification_gate(tmp_path):
    args = benchmark._parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--quarter-mesh",
            str(tmp_path / "quarter.msh"),
        ]
    )

    assert args.axisym_backend == "cpu"
    assert args.cpu_field == "numba"
    assert args.azimuth_min == 64
    assert args.qualification_ratio == 0.5
    assert args.meridian_refinement_factor == 1
    assert args.meridian_refinement_factors is None


def test_parse_args_accepts_fixed_geometry_convergence_ladder(tmp_path):
    args = benchmark._parse_args(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--quarter-mesh",
            str(tmp_path / "quarter.msh"),
            "--meridian-refinement-factor",
            "4",
            "--meridian-refinement-factors",
            "1,2,4,8",
        ]
    )

    assert args.meridian_refinement_factor == 4
    assert args.meridian_refinement_factors == (1, 2, 4, 8)


def test_resonance_comparison_requires_explicit_chief_points(tmp_path):
    with pytest.raises(SystemExit):
        benchmark._parse_args(
            [
                "--config",
                str(tmp_path / "config.json"),
                "--quarter-mesh",
                str(tmp_path / "quarter.msh"),
                "--resonance-comparison",
            ]
        )


def test_load_chief_points_records_caller_owned_content_hash(tmp_path):
    path = tmp_path / "chief-points.json"
    path.write_text("[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]", encoding="utf-8")

    points, provenance = benchmark._load_chief_points(path)

    assert points.shape == (2, 3)
    assert provenance["source"] == "caller_supplied_json"
    assert provenance["point_count"] == 2
    assert len(provenance["sha256"]) == 64


def test_load_chief_points_accepts_windows_utf8_bom(tmp_path):
    path = tmp_path / "chief-points-bom.json"
    path.write_bytes(b"\xef\xbb\xbf[[0.01, 0.02, 0.03]]")

    points, provenance = benchmark._load_chief_points(path)

    assert points.tolist() == [[0.01, 0.02, 0.03]]
    assert provenance["point_count"] == 1


def test_phase_rows_separate_reference_nulls_without_changing_official_metric():
    reference = _result(
        np.array([[1.0 + 0.0j, 1.0e-5 + 0.0j]]), np.array([[0.0, -6.0]])
    )
    candidate = _result(
        np.array([[1.0 + 0.0j, -1.0e-5 + 0.0j]]), np.array([[0.0, -6.0]])
    )

    row = benchmark._accuracy_by_frequency(candidate, reference, [1000.0])[0]

    assert row["phase_rms_degrees_above_floor"] == 0.0
    assert row["significant_field_sample_count"] == 1
    assert row["null_region_sample_count"] == 1
    assert row["null_region_phase_rms_degrees"] == pytest.approx(180.0)
    assert row["phase_sample_count_above_1e-3_reference_peak"] == 1
    assert row["phase_sample_count_above_1e-2_reference_peak"] == 1


def test_phase_diagnostics_exclude_exact_zero_reference_nulls_without_nan():
    reference = _result(
        np.array([[1.0 + 0.0j, 0.0 + 0.0j]]), np.array([[0.0, -6.0]])
    )
    candidate = _result(
        np.array([[1.0 + 0.0j, 1.0 + 0.0j]]), np.array([[0.0, -6.0]])
    )

    row = benchmark._accuracy_by_frequency(candidate, reference, [1000.0])[0]

    assert row["phase_rms_degrees_above_floor"] == 0.0
    assert row["significant_field_valid_phase_sample_count"] == 1
    assert row["null_region_sample_count"] == 1
    assert row["null_region_valid_phase_sample_count"] == 0
    assert row["null_region_undefined_zero_reference_sample_count"] == 1
    assert row["null_region_phase_rms_degrees"] == 0.0
    assert np.isfinite(row["null_region_phase_rms_degrees"])


def test_geometry_distance_compares_quarter_vertices_against_matching_tag_curve():
    meridian = MeridianMesh.from_polyline(
        np.array([[0.0, 0.0], [1.0, 0.0]]), tags=2
    )
    aligned_quarter = SimpleNamespace(
        physical_tags=np.array([2]),
        grid=SimpleNamespace(
            vertices=np.array(
                [[0.0, 0.5, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            elements=np.array([[0], [1], [2]]),
        ),
    )
    mismatched_quarter = SimpleNamespace(
        physical_tags=np.array([2]),
        grid=SimpleNamespace(
            vertices=np.array(
                [[0.0, 0.5, 1.0], [0.0, 0.0, 0.0], [0.0, 0.003, 0.0]]
            ),
            elements=np.array([[0], [1], [2]]),
        ),
    )

    aligned = benchmark._meridian_quarter_geometry_distance(meridian, aligned_quarter)
    mismatched = benchmark._meridian_quarter_geometry_distance(
        meridian, mismatched_quarter
    )

    assert aligned["max_distance_m"] == 0.0
    assert aligned["per_tag"]["2"]["sample_count"] == 3
    assert mismatched["max_distance_m"] == pytest.approx(0.003)


def test_geometry_contract_keeps_coarse_chord_distance_diagnostic_only():
    coarse = MeridianMesh.from_polyline(
        np.array([[0.0, 0.0], [1.0, 0.0]]), tags=1
    )
    reference = MeridianMesh.from_polyline(
        np.array([[0.0, 0.0], [0.5, 0.003], [1.0, 0.0]]), tags=[1, 1]
    )
    quarter = SimpleNamespace(
        physical_tags=np.array([1]),
        grid=SimpleNamespace(
            vertices=np.array(
                [[0.0, 0.5, 1.0], [0.0, 0.0, 0.0], [0.0, 0.003, 0.0]]
            ),
            elements=np.array([[0], [1], [2]]),
        ),
    )

    diagnostics = benchmark._geometry_contract_diagnostics(
        coarse, reference, quarter, None
    )

    assert diagnostics["coarse_meridian_panel_distance"]["max_distance_m"] == pytest.approx(
        0.003
    )
    assert diagnostics["generating_curve_reference_distance"]["max_distance_m"] == 0.0
    assert diagnostics["coarse_meridian_panel_distance"][
        "max_distance_over_median_quarter_edge"
    ] > 0.0


def test_flat_mouth_closure_requires_actual_wall_triangles_on_the_contract_span():
    meridian = MeridianMesh.from_polyline(
        np.array([[0.5, 0.0], [1.0, 0.0], [1.0, 0.003]]), tags=[1, 1]
    )
    quarter = SimpleNamespace(
        physical_tags=np.array([1]),
        grid=SimpleNamespace(
            vertices=np.array(
                [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0015, 0.003]]
            ),
            elements=np.array([[0], [1], [2]]),
        ),
    )

    diagnostics = benchmark._flat_mouth_closure_diagnostics(
        meridian,
        quarter,
        {"sourceSegmentCount": 0, "innerSegmentCount": 1, "mouthRimSegmentCount": 1},
    )

    assert diagnostics["checked"] is True
    assert diagnostics["reference_closure_max_distance_to_straight_chord_m"] == 0.0
    assert diagnostics["quarter_wall_endpoint_max_distance_m"] == 0.0
    assert diagnostics["quarter_wall_triangles_on_flat_closure"] == 1
    assert diagnostics["flat_closure_covers_full_span"] is True
    assert diagnostics["passes"] is True

    # An adjacent wall panel may have two different azimuthal vertices at one
    # mouth-ring endpoint. Its rho/z projection has zero chord span, so it must
    # not be mistaken for a closure face.
    with_adjacent_wall = SimpleNamespace(
        physical_tags=np.array([1, 1]),
        grid=SimpleNamespace(
            vertices=np.array(
                [
                    [1.0, 0.0, 1.0, 0.0, 0.98],
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0015, 0.003, 0.0, -0.001],
                ]
            ),
            elements=np.array([[0, 0], [1, 3], [2, 4]]),
        ),
    )
    with_adjacent = benchmark._flat_mouth_closure_diagnostics(
        meridian,
        with_adjacent_wall,
        {"sourceSegmentCount": 0, "innerSegmentCount": 1, "mouthRimSegmentCount": 1},
    )
    assert with_adjacent["quarter_wall_closure_region_triangle_count"] == 1
    assert with_adjacent["passes"] is True

    curved_quarter = SimpleNamespace(
        physical_tags=np.array([1]),
        grid=SimpleNamespace(
            vertices=np.array(
                [[1.0, 0.998, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0015, 0.003]]
            ),
            elements=np.array([[0], [1], [2]]),
        ),
    )
    curved = benchmark._flat_mouth_closure_diagnostics(
        meridian,
        curved_quarter,
        {"sourceSegmentCount": 0, "innerSegmentCount": 1, "mouthRimSegmentCount": 1},
    )
    assert curved["quarter_wall_endpoint_max_distance_m"] == 0.0
    assert curved["quarter_wall_triangles_on_flat_closure"] == 0

    # One correctly flat face must not hide an adjacent bulged closure face.
    mixed_quarter = SimpleNamespace(
        physical_tags=np.array([1, 1]),
        grid=SimpleNamespace(
            vertices=np.array(
                [
                    [1.0, 1.0, 1.0, 0.998],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0015, 0.003, 0.0015],
                ]
            ),
            elements=np.array([[0, 0], [1, 2], [2, 3]]),
        ),
    )
    mixed = benchmark._flat_mouth_closure_diagnostics(
        meridian,
        mixed_quarter,
        {"sourceSegmentCount": 0, "innerSegmentCount": 1, "mouthRimSegmentCount": 1},
    )
    assert mixed["quarter_wall_triangles_on_flat_closure"] == 1
    assert mixed["quarter_wall_closure_region_triangle_count"] == 2
    assert mixed["quarter_wall_closure_region_max_distance_m"] == pytest.approx(0.002)
    assert mixed["passes"] is False


def test_flat_mouth_closure_rejects_subdivided_bulge_after_a_short_flat_strip():
    # Expected radial closure: (160, 125) -> (154, 125) mm. The mesh starts
    # correctly for 1 mm, then follows the continuous bulge
    # (159,125)->(157,122)->(154,125) mm. No curved triangle has two points on
    # the expected chord, so span coverage—not only candidate quality—must fail.
    meridian = MeridianMesh.from_polyline(
        np.array([[0.160, 0.124], [0.160, 0.125], [0.154, 0.125]]), tags=[1, 1]
    )
    quarter = SimpleNamespace(
        physical_tags=np.array([1, 1, 1]),
        grid=SimpleNamespace(
            vertices=np.array(
                [
                    [0.160, 0.0, 0.159, 0.157, 0.0, 0.154, 0.0],
                    [0.0, 0.159, 0.0, 0.0, 0.157, 0.0, 0.154],
                    [0.125, 0.125, 0.125, 0.122, 0.122, 0.125, 0.125],
                ]
            ),
            elements=np.array([[0, 2, 3], [1, 3, 5], [2, 4, 6]]),
        ),
    )

    diagnostics = benchmark._flat_mouth_closure_diagnostics(
        meridian,
        quarter,
        {"sourceSegmentCount": 0, "innerSegmentCount": 1, "mouthRimSegmentCount": 1},
    )

    assert diagnostics["quarter_wall_triangles_on_flat_closure"] == 1
    assert diagnostics["flat_closure_parameter_min"] == pytest.approx(0.0)
    assert diagnostics["flat_closure_parameter_max"] == pytest.approx(1.0 / 6.0)
    assert diagnostics["flat_closure_covers_full_span"] is False
    assert diagnostics["passes"] is False


def test_flat_mouth_closure_rejects_two_endpoint_strips_with_a_middle_gap():
    meridian = MeridianMesh.from_polyline(
        np.array([[0.160, 0.124], [0.160, 0.125], [0.154, 0.125]]), tags=[1, 1]
    )
    # Flat strips cover t=[0, 1/6] and t=[5/6, 1]. The middle triangles are a
    # subdivided bulge and do not qualify as flat closure candidates.
    quarter = SimpleNamespace(
        physical_tags=np.array([1, 1, 1, 1]),
        grid=SimpleNamespace(
            vertices=np.array(
                [
                    [0.160, 0.0, 0.159, 0.157, 0.0, 0.155, 0.154, 0.0, 0.154],
                    [0.0, 0.159, 0.0, 0.0, 0.157, 0.0, 0.0, 0.154, 0.0],
                    [0.125, 0.125, 0.125, 0.122, 0.122, 0.125, 0.125, 0.125, 0.125],
                ]
            ),
            elements=np.array([[0, 2, 3, 5], [1, 3, 5, 7], [2, 4, 6, 8]]),
        ),
    )

    diagnostics = benchmark._flat_mouth_closure_diagnostics(
        meridian,
        quarter,
        {"sourceSegmentCount": 0, "innerSegmentCount": 1, "mouthRimSegmentCount": 1},
    )

    assert diagnostics["flat_closure_parameter_intervals"][0] == pytest.approx(
        (0.0, 1.0 / 6.0)
    )
    assert diagnostics["flat_closure_parameter_intervals"][1] == pytest.approx(
        (5.0 / 6.0, 1.0)
    )
    assert diagnostics["flat_closure_largest_gap"] == pytest.approx(2.0 / 3.0)
    assert diagnostics["flat_closure_covers_full_span"] is False
    assert diagnostics["passes"] is False


def test_high_resolution_generating_curve_only_refines_mesh_target_lengths():
    raw = {
        "mesh": {"throat_res_mm": 4.0, "mouth_res_mm": 24.0, "rear_res_mm": 16.0},
        "formula": "OSSE",
    }

    refined = benchmark._high_resolution_generating_curve_config(raw)

    assert raw["mesh"]["mouth_res_mm"] == 24.0
    assert refined["formula"] == "OSSE"
    assert refined["mesh"] == {
        "throat_res_mm": 0.5,
        "mouth_res_mm": 3.0,
        "rear_res_mm": 2.0,
    }


def test_high_resolution_generating_curve_honors_mesher_resolution_aliases():
    raw = {
        "mesh": {
            "throatResolution": 6.0,
            "mouthResolution": 10.0,
            "rearResolution": 25.0,
        }
    }

    refined = benchmark._high_resolution_generating_curve_config(raw)

    assert refined["mesh"]["throat_res_mm"] == pytest.approx(0.75)
    assert refined["mesh"]["mouth_res_mm"] == pytest.approx(1.25)
    assert refined["mesh"]["rear_res_mm"] == pytest.approx(3.125)


def test_meridian_subdivision_preserves_tagged_revolved_area_and_normals():
    base = MeridianMesh.from_polyline(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), tags=[2, 1]
    )

    refined = benchmark._subdivide_meridian(base, 4)
    provenance = benchmark._meridian_refinement_provenance(base, refined, 4)

    assert refined.segment_count == 8
    assert refined.node_count == base.node_count + base.segment_count * 3
    assert np.allclose(refined.nodes[: base.node_count], base.nodes)
    assert refined.segments[0, 0] == 0
    assert refined.segments[-1, 1] == base.node_count - 1
    # The retained original joint keeps both child chains connected.
    assert np.count_nonzero(refined.segments == 1) == 2
    assert refined.physical_tags.tolist() == [2, 2, 2, 2, 1, 1, 1, 1]
    assert np.allclose(refined.normals[:4], base.normals[0])
    assert np.allclose(refined.normals[4:], base.normals[1])
    assert provenance["expected_segment_count"] == 8
    assert all(
        error == pytest.approx(0.0)
        for error in provenance["surface_area_relative_error_by_tag"].values()
    )


def test_subdivided_closed_meridian_runs_circsym_smoke_solve():
    base = MeridianMesh.from_polyline(
        np.array([[0.0, 0.03], [0.03, 0.0], [0.0, -0.03]]), tags=2
    )
    refined = benchmark._subdivide_meridian(base, 4)

    result = solve_circsym_frequencies(
        refined,
        np.array([500.0]),
        SolveConfig(
            velocity_sources={2: 1.0},
            observation=ObservationConfig(angle_count=3),
        ),
    )

    assert refined.nodes[refined.segments[0, 0], 0] == 0.0
    assert refined.nodes[refined.segments[-1, 1], 0] == 0.0
    assert np.all(np.isfinite(result.pressure_complex))


def test_convergence_ladder_reuses_timed_rung_and_compares_all_fixed_geometry_rungs():
    base = MeridianMesh.from_polyline(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), tags=[2, 1]
    )
    candidate = _result(np.ones((1, 2), dtype=np.complex128), np.zeros((1, 2)))
    calls: list[int] = []

    def solve(mesh, frequencies, config):
        del frequencies, config
        calls.append(mesh.segment_count)
        pressure = np.full((1, 2), mesh.segment_count, dtype=np.complex128)
        return _result(pressure, np.zeros((1, 2)))

    ladder = benchmark._run_meridian_convergence_ladder(
        base_meridian=base,
        factors=(1, 2, 4),
        candidate_factor=1,
        candidate_result=candidate,
        frequencies=np.array([1000.0]),
        axisym_config=object(),
        quarter_result=candidate,
        solve=solve,
    )

    assert calls == [4, 8]
    assert ladder["reference_factor"] == 4
    assert ladder["timed_candidate_factor"] == 1
    assert [row["refinement"]["segment_count"] for row in ladder["rows"]] == [
        2,
        4,
        8,
    ]
    assert ladder["rows"][0]["additional_wall_seconds"] is None


def test_convergence_ladder_only_runs_when_explicitly_requested():
    assert benchmark._requested_convergence_ladder_factors(None, 4) is None
    assert benchmark._requested_convergence_ladder_factors((1, 2, 8), 4) == (
        1,
        2,
        4,
        8,
    )


def test_resonance_comparison_pairs_real_k_chief_with_each_complex_k_shift():
    calls: list[tuple[str, str, bool]] = []
    result = _result(np.ones((1, 2), dtype=np.complex128), np.zeros((1, 2)))

    def solve_axisym(mesh, frequencies, config):
        del mesh, frequencies
        calls.append(("axisym", config.formulation, config.chief_points is not None))
        return result

    def solve_quarter(mesh, frequencies, config):
        del mesh, frequencies
        calls.append(("quarter", config.formulation, config.chief_points is not None))
        return result

    report = benchmark._run_resonance_comparison(
        enabled=True,
        chief_points=np.array([[0.01, 0.02, 0.03]]),
        chief_provenance={"source": "caller_supplied_json", "point_count": 1},
        chief_weight=2.0,
        complex_k_shifts=(0.001, 0.005),
        meridian=object(),
        quarter=object(),
        frequencies=np.array([1000.0]),
        axisym_config=SolveConfig(),
        quarter_config=SolveConfig(),
        solve_axisym=solve_axisym,
        solve_quarter=solve_quarter,
    )

    assert report["qualification_effect"] == "none"
    assert report["chief"]["weight"] == 2.0
    assert [row["complex_k_shift"] for row in report["complex_k_shift_ladder"]] == [
        0.001,
        0.005,
    ]
    assert calls == [
        ("axisym", "standard", True),
        ("quarter", "standard", True),
        ("axisym", "complex_k", False),
        ("quarter", "complex_k", False),
        ("axisym", "complex_k", False),
        ("quarter", "complex_k", False),
    ]


def test_overall_qualification_requires_compact_cpu_parity():
    common = {
        "speed_ratio": True,
        "half_second": True,
        "numerical_gate": True,
    }

    assert benchmark._passes_overall_qualification(
        **common,
        compact_cpu_parity=True,
    )
    assert not benchmark._passes_overall_qualification(
        **common,
        compact_cpu_parity=False,
    )


def test_matched_input_validation_rejects_wrong_quarter_extent():
    meridian = SimpleNamespace(
        physical_tags=np.array([1, 2]),
        nodes=np.array([[0.0, -0.01], [0.16, 0.17]]),
        segment_geometry=lambda: SimpleNamespace(area_weights=np.array([1.0, 1.0])),
    )
    quarter = SimpleNamespace(
        physical_tags=np.array([1, 2]),
        grid=SimpleNamespace(
            vertices=np.array(
                [
                    [0.0, 0.10, 0.0, 0.10],
                    [0.0, 0.0, 0.10, 0.10],
                    [-0.01, -0.01, 0.17, 0.17],
                ]
            ),
            elements=np.array([[0, 1], [1, 2], [2, 3]]),
        ),
    )
    config = {
        "mode": "freestanding",
        "cross_section": {"exponent": 2.0, "aspectRatio": 1.0},
        "morph": {"morphTarget": 0.0},
    }

    with pytest.raises(ValueError, match="radial extent"):
        benchmark._validate_matched_inputs(config, meridian, quarter)

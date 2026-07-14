from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from contest_solution.golden_plus_repair import (
    RepairConfig,
    calculate_bbox_area,
    calculate_hpwl,
    enumerate_factor_shapes,
    repair_fixed_topology,
    soft_counts,
)


def _repair(positions, areas, constraints, *, b2b=(), p2b=(), pins=(), config=None):
    return repair_fixed_topology(
        positions,
        areas,
        b2b,
        p2b,
        pins,
        constraints,
        config=config,
        return_report=True,
    )


def test_exact_factor_enumeration_is_oriented_and_aspect_bounded():
    assert set(enumerate_factor_shapes(12)) == {
        (2.0, 6.0),
        (6.0, 2.0),
        (3.0, 4.0),
        (4.0, 3.0),
    }
    assert enumerate_factor_shapes(10.5) == ()
    assert enumerate_factor_shapes(7) == ()


def test_boundary_slide_removes_violation_and_reduces_hpwl():
    positions = [
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (4.0, 1.0, 1.0, 1.0),
    ]
    constraints = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
    ]
    repaired, report = _repair(
        positions,
        [1.0, 1.0, 1.0],
        constraints,
        p2b=[(0, 1, 1.0)],
        pins=[(5.0, 0.5)],
    )

    assert repaired[1] == (4.0, 0.0, 1.0, 1.0)
    assert report.changed
    assert report.accepted == {"boundary": 1, "mib": 0, "grouping": 0}
    assert report.before.soft.boundary == 1
    assert report.after.soft.boundary == 0
    assert report.after.hpwl < report.before.hpwl
    assert report.after.bbox_area == report.before.bbox_area


def test_boundary_move_with_hpwl_regression_is_rejected_exactly():
    positions = [
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (4.0, 1.0, 1.0, 1.0),
    ]
    constraints = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0],
    ]
    repaired, report = _repair(
        positions,
        [1.0, 1.0, 1.0],
        constraints,
        p2b=[(0, 1, 1.0)],
        pins=[(1.5, 0.5)],
    )

    assert repaired == positions
    assert not report.changed
    assert report.fallback_reason == "no_candidate_passed_all_gates"


def test_compatible_mib_group_is_repaired_with_common_exact_shape():
    square = 40.0**0.5
    positions = [
        (0.0, 0.0, square, square),
        (20.0, 0.0, square, square),
        (40.0, 0.0, 4.0, 10.0),
        (0.0, 20.0, 1.0, 1.0),
    ]
    constraints = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    config = replace(
        RepairConfig(), enable_boundary=False, enable_mib=True, enable_grouping=False
    )
    repaired, report = _repair(
        positions, [40.0, 40.0, 40.0, 1.0], constraints, config=config
    )

    assert repaired[0][2:] == repaired[1][2:] == repaired[2][2:] == (4.0, 10.0)
    assert report.accepted["mib"] == 1
    assert report.before.soft.mib == 1
    assert report.after.soft.mib == 0
    assert report.after.hpwl == report.before.hpwl
    assert report.after.bbox_area <= report.before.bbox_area


def test_mib_repair_does_not_change_fixed_or_preplaced_geometry():
    square = 40.0**0.5
    positions = [
        (0.0, 0.0, 4.0, 10.0),
        (20.0, 0.0, square, square),
        (40.0, 0.0, square, square),
        (0.0, 20.0, 1.0, 1.0),
    ]
    constraints = [
        [1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    config = replace(
        RepairConfig(), enable_boundary=False, enable_mib=True, enable_grouping=False
    )
    repaired, report = _repair(
        positions, [40.0, 40.0, 40.0, 1.0], constraints, config=config
    )

    assert repaired[0] == positions[0]
    assert repaired[1][2:] == repaired[2][2:] == (4.0, 10.0)
    assert report.output_feasible


def test_rigid_cluster_translation_preserves_component_and_connects_group():
    positions = [
        (0.0, 0.0, 1.0, 1.0),
        (3.0, 0.0, 1.0, 1.0),
        (0.0, 2.0, 1.0, 1.0),
    ]
    constraints = [
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    repaired, report = _repair(
        positions, [1.0, 1.0, 1.0], constraints, b2b=[(0, 1, 1.0)]
    )

    assert soft_counts(repaired, constraints).grouping == 0
    assert report.accepted["grouping"] == 1
    assert report.after.hpwl < report.before.hpwl
    assert report.after.bbox_area < report.before.bbox_area


def test_preplaced_cluster_component_is_never_translated():
    positions = [(0.0, 0.0, 1.0, 1.0), (3.0, 0.0, 1.0, 1.0)]
    constraints = [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0]]
    repaired, report = _repair(positions, [1.0, 1.0], constraints)

    assert repaired == positions
    assert not report.changed


def test_infeasible_incumbent_fails_closed_to_original():
    positions = [(0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 2.0, 2.0)]
    repaired, report = _repair(
        positions,
        [4.0, 4.0],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    )

    assert repaired == positions
    assert not report.input_feasible
    assert report.fallback_reason == "incumbent_not_hard_feasible"


def test_mechanism_ablation_disables_other_candidate_families():
    positions = [(0.0, 0.0, 1.0, 4.0), (5.0, 0.0, 2.0, 2.0)]
    constraints = [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
    config = replace(
        RepairConfig(), enable_boundary=True, enable_mib=False, enable_grouping=False
    )
    repaired, report = _repair(positions, [4.0, 4.0], constraints, config=config)

    assert repaired == positions
    assert report.attempted["mib"] == 0


def test_mib_only_uniform_output_uses_constant_work_fast_path():
    positions = [(0.0, 0.0, 2.0, 2.0), (3.0, 0.0, 2.0, 2.0)]
    constraints = [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]
    config = replace(
        RepairConfig(), enable_boundary=False, enable_mib=True, enable_grouping=False
    )
    repaired, report = _repair(positions, [4.0, 4.0], constraints, config=config)

    assert repaired == positions
    assert report.fallback_reason == "no_safe_mib_pattern_fast_path"
    assert report.before is None
    assert report.after is None
    assert report.attempted == {"boundary": 0, "mib": 0, "grouping": 0}


def test_public_metric_primitives_match_hand_calculation():
    positions = [(0.0, 0.0, 2.0, 2.0), (3.0, 1.0, 2.0, 2.0)]
    b2b = [(0, 1, 2.0)]
    p2b = [(0, 0, 0.5)]
    pins = [(0.0, 0.0)]

    assert calculate_hpwl(positions, b2b, p2b, pins) == 9.0
    assert calculate_bbox_area(positions) == 15.0


def test_mib_repair_matches_real_torch_and_packaging_tensor_stub():
    torch = pytest.importorskip("torch")
    stub_path = Path(__file__).resolve().parents[1] / "packaging" / "torch_stub.py"
    spec = importlib.util.spec_from_file_location("submission_torch_stub", stub_path)
    assert spec is not None and spec.loader is not None
    torch_stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(torch_stub)

    square = 40.0**0.5
    positions = [
        (0.0, 0.0, square, square),
        (20.0, 0.0, square, square),
        (40.0, 0.0, 4.0, 10.0),
        (0.0, 20.0, 1.0, 1.0),
    ]
    areas = [40.0, 40.0, 40.0, 1.0]
    constraints = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    targets = [[-1.0] * 4 for _ in positions]
    config = RepairConfig(
        enable_boundary=False,
        enable_mib=True,
        enable_grouping=False,
        require_safe_mib_pattern=True,
    )

    real = repair_fixed_topology(
        positions,
        torch.tensor(areas),
        [],
        [],
        torch.empty((0, 2)),
        torch.tensor(constraints),
        torch.tensor(targets),
        config=config,
    )
    packaged = repair_fixed_topology(
        positions,
        torch_stub.Tensor(areas),
        [],
        [],
        torch_stub.Tensor([]),
        torch_stub.Tensor(constraints),
        torch_stub.Tensor(targets),
        config=config,
    )

    assert packaged == real
    assert real[0][2:] == real[1][2:] == real[2][2:] == (4.0, 10.0)

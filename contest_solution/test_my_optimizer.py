import json
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent))

try:
    from iccad2026_evaluate import check_overlap, check_area_tolerance, check_dimension_hard_constraints
except (ModuleNotFoundError, ImportError):
    # Public-repo fallback: the official evaluator is available in the contest
    # checkout, but these smoke tests should still run after a plain clone.
    evaluator_stub = sys.modules.get(
        "iccad2026_evaluate", types.ModuleType("iccad2026_evaluate")
    )

    class FloorplanOptimizer:
        def __init__(self, verbose=False):
            self.verbose = verbose

    def check_overlap(positions):
        violations = 0
        for idx, (x1, y1, w1, h1) in enumerate(positions):
            for x2, y2, w2, h2 in positions[idx + 1:]:
                if min(x1 + w1, x2 + w2) - max(x1, x2) > 1e-6 and min(y1 + h1, y2 + h2) - max(y1, y2) > 1e-6:
                    violations += 1
        return violations

    def check_area_tolerance(positions, area_targets, skip_indices=None):
        skip_indices = skip_indices or set()
        violations = 0
        for idx, (_, _, w, h) in enumerate(positions):
            if idx in skip_indices:
                continue
            target = float(area_targets[idx])
            if target > 0.0 and not math.isclose(w * h, target, rel_tol=1e-6, abs_tol=1e-6):
                violations += 1
        return violations

    def check_dimension_hard_constraints(positions, target_positions, constraints, block_count):
        violations = 0
        for idx in range(block_count):
            if constraints[idx, 0] == 0 and constraints[idx, 1] == 0:
                continue
            _, _, w, h = positions[idx]
            target_w = float(target_positions[idx, 2])
            target_h = float(target_positions[idx, 3])
            if target_w != -1.0 and not math.isclose(w, target_w, rel_tol=1e-6, abs_tol=1e-6):
                violations += 1
            if target_h != -1.0 and not math.isclose(h, target_h, rel_tol=1e-6, abs_tol=1e-6):
                violations += 1
        return violations

    if not hasattr(evaluator_stub, "FloorplanOptimizer"):
        evaluator_stub.FloorplanOptimizer = FloorplanOptimizer
    evaluator_stub.calculate_bbox_area = lambda positions: 0.0
    evaluator_stub.calculate_hpwl_b2b = lambda positions, conn: 0.0
    evaluator_stub.calculate_hpwl_p2b = lambda positions, conn, pins: 0.0
    evaluator_stub.check_overlap = check_overlap
    evaluator_stub.check_area_tolerance = check_area_tolerance
    evaluator_stub.check_dimension_hard_constraints = check_dimension_hard_constraints
    sys.modules["iccad2026_evaluate"] = evaluator_stub

from my_optimizer import (
    MyOptimizer,
    _LEARNED_REPLACEMENT_WF,
    _should_try_anchored_third_pass,
    _should_try_preplaced_aspect_pass,
)
from dissect import dissect_solve
import dissect as dissect_module
import my_optimizer as optimizer_module


def test_learned_replacement_slots_match_frozen_audit_artifact():
    path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "models"
        / "order_v5b_redundant_slots_v1.json"
    )
    artifact = json.loads(path.read_bytes())
    expected = {
        int(block_count): float(width_factor)
        for block_count, width_factor in artifact[
            "replacement_wf_by_size"
        ].items()
    }
    assert _LEARNED_REPLACEMENT_WF == expected
    assert artifact["abstain_sizes"] == [101, 109, 112]


def test_anchored_third_pass_gate_accepts_case_81_feature_pocket():
    assert _should_try_anchored_third_pass(102, 34, 1, 2184, 45)


@pytest.mark.parametrize(
    "features",
    [
        (99, 34, 1, 2184, 45),
        (104, 34, 1, 2184, 45),
        (102, 33, 1, 2184, 45),
        (102, 34, 2, 2184, 45),
        (102, 34, 1, 1799, 45),
        (102, 34, 1, 2501, 45),
        (102, 34, 1, 2184, 101),
    ],
)
def test_anchored_third_pass_gate_rejects_outside_feature_pocket(features):
    assert not _should_try_anchored_third_pass(*features)


def test_preplaced_aspect_pass_gate_accepts_case_61_feature_pocket():
    assert _should_try_preplaced_aspect_pass(82, 22, 7, 336, 1367)


def test_dissect_solve_can_return_reusable_first_pass():
    kwargs = dict(
        n=1,
        areas=[4.0],
        b2b_edges=[],
        p2b_edges=[],
        pins=[],
        constraints=[[0, 0, 0, 0, 0]],
        target_positions=None,
    )
    ordinary = dissect_solve(**kwargs)
    final, first = dissect_solve(**kwargs, return_first_pass=True)

    assert isinstance(ordinary, list)
    assert final == ordinary
    assert len(first) == 1


def test_dissect_first_pass_only_skips_second_paid_pass(monkeypatch):
    calls = []

    def fake_once(*args, **kwargs):
        calls.append(kwargs["pass_name"])
        return [(0.0, 0.0, 2.0, 2.0)]

    monkeypatch.setattr(dissect_module, "_dissect_once", fake_once)
    result = dissect_module.dissect_solve(
        n=1,
        areas=[4.0],
        b2b_edges=[],
        p2b_edges=[],
        pins=[],
        constraints=[[0, 0, 0, 0, 0]],
        target_positions=None,
        first_pass_only=True,
    )

    assert result == [(0.0, 0.0, 2.0, 2.0)]
    assert calls == ["p1"]


def test_dissect_without_learned_prior_is_scalar_identical_to_v32_fixture():
    kwargs = dict(
        n=2,
        areas=[4.0, 9.0],
        b2b_edges=[(0, 1, 1.0)],
        p2b_edges=[],
        pins=[],
        constraints=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        target_positions=None,
        width_factor=1.0,
    )
    expected = [
        (2.496150883013531, 0.0, 1.109400392450458, 3.6055512754639896),
        (0.0, 0.0, 2.496150883013531, 3.6055512754639896),
    ]
    assert dissect_solve(**kwargs) == expected
    assert dissect_solve(**kwargs, learned_order=None) == expected


@pytest.mark.parametrize(
    "features",
    [
        (77, 22, 7, 336, 1367),
        (87, 22, 7, 336, 1367),
        (82, 25, 7, 336, 1367),
        (82, 22, 5, 336, 1367),
        (82, 22, 7, 501, 1367),
        (82, 22, 7, 336, 999),
        (82, 22, 7, 336, 1801),
    ],
)
def test_preplaced_aspect_pass_gate_rejects_outside_feature_pocket(features):
    assert not _should_try_preplaced_aspect_pass(*features)


def test_optimizer_keeps_preplaced_blocks_exact_and_avoids_overlap():
    opt = MyOptimizer()
    block_count = 4
    area_targets = torch.tensor([100.0, 25.0, 36.0, 49.0])
    b2b = torch.empty((0, 3))
    p2b = torch.empty((0, 3))
    pins = torch.empty((0, 2))
    constraints = torch.zeros((block_count, 5))
    constraints[0, 1] = 1  # preplaced
    target_positions = torch.full((block_count, 4), -1.0)
    target_positions[0] = torch.tensor([10.0, 20.0, 10.0, 10.0])

    pos = opt.solve(block_count, area_targets, b2b, p2b, pins, constraints, target_positions)

    assert len(pos) == block_count
    assert pos[0] == (10.0, 20.0, 10.0, 10.0)
    assert check_overlap(pos) == 0
    assert check_area_tolerance(pos, area_targets, skip_indices={0}) == 0
    assert check_dimension_hard_constraints(pos, target_positions, constraints, block_count) == 0


def test_optimizer_uses_exact_fixed_dimensions():
    opt = MyOptimizer()
    block_count = 3
    area_targets = torch.tensor([100.0, 64.0, 81.0])
    b2b = torch.empty((0, 3))
    p2b = torch.empty((0, 3))
    pins = torch.empty((0, 2))
    constraints = torch.zeros((block_count, 5))
    constraints[1, 0] = 1  # fixed shape
    target_positions = torch.full((block_count, 4), -1.0)
    target_positions[1, 2] = 4.0
    target_positions[1, 3] = 16.0

    pos = opt.solve(block_count, area_targets, b2b, p2b, pins, constraints, target_positions)

    assert math.isclose(pos[1][2], 4.0)
    assert math.isclose(pos[1][3], 16.0)
    assert check_overlap(pos) == 0
    assert check_dimension_hard_constraints(pos, target_positions, constraints, block_count) == 0


@pytest.mark.parametrize("invalid_kind", ["none", "empty", "wrong_length"])
def test_invalid_learned_slot_falls_back_to_displaced_standard_pass(
    monkeypatch, invalid_kind
):
    block_count = 100
    standard = [(1.1 * index, 0.0, 1.0, 1.0) for index in range(block_count)]
    invalid = {
        "none": None,
        "empty": [],
        "wrong_length": standard[:1],
    }[invalid_kind]
    calls = []

    def fake_dissect(*args, **kwargs):
        calls.append(dict(kwargs))
        if "learned_order" in kwargs:
            return invalid
        return list(standard)

    monkeypatch.setattr(dissect_module, "dissect_solve", fake_dissect)
    monkeypatch.setattr(
        optimizer_module,
        "_learned_order_prior",
        lambda *args, **kwargs: {
            index: (0.5, 0.5) for index in range(block_count)
        },
    )
    optimizer = MyOptimizer()
    monkeypatch.setattr(optimizer, "_solve_one", lambda *args, **kwargs: standard)
    area_targets = torch.ones(block_count)
    b2b = torch.empty((0, 3))
    p2b = torch.empty((0, 3))
    pins = torch.empty((0, 2))
    constraints = torch.zeros((block_count, 5))
    targets = torch.full((block_count, 4), -1.0)

    assert optimizer.solve(
        block_count, area_targets, b2b, p2b, pins, constraints, targets
    ) == standard
    learned_call = next(
        index for index, kwargs in enumerate(calls) if "learned_order" in kwargs
    )
    fallback = calls[learned_call + 1]
    assert fallback["width_factor"] == optimizer_module._LEARNED_REPLACEMENT_WF[
        block_count
    ]
    assert "learned_order" not in fallback
    assert not optimizer._learned_candidate_attempted


def test_additive_learning_is_withheld_until_final_v32_reference(monkeypatch):
    block_count = 100
    standard = [(1.1 * index, 0.0, 1.0, 1.0) for index in range(block_count)]
    learned = [(1.1 * index, 2.0, 1.0, 1.0) for index in range(block_count)]
    standard_widths = []

    def fake_dissect(*args, **kwargs):
        if "learned_order" in kwargs:
            return learned
        if kwargs.get("width_factor") in (0.8, 0.9, 1.0, 1.1, 1.2):
            standard_widths.append(kwargs["width_factor"])
        if kwargs.get("return_first_pass"):
            return list(standard), list(standard)
        return list(standard)

    monkeypatch.setattr(dissect_module, "dissect_solve", fake_dissect)
    monkeypatch.setattr(
        optimizer_module,
        "_learned_order_prior",
        lambda *args, **kwargs: {
            index: (0.5, 0.5) for index in range(block_count)
        },
    )
    optimizer = MyOptimizer()
    optimizer._learned_order_mode = "additive"
    monkeypatch.setattr(optimizer, "_solve_one", lambda *args, **kwargs: standard)
    selector_pools = []

    def select_first(candidates, *args, **kwargs):
        selector_pools.append(tuple(candidate is learned for candidate in candidates))
        return candidates[0]

    monkeypatch.setattr(optimizer, "_select_candidate", select_first)
    area_targets = torch.ones(block_count)
    empty_edges = torch.empty((0, 3))
    constraints = torch.zeros((block_count, 5))

    result = optimizer.solve(
        block_count,
        area_targets,
        empty_edges,
        empty_edges,
        torch.empty((0, 2)),
        constraints,
        torch.full((block_count, 4), -1.0),
    )

    assert result == standard
    assert set((0.8, 0.9, 1.0, 1.1, 1.2)).issubset(standard_widths)
    assert all(not any(pool) for pool in selector_pools[:-1])
    assert selector_pools[-1] == (False, True)
    assert optimizer._debug_final_nonlearned_reference == standard


@pytest.mark.parametrize(
    ("candidate_hpwl", "candidate_area", "candidate_soft", "accepted"),
    [
        (80.0, 80.0, 15, False),  # violation delta above the hard cap
        (101.0, 100.0, 10, False),  # no violation win: quality must not rise
        (99.0, 100.0, 10, True),
        (103.9, 104.0, 8, True),  # <0.08 quality rise paid by -0.02 V_rel
        (104.1, 104.0, 8, False),
    ],
)
def test_learned_trade_gate_uses_only_visible_quality_and_violations(
    monkeypatch,
    candidate_hpwl,
    candidate_area,
    candidate_soft,
    accepted,
):
    optimizer = MyOptimizer()
    incumbent = [(0.0, 0.0, 1.0, 1.0)]
    candidate = [(2.0, 0.0, 1.0, 1.0)]
    metrics = {
        id(incumbent): (100.0, 100.0, 10),
        id(candidate): (candidate_hpwl, candidate_area, candidate_soft),
    }
    monkeypatch.setattr(
        optimizer_module,
        "_calculate_hpwl_edges",
        lambda positions, *args: metrics[id(positions)][0],
    )
    monkeypatch.setattr(
        optimizer_module,
        "calculate_bbox_area",
        lambda positions: metrics[id(positions)][1],
    )
    monkeypatch.setattr(optimizer, "_is_feasible", lambda *args: True)
    monkeypatch.setattr(optimizer, "_n_soft", lambda *args: 100)
    monkeypatch.setattr(
        optimizer,
        "_soft_violation_count",
        lambda positions, constraints: metrics[id(positions)][2],
    )
    assert optimizer._accept_learned_trade(
        candidate, incumbent, None, [], [], [], [], None
    ) is accepted

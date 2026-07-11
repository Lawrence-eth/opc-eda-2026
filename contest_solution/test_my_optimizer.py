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
    _should_try_anchored_third_pass,
    _should_try_preplaced_aspect_pass,
)
from dissect import dissect_solve


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

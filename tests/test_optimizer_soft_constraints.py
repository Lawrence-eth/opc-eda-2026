import importlib
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_optimizer():
    """Import the public optimizer with a tiny evaluator stub.

    The contest checkout provides iccad2026_evaluate.py at evaluation time.
    These regression tests exercise optimizer-local helpers, so a minimal stub
    keeps the public test suite runnable without vendoring the official repo.
    """

    if "iccad2026_evaluate" not in sys.modules:
        stub = types.ModuleType("iccad2026_evaluate")

        class FloorplanOptimizer:
            def __init__(self, verbose=False):
                self.verbose = verbose

        stub.FloorplanOptimizer = FloorplanOptimizer
        stub.calculate_bbox_area = lambda positions: 0.0
        stub.calculate_hpwl_b2b = lambda positions, conn: 0.0
        stub.calculate_hpwl_p2b = lambda positions, conn, pins: 0.0
        sys.modules["iccad2026_evaluate"] = stub

    solution_dir = Path(__file__).resolve().parents[1] / "contest_solution"
    sys.path.insert(0, str(solution_dir))
    return importlib.import_module("my_optimizer")


optimizer_module = _load_optimizer()
MyOptimizer = optimizer_module.MyOptimizer
calculate_hpwl_edges = optimizer_module._calculate_hpwl_edges
tensor_to_list = optimizer_module._tensor_to_list
should_try_anchored_third_pass = (
    optimizer_module._should_try_anchored_third_pass
)
should_try_preplaced_aspect_pass = (
    optimizer_module._should_try_preplaced_aspect_pass
)
dissect_module = importlib.import_module("dissect")
topology_polish_module = importlib.import_module("topology_polish")


def _constraints(block_count):
    return torch.zeros((block_count, 5), dtype=torch.float32)


def test_list_hpwl_matches_official_arithmetic_for_lists_and_tensors():
    positions = [
        (0.0, 1.0, 2.0, 4.0),
        (5.0, 3.0, 6.0, 2.0),
        (2.0, 8.0, 4.0, 2.0),
    ]
    b2b = [(0, 1, 1.5), (1, 2, 0.25)]
    p2b = [(0, 0, 2.0), (1, 2, 0.75)]
    pins = [(10.0, 4.0), (1.0, 12.0)]

    centers = [(x + w / 2, y + h / 2) for x, y, w, h in positions]
    b2b_total = sum(
        weight * (abs(centers[b][0] - centers[a][0])
                  + abs(centers[b][1] - centers[a][1]))
        for a, b, weight in b2b
    )
    p2b_total = sum(
        weight * (abs(pins[pin][0] - centers[block][0])
                  + abs(pins[pin][1] - centers[block][1]))
        for pin, block, weight in p2b
    )
    expected = b2b_total + p2b_total

    assert calculate_hpwl_edges(positions, b2b, p2b, pins) == expected
    assert calculate_hpwl_edges(
        positions,
        torch.tensor(b2b),
        torch.tensor(p2b),
        torch.tensor(pins),
    ) == expected


def test_tensor_to_list_supports_packaging_stub_protocol():
    class StubTensor:
        def tolist(self):
            return [[1.0, 2.0]]

    assert tensor_to_list(StubTensor()) == [[1.0, 2.0]]


def test_anchored_third_pass_feature_gate_is_narrow():
    assert should_try_anchored_third_pass(102, 34, 1, 2184, 45)
    rejected = [
        (99, 34, 1, 2184, 45),
        (104, 34, 1, 2184, 45),
        (102, 33, 1, 2184, 45),
        (102, 34, 2, 2184, 45),
        (102, 34, 1, 1799, 45),
        (102, 34, 1, 2501, 45),
        (102, 34, 1, 2184, 101),
    ]
    assert not any(should_try_anchored_third_pass(*case) for case in rejected)


def test_preplaced_aspect_pass_feature_gate_is_narrow():
    assert should_try_preplaced_aspect_pass(82, 22, 7, 336, 1367)
    rejected = [
        (77, 22, 7, 336, 1367),
        (87, 22, 7, 336, 1367),
        (82, 25, 7, 336, 1367),
        (82, 22, 5, 336, 1367),
        (82, 22, 7, 501, 1367),
        (82, 22, 7, 336, 999),
        (82, 22, 7, 336, 1801),
    ]
    assert not any(should_try_preplaced_aspect_pass(*case) for case in rejected)


def test_group_components_requires_exact_positive_length_edge_contact():
    opt = MyOptimizer()
    exact = [(0.0, 0.0, 1.0, 1.0), (1.0, 0.25, 1.0, 0.5)]
    tiny_gap = [(0.0, 0.0, 1.0, 1.0), (1.0 + 5e-7, 0.25, 1.0, 0.5)]
    corner_only = [(0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)]

    assert opt._group_components(exact, [0, 1]) == 1
    assert opt._group_components(tiny_gap, [0, 1]) == 2
    assert opt._group_components(corner_only, [0, 1]) == 2


def test_fixed_topology_polish_reduces_hpwl_without_changing_bbox_or_dims():
    positions = [
        (0.0, 0.0, 1.0, 1.0),
        (4.0, 0.0, 1.0, 1.0),
        (9.0, 0.0, 1.0, 1.0),
    ]
    constraints = [[0.0] * 5 for _ in positions]
    polished = topology_polish_module.polish_fixed_topology(
        positions,
        [(1, 2, 1.0)],
        [],
        [],
        constraints,
    )

    assert polished[1][0] == 8.0
    assert [(p[2], p[3]) for p in polished] == [(1.0, 1.0)] * 3
    assert min(p[0] for p in polished) == 0.0
    assert max(p[0] + p[2] for p in polished) == 10.0
    assert MyOptimizer()._is_feasible(
        polished, torch.tensor(constraints), torch.ones(3), None
    )


def test_fixed_topology_polish_keeps_preplaced_coordinate():
    positions = [
        (0.0, 0.0, 1.0, 1.0),
        (4.0, 0.0, 1.0, 1.0),
        (9.0, 0.0, 1.0, 1.0),
    ]
    constraints = [[0.0] * 5 for _ in positions]
    constraints[1][1] = 1.0
    polished = topology_polish_module.polish_fixed_topology(
        positions,
        [(1, 2, 1.0)],
        [],
        [],
        constraints,
    )

    assert polished[1][:2] == positions[1][:2]


def test_clamped_row_backfill_uses_queued_unit_that_fits_short_slab():
    case = dissect_module.Case(
        3,
        [40.0, 200.0, 4.0],
        [[0, 0, 0, 0, 0]] * 3,
        None,
    )
    head = dissect_module.Unit([0], "block", case)
    large = dissect_module.Unit([1], "block", case)
    small = dissect_module.Unit([2], "block", case)
    obstacle = [(8.0, 2.0, 2.0, 2.0)]

    without_backfill = {}
    dissect_module.fill_region(
        case,
        [large, small],
        0.0,
        10.0,
        0.0,
        obstacle,
        without_backfill,
        l_queue=[head],
    )
    with_backfill = {}
    dissect_module.fill_region(
        case,
        [large, small],
        0.0,
        10.0,
        0.0,
        obstacle,
        with_backfill,
        l_queue=[head],
        clamped_backfill=True,
    )

    assert without_backfill[2][1] == 2.0
    assert with_backfill[2][1] == 0.0


def test_active_slab_aspect_limit_controls_legal_short_slab_fill():
    case = dissect_module.Case(
        1,
        [4.0],
        [[0, 0, 0, 0, 0]],
        None,
    )
    obstacle = [(8.0, 0.0, 2.0, 0.5)]

    default_fill = {}
    dissect_module.fill_region(
        case,
        [dissect_module.Unit([0], "block", case)],
        0.0,
        10.0,
        0.0,
        obstacle,
        default_fill,
    )
    relaxed_fill = {}
    dissect_module.fill_region(
        case,
        [dissect_module.Unit([0], "block", case)],
        0.0,
        10.0,
        0.0,
        obstacle,
        relaxed_fill,
        active_slab_max_aspect=18.0,
    )

    assert default_fill[0] == (0.0, 0.5, 10.0, 0.4)
    assert relaxed_fill[0] == (0.0, 0.0, 8.0, 0.5)


def test_group_components_require_shared_edge_not_corner_touch():
    opt = MyOptimizer()
    positions = [
        (0.0, 0.0, 2.0, 2.0),
        (2.0, 0.0, 3.0, 2.0),
        (5.0, 2.0, 1.0, 1.0),
    ]

    assert opt._group_components(positions, [0, 1, 2]) == 2


def test_soft_violation_count_handles_exact_corners_and_edges():
    opt = MyOptimizer()
    constraints = _constraints(3)
    constraints[0, 4] = 5   # left + top
    constraints[1, 4] = 8   # bottom
    constraints[2, 4] = 2   # right
    positions = [
        (0.0, 2.0, 2.0, 1.0),
        (0.0, 0.0, 2.0, 2.0),
        (2.0, 0.0, 1.0, 3.0),
    ]

    assert opt._soft_violation_count(positions, constraints) == 0

    shifted = [(0.25, 2.0, 2.0, 1.0), positions[1], positions[2]]
    assert opt._soft_violation_count(shifted, constraints) == 1


def test_mib_dimensions_normalize_only_when_areas_are_compatible():
    opt = MyOptimizer()
    target_positions = torch.full((2, 4), -1.0)
    constraints = _constraints(2)
    constraints[:, 2] = 1

    compatible = opt._choose_dimensions(2, torch.tensor([100.0, 100.5]), constraints, target_positions)
    assert compatible[0] == compatible[1]
    assert math.isclose(compatible[0][0] * compatible[0][1], 100.25)

    incompatible = opt._choose_dimensions(2, torch.tensor([100.0, 121.0]), constraints, target_positions)
    assert incompatible[0] != incompatible[1]
    assert math.isclose(incompatible[0][0] * incompatible[0][1], 100.0)
    assert math.isclose(incompatible[1][0] * incompatible[1][1], 121.0)


def test_boundary_cluster_pack_keeps_edge_members_on_edge_and_mates_inward():
    opt = MyOptimizer()
    dims = [(2.0, 3.0), (1.5, 2.0), (4.0, 2.0)]
    area_targets = torch.tensor([6.0, 3.0, 8.0])

    local, unit_w, unit_h = opt._boundary_cluster_local_pack(
        bmembers=[0, 1],
        mates=[2],
        code=1,
        dims=dims,
        area_targets=area_targets,
        b2b_connectivity=torch.empty((0, 3)),
        p2b_connectivity=torch.empty((0, 3)),
    )

    assert local[0][0] == 0.0
    assert local[1][0] == 0.0
    assert local[2][0] >= max(dims[0][0], dims[1][0])
    assert unit_w >= local[2][0] + local[2][2]
    assert unit_h >= max(local[i][1] + local[i][3] for i in local)
    assert opt._group_components([local[i] for i in range(3)], [0, 1, 2]) == 1

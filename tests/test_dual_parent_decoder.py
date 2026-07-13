from dataclasses import replace

import pytest

from contest_solution.dual_parent_decoder import (
    DualParentError,
    compare_geometry,
    decode_dual_parent,
    enumerate_oriented_factor_shapes,
    extract_oracle_labels,
    hard_targets_from_golden,
)


def _structural_example(*, translated=False):
    """A tree whose side-1 relation points downward in physical y.

    This captures a real FloorSet counterexample: in worker_0/layouts_0,
    edge (17, 11, side=1) has parent (w=15,h=6,x=99,y=113) and child
    (w=9,h=18,x=99,y=83).  Side 1 means equal x only; using it as a skyline
    or vertical-parent relation is structurally lossy.
    """

    dx, dy = (10.0, 20.0) if translated else (0.0, 0.0)
    rectangles = [
        (dx + 0.0, dy + 0.0, 2.0, 2.0),
        (dx + 2.0, dy + 0.0, 2.0, 1.0),
        (dx + 2.0, dy + 3.0, 1.0, 1.0),
        (dx + 2.0, dy + 1.0, 1.0, 2.0),
    ]
    # Block 2 is the side-1 child of block 1 even though it is above block 1;
    # block 3 is a side-1 child of block 2 even though it is below block 2.
    tree = [[0, 1, 0], [1, 2, 1], [2, 3, 1]]
    areas = [4.0, 2.0, 1.0, 2.0]
    constraints = [[0.0] * 5 for _ in rectangles]
    return areas, constraints, tree, rectangles


def test_enumerates_complete_oriented_exact_factor_shapes():
    shapes = enumerate_oriented_factor_shapes(12)
    assert set(shapes) == {(2.0, 6.0), (6.0, 2.0), (3.0, 4.0), (4.0, 3.0)}
    assert enumerate_oriented_factor_shapes(10.5) == ()
    assert enumerate_oriented_factor_shapes(7) == ()  # 1x7 violates 3:1


def test_independent_vertical_support_reconstructs_side_one_counterexample():
    areas, constraints, tree, golden = _structural_example()
    labels = extract_oracle_labels(areas, constraints, tree, golden)

    horizontal_parent = {
        relation.child: relation.parent for relation in labels.horizontal
    }
    assert horizontal_parent[2] == 1
    assert labels.vertical_supports[2] == 3
    assert golden[horizontal_parent[2]][1] < golden[2][1]
    assert labels.vertical_supports[3] == 1
    assert golden[horizontal_parent[3]][1] > golden[3][1]

    decoded = decode_dual_parent(labels, constraints=constraints, hard_targets=[])
    comparison = compare_geometry(decoded, golden)
    assert comparison.is_exact()
    assert comparison.max_coordinate_delta == 0.0
    assert comparison.max_dimension_delta == 0.0


def test_preplaced_anchor_recovers_global_translation_and_fixed_dimensions():
    areas, constraints, tree, golden = _structural_example(translated=True)
    constraints[0][1] = 1.0  # preplaced root fixes x/y and dimensions
    constraints[1][0] = 1.0  # fixed block fixes dimensions only
    labels = extract_oracle_labels(areas, constraints, tree, golden)
    hard_targets = hard_targets_from_golden(constraints, golden)

    decoded = decode_dual_parent(
        labels, constraints=constraints, hard_targets=hard_targets
    )
    assert compare_geometry(decoded, golden).is_exact()
    assert decoded[0] == golden[0]
    assert decoded[1][2:] == golden[1][2:]


def test_corrupt_mib_label_is_classified_but_geometry_remains_diagnostic():
    areas, constraints, tree, golden = _structural_example()
    constraints[1][2] = 1.0
    constraints[2][2] = 1.0
    labels = extract_oracle_labels(areas, constraints, tree, golden)
    assert labels.mib_inconsistent_groups == (1,)

    with pytest.raises(DualParentError) as exc_info:
        decode_dual_parent(labels, constraints=constraints, hard_targets=[])
    assert exc_info.value.code == "mib_shape_mismatch"

    decoded = decode_dual_parent(
        labels,
        constraints=constraints,
        hard_targets=[],
        enforce_mib=False,
    )
    assert compare_geometry(decoded, golden).is_exact()


def test_extraction_rejects_a_false_horizontal_label_with_stable_taxonomy():
    areas, constraints, tree, golden = _structural_example()
    broken = list(golden)
    x, y, width, height = broken[2]
    broken[2] = (x + 1.0, y, width, height)
    with pytest.raises(DualParentError) as exc_info:
        extract_oracle_labels(areas, constraints, tree, broken)
    assert exc_info.value.code == "horizontal_relation_mismatch"


def test_decoder_rejects_vertical_support_cycles():
    areas, constraints, tree, golden = _structural_example()
    labels = extract_oracle_labels(areas, constraints, tree, golden)
    cyclic = replace(labels, vertical_supports=(1, 0, 3, 1))
    with pytest.raises(DualParentError) as exc_info:
        decode_dual_parent(cyclic, constraints=constraints, hard_targets=[])
    assert exc_info.value.code == "vertical_support_cycle"


def test_decoder_rejects_dimension_that_disagrees_with_selected_shape_category():
    areas, constraints, tree, golden = _structural_example()
    labels = extract_oracle_labels(areas, constraints, tree, golden)
    dimensions = list(labels.dimensions)
    dimensions[0] = (1.0, 4.0)
    inconsistent = replace(labels, dimensions=tuple(dimensions))
    with pytest.raises(DualParentError) as exc_info:
        decode_dual_parent(inconsistent, constraints=constraints, hard_targets=[])
    assert exc_info.value.code == "selected_shape_dimension_mismatch"


def test_bstar_tree_requires_one_child_per_parent_side_slot():
    areas, constraints, _tree, golden = _structural_example()
    malformed_tree = [[0, 1, 0], [0, 2, 0], [2, 3, 1]]
    with pytest.raises(DualParentError) as exc_info:
        extract_oracle_labels(areas, constraints, malformed_tree, golden)
    assert exc_info.value.code == "duplicate_bstar_child_slot"

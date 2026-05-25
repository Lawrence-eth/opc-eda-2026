"""TDD tests for Sequence Pair (SP) label extraction.

These tests intentionally define expected behavior before implementation.
They should fail until the corresponding API is implemented.
"""

from __future__ import annotations

import pytest

from scripts import sp_labels


@pytest.fixture
def rectangles_abc() -> list[tuple[float, float, float, float]]:
    """[w, h, x, y] blocks with known relations.

    A: (w=10, h=5, x=0,  y=0)
    B: (w=8,  h=5, x=12, y=0)  -> right of A
    C: (w=10, h=4, x=0,  y=7)  -> above A
    """
    return [
        (10.0, 5.0, 0.0, 0.0),
        (8.0, 5.0, 12.0, 0.0),
        (10.0, 4.0, 0.0, 7.0),
    ]


def _is_valid_permutation(order: list[int], n: int) -> bool:
    return len(order) == n and sorted(order) == list(range(n))


def _no_overlap(placed: list[tuple[float, float, float, float]]) -> bool:
    for i in range(len(placed)):
        xi, yi, wi, hi = placed[i]
        for j in range(i + 1, len(placed)):
            xj, yj, wj, hj = placed[j]
            x_overlap = xi < (xj + wj) and xj < (xi + wi)
            y_overlap = yi < (yj + hj) and yj < (yi + hi)
            if x_overlap and y_overlap:
                return False
    return True


def test_rectangles_to_relations(rectangles_abc):
    """Extract pairwise left/right/below/above labels including edge-touch cases."""
    relations = sp_labels.extract_pairwise_relations(rectangles_abc)

    # A(0) vs B(1)
    assert relations[0][1] == "left"
    assert relations[1][0] == "right"

    # A(0) vs C(2)
    assert relations[0][2] == "below"
    assert relations[2][0] == "above"

    # Touching case: D touches A on right edge (x+w == other_x)
    rectangles_touch = [
        (10.0, 5.0, 0.0, 0.0),
        (3.0, 3.0, 10.0, 0.0),
    ]
    touch_rel = sp_labels.extract_pairwise_relations(rectangles_touch)
    assert touch_rel[0][1] == "left"
    assert touch_rel[1][0] == "right"

    # Ambiguous diagonal case should be deterministic.
    rectangles_ambiguous = [
        (4.0, 4.0, 0.0, 0.0),
        (4.0, 4.0, 3.0, 3.0),
    ]
    amb_rel = sp_labels.extract_pairwise_relations(rectangles_ambiguous)
    assert amb_rel[0][1] in {"left", "right", "below", "above"}
    assert amb_rel[1][0] in {"left", "right", "below", "above"}
    assert amb_rel[0][1] != amb_rel[1][0]


def test_relations_to_sequence_pair(rectangles_abc):
    """Convert relation labels to two valid SP permutations."""
    relations = sp_labels.extract_pairwise_relations(rectangles_abc)
    sp_plus, sp_minus = sp_labels.relations_to_sequence_pair(relations)

    n = len(rectangles_abc)
    assert _is_valid_permutation(sp_plus, n)
    assert _is_valid_permutation(sp_minus, n)

    # Horizontal: A left of B => A before B in both sequences.
    assert sp_plus.index(0) < sp_plus.index(1)
    assert sp_minus.index(0) < sp_minus.index(1)

    # Vertical: A below C => A before C in plus, after C in minus.
    assert sp_plus.index(0) < sp_plus.index(2)
    assert sp_minus.index(0) > sp_minus.index(2)


def test_sequence_pair_packing(rectangles_abc):
    """Pack from SP and verify overlap-free valid placement."""
    relations = sp_labels.extract_pairwise_relations(rectangles_abc)
    sp_plus, sp_minus = sp_labels.relations_to_sequence_pair(relations)

    placed = sp_labels.pack_from_sequence_pair(rectangles_abc, sp_plus, sp_minus)

    assert len(placed) == 3
    assert _no_overlap(placed)

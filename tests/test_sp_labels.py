"""TDD tests for Sequence Pair (SP) topology label extraction.

These tests define the expected API and behavior for:
1. Converting ground-truth floorplan rectangles into pairwise topological relations.
2. Converting relations into two valid Sequence Pair permutations.
3. Packing from a Sequence Pair to verify overlap-free reconstruction.

All tests are expected to FAIL until the implementation is written.
"""

from __future__ import annotations

import math

import pytest

# The implementation module does not exist yet — this is TDD.
from scripts import sp_labels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rect_a():
    """Block A: w=10, h=5, x=0, y=0."""
    return (10.0, 5.0, 0.0, 0.0)


@pytest.fixture
def rect_b():
    """Block B: w=8, h=5, x=12, y=0 — strictly to the RIGHT of A."""
    return (8.0, 5.0, 12.0, 0.0)


@pytest.fixture
def rect_c():
    """Block C: w=10, h=4, x=0, y=7 — strictly ABOVE A."""
    return (10.0, 4.0, 0.0, 7.0)


@pytest.fixture
def rect_d():
    """Block D: w=6, h=3, x=12, y=7 — to the right of A and above B."""
    return (6.0, 3.0, 12.0, 7.0)


@pytest.fixture
def rect_e_touching_right():
    """Block E: w=5, h=5, x=10, y=0 — touches A on the right edge."""
    return (5.0, 5.0, 10.0, 0.0)


@pytest.fixture
def rect_f_touching_above():
    """Block F: w=10, h=3, x=0, y=5 — touches A on the top edge."""
    return (10.0, 3.0, 0.0, 5.0)


@pytest.fixture
def rect_g_ambiguous():
    """Block G: w=2, h=2, x=11, y=6 — ambiguous diagonal placement."""
    return (2.0, 2.0, 11.0, 6.0)


# ---------------------------------------------------------------------------
# 1. rectangles → pairwise relations
# ---------------------------------------------------------------------------

def test_rectangles_to_relations_basic_horizontal(rect_a, rect_b):
    """Two non-overlapping rectangles side-by-side should classify as left/right."""
    rectangles = [rect_a, rect_b]
    relations = sp_labels.rectangles_to_relations(rectangles)

    # Block 0 (A) is to the LEFT of block 1 (B)
    assert relations[0][1] == "left"
    # Block 1 (B) is to the RIGHT of block 0 (A)
    assert relations[1][0] == "right"


def test_rectangles_to_relations_basic_vertical(rect_a, rect_c):
    """Two non-overlapping rectangles stacked vertically should classify as below/above."""
    rectangles = [rect_a, rect_c]
    relations = sp_labels.rectangles_to_relations(rectangles)

    # Block 0 (A) is BELOW block 1 (C)
    assert relations[0][1] == "below"
    # Block 1 (C) is ABOVE block 0 (A)
    assert relations[1][0] == "above"


def test_rectangles_to_relations_four_blocks(rect_a, rect_b, rect_c, rect_d):
    """A 2x2 grid should produce consistent pairwise relations."""
    rectangles = [rect_a, rect_b, rect_c, rect_d]
    relations = sp_labels.rectangles_to_relations(rectangles)

    # A (0) left of B (1)
    assert relations[0][1] == "left"
    # A (0) below C (2)
    assert relations[0][2] == "below"
    # B (1) below D (3)
    assert relations[1][3] == "below"
    # C (2) left of D (3)
    assert relations[2][3] == "left"

    # Inverse relations must be consistent
    assert relations[1][0] == "right"
    assert relations[2][0] == "above"
    assert relations[3][1] == "above"
    assert relations[3][2] == "right"


def test_rectangles_to_relations_touching_right_is_right(rect_a, rect_e_touching_right):
    """Touching on the right edge (x+w == other_x) should still be 'right'."""
    rectangles = [rect_a, rect_e_touching_right]
    relations = sp_labels.rectangles_to_relations(rectangles)

    # A at (0,0) w=10, E at (10,0) — they touch. E is RIGHT of A.
    assert relations[0][1] == "left"
    assert relations[1][0] == "right"


def test_rectangles_to_relations_touching_above_is_above(rect_a, rect_f_touching_above):
    """Touching on the top edge (y+h == other_y) should still be 'above'."""
    rectangles = [rect_a, rect_f_touching_above]
    relations = sp_labels.rectangles_to_relations(rectangles)

    # A at (0,0) h=5, F at (0,5) — they touch. F is ABOVE A.
    assert relations[0][1] == "below"
    assert relations[1][0] == "above"


def test_rectangles_to_relations_diagonal_deterministic_tie_break(rect_a, rect_g_ambiguous):
    """Diagonal placement must resolve to a deterministic relation via tie-breaking."""
    rectangles = [rect_a, rect_g_ambiguous]
    relations = sp_labels.rectangles_to_relations(rectangles)

    # G is at (11,6): to the right and above A.
    # The function must pick ONE deterministic relation.
    rel = relations[0][1]
    assert rel in ("left", "right", "below", "above")
    # The inverse must be consistent
    inverse = relations[1][0]
    assert inverse == sp_labels._inverse_relation(rel)


def test_rectangles_to_relations_self_is_none(rect_a):
    """Diagonal entries (i vs i) should be None."""
    rectangles = [rect_a]
    relations = sp_labels.rectangles_to_relations(rectangles)
    assert relations[0][0] is None


def test_rectangles_to_relations_returns_square_matrix(rect_a, rect_b, rect_c):
    """Output must be an n×n matrix."""
    rectangles = [rect_a, rect_b, rect_c]
    relations = sp_labels.rectangles_to_relations(rectangles)
    assert len(relations) == 3
    assert all(len(row) == 3 for row in relations)


# ---------------------------------------------------------------------------
# 2. relations → Sequence Pair permutations
# ---------------------------------------------------------------------------

def test_relations_to_sequence_pair_two_blocks_horizontal():
    """Two blocks side-by-side should produce valid SP permutations."""
    # 0 is left of 1
    relations = [
        [None, "left"],
        ["right", None],
    ]
    sp_plus, sp_minus = sp_labels.relations_to_sequence_pair(relations)

    # Both permutations must contain exactly [0, 1]
    assert sorted(sp_plus) == [0, 1]
    assert sorted(sp_minus) == [0, 1]
    # 0 before 1 in both sequences → 0 is left of 1
    assert sp_plus.index(0) < sp_plus.index(1)
    assert sp_minus.index(0) < sp_minus.index(1)


def test_relations_to_sequence_pair_two_blocks_vertical():
    """Two blocks stacked vertically should produce valid SP permutations."""
    # 0 is below 1
    relations = [
        [None, "below"],
        ["above", None],
    ]
    sp_plus, sp_minus = sp_labels.relations_to_sequence_pair(relations)

    assert sorted(sp_plus) == [0, 1]
    assert sorted(sp_minus) == [0, 1]
    # 0 before 1 in Γ+ and after 1 in Γ− → 0 is below 1
    assert sp_plus.index(0) < sp_plus.index(1)
    assert sp_minus.index(0) > sp_minus.index(1)


def test_relations_to_sequence_pair_four_blocks():
    """A 2x2 grid should produce two valid topological orders."""
    # Layout:
    #   C(2)  D(3)
    #   A(0)  B(1)
    relations = [
        [None, "left", "below", "below"],
        ["right", None, "below", "below"],
        ["above", "above", None, "left"],
        ["above", "above", "right", None],
    ]
    sp_plus, sp_minus = sp_labels.relations_to_sequence_pair(relations)

    assert sorted(sp_plus) == [0, 1, 2, 3]
    assert sorted(sp_minus) == [0, 1, 2, 3]

    # Verify SP semantics against the relations
    _assert_sp_matches_relations(sp_plus, sp_minus, relations)


def test_relations_to_sequence_pair_is_consistent_with_inverse():
    """If i is left of j, the SP must reflect that regardless of how we query."""
    relations = [
        [None, "left", "below"],
        ["right", None, "below"],
        ["above", "above", None],
    ]
    sp_plus, sp_minus = sp_labels.relations_to_sequence_pair(relations)
    _assert_sp_matches_relations(sp_plus, sp_minus, relations)


# ---------------------------------------------------------------------------
# 3. Sequence Pair → packing (longest-path algorithm)
# ---------------------------------------------------------------------------

def test_sequence_pair_packing_two_blocks_horizontal(rect_a, rect_b):
    """Packing two side-by-side blocks from SP should place B to the right of A."""
    rectangles = [rect_a, rect_b]
    sp_plus = [0, 1]
    sp_minus = [0, 1]

    positions = sp_labels.pack_sequence_pair(rectangles, sp_plus, sp_minus)

    # positions should be list of (x, y, w, h)
    assert len(positions) == 2
    ax, ay, aw, ah = positions[0]
    bx, by, bw, bh = positions[1]

    # A stays at origin
    assert ax == 0.0
    assert ay == 0.0
    # B is to the right of A, same y
    assert bx >= ax + aw
    assert by == ay
    # No overlap
    assert bx >= ax + aw or by >= ay + ah


def test_sequence_pair_packing_two_blocks_vertical(rect_a, rect_c):
    """Packing two stacked blocks from SP should place C above A."""
    rectangles = [rect_a, rect_c]
    # 0 below 1: 0 before 1 in Γ+, 0 after 1 in Γ−
    sp_plus = [0, 1]
    sp_minus = [1, 0]

    positions = sp_labels.pack_sequence_pair(rectangles, sp_plus, sp_minus)

    ax, ay, aw, ah = positions[0]
    cx, cy, cw, ch = positions[1]

    assert ax == 0.0
    assert ay == 0.0
    assert cx == ax  # same x alignment for this simple case
    assert cy >= ay + ah


def test_sequence_pair_packing_four_blocks(rect_a, rect_b, rect_c, rect_d):
    """Packing a 2x2 grid should produce an overlap-free placement."""
    rectangles = [rect_a, rect_b, rect_c, rect_d]
    # Relations:
    #   C(2)  D(3)
    #   A(0)  B(1)
    # A left of B, A below C, B below D, C left of D
    relations = [
        [None, "left", "below", "below"],
        ["right", None, "below", "below"],
        ["above", "above", None, "left"],
        ["above", "above", "right", None],
    ]
    sp_plus, sp_minus = sp_labels.relations_to_sequence_pair(relations)
    positions = sp_labels.pack_sequence_pair(rectangles, sp_plus, sp_minus)

    assert len(positions) == 4
    _assert_no_overlap(positions)


def test_sequence_pair_packing_reconstructs_original_layout(rect_a, rect_b, rect_c, rect_d):
    """Round-trip: rectangles → relations → SP → packing should reconstruct original layout."""
    original = [rect_a, rect_b, rect_c, rect_d]

    relations = sp_labels.rectangles_to_relations(original)
    sp_plus, sp_minus = sp_labels.relations_to_sequence_pair(relations)
    packed = sp_labels.pack_sequence_pair(original, sp_plus, sp_minus)

    # For a feasible floorplan, packed positions should match original within tolerance.
    # Fixtures are (w, h, x, y); packed positions are (x, y, w, h).
    for (ow, oh, ox, oy), (px, py, pw, ph) in zip(original, packed):
        assert math.isclose(ox, px, abs_tol=1e-6)
        assert math.isclose(oy, py, abs_tol=1e-6)
        assert math.isclose(ow, pw, abs_tol=1e-6)
        assert math.isclose(oh, ph, abs_tol=1e-6)


def test_sequence_pair_packing_with_touching_blocks(rect_a, rect_e_touching_right, rect_f_touching_above):
    """Touching blocks should pack without overlap and preserve adjacency."""
    rectangles = [rect_a, rect_e_touching_right, rect_f_touching_above]
    relations = sp_labels.rectangles_to_relations(rectangles)
    sp_plus, sp_minus = sp_labels.relations_to_sequence_pair(relations)
    positions = sp_labels.pack_sequence_pair(rectangles, sp_plus, sp_minus)

    assert len(positions) == 3
    _assert_no_overlap(positions)

    # E touches A on the right
    ax, ay, aw, ah = positions[0]
    ex, ey, ew, eh = positions[1]
    assert math.isclose(ex, ax + aw, abs_tol=1e-6)

    # F touches A on the top
    fx, fy, fw, fh = positions[2]
    assert math.isclose(fy, ay + ah, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 4. Edge cases and invariants
# ---------------------------------------------------------------------------

def test_inverse_relation_symmetry():
    """_inverse_relation must be a proper involution."""
    pairs = [
        ("left", "right"),
        ("right", "left"),
        ("below", "above"),
        ("above", "below"),
    ]
    for a, b in pairs:
        assert sp_labels._inverse_relation(a) == b
        assert sp_labels._inverse_relation(b) == a
        assert sp_labels._inverse_relation(sp_labels._inverse_relation(a)) == a


def test_relations_to_sequence_pair_empty():
    """Zero blocks should produce empty permutations."""
    sp_plus, sp_minus = sp_labels.relations_to_sequence_pair([])
    assert sp_plus == []
    assert sp_minus == []


def test_relations_to_sequence_pair_single_block():
    """One block should produce single-element permutations."""
    sp_plus, sp_minus = sp_labels.relations_to_sequence_pair([[None]])
    assert sp_plus == [0]
    assert sp_minus == [0]


def test_pack_sequence_pair_empty():
    """Packing zero blocks should return an empty list."""
    positions = sp_labels.pack_sequence_pair([], [], [])
    assert positions == []


def test_pack_sequence_pair_single_block(rect_a):
    """Packing one block should place it at the origin."""
    positions = sp_labels.pack_sequence_pair([rect_a], [0], [0])
    assert len(positions) == 1
    x, y, w, h = positions[0]
    assert x == 0.0
    assert y == 0.0
    assert w == rect_a[0]
    assert h == rect_a[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_sp_matches_relations(sp_plus, sp_minus, relations):
    """Verify that two permutations encode the given relation matrix."""
    n = len(relations)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rel = relations[i][j]
            i_in_plus = sp_plus.index(i)
            j_in_plus = sp_plus.index(j)
            i_in_minus = sp_minus.index(i)
            j_in_minus = sp_minus.index(j)

            if rel == "left":
                assert i_in_plus < j_in_plus and i_in_minus < j_in_minus
            elif rel == "right":
                assert i_in_plus > j_in_plus and i_in_minus > j_in_minus
            elif rel == "below":
                assert i_in_plus < j_in_plus and i_in_minus > j_in_minus
            elif rel == "above":
                assert i_in_plus > j_in_plus and i_in_minus < j_in_minus
            else:
                pytest.fail(f"Unexpected relation '{rel}' at ({i}, {j})")


def _assert_no_overlap(positions):
    """Check that no pair of rectangles overlaps."""
    n = len(positions)
    for i in range(n):
        ix, iy, iw, ih = positions[i]
        for j in range(i + 1, n):
            jx, jy, jw, jh = positions[j]
            # Check for overlap in both x and y
            x_overlap = ix < jx + jw and jx < ix + iw
            y_overlap = iy < jy + jh and jy < iy + ih
            assert not (x_overlap and y_overlap), (
                f"Overlap between block {i} at ({ix},{iy},{iw},{ih}) "
                f"and block {j} at ({jx},{jy},{jw},{jh})"
            )

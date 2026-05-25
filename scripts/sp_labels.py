"""Stub module for Sequence Pair (SP) topology label extraction.

This module will be implemented to make the TDD tests in
``tests/test_sp_labels.py`` pass.  Every function currently raises
``NotImplementedError``.
"""

from __future__ import annotations

from typing import List, Tuple


Rect = Tuple[float, float, float, float]
RelationMatrix = List[List[str | None]]
Permutation = List[int]


def rectangles_to_relations(rectangles: List[Rect]) -> RelationMatrix:
    """Convert a list of [w, h, x, y] rectangles into pairwise topological relations.

    Args:
        rectangles: List of (width, height, x, y) tuples.

    Returns:
        n×n matrix where entry [i][j] is one of:
        ``"left"``, ``"right"``, ``"below"``, ``"above"``, or ``None`` (diagonal).
    """
    raise NotImplementedError("rectangles_to_relations is not yet implemented")


def relations_to_sequence_pair(relations: RelationMatrix) -> Tuple[Permutation, Permutation]:
    """Convert pairwise relations into two Sequence Pair permutations.

    Args:
        relations: n×n relation matrix from :func:`rectangles_to_relations`.

    Returns:
        (Γ+, Γ−) — two permutations of block indices that encode the
        topological relations via the standard Sequence Pair semantics.
    """
    raise NotImplementedError("relations_to_sequence_pair is not yet implemented")


def pack_sequence_pair(
    rectangles: List[Rect],
    sp_plus: Permutation,
    sp_minus: Permutation,
) -> List[Rect]:
    """Pack rectangles using the longest-path algorithm for a given Sequence Pair.

    Args:
        rectangles: List of (width, height, x, y) tuples.
        sp_plus: First permutation (Γ+).
        sp_minus: Second permutation (Γ−).

    Returns:
        List of packed (x, y, width, height) positions.
    """
    raise NotImplementedError("pack_sequence_pair is not yet implemented")


def _inverse_relation(rel: str) -> str:
    """Return the inverse of a topological relation.

    Args:
        rel: One of ``"left"``, ``"right"``, ``"below"``, ``"above"``.

    Returns:
        The inverse relation (e.g. ``"right"`` → ``"left"``).
    """
    raise NotImplementedError("_inverse_relation is not yet implemented")

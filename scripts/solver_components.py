#!/usr/bin/env python3
"""Authoritative list of source modules used by the live solver.

The official-copy scripts, synchronization audit, and holdout provenance all
consume this registry.  Keep implementation modules and generated deployment
artifacts adjacent here when adding a new live model family.
"""

from __future__ import annotations


SOLVER_ENTRYPOINT = "my_optimizer.py"

LIVE_SOLVER_COMPONENTS = (
    SOLVER_ENTRYPOINT,
    "dissect.py",
    "topology_polish.py",
    "learned_order.py",
    "order_model_v5b.py",
    "golden_plus_repair.py",
)


def main() -> None:
    """Print one component per line for the Bash copy workflows."""

    print("\n".join(LIVE_SOLVER_COMPONENTS))


if __name__ == "__main__":
    main()

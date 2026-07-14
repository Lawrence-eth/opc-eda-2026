#!/usr/bin/env python3
"""Executable entry point for the ICCAD 2026 Problem C submission.

Protocol (matches the organizers' op_wrapper.py):
  stdin : one JSON object with keys block_count, area_targets,
          b2b_connectivity, p2b_connectivity, pins_pos, constraints,
          target_positions (possibly null)
  stdout: {"positions": [[x, y, w, h], ...]}  (exactly block_count entries)

Never raises: one crashed case costs 10.0 (weighted, catastrophic on large
cases), so any internal failure falls back to a conservative always-feasible
layout — preplaced blocks at their exact (x,y,w,h), fixed-shape blocks at
their exact (w,h), soft blocks as sqrt-area squares, packed in shelf rows in
a strip strictly to the right of every preplaced block so no overlap is
possible by construction.
"""

import hashlib
import json
import math
import os
import sys

# Local stubs shadow the real packages inside the bundle: `torch` is
# packaging/torch_stub.py and `iccad2026_evaluate` is packaging/eval_stub.py,
# both shipped next to this file. my_optimizer.py is used unmodified.
import torch
from my_optimizer import (
    MyOptimizer,
    _LEARNED_REPLACEMENT_WF,
    _compiled_learned_model,
    _learned_order_prior,
)


_EXPECTED_MODEL_PAYLOAD_SHA256 = (
    "c94b4af92a7088f04206a5fa20dfbf807f945d9bdd80d9ffcbdc0b8b45f18beb"
)


def _fallback(block_count, area_targets, constraints, target_positions):
    """Always-hard-feasible layout from plain-list inputs."""
    rects = [None] * block_count
    preplaced = []
    for i in range(block_count):
        tp = (target_positions[i] if target_positions is not None
              else (-1.0, -1.0, -1.0, -1.0))
        is_pre = constraints is not None and float(constraints[i][1]) != 0
        if is_pre and float(tp[0]) != -1:
            r = (float(tp[0]), float(tp[1]), float(tp[2]), float(tp[3]))
            rects[i] = r
            preplaced.append(r)

    x0, y0 = 0.0, 0.0
    if preplaced:
        x0 = max(x + w for x, y, w, h in preplaced) + 1.0
        y0 = min(y for x, y, w, h in preplaced)

    rest = [i for i in range(block_count) if rects[i] is None]
    dims = {}
    for i in rest:
        tp = (target_positions[i] if target_positions is not None
              else (-1.0, -1.0, -1.0, -1.0))
        if float(tp[2]) != -1 and float(tp[3]) != -1:
            dims[i] = (float(tp[2]), float(tp[3]))
        else:
            a = float(area_targets[i])
            if not (a > 0):
                a = 1.0
            s = math.sqrt(a)
            dims[i] = (s, s)

    total = sum(w * h for w, h in dims.values()) or 1.0
    row_w = max(math.sqrt(total) * 1.3,
                max((w for w, _ in dims.values()), default=1.0))
    x, y, row_h = x0, y0, 0.0
    for i in sorted(rest, key=lambda i: -dims[i][1]):
        w, h = dims[i]
        if x > x0 and (x - x0) + w > row_w:
            x = x0
            y += row_h
            row_h = 0.0
        rects[i] = (x, y, w, h)
        x += w
        row_h = max(row_h, h)
    return rects


def _positions_ok(positions, block_count):
    if not isinstance(positions, list) or len(positions) != block_count:
        return False
    for p in positions:
        if p is None or len(p) != 4:
            return False
        for v in p:
            if not math.isfinite(float(v)):
                return False
    return True


def _canonical_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _learned_self_test():
    """Exercise the sealed model, prior, raw candidate, and final selector."""

    from dissect import dissect_solve
    from order_model_v5b import MODEL

    eligible_sizes = sorted(
        size for size in _LEARNED_REPLACEMENT_WF if 100 <= size <= 120
    )
    if not eligible_sizes or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1 or size > 120
        for size in eligible_sizes
    ):
        raise RuntimeError("learned replacement eligibility map is empty or malformed")
    n = eligible_sizes[0]
    areas = [1.0 + 0.01 * (index % 7) for index in range(n)]
    b2b = [
        [index, index + 1, 1.0 + 0.1 * (index % 5)]
        for index in range(n - 1)
    ]
    pins = [[0.0, 0.0], [50.0, 30.0], [100.0, 80.0]]
    p2b = [
        [index % len(pins), index, 1.0]
        for index in range(0, n, 9)
    ]
    constraints = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(n)]
    model_payload = dict(MODEL)
    declared_model_digest = model_payload.pop("payload_sha256", None)
    model_payload_digest = _canonical_sha256(model_payload)
    if (
        declared_model_digest != _EXPECTED_MODEL_PAYLOAD_SHA256
        or model_payload_digest != _EXPECTED_MODEL_PAYLOAD_SHA256
    ):
        raise RuntimeError("packaged learned model does not match the sealed payload")

    compiled_model = _compiled_learned_model()
    if compiled_model is None:
        raise RuntimeError("packaged learned model did not compile")
    prior = _learned_order_prior(
        n,
        areas,
        b2b,
        p2b,
        pins,
        constraints,
        None,
    )
    if prior is None or len(prior) != n:
        raise RuntimeError("packaged learned prior was not produced")
    prior_rows = [[float(value) for value in prior[index]] for index in range(n)]

    # These are the exact learned-slot arguments used by MyOptimizer for this
    # eligible heavy fixture. Hashing the raw candidate prevents a healthy
    # legacy candidate from hiding a missing or divergent learned module.
    raw_candidate = dissect_solve(
        n,
        areas,
        b2b,
        p2b,
        pins,
        constraints,
        None,
        width_factor=1.1,
        edge_order_mode="bary",
        band_order_mode="pinx",
        clamped_backfill=False,
        active_slab_max_aspect=12.0,
        learned_order=prior,
        learned_prior_weight=0.65,
    )
    if not _positions_ok(raw_candidate, n):
        raise RuntimeError("packaged learned candidate was not produced")
    raw_rows = [[float(value) for value in row] for row in raw_candidate]
    optimizer = MyOptimizer(verbose=False)
    positions = optimizer.solve(
        n,
        torch.Tensor(areas),
        b2b,
        p2b,
        torch.Tensor(pins),
        torch.Tensor(constraints),
        None,
    )
    if not _positions_ok(positions, n):
        raise RuntimeError("packaged learned fixture returned invalid positions")
    final_rows = [[float(value) for value in row] for row in positions]
    if not optimizer._learned_candidate_attempted:
        raise RuntimeError("packaged learned candidate was not attempted")

    # A heavy production solve is intentionally outside the replacement map.
    # It proves that a real learned-range abstention keeps the legacy path live,
    # while choosing the module probe from the map avoids pinning it to n=100.
    abstention_sizes = [
        size for size in range(100, 121) if size not in _LEARNED_REPLACEMENT_WF
    ]
    if not abstention_sizes:
        raise RuntimeError("learned replacement map has no heavy abstention case")
    abstention_n = abstention_sizes[0]
    abstention_optimizer = MyOptimizer(verbose=False)
    abstention_positions = abstention_optimizer.solve(
        abstention_n,
        torch.Tensor([1.0] * abstention_n),
        [],
        [],
        torch.Tensor([]),
        torch.Tensor([[0.0] * 5 for _ in range(abstention_n)]),
        None,
    )
    if (
        not _positions_ok(abstention_positions, abstention_n)
        or abstention_optimizer._learned_candidate_attempted
    ):
        raise RuntimeError("learned abstention did not preserve the legacy fallback")

    return {
        "model_payload_sha256": model_payload_digest,
        "compiled_model_sha256": _canonical_sha256(compiled_model),
        "prior_sha256": _canonical_sha256(prior_rows),
        "raw_candidate_sha256": _canonical_sha256(raw_rows),
        "candidate_attempted": True,
        "candidate_selected": bool(optimizer._learned_candidate_selected),
        "production_eligible_block_count": n,
        "abstention_block_count": abstention_n,
        "abstention_verified": True,
        "final_positions": final_rows,
    }


def _mib_self_test():
    """Exercise the direct safe-MIB repair fixture used by shim parity tests."""

    from golden_plus_repair import RepairConfig, repair_fixed_topology

    square = 40.0 ** 0.5
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
    repaired = repair_fixed_topology(
        positions,
        torch.Tensor(areas),
        [],
        [],
        torch.Tensor([]),
        torch.Tensor(constraints),
        torch.Tensor(targets),
        config=RepairConfig(
            enable_boundary=False,
            enable_mib=True,
            enable_grouping=False,
            require_safe_mib_pattern=True,
        ),
    )
    rows = [[float(value) for value in row] for row in repaired]
    if rows == [list(row) for row in positions] or any(
        row[2:] != [4.0, 10.0] for row in rows[:3]
    ):
        raise RuntimeError("packaged safe-MIB repair did not execute")
    return {
        "repaired": True,
        "positions": rows,
        "positions_sha256": _canonical_sha256(rows),
    }


def _live_module_self_test():
    return {
        "schema_version": 1,
        "learned": _learned_self_test(),
        "safe_mib": _mib_self_test(),
    }


def main():
    if sys.argv[1:] == ["--self-test-live-modules"]:
        sys.stdout.write(json.dumps(_live_module_self_test()))
        sys.stdout.flush()
        return
    payload = json.loads(sys.stdin.read())
    n = int(payload["block_count"])
    at_l = payload.get("area_targets") or []
    b2b_l = payload.get("b2b_connectivity") or []
    p2b_l = payload.get("p2b_connectivity") or []
    pins_l = payload.get("pins_pos") or []
    con_l = payload.get("constraints")
    tp_l = payload.get("target_positions")

    try:
        area_targets = torch.Tensor(at_l)
        constraints = torch.Tensor(con_l) if con_l is not None else None
        pins_pos = torch.Tensor(pins_l)
        target_positions = torch.Tensor(tp_l) if tp_l is not None else None

        # Connectivity stays as plain lists: my_optimizer._b2b_edges /
        # _p2b_edges take the non-tensor branch and iterate them directly.
        opt = MyOptimizer(verbose=False)
        positions = opt.solve(n, area_targets, b2b_l, p2b_l, pins_pos,
                              constraints, target_positions)
        if not _positions_ok(positions, n):
            positions = _fallback(n, at_l, con_l, tp_l)
    except Exception:
        if os.environ.get("SOLVER_DEBUG"):
            import traceback
            traceback.print_exc(file=sys.stderr)
        positions = _fallback(n, at_l, con_l, tp_l)

    out = [[float(p[0]), float(p[1]), float(p[2]), float(p[3])]
           for p in positions]
    sys.stdout.write(json.dumps({"positions": out}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()

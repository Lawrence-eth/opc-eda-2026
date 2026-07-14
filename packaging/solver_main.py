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

import json
import math
import os
import sys

# Local stubs shadow the real packages inside the bundle: `torch` is
# packaging/torch_stub.py and `iccad2026_evaluate` is packaging/eval_stub.py,
# both shipped next to this file. my_optimizer.py is used unmodified.
import torch
from my_optimizer import MyOptimizer, _compiled_learned_model


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


def _learned_self_test():
    """Exercise the live learned path for source/binary package parity."""
    n = 100
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
    result = {
        "model_compiled": _compiled_learned_model() is not None,
        "learned_attempted": bool(optimizer._learned_candidate_attempted),
        "positions": [list(map(float, row)) for row in positions],
    }
    if not result["model_compiled"] or not result["learned_attempted"]:
        raise RuntimeError("packaged learned path did not execute")
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


def main():
    if sys.argv[1:] == ["--self-test-learned"]:
        _learned_self_test()
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

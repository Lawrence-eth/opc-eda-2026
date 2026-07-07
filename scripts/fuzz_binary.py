#!/usr/bin/env python3
"""Feasibility fuzz of the packaged submission binary over FloorSet training
instances, through the exact op_wrapper JSON protocol.

Complements scripts/n9_robustness.py (which fuzzed solve() in-process): this
exercises the shipped artifact — JSON parsing, the torch-free stubs, the
PyInstaller bundle — end to end.

Usage:
    python3 scripts/fuzz_binary.py [--num 400] [--seed 7] [--binary PATH]
"""
import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "FloorSet"))

import torch  # noqa: E402
from lite_dataset import FloorplanDatasetLite  # noqa: E402


def hard_feasible(pos, n, areas, constraints, tp):
    if pos is None or len(pos) != n:
        return f"bad length {None if pos is None else len(pos)} != {n}"
    for i in range(n):
        x, y, w, h = pos[i]
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            return f"non-finite rect {i}"
        is_fixed = constraints[i][0] != 0
        is_pre = constraints[i][1] != 0
        if is_pre:
            tx, ty, tw, th = tp[i]
            if (abs(x - tx) > 1e-4 or abs(y - ty) > 1e-4
                    or abs(w - tw) > 1e-4 or abs(h - th) > 1e-4):
                return f"preplaced mismatch {i}"
        elif is_fixed:
            tw, th = tp[i][2], tp[i][3]
            if abs(w - tw) > 1e-4 or abs(h - th) > 1e-4:
                return f"fixed dims mismatch {i}"
        else:
            a = areas[i]
            if a > 0 and abs(w * h - a) / a > 0.01:
                return f"area viol {i}: {w*h} vs {a}"
    for i in range(n):
        xi, yi, wi, hi = pos[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = pos[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox > 1e-6 and oy > 1e-6:
                return f"overlap {i},{j}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--binary", default=str(
        ROOT / "submission" / "dist" / "my_optimizer" / "my_optimizer"))
    args = ap.parse_args()

    ds = FloorplanDatasetLite(str(ROOT / "external" / "FloorSet") + "/")
    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(ds)), args.num)

    fails, runtimes, ns = [], [], []
    for k, idx in enumerate(idxs):
        s = ds[idx]
        area_target, b2b, p2b, pins, constraints = s["input"]
        fp_sol = s["label"][1]  # n x 4 (w, h, x, y)
        n = int((area_target != -1).sum().item())
        con = constraints[:n].tolist()
        areas = area_target[:n].tolist()
        # target_positions built like the evaluator: -1 default; fixed get
        # (w,h); preplaced get (x,y,w,h) — from the golden solution fields.
        tp = [[-1.0, -1.0, -1.0, -1.0] for _ in range(n)]
        for i in range(n):
            w, h, x, y = [float(v) for v in fp_sol[i][:4]]
            if con[i][1] != 0:
                tp[i] = [x, y, w, h]
            elif con[i][0] != 0:
                tp[i][2], tp[i][3] = w, h
        payload = {"block_count": n, "area_targets": areas,
                   "b2b_connectivity": b2b.tolist(), "p2b_connectivity": p2b.tolist(),
                   "pins_pos": pins.tolist(), "constraints": con,
                   "target_positions": tp}
        t0 = time.time()
        try:
            proc = subprocess.run([args.binary], input=json.dumps(payload),
                                  text=True, capture_output=True, timeout=60,
                                  check=True)
            pos = json.loads(proc.stdout)["positions"]
        except Exception as e:
            fails.append((idx, n, f"protocol failure: {e}"))
            continue
        dt = time.time() - t0
        runtimes.append(dt)
        ns.append(n)
        err = hard_feasible(pos, n, areas, con, tp)
        if err:
            fails.append((idx, n, err))
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{args.num} ok so far, {len(fails)} failures", flush=True)

    print("\n=== BINARY FUZZ RESULT ===")
    print(f"instances: {len(idxs)}  failures: {len(fails)}")
    if runtimes:
        runtimes.sort()
        print(f"runtime: avg {sum(runtimes)/len(runtimes):.3f}s "
              f"p95 {runtimes[int(0.95*len(runtimes))]:.3f}s max {runtimes[-1]:.3f}s "
              f"(n up to {max(ns)})")
    for f in fails[:10]:
        print("  FAIL:", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

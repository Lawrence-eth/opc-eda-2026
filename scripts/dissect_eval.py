#!/usr/bin/env python3
"""Offline evaluation of the dissection engine (CAMPAIGN_GOLDEN G2-G4) on the
100 validation cases: hard feasibility, utilization, gaps vs baseline, soft
violations with evaluator semantics, and per-case comparison against v9's
locked per-case costs.

Usage:
    .venv/bin/python scripts/dissect_eval.py [--cases 0,5,99 | --all] [--wf 1.0]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "FloorSet"))
sys.path.insert(0, str(ROOT / "external" / "FloorSet" / "iccad2026contest"))
sys.path.insert(0, str(ROOT / "contest_solution"))

import torch  # noqa: E402
from iccad2026_evaluate import ContestEvaluator, evaluate_solution  # noqa: E402
from dissect import dissect_solve  # noqa: E402


def edges(t):
    if t is None:
        return []
    return [(int(a), int(b), abs(float(w))) for a, b, w, *_ in t.tolist()
            if a != -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="all")
    ap.add_argument("--wf", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import os
    os.chdir(str(ROOT / "external" / "FloorSet" / "iccad2026contest"))
    ev = ContestEvaluator(data_path="../", verbose=False)
    ev._load_dataset()

    ids = (range(100) if args.cases == "all"
           else [int(c) for c in args.cases.split(",")])

    v9 = {r["test_id"]: r for r in json.load(
        open(ROOT / "results" / "v9_locked.json"))["test_results"]}

    rows = []
    for idx in ids:
        sample = ev.dataset[idx]
        inputs, labels = sample["input"], sample["label"]
        area, b2b, p2b, pins, con = inputs
        n = int((area != -1).sum().item())
        baseline, target_pos = ev._extract_baseline(idx, labels, b2b, p2b, pins, n)
        tp = [[-1.0] * 4 for _ in range(n)]
        nc = con.shape[1] if con is not None and con.dim() > 1 else 0
        for i in range(n):
            if nc > 1 and con[i, 1] != 0:
                tp[i] = [float(v) for v in target_pos[i]]
            elif nc > 0 and con[i, 0] != 0:
                tp[i][2], tp[i][3] = float(target_pos[i][2]), float(target_pos[i][3])

        import time
        t0 = time.time()
        pos = dissect_solve(n, area[:n].tolist(), edges(b2b), edges(p2b),
                            pins.tolist(), con[:n].tolist(), tp,
                            width_factor=args.wf)
        rt = time.time() - t0
        m = evaluate_solution({"positions": pos, "runtime": 1.0}, baseline,
                              con, b2b, p2b, pins, area, target_pos, 1.0)
        tot_a = sum(pos[i][2] * pos[i][3] for i in range(n))
        bw = max(p[0] + p[2] for p in pos) - min(p[0] for p in pos)
        bh = max(p[1] + p[3] for p in pos) - min(p[1] for p in pos)
        rows.append(dict(test_id=idx, n=n, feas=m.is_feasible,
                         hg=m.hpwl_gap, ag=m.area_gap, vr=m.violations_relative,
                         cost=m.cost, util=tot_a / (bw * bh), rt=rt,
                         v9_cost=v9[idx]["cost"]))

    feas = sum(r["feas"] for r in rows)
    W = lambda r: math.exp(r["n"] / 12)
    Z = sum(W(r) for r in rows)
    tot = sum(r["cost"] * W(r) for r in rows) / Z
    v9tot = sum(r["v9_cost"] * W(r) for r in rows) / Z
    wins = sum(1 for r in rows if r["cost"] < r["v9_cost"] - 1e-9)
    print(f"cases {len(rows)} feasible {feas} | dissect weighted {tot:.4f} "
          f"vs v9 {v9tot:.4f} | dissect wins {wins}/{len(rows)}")
    big = [r for r in rows if r["n"] >= 100]
    if big:
        print("n>=100: util avg %.3f | hg avg %.3f | ag avg %.3f | vr avg %.3f" % (
            sum(r["util"] for r in big) / len(big),
            sum(r["hg"] for r in big) / len(big),
            sum(r["ag"] for r in big) / len(big),
            sum(r["vr"] for r in big) / len(big)))
    infeas = [r for r in rows if not r["feas"]]
    for r in infeas[:8]:
        print("  INFEASIBLE:", r["test_id"], "n", r["n"])
    worst = sorted(rows, key=lambda r: r["v9_cost"] - r["cost"])[:6]
    for r in worst:
        print("  worst vs v9: case %d n=%d cost %.3f v9 %.3f (hg %.2f ag %.2f vr %.3f util %.2f)"
              % (r["test_id"], r["n"], r["cost"], r["v9_cost"], r["hg"], r["ag"], r["vr"], r["util"]))
    if args.out:
        json.dump(rows, open(str(ROOT / args.out) if not args.out.startswith("/") else args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

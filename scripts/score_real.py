#!/usr/bin/env python3
"""Runtime-adjusted scoring tool for FloorSet ICCAD-2026.

Reads result JSON files and recomputes weighted totals under various
assumed median runtimes, revealing the real leaderboard impact of runtime.

Usage:
    python3 scripts/score_real.py results/current.json [results/baseline.json]
"""
import json
import math
import sys


def compute_scores(cases, median):
    """Compute weighted total for a given assumed median runtime."""
    ns = [c['block_count'] for c in cases]
    mx = max(ns)
    Z = sum(math.exp((n - mx) / 12) for n in ns)

    total = 0.0
    for c in cases:
        n = c['block_count']
        hg = max(0.0, c.get('hpwl_gap', 0))
        ag = max(0.0, c.get('area_gap', 0))
        vr = c.get('violations_relative', 0)
        rt = c.get('runtime_seconds', 1.0)

        quality = 1.0 + 0.5 * (hg + ag)
        soft = math.exp(2.0 * vr)
        runtime_mult = max(0.7, (rt / max(median, 0.01)) ** 0.3)
        cost = quality * soft * runtime_mult

        # Feasibility check (simplified)
        if not c.get('is_feasible', True):
            cost = 10.0

        w = math.exp((n - mx) / 12) / Z
        total += cost * w

    return total


def analyze_result(label, cases):
    """Analyze a result file under various median assumptions."""
    medians = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

    # Compute self-relative median (our own per-n median)
    from collections import defaultdict
    rt_by_n = defaultdict(list)
    for c in cases:
        rt_by_n[c['block_count']].append(c.get('runtime_seconds', 1.0))
    self_medians = {}
    for n, rts in rt_by_n.items():
        rts.sort()
        self_medians[n] = rts[len(rts) // 2]

    # Self-relative score
    self_total = 0.0
    ns = [c['block_count'] for c in cases]
    mx = max(ns)
    Z = sum(math.exp((n - mx) / 12) for n in ns)
    for c in cases:
        n = c['block_count']
        hg = max(0.0, c.get('hpwl_gap', 0))
        ag = max(0.0, c.get('area_gap', 0))
        vr = c.get('violations_relative', 0)
        rt = c.get('runtime_seconds', 1.0)
        median = self_medians.get(n, 1.0)
        quality = 1.0 + 0.5 * (hg + ag)
        soft = math.exp(2.0 * vr)
        runtime_mult = max(0.7, (rt / max(median, 0.01)) ** 0.3)
        cost = quality * soft * runtime_mult
        if not c.get('is_feasible', True):
            cost = 10.0
        w = math.exp((n - mx) / 12) / Z
        self_total += cost * w

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Header
    header = f"{'median':>8}"
    for m in medians:
        header += f"  {m:>5.1f}s"
    header += f"  {'self':>8}"
    print(header)
    print("-" * len(header))

    # Scores
    row = f"{'score':>8}"
    for m in medians:
        s = compute_scores(cases, m)
        row += f"  {s:>5.3f}"
    row += f"  {self_total:>8.3f}"
    print(row)

    # Runtime stats
    rts = [c.get('runtime_seconds', 0) for c in cases]
    print(f"\n  Runtime: sum={sum(rts):.1f}s  avg={sum(rts)/len(rts):.2f}s  max={max(rts):.1f}s  median={sorted(rts)[len(rts)//2]:.2f}s")
    print(f"  Feasible: {sum(c.get('is_feasible', True) for c in cases)}/{len(cases)}")

    # Per-band breakdown
    bands = [(21, 80), (81, 100), (101, 115), (116, 120)]
    print(f"\n  {'band':>9} {'wt%':>5} {'cost':>5} {'hpwlg':>6} {'areag':>6} {'vrel':>5} {'avg_rt':>6}")
    for lo, hi in bands:
        g = [c for c in cases if lo <= c['block_count'] <= hi]
        if not g:
            continue
        wt = sum(math.exp((c['block_count'] - mx) / 12) / Z for c in g)
        print(f"  {lo:3d}-{hi:3d} {100*wt:5.1f} {sum(c.get('cost', 0) for c in g)/len(g):5.2f} "
              f"{sum(c.get('hpwl_gap', 0) for c in g)/len(g):6.2f} "
              f"{sum(c.get('area_gap', 0) for c in g)/len(g):6.2f} "
              f"{sum(c.get('violations_relative', 0) for c in g)/len(g):5.3f} "
              f"{sum(c.get('runtime_seconds', 0) for c in g)/len(g):6.2f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/score_real.py results/current.json [results/baseline.json]")
        sys.exit(1)

    current_file = sys.argv[1]
    with open(current_file) as f:
        current = json.load(f)
    cases_current = current['test_results']
    analyze_result(f"Current: {current_file}", cases_current)

    if len(sys.argv) >= 3:
        baseline_file = sys.argv[2]
        with open(baseline_file) as f:
            baseline = json.load(f)
        cases_baseline = baseline['test_results']
        analyze_result(f"Baseline: {baseline_file}", cases_baseline)

        # Comparison
        print(f"\n{'='*60}")
        print(f"  COMPARISON (current vs baseline)")
        print(f"{'='*60}")
        medians = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
        header = f"{'median':>8}"
        for m in medians:
            header += f"  {m:>5.1f}s"
        print(header)
        print("-" * len(header))

        row_cur = f"{'cur':>8}"
        row_base = f"{'base':>8}"
        row_delta = f"{'delta':>8}"
        for m in medians:
            sc = compute_scores(cases_current, m)
            sb = compute_scores(cases_baseline, m)
            row_cur += f"  {sc:>5.3f}"
            row_base += f"  {sb:>5.3f}"
            row_delta += f"  {sc-sb:>+5.3f}"
        print(row_base)
        print(row_cur)
        print(row_delta)


if __name__ == '__main__':
    main()

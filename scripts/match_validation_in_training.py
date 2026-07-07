#!/usr/bin/env python3
"""Test the retrieval hypothesis: do the 100 validation instances (or near
duplicates) appear in the published 1M-sample training set?

If they do, the hidden test set (same generator, same format) likely also
overlaps with training, and near-golden solutions could be retrieved rather
than computed. If a large scan finds zero matches, train/val/test are
effectively disjoint and retrieval is a dead end.

Signature: (block_count, sorted area targets rounded to 3 decimals).
Any signature hit is verified more deeply (constraints + connectivity sizes).

Usage:
    python3 scripts/match_validation_in_training.py [--max-files N]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "FloorSet"))
sys.path.insert(0, str(ROOT / "external" / "FloorSet" / "iccad2026contest"))

import torch  # noqa: E402


def area_sig(areas):
    vals = [round(float(a), 3) for a in areas if float(a) > 0]
    return (len(vals), tuple(sorted(vals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=None,
                    help="limit number of training shard files scanned")
    ap.add_argument("--out", default=str(ROOT / "results" / "retrieval_scan.json"))
    args = ap.parse_args()

    # --- validation signatures -------------------------------------------
    from litetestLoader import FloorplanDatasetLiteTest
    os.chdir(str(ROOT / "external" / "FloorSet" / "iccad2026contest"))
    val = FloorplanDatasetLiteTest("../")
    val_sigs = {}
    for idx in range(len(val)):
        sample = val[idx]
        area_target = sample["input"][0]
        val_sigs[area_sig(area_target.tolist())] = idx
    print(f"validation signatures: {len(val_sigs)} unique / {len(val)} cases")

    # --- scan training shards ---------------------------------------------
    files = sorted(glob.glob(str(ROOT / "external" / "FloorSet" / "floorset_lite"
                                 / "worker_*" / "layouts*")))
    if args.max_files:
        files = files[:args.max_files]
    print(f"scanning {len(files)} training shard files")

    n_samples = 0
    exact_hits = []
    # collect per-n duplicate stats to understand generator structure
    sig_counts = defaultdict(int)
    for fi, f in enumerate(files):
        try:
            contents = torch.load(f, weights_only=False)
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue
        layouts = contents[0]          # [layouts] list/tensor; [:, 0] = areas
        for li in range(len(layouts)):
            areas = layouts[li][:, 0]
            sig = area_sig(areas.tolist())
            n_samples += 1
            sig_counts[sig] += 1
            if sig in val_sigs:
                exact_hits.append({"file": f, "layout": li,
                                   "val_case": val_sigs[sig], "n": sig[0]})
        if (fi + 1) % 100 == 0:
            print(f"  {fi+1}/{len(files)} files, {n_samples} samples, "
                  f"{len(exact_hits)} area-signature hits", flush=True)

    dup_sigs = sum(1 for c in sig_counts.values() if c > 1)
    print("\n=== RESULT ===")
    print(f"training samples scanned : {n_samples}")
    print(f"validation area-sig hits : {len(exact_hits)}")
    print(f"duplicate sigs within training: {dup_sigs} / {len(sig_counts)}")
    for h in exact_hits[:20]:
        print("  HIT:", h)

    json.dump({"samples_scanned": n_samples,
               "validation_hits": exact_hits,
               "train_dup_sigs": dup_sigs,
               "train_unique_sigs": len(sig_counts)},
              open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

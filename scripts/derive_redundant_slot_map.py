#!/usr/bin/env python3
"""Derive a per-size zero-win standard-slot replacement map.

Inputs must be legacy-control artifacts from ``audit_standard_slot_usage.py``.
The tie rule is fixed before confirmation: rank slots by total calibration
selection count (then numeric width), and choose the first zero-count slot for
each block count. A size with no zero-count slot is omitted (abstention).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SLOTS = (0.8, 0.9, 1.0, 1.1, 1.2)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audits", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_size = defaultdict(Counter)
    total = Counter()
    descriptors = []
    support = []
    for path in args.audits:
        payload = json.loads(path.read_bytes())
        if payload.get("mode") != "legacy_deployed_selector_standard_slot_usage":
            raise ValueError(f"not a standard-slot audit: {path}")
        config = payload.get("config", {})
        if config.get("learned_enabled") is not False:
            raise ValueError(f"learned solver was enabled in {path}")
        if config.get("golden_cost_computed") is not False:
            raise ValueError(f"golden cost was computed in {path}")
        rows = payload.get("cases", [])
        if len(rows) != 105:
            raise ValueError(f"expected 105 calibration cases in {path}")
        for row in rows:
            wf = row.get("selected_standard_wf")
            if wf is not None:
                wf = float(wf)
                if wf not in SLOTS:
                    raise ValueError(f"unknown standard slot {wf} in {path}")
                by_size[int(row["block_count"])][wf] += 1
                total[wf] += 1
        manifest = config.get("manifest", {})
        descriptor = {
            "sha256": _sha256(path),
            "fold": int(config["fold"]),
            "manifest_sha256": manifest.get("sha256"),
            "require_mib_input_compatible": manifest.get(
                "fold_metadata", {}
            ).get("require_mib_input_compatible"),
            "case_count": len(rows),
        }
        descriptors.append(descriptor)
        support.append(
            {
                **descriptor,
                "counts_by_size": payload["counts_by_size"],
            }
        )

    preference = sorted(SLOTS, key=lambda wf: (total[wf], wf))
    replacement = {}
    abstentions = []
    for n in range(100, 121):
        chosen = next((wf for wf in preference if by_size[n][wf] == 0), None)
        if chosen is None:
            abstentions.append(n)
        else:
            replacement[str(n)] = chosen
    result = {
        "schema_version": 1,
        "mode": "legacy_v32_zero_selected_slot_calibration",
        "contract": {
            "folds": [0, 1, 2],
            "domains": ["mib_input_compatible", "raw_hash"],
            "uses_learned_outputs": False,
            "uses_golden_costs": False,
            "uses_public_cases": False,
            "tie_rule": "ascending(total_selected_count, numeric_width_factor)",
            "abstain_when_no_zero_selected_slot": True,
        },
        "input_artifacts": descriptors,
        "total_selected_counts": {str(wf): total[wf] for wf in SLOTS},
        "tie_preference": preference,
        "counts_by_size": {
            str(n): {str(wf): by_size[n][wf] for wf in SLOTS}
            for n in range(100, 121)
        },
        "replacement_wf_by_size": replacement,
        "abstain_sizes": abstentions,
        "support": support,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "tie_preference": preference,
        "replacement_wf_by_size": replacement,
        "abstain_sizes": abstentions,
    }, indent=2))


if __name__ == "__main__":
    main()

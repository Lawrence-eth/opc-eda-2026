#!/usr/bin/env python3
"""Compare matched solver results with source-clustered uncertainty.

Comparison is fail-closed: baseline and candidate artifacts must bind the same
canonical case identities, inputs, labels, fold manifest, dataset, evaluator,
evaluation harness, runtime mode, and oracle-selection mode.  Solver source
provenance is intentionally allowed to differ.  When a fold manifest is
supplied, bootstrap resampling happens by source ``.th`` file rather than by
layout so nearby configurations from one source cannot create false
confidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


_CASE_IDENTITY_FIELDS = (
    "case_id",
    "source_file",
    "file_offset",
    "sample_index",
    "block_count",
)
_CASE_DIGEST_FIELDS = ("input_sha256", "scoring_label_sha256")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_descriptor(path):
    path = Path(path)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _require_mapping(value, context):
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _require_field(mapping, field, context):
    if field not in mapping or mapping[field] is None:
        raise ValueError(f"{context} is missing required field {field}")
    return mapping[field]


def _require_sha256(value, context):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_git_commit(value, context):
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase full Git commit")
    return value


def _validate_case_contract(row, context):
    row = _require_mapping(row, context)
    for field in _CASE_IDENTITY_FIELDS:
        _require_field(row, field, context)
    for field in _CASE_DIGEST_FIELDS:
        _require_sha256(_require_field(row, field, context), f"{context}.{field}")

    case_id = row["case_id"]
    source_file = row["source_file"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"{context}.case_id must be a non-empty string")
    if not isinstance(source_file, str) or not source_file:
        raise ValueError(f"{context}.source_file must be a non-empty string")
    for field in ("file_offset", "sample_index", "block_count"):
        if isinstance(row[field], bool) or not isinstance(row[field], int):
            raise ValueError(f"{context}.{field} must be an integer")
    file_offset = row["file_offset"]
    sample_index = row["sample_index"]
    block_count = row["block_count"]
    if file_offset < 0 or sample_index < 0 or block_count <= 0:
        raise ValueError(f"{context} has an out-of-range identity field")
    if case_id != f"{source_file}#{file_offset}":
        raise ValueError(
            f"{context}.case_id does not match source_file/file_offset: {case_id}"
        )
    if not isinstance(row.get("is_feasible"), bool):
        raise ValueError(f"{context}.is_feasible must be boolean")
    try:
        cost = float(row["cost"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context}.cost must be numeric") from error
    if not math.isfinite(cost):
        raise ValueError(f"{context}.cost must be finite")


def _case_contract(row):
    return {field: row[field] for field in _CASE_IDENTITY_FIELDS + _CASE_DIGEST_FIELDS}


def weighted_score(rows):
    """Return the contest's exp(n/12)-weighted mean case cost."""
    if not rows:
        raise ValueError("cannot score an empty result")
    max_blocks = max(int(row["block_count"]) for row in rows)
    weighted = [
        (math.exp((int(row["block_count"]) - max_blocks) / 12.0), float(row["cost"]))
        for row in rows
    ]
    return sum(weight * cost for weight, cost in weighted) / sum(
        weight for weight, _cost in weighted
    )


def _load_cases(path):
    path = Path(path)
    payload = path.read_bytes()
    data = json.loads(payload)
    rows = data.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no non-empty cases list")
    indexed = {}
    for position, row in enumerate(rows):
        _validate_case_contract(row, f"{path}.cases[{position}]")
        identity = _case_key(row)
        if identity in indexed:
            raise ValueError(f"{path} contains duplicate case identity {identity}")
        indexed[identity] = row
    if len(indexed) != len(rows):
        raise ValueError(f"{path} contains duplicate case identities")
    artifact = {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    return data, indexed, artifact


def _case_key(row):
    return f"case:{row['case_id']}"


def _source_map(manifest_path):
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    mapping = {}
    for manifest in data.get("manifests", [data]):
        for case in manifest["cases"]:
            if case.get("case_id") is None or case.get("source_file") is None:
                raise ValueError("manifest case lacks stable identity/source_file")
            identity = _case_key(case)
            if identity in mapping:
                raise ValueError(f"manifest repeats case {identity}")
            mapping[identity] = str(case["source_file"])
    return mapping


def _extract_evaluation_contract(data, rows, path):
    config = _require_mapping(data.get("config"), f"{path}.config")
    provenance = _require_mapping(data.get("provenance"), f"{path}.provenance")
    manifest = _require_mapping(
        _require_field(config, "manifest", f"{path}.config"),
        f"{path}.config.manifest",
    )
    manifest_context = f"{path}.config.manifest"

    manifest_sha256 = _require_sha256(
        _require_field(manifest, "sha256", manifest_context),
        f"{manifest_context}.sha256",
    )
    manifest_schema_version = _require_field(manifest, "schema_version", manifest_context)
    if (
        isinstance(manifest_schema_version, bool)
        or not isinstance(manifest_schema_version, int)
        or manifest_schema_version < 2
    ):
        raise ValueError(f"{manifest_context}.schema_version must be an integer >= 2")
    manifest_fold = int(_require_field(manifest, "fold", manifest_context))
    fold = int(_require_field(config, "fold", f"{path}.config"))
    if manifest_fold != fold:
        raise ValueError(
            f"{path} config/manifest fold mismatch: {fold} != {manifest_fold}"
        )

    fold_metadata = _require_mapping(
        _require_field(manifest, "fold_metadata", manifest_context),
        f"{manifest_context}.fold_metadata",
    )
    metadata_fold = int(
        _require_field(fold_metadata, "fold", f"{manifest_context}.fold_metadata")
    )
    if metadata_fold != fold:
        raise ValueError(
            f"{path} config/fold-metadata mismatch: {fold} != {metadata_fold}"
        )
    case_count = int(
        _require_field(
            fold_metadata, "case_count", f"{manifest_context}.fold_metadata"
        )
    )
    if case_count != len(rows):
        raise ValueError(
            f"{path} manifest case_count mismatch: {case_count} != {len(rows)}"
        )
    source_file_count = int(
        _require_field(
            fold_metadata,
            "source_file_count",
            f"{manifest_context}.fold_metadata",
        )
    )
    actual_source_file_count = len({row["source_file"] for row in rows.values()})
    if source_file_count != actual_source_file_count:
        raise ValueError(
            f"{path} manifest source_file_count mismatch: "
            f"{source_file_count} != {actual_source_file_count}"
        )

    generation = _require_mapping(
        _require_field(manifest, "generation", manifest_context),
        f"{manifest_context}.generation",
    )
    for field in ("min_blocks", "max_blocks", "num_folds", "per_size", "seed"):
        _require_field(generation, field, f"{manifest_context}.generation")
    dataset = _require_mapping(
        _require_field(manifest, "dataset", manifest_context),
        f"{manifest_context}.dataset",
    )
    dataset_context = f"{manifest_context}.dataset"
    for field in (
        "name",
        "official_floorset_commit",
        "loader",
        "layouts_per_file",
        "source_file_count",
        "source_inventory_sha256",
    ):
        _require_field(dataset, field, dataset_context)
    inventory_sha256 = _require_sha256(
        dataset["source_inventory_sha256"],
        f"{dataset_context}.source_inventory_sha256",
    )
    resolved_inventory_sha256 = _require_sha256(
        _require_field(manifest, "resolved_inventory_sha256", manifest_context),
        f"{manifest_context}.resolved_inventory_sha256",
    )
    if resolved_inventory_sha256 != inventory_sha256:
        raise ValueError(
            f"{path} resolved dataset inventory does not match manifest dataset"
        )

    evaluator_sha256 = _require_sha256(
        _require_field(provenance, "evaluator_sha256", f"{path}.provenance"),
        f"{path}.provenance.evaluator_sha256",
    )
    harness_sha256 = _require_sha256(
        _require_field(
            provenance, "evaluation_harness_sha256", f"{path}.provenance"
        ),
        f"{path}.provenance.evaluation_harness_sha256",
    )
    official_git = _require_mapping(
        _require_field(provenance, "official_floorset_git", f"{path}.provenance"),
        f"{path}.provenance.official_floorset_git",
    )
    official_commit = _require_git_commit(
        _require_field(
            official_git, "commit", f"{path}.provenance.official_floorset_git"
        ),
        f"{path}.provenance.official_floorset_git.commit",
    )
    tracked_dirty = _require_field(
        official_git,
        "tracked_dirty",
        f"{path}.provenance.official_floorset_git",
    )
    if not isinstance(tracked_dirty, bool):
        raise ValueError(
            f"{path}.provenance.official_floorset_git.tracked_dirty must be boolean"
        )
    if tracked_dirty:
        raise ValueError(f"{path} official FloorSet checkout has tracked changes")
    if official_commit != dataset["official_floorset_commit"]:
        raise ValueError(
            f"{path} official FloorSet commit does not match manifest dataset"
        )

    runtime_factor_mode = _require_field(
        config, "runtime_factor_mode", f"{path}.config"
    )
    oracle_baseline_selector = _require_field(
        config, "oracle_baseline_selector", f"{path}.config"
    )
    if not isinstance(oracle_baseline_selector, bool):
        raise ValueError(f"{path}.config.oracle_baseline_selector must be boolean")
    if not isinstance(runtime_factor_mode, str) or not runtime_factor_mode:
        raise ValueError(f"{path}.config.runtime_factor_mode must be non-empty")
    if "require_golden_mib_clean" not in config:
        raise ValueError(
            f"{path}.config is missing required field require_golden_mib_clean"
        )
    require_golden_mib_clean = config["require_golden_mib_clean"]
    if require_golden_mib_clean is not None and not isinstance(
        require_golden_mib_clean, bool
    ):
        raise ValueError(
            f"{path}.config.require_golden_mib_clean must be boolean or null"
        )

    pair_contract = {
        "fold": fold,
        "manifest": manifest,
        "runtime_factor_mode": runtime_factor_mode,
        "oracle_baseline_selector": oracle_baseline_selector,
        "require_golden_mib_clean": require_golden_mib_clean,
        "evaluator_sha256": evaluator_sha256,
        "evaluation_harness_sha256": harness_sha256,
        "official_floorset_commit": official_commit,
        "official_floorset_tracked_dirty": tracked_dirty,
    }
    stable_contract = {
        "manifest_sha256": manifest_sha256,
        "manifest_schema_version": manifest_schema_version,
        "manifest_generation": generation,
        "dataset": dataset,
        "resolved_inventory_sha256": resolved_inventory_sha256,
        "runtime_factor_mode": runtime_factor_mode,
        "oracle_baseline_selector": oracle_baseline_selector,
        "require_golden_mib_clean": require_golden_mib_clean,
        "evaluator_sha256": evaluator_sha256,
        "evaluation_harness_sha256": harness_sha256,
        "official_floorset_commit": official_commit,
        "official_floorset_tracked_dirty": tracked_dirty,
    }
    return pair_contract, stable_contract


def _require_equal_contract(baseline, candidate, context):
    if baseline != candidate:
        raise ValueError(f"{context} evaluation contract mismatch")


def _percentile(sorted_values, probability):
    if not sorted_values:
        raise ValueError("cannot take percentile of empty values")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def compare_result_pairs(
    baseline_paths,
    candidate_paths,
    *,
    source_by_sample=None,
    source_manifest_path=None,
    bootstrap_samples=10_000,
    seed=20260710,
):
    if len(baseline_paths) != len(candidate_paths) or not baseline_paths:
        raise ValueError("baseline and candidate path counts must match and be non-zero")

    baseline_artifacts = []
    candidate_artifacts = []
    source_manifest_artifact = (
        _artifact_descriptor(source_manifest_path) if source_manifest_path else None
    )
    pooled_baseline = []
    pooled_candidate = []
    fold_rows = []
    seen_samples = set()
    seen_folds = set()
    stable_contract = None
    for fold, (baseline_path, candidate_path) in enumerate(
        zip(baseline_paths, candidate_paths)
    ):
        baseline_data, baseline, baseline_artifact = _load_cases(baseline_path)
        candidate_data, candidate, candidate_artifact = _load_cases(candidate_path)
        baseline_artifacts.append(baseline_artifact)
        candidate_artifacts.append(candidate_artifact)
        baseline_pair_contract, baseline_stable_contract = _extract_evaluation_contract(
            baseline_data, baseline, baseline_path
        )
        candidate_pair_contract, candidate_stable_contract = _extract_evaluation_contract(
            candidate_data, candidate, candidate_path
        )
        _require_equal_contract(
            baseline_pair_contract,
            candidate_pair_contract,
            f"fold pair {fold}",
        )
        _require_equal_contract(
            baseline_stable_contract,
            candidate_stable_contract,
            f"fold pair {fold} stable",
        )
        if stable_contract is None:
            stable_contract = baseline_stable_contract
        else:
            _require_equal_contract(
                stable_contract,
                baseline_stable_contract,
                f"fold pair {fold} cross-fold",
            )

        if baseline.keys() != candidate.keys():
            missing = sorted(baseline.keys() - candidate.keys())[:5]
            extra = sorted(candidate.keys() - baseline.keys())[:5]
            raise ValueError(
                f"fold {fold} sample mismatch: missing={missing}, extra={extra}"
            )
        overlap = seen_samples.intersection(baseline)
        if overlap:
            raise ValueError(f"samples occur in multiple result pairs: {sorted(overlap)[:5]}")
        seen_samples.update(baseline)

        baseline_rows = [baseline[index] for index in sorted(baseline)]
        candidate_rows = [candidate[index] for index in sorted(candidate)]
        for old, new in zip(baseline_rows, candidate_rows):
            if _case_contract(old) != _case_contract(new):
                raise ValueError(f"case contract mismatch for {_case_key(old)}")
            if source_by_sample is not None:
                identity = _case_key(old)
                if identity not in source_by_sample:
                    raise ValueError(f"source manifest is missing case {identity}")
                if str(source_by_sample[identity]) != str(old["source_file"]):
                    raise ValueError(
                        f"source manifest mismatch for {identity}: "
                        f"{source_by_sample[identity]} != {old['source_file']}"
                    )
        base_score = weighted_score(baseline_rows)
        cand_score = weighted_score(candidate_rows)
        fold_id = baseline_pair_contract["fold"]
        if fold_id in seen_folds:
            raise ValueError(f"fold {fold_id} occurs in multiple result pairs")
        seen_folds.add(fold_id)
        fold_rows.append(
            {
                "fold": fold_id,
                "cases": len(baseline_rows),
                "baseline_score": base_score,
                "candidate_score": cand_score,
                "delta_candidate_minus_baseline": cand_score - base_score,
                "wins": sum(
                    float(new["cost"]) < float(old["cost"]) - 1e-12
                    for old, new in zip(baseline_rows, candidate_rows)
                ),
                "losses": sum(
                    float(new["cost"]) > float(old["cost"]) + 1e-12
                    for old, new in zip(baseline_rows, candidate_rows)
                ),
                "baseline_feasible": sum(bool(row["is_feasible"]) for row in baseline_rows),
                "candidate_feasible": sum(bool(row["is_feasible"]) for row in candidate_rows),
            }
        )
        pooled_baseline.extend(baseline_rows)
        pooled_candidate.extend(candidate_rows)

    if source_manifest_artifact is not None:
        if source_by_sample is None:
            raise ValueError("source_manifest_path requires source_by_sample")
        if source_manifest_artifact["sha256"] != stable_contract["manifest_sha256"]:
            raise ValueError(
                "source manifest artifact hash does not match result evaluation contract"
            )

    baseline_score = weighted_score(pooled_baseline)
    candidate_score = weighted_score(pooled_candidate)
    baseline_by_index = {_case_key(row): row for row in pooled_baseline}
    candidate_by_index = {_case_key(row): row for row in pooled_candidate}

    groups = defaultdict(list)
    for index in sorted(baseline_by_index):
        source = source_by_sample.get(index) if source_by_sample else None
        groups[source or index].append(index)
    group_names = sorted(groups)
    rng = random.Random(seed)
    bootstrap_deltas = []
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(group_names) for _ in group_names]
        baseline_draw = []
        candidate_draw = []
        for group in sampled:
            for index in groups[group]:
                baseline_draw.append(baseline_by_index[index])
                candidate_draw.append(candidate_by_index[index])
        bootstrap_deltas.append(
            weighted_score(candidate_draw) - weighted_score(baseline_draw)
        )
    bootstrap_deltas.sort()

    # The hidden/public suite contains one case for each block count.  Drawing
    # one held-out case per n produces a more faithful pseudo-test distribution
    # than an ordinary iid layout bootstrap.
    indices_by_size = defaultdict(list)
    for index, row in baseline_by_index.items():
        indices_by_size[int(row["block_count"])].append(index)
    pseudo_rng = random.Random(seed + 1)
    pseudo_test_deltas = []
    for _ in range(bootstrap_samples):
        drawn_indices = [
            pseudo_rng.choice(indices_by_size[block_count])
            for block_count in sorted(indices_by_size)
        ]
        pseudo_test_deltas.append(
            weighted_score([candidate_by_index[index] for index in drawn_indices])
            - weighted_score([baseline_by_index[index] for index in drawn_indices])
        )
    pseudo_test_deltas.sort()

    max_blocks = max(int(row["block_count"]) for row in pooled_baseline)
    denominator = sum(
        math.exp((int(row["block_count"]) - max_blocks) / 12.0)
        for row in pooled_baseline
    )
    score_contributions = sorted(
        (
            math.exp((int(baseline_by_index[index]["block_count"]) - max_blocks) / 12.0)
            * (
                float(candidate_by_index[index]["cost"])
                - float(baseline_by_index[index]["cost"])
            )
            / denominator
        )
        for index in baseline_by_index
    )
    tail_count = max(1, math.ceil(0.05 * len(score_contributions)))
    regression_tail = score_contributions[-tail_count:]

    wins = sum(
        float(candidate_by_index[index]["cost"])
        < float(baseline_by_index[index]["cost"]) - 1e-12
        for index in baseline_by_index
    )
    losses = sum(
        float(candidate_by_index[index]["cost"])
        > float(baseline_by_index[index]["cost"]) + 1e-12
        for index in baseline_by_index
    )
    return {
        "schema_version": 2,
        "evaluation_contract": stable_contract,
        "input_result_artifacts": {
            "baseline": baseline_artifacts,
            "candidate": candidate_artifacts,
        },
        **(
            {"source_manifest_artifact": source_manifest_artifact}
            if source_manifest_artifact is not None
            else {}
        ),
        "cases": len(pooled_baseline),
        "source_clusters": len(groups),
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "delta_candidate_minus_baseline": candidate_score - baseline_score,
        "wins": wins,
        "losses": losses,
        "ties": len(pooled_baseline) - wins - losses,
        "baseline_feasible": sum(bool(row["is_feasible"]) for row in pooled_baseline),
        "candidate_feasible": sum(bool(row["is_feasible"]) for row in pooled_candidate),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "cluster_unit": "source_file" if source_by_sample else "sample",
            "delta_ci95": [
                _percentile(bootstrap_deltas, 0.025),
                _percentile(bootstrap_deltas, 0.975),
            ],
            "probability_candidate_improves": sum(
                delta < 0.0 for delta in bootstrap_deltas
            ) / len(bootstrap_deltas),
        },
        "pseudo_test_one_per_block_count": {
            "samples": bootstrap_samples,
            "seed": seed + 1,
            "block_counts": sorted(indices_by_size),
            "delta_ci95": [
                _percentile(pseudo_test_deltas, 0.025),
                _percentile(pseudo_test_deltas, 0.975),
            ],
            "probability_candidate_improves": sum(
                delta < 0.0 for delta in pseudo_test_deltas
            ) / len(pseudo_test_deltas),
            "worst_sampled_delta": pseudo_test_deltas[-1],
        },
        "tail_risk": {
            "worst_case_score_contribution": score_contributions[-1],
            "regression_cvar_5pct_score_contribution": sum(regression_tail)
            / len(regression_tail),
        },
        "folds": fold_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    result = compare_result_pairs(
        args.baseline,
        args.candidate,
        source_by_sample=_source_map(args.manifest) if args.manifest else None,
        source_manifest_path=args.manifest,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure solver quality on a stratified, unseen FloorSet training holdout.

The contest validation set has one public instance per block count and is easy
to overfit.  This gate samples different training instances for every size in
the requested range, scores both the solver and the supplied layout with the
official evaluator, and reports MIB-corrupted samples separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_DIR = ROOT / "contest_solution"
OFFICIAL_ROOT = ROOT / "external" / "FloorSet"
OFFICIAL_CONTEST_DIR = OFFICIAL_ROOT / "iccad2026contest"
SCRIPTS_DIR = ROOT / "scripts"
# Keep the official contest directory first: result provenance records the
# official evaluator, so importing the convenience working copy from
# ``contest_solution`` would make that provenance false even when the scoring
# functions happen to be identical.
sys.path[:0] = [
    str(OFFICIAL_CONTEST_DIR),
    str(OFFICIAL_ROOT),
    str(SOLUTION_DIR),
    str(SCRIPTS_DIR),
]

from lite_dataset import FloorplanDatasetLite  # noqa: E402
from build_holdout_folds import (  # noqa: E402
    _canonical_relative,
    _case_metadata,
    _fold_for_file,
    _git_commit,
    _input_sha256,
    _inventory_sha256,
)
from solver_components import LIVE_SOLVER_COMPONENTS  # noqa: E402


def _load_official_evaluator():
    """Load the pinned evaluator by path, independent of ``sys.modules``.

    Test collection and solver imports can legitimately cache the repository's
    convenience evaluator under the public ``iccad2026_evaluate`` name.  The
    holdout harness must never silently inherit that module while claiming the
    hash of the pinned official file in its provenance.
    """
    path = OFFICIAL_CONTEST_DIR / "iccad2026_evaluate.py"
    spec = importlib.util.spec_from_file_location(
        "_holdout_official_iccad2026_evaluate", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OFFICIAL_EVALUATOR = _load_official_evaluator()
compute_total_score = _OFFICIAL_EVALUATOR.compute_total_score
evaluate_solution = _OFFICIAL_EVALUATOR.evaluate_solution


def _load_optimizer(solver_dir: Path):
    solver_dir = solver_dir.resolve()
    sys.path.insert(0, str(solver_dir))
    optimizer_path = solver_dir / "my_optimizer.py"
    spec = importlib.util.spec_from_file_location("holdout_optimizer", optimizer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load optimizer from {optimizer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyOptimizer(verbose=False)


def _positions_from_training_label(fp_sol: torch.Tensor, n: int):
    """Convert training [w,h,x,y] rows to evaluator [x,y,w,h] tuples."""
    return [
        (
            float(fp_sol[i, 2]),
            float(fp_sol[i, 3]),
            float(fp_sol[i, 0]),
            float(fp_sol[i, 1]),
        )
        for i in range(n)
    ]


def _optimizer_targets(constraints, golden_positions, n):
    out = torch.full((n, 4), -1.0)
    ncols = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0
    for i in range(n):
        is_fixed = ncols > 0 and constraints[i, 0] != 0
        is_preplaced = ncols > 1 and constraints[i, 1] != 0
        x, y, w, h = golden_positions[i]
        if is_preplaced:
            out[i] = torch.tensor((x, y, w, h))
        elif is_fixed:
            out[i, 2] = w
            out[i, 3] = h
    return out


def _baseline(metrics):
    return {
        "area_baseline": float(metrics[0]),
        "hpwl_baseline": float(metrics[-2]) + float(metrics[-1]),
    }


def _golden_mib_violation_count(sample, n):
    constraints = sample["input"][4]
    fp_sol = sample["label"][1]
    if constraints is None or constraints.dim() < 2 or constraints.shape[1] <= 2:
        return 0
    mib = constraints[:n, 2]
    max_group = int(mib.max().item()) if mib.numel() else 0
    violations = 0
    for group_id in range(1, max_group + 1):
        indices = torch.where(mib == group_id)[0].tolist()
        shapes = {
            (round(float(fp_sol[i, 0]), 4), round(float(fp_sol[i, 1]), 4))
            for i in indices
        }
        violations += max(0, len(shapes) - 1)
    return violations


def _summary(rows):
    if not rows:
        return {"cases": 0}
    costs = [r["cost"] for r in rows]
    counts = [r["block_count"] for r in rows]
    max_n = max(counts)
    weights = [math.exp((n - max_n) / 12.0) for n in counts]
    z = sum(weights)

    def weighted(key):
        return sum(w * float(r[key]) for w, r in zip(weights, rows)) / z

    runtimes = [r["runtime_seconds"] for r in rows]
    return {
        "cases": len(rows),
        "feasible": sum(bool(r["is_feasible"]) for r in rows),
        "total_score": compute_total_score(costs, counts),
        "weighted_hpwl_gap_clamped": weighted("hpwl_gap_clamped"),
        "weighted_area_gap_clamped": weighted("area_gap_clamped"),
        "weighted_violations_relative": weighted("violations_relative"),
        "runtime_mean": statistics.fmean(runtimes),
        "runtime_p95": sorted(runtimes)[max(0, math.ceil(0.95 * len(runtimes)) - 1)],
        "runtime_max": max(runtimes),
    }


def _portable_path(path: Path | None):
    """Keep result metadata reproducible when a path lives in this checkout."""
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _clone_value(value):
    return value.clone() if hasattr(value, "clone") else value


def _load_manifest_fold(path: Path, fold: int):
    raw = path.read_bytes()
    data = json.loads(raw)
    manifests = data.get("manifests", [data])
    fold_ids = [int(row.get("fold", fold)) for row in manifests]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError(f"manifest {path} contains duplicate fold IDs")
    matches = [row for row in manifests if int(row.get("fold", fold)) == fold]
    if len(matches) != 1:
        raise ValueError(f"manifest {path} has {len(matches)} entries for fold {fold}")

    seen_cases = set()
    source_fold = {}
    for manifest in manifests:
        manifest_fold = int(manifest.get("fold", fold))
        for case in manifest.get("cases", []):
            identity = str(case.get("case_id", case.get("sample_index")))
            if identity in seen_cases:
                raise ValueError(f"manifest {path} repeats case {identity}")
            seen_cases.add(identity)
            source = case.get("source_file")
            if source is not None:
                prior = source_fold.setdefault(str(source), manifest_fold)
                if prior != manifest_fold:
                    raise ValueError(f"manifest {path} leaks source {source} across folds")
    return data, matches[0], hashlib.sha256(raw).hexdigest()


def _manifest_indices(path: Path, fold: int):
    """Compatibility helper used by focused tests and legacy callers."""
    _data, manifest, _sha256 = _load_manifest_fold(path, fold)
    indices = [int(row["sample_index"]) for row in manifest["cases"]]
    if len(indices) != len(set(indices)):
        raise ValueError(f"manifest {path} fold {fold} contains duplicate samples")
    return indices


def _resolve_manifest_cases(dataset, data_root: Path, path: Path, fold: int):
    """Resolve stable source+offset identities and fail closed on stale data."""
    data, manifest, manifest_sha256 = _load_manifest_fold(path, fold)
    if int(data.get("schema_version", 0)) < 3:
        raise ValueError(
            f"manifest {path} predates complete optimizer/scoring identity schema"
        )
    if data.get("split_unit") != "source_file":
        raise ValueError(f"manifest {path} split_unit must be source_file")
    dataset.all_files = sorted(str(Path(item).resolve()) for item in dataset.all_files)
    if hasattr(dataset, "cached_file_idx"):
        dataset.cached_file_idx = -1
    inventory_sha256 = _inventory_sha256(dataset, data_root)
    dataset_contract = data.get("dataset", {})
    expected_commit = dataset_contract.get("official_floorset_commit")
    resolved_commit = _git_commit(data_root)
    if not expected_commit or resolved_commit != expected_commit:
        raise ValueError(
            f"FloorSet commit does not match manifest: {resolved_commit} != "
            f"{expected_commit}"
        )
    expected_inventory = dataset_contract.get("source_inventory_sha256")
    if expected_inventory != inventory_sha256:
        raise ValueError(
            f"dataset inventory does not match manifest: {inventory_sha256} != "
            f"{expected_inventory}"
        )
    if int(data["dataset"]["layouts_per_file"]) != int(dataset.layouts_per_file):
        raise ValueError("dataset layouts_per_file does not match manifest")
    if int(data["dataset"]["source_file_count"]) != len(dataset.all_files):
        raise ValueError("dataset source-file count does not match manifest")
    relative_to_index = {}
    for file_index, path_string in enumerate(dataset.all_files):
        relative = _canonical_relative(Path(path_string), data_root)
        if relative in relative_to_index:
            raise ValueError(f"dataset repeats source path {relative}")
        relative_to_index[relative] = file_index

    generation = data.get("generation", {})
    seed = int(generation["seed"])
    num_folds = int(generation["num_folds"])
    selected = []
    for case in manifest["cases"]:
        relative = str(case["source_file"]).replace("\\", "/")
        file_index = relative_to_index.get(relative)
        if file_index is None:
            raise ValueError(f"manifest source is absent from dataset: {relative}")
        if _fold_for_file(relative, seed, num_folds) != fold:
            raise ValueError(f"source {relative} does not hash to fold {fold}")
        offset = int(case["file_offset"])
        if not 0 <= offset < int(dataset.layouts_per_file):
            raise ValueError(f"invalid file offset for {relative}: {offset}")
        sample_index = file_index * int(dataset.layouts_per_file) + offset
        sample = dataset[sample_index]
        n = int((sample["input"][0] != -1).sum().item())
        if n != int(case["block_count"]):
            raise ValueError(f"block count changed for {relative}#{offset}: {n}")
        input_sha256 = _input_sha256(sample)
        if input_sha256 != case.get("input_sha256"):
            raise ValueError(f"input digest changed for {relative}#{offset}")
        replayed = _case_metadata(sample, sample_index, relative, offset, n)
        for key in (
            "case_id",
            "fixed_blocks",
            "preplaced_blocks",
            "mib_groups",
            "cluster_groups",
            "boundary_blocks",
            "b2b_edges",
            "p2b_edges",
            "area_sum",
            "optimizer_target_sha256",
            "scoring_label_sha256",
        ):
            if replayed[key] != case.get(key):
                raise ValueError(f"metadata {key} changed for {relative}#{offset}")
        selected.append((sample_index, sample, case))
    expected_per_size = int(generation["per_size"])
    counts = defaultdict(int)
    for _sample_index, _sample, case in selected:
        counts[int(case["block_count"])] += 1
    expected_sizes = range(
        int(generation["min_blocks"]), int(generation["max_blocks"]) + 1
    )
    if any(counts[size] != expected_per_size for size in expected_sizes):
        raise ValueError(f"fold {fold} does not satisfy exact per-size quotas")
    if int(manifest["case_count"]) != len(selected):
        raise ValueError(f"fold {fold} case_count metadata is stale")
    if int(manifest["source_file_count"]) != len(
        {case["source_file"] for _index, _sample, case in selected}
    ):
        raise ValueError(f"fold {fold} source_file_count metadata is stale")
    return selected, {
        "sha256": manifest_sha256,
        "schema_version": int(data["schema_version"]),
        "fold": fold,
        "fold_metadata": {key: value for key, value in manifest.items() if key != "cases"},
        "generation": generation,
        "dataset": data.get("dataset", {}),
        "resolved_inventory_sha256": inventory_sha256,
        "resolved_official_floorset_commit": resolved_commit,
    }


def _validate_solver_positions(positions, block_count: int):
    """Return canonical rectangles or reject an incomplete/malformed output."""
    if positions is None or isinstance(positions, (str, bytes, dict)):
        raise ValueError("solver positions must be a sequence of rectangles")
    try:
        row_count = len(positions)
    except TypeError as exc:
        raise ValueError("solver positions must be a sequence of rectangles") from exc
    if row_count != block_count:
        raise ValueError(
            f"solver returned {row_count} positions for {block_count} blocks"
        )

    canonical = []
    for index, row in enumerate(positions):
        if isinstance(row, (str, bytes, dict)):
            raise ValueError(f"solver position {index} is not an x/y/w/h row")
        try:
            width = len(row)
        except TypeError as exc:
            raise ValueError(f"solver position {index} is not an x/y/w/h row") from exc
        if width != 4:
            raise ValueError(f"solver position {index} must have exactly four values")
        try:
            rectangle = tuple(float(row[column]) for column in range(4))
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"solver position {index} must be numeric") from exc
        if not all(math.isfinite(value) for value in rectangle):
            raise ValueError(f"solver position {index} contains a non-finite value")
        if rectangle[2] <= 0.0 or rectangle[3] <= 0.0:
            raise ValueError(f"solver position {index} must have positive width and height")
        canonical.append(rectangle)
    return canonical


def _file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(path: Path):
    digest = hashlib.sha256()
    for source in sorted(path.glob("*.py")):
        digest.update(source.name.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(_file_sha256(source)))
    return digest.hexdigest()


def _solver_component_hashes(path: Path):
    missing = [
        name for name in LIVE_SOLVER_COMPONENTS if not (path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "live solver component(s) missing from "
            f"{path}: {', '.join(missing)}"
        )
    return {
        name: _file_sha256(path / name)
        for name in LIVE_SOLVER_COMPONENTS
    }


def _git_state(path: Path):
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        tracked_status = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {
            "commit": commit,
            "dirty": bool(status),
            "tracked_dirty": bool(tracked_status),
            "has_untracked": any(line.startswith("??") for line in status.splitlines()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _metric_row(
    sample_index, n, metrics, runtime, golden_mib_violations, identity=None
):
    row = {
        "sample_index": sample_index,
        "block_count": n,
        "is_feasible": bool(metrics.is_feasible),
        "cost": float(metrics.cost),
        "hpwl_gap": float(metrics.hpwl_gap),
        "hpwl_gap_clamped": max(0.0, float(metrics.hpwl_gap)),
        "area_gap": float(metrics.area_gap),
        "area_gap_clamped": max(0.0, float(metrics.area_gap)),
        "violations_relative": float(metrics.violations_relative),
        "boundary_violations": int(metrics.boundary_violations),
        "grouping_violations": int(metrics.grouping_violations),
        "mib_violations": int(metrics.mib_violations),
        "golden_mib_violations": int(golden_mib_violations),
        "runtime_seconds": float(runtime),
    }
    if identity is not None:
        row.update(
            {
                "case_id": identity["case_id"],
                "source_file": identity["source_file"],
                "file_offset": int(identity["file_offset"]),
                "input_sha256": identity["input_sha256"],
                "optimizer_target_sha256": identity["optimizer_target_sha256"],
                "scoring_label_sha256": identity["scoring_label_sha256"],
            }
        )
    return row


def collect_stratified(
    dataset, min_blocks, max_blocks, per_size, seed, max_files,
    require_golden_mib_clean=False,
):
    # glob order is filesystem-dependent; make sampling stable across machines.
    dataset.all_files = sorted(dataset.all_files)
    file_ids = list(range(len(dataset.all_files)))
    random.Random(seed).shuffle(file_ids)
    buckets = {n: [] for n in range(min_blocks, max_blocks + 1)}
    scanned_files = 0
    for file_idx in file_ids[:max_files]:
        scanned_files += 1
        base = file_idx * dataset.layouts_per_file
        for offset in range(dataset.layouts_per_file):
            sample_index = base + offset
            sample = dataset[sample_index]
            area_target = sample["input"][0]
            n = int((area_target != -1).sum().item())
            if n not in buckets or len(buckets[n]) >= per_size:
                continue
            if require_golden_mib_clean and _golden_mib_violation_count(sample, n) != 0:
                continue
            buckets[n].append((sample_index, sample))
        if all(len(rows) >= per_size for rows in buckets.values()):
            break
    missing = {n: per_size - len(rows) for n, rows in buckets.items() if len(rows) < per_size}
    if missing:
        raise RuntimeError(f"stratified sample incomplete after {scanned_files} files: {missing}")
    selected = [item for n in sorted(buckets) for item in buckets[n]]
    return selected, scanned_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "external" / "FloorSet")
    parser.add_argument("--solver-dir", type=Path, default=SOLUTION_DIR)
    parser.add_argument(
        "--learned-mode",
        choices=(
            "solver-default", "off", "replacement", "additive",
            "additive_first_pass",
        ),
        default="solver-default",
        help="research override for solvers exposing _learned_order_mode",
    )
    parser.add_argument("--min-blocks", type=int, default=100)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--per-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--require-golden-mib-clean", action="store_true")
    parser.add_argument(
        "--indices-from", type=Path,
        help="reuse sample_index values from a previous holdout result",
    )
    parser.add_argument(
        "--fold-manifest", type=Path,
        help="evaluate sample indices from a source-file-disjoint fold manifest",
    )
    parser.add_argument("--fold", type=int, default=0, help="fold ID in --fold-manifest")
    parser.add_argument(
        "--oracle-baseline-selector", action="store_true",
        help="diagnostic only: expose the golden HPWL/area baseline to candidate selection",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "training_holdout.json")
    args = parser.parse_args()

    dataset = FloorplanDatasetLite(str(args.data_root))
    dataset.all_files = sorted(str(Path(item).resolve()) for item in dataset.all_files)
    if hasattr(dataset, "cached_file_idx"):
        dataset.cached_file_idx = -1
    manifest_provenance = None
    if args.indices_from and args.fold_manifest:
        parser.error("--indices-from and --fold-manifest are mutually exclusive")
    if args.fold_manifest:
        selected, manifest_provenance = _resolve_manifest_cases(
            dataset, args.data_root, args.fold_manifest, args.fold
        )
        scanned_files = 0
    elif args.indices_from:
        prior = json.loads(args.indices_from.read_text())
        indices = [int(row["sample_index"]) for row in prior["cases"]]
        selected = [(index, dataset[index], None) for index in indices]
        scanned_files = 0
    else:
        sampled, scanned_files = collect_stratified(
            dataset, args.min_blocks, args.max_blocks, args.per_size, args.seed,
            args.max_files, args.require_golden_mib_clean,
        )
        selected = [(index, sample, None) for index, sample in sampled]
    optimizer = _load_optimizer(args.solver_dir)
    if args.learned_mode != "solver-default":
        if not hasattr(optimizer, "_learned_order_mode"):
            parser.error("solver does not expose _learned_order_mode")
        optimizer._learned_order_mode = args.learned_mode
        optimizer._learned_order_enabled = args.learned_mode != "off"
    solver_rows = []
    golden_rows = []

    for ordinal, (sample_index, sample, identity) in enumerate(selected, 1):
        area_target, b2b, p2b, pins, constraints = sample["input"]
        _, fp_sol, stored_metrics = sample["label"]
        # Give the optimizer private tensors. Evaluation always consumes a
        # second pristine copy, so an accidental in-place mutation cannot make
        # a candidate pass its own altered constraints.
        solve_area = _clone_value(area_target)
        solve_b2b = _clone_value(b2b)
        solve_p2b = _clone_value(p2b)
        solve_pins = _clone_value(pins)
        solve_constraints = _clone_value(constraints)
        eval_area = _clone_value(area_target)
        eval_b2b = _clone_value(b2b)
        eval_p2b = _clone_value(p2b)
        eval_pins = _clone_value(pins)
        eval_constraints = _clone_value(constraints)
        n = int((area_target != -1).sum().item())
        golden_positions = _positions_from_training_label(fp_sol, n)
        opt_targets = _optimizer_targets(eval_constraints, golden_positions, n)
        baseline = _baseline(stored_metrics)

        golden_metrics = evaluate_solution(
            {"positions": golden_positions, "runtime": 1.0},
            baseline,
            eval_constraints,
            eval_b2b,
            eval_p2b,
            eval_pins,
            eval_area,
            golden_positions,
            median_runtime=1.0,
        )
        if args.oracle_baseline_selector:
            optimizer._baselines_by_n = {
                n: (baseline["hpwl_baseline"], baseline["area_baseline"])
            }
        t0 = time.perf_counter()
        positions = optimizer.solve(
            n,
            solve_area,
            solve_b2b,
            solve_p2b,
            solve_pins,
            solve_constraints,
            opt_targets.clone(),
        )
        runtime = time.perf_counter() - t0
        positions = _validate_solver_positions(positions, n)
        solver_metrics = evaluate_solution(
            {"positions": positions, "runtime": 1.0},
            baseline,
            eval_constraints,
            eval_b2b,
            eval_p2b,
            eval_pins,
            eval_area,
            golden_positions,
            median_runtime=1.0,
        )
        solver_rows.append(
            _metric_row(
                sample_index,
                n,
                solver_metrics,
                runtime,
                golden_metrics.mib_violations,
                identity,
            )
        )
        golden_rows.append(
            _metric_row(
                sample_index,
                n,
                golden_metrics,
                0.0,
                golden_metrics.mib_violations,
                identity,
            )
        )
        if ordinal % 10 == 0 or ordinal == len(selected):
            print(f"scored {ordinal}/{len(selected)}", flush=True)

    clean_solver = [r for r in solver_rows if r["golden_mib_violations"] == 0]
    clean_golden = [r for r in golden_rows if r["golden_mib_violations"] == 0]
    by_size = defaultdict(list)
    for row in solver_rows:
        by_size[row["block_count"]].append(row)
    if manifest_provenance:
        generation = manifest_provenance["generation"]
        reported_min_blocks = int(generation["min_blocks"])
        reported_max_blocks = int(generation["max_blocks"])
        reported_per_size = int(generation["per_size"])
        reported_seed = int(generation["seed"])
    else:
        reported_min_blocks = args.min_blocks
        reported_max_blocks = args.max_blocks
        reported_per_size = args.per_size
        reported_seed = args.seed
    evaluator_path = OFFICIAL_ROOT / "iccad2026contest" / "iccad2026_evaluate.py"
    result = {
        "config": {
            "min_blocks": reported_min_blocks,
            "max_blocks": reported_max_blocks,
            "per_size": reported_per_size,
            "seed": reported_seed,
            "scanned_files": scanned_files,
            "require_golden_mib_clean": (
                None if manifest_provenance else args.require_golden_mib_clean
            ),
            "solver_dir": _portable_path(args.solver_dir),
            "learned_mode": args.learned_mode,
            "indices_from": _portable_path(args.indices_from),
            "fold_manifest": _portable_path(args.fold_manifest),
            "fold": args.fold if args.fold_manifest else None,
            "manifest": manifest_provenance,
            "oracle_baseline_selector": args.oracle_baseline_selector,
            "runtime_factor_mode": "neutral_rf_1",
        },
        "provenance": {
            "evaluation_harness_sha256": _file_sha256(Path(__file__)),
            "solver_source_sha256": _source_tree_sha256(args.solver_dir),
            "solver_component_sha256": _solver_component_hashes(args.solver_dir),
            "solver_git": _git_state(args.solver_dir),
            "evaluator_sha256": _file_sha256(evaluator_path),
            "official_floorset_git": _git_state(OFFICIAL_ROOT),
        },
        "solver_all": _summary(solver_rows),
        "golden_all": _summary(golden_rows),
        "solver_golden_mib_clean": _summary(clean_solver),
        "golden_mib_clean": _summary(clean_golden),
        "golden_mib_violation_cases": len(solver_rows) - len(clean_solver),
        "solver_by_size": {str(n): _summary(rows) for n, rows in sorted(by_size.items())},
        "cases": solver_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "solver_by_size")}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

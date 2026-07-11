#!/usr/bin/env python3
"""Build immutable, source-file-disjoint FloorSet holdout manifests.

The 112 layouts stored in one ``.th`` file can share generation context.  A
layout-level random split therefore overstates generalization.  This tool hashes
whole source files into folds, then fills an exact per-block-count quota while
optionally filtering MIB groups that cannot share one shape from input-visible
area tolerances and hard fixed/preplaced targets.

Free-block golden positions, trees, and metrics never affect admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOORSET = ROOT / "external" / "FloorSet"
sys.path.insert(0, str(FLOORSET))

from lite_dataset import FloorplanDatasetLite  # noqa: E402


def _scalar(value):
    return float(value.item()) if hasattr(value, "item") else float(value)


def _fold_for_file(relative_path: str, seed: int, num_folds: int) -> int:
    digest = hashlib.sha256(f"{seed}:{relative_path}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % num_folds


def _hash_offset(relative_path: str, seed: int, layouts_per_file: int) -> int:
    """Choose one layout without inspecting either its input or label."""
    digest = hashlib.sha256(f"offset:{seed}:{relative_path}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % layouts_per_file


def _canonical_relative(path: Path, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"dataset source lies outside data root: {path}") from exc


def _git_commit(path: Path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _update_value_digest(digest, value):
    if value is None:
        digest.update(b"none\0")
        return
    if hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        return
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest.update(payload.encode("utf-8"))


def _input_sha256(sample) -> str:
    digest = hashlib.sha256()
    for value in sample["input"]:
        _update_value_digest(digest, value)
        digest.update(b"\xff")
    return digest.hexdigest()


def _optimizer_target_rows(sample, n: int):
    """Return the fixed/preplaced target rows exposed to the optimizer.

    FloorSet stores these contest inputs inside ``fp_sol`` rather than the
    sample's input tuple.  Keeping the derivation here lets the manifest bind
    the *complete* optimizer input without admitting free-block golden data.
    """
    constraints = sample["input"][4]
    fp_sol = sample["label"][1]
    columns = (
        constraints.shape[1]
        if constraints is not None and len(getattr(constraints, "shape", ())) > 1
        else 0
    )
    rows = [[-1.0, -1.0, -1.0, -1.0] for _ in range(n)]
    for i in range(n):
        fixed = columns > 0 and _scalar(constraints[i, 0]) != 0.0
        preplaced = columns > 1 and _scalar(constraints[i, 1]) != 0.0
        if preplaced:
            width = _scalar(fp_sol[i, 0])
            height = _scalar(fp_sol[i, 1])
            rows[i] = [
                _scalar(fp_sol[i, 2]),
                _scalar(fp_sol[i, 3]),
                width,
                height,
            ]
        elif fixed:
            width = _scalar(fp_sol[i, 0])
            height = _scalar(fp_sol[i, 1])
            rows[i][2:] = [width, height]
    return rows


def _optimizer_target_sha256(sample, n: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"floorset_optimizer_targets_v1\0")
    _update_value_digest(digest, _optimizer_target_rows(sample, n))
    return digest.hexdigest()


def _scoring_label_sha256(sample, n: int) -> str:
    """Bind exactly the golden layout and metrics consumed by the scorer."""
    _tree, fp_sol, metrics = sample["label"]
    digest = hashlib.sha256()
    digest.update(b"floorset_scoring_label_v1\0")
    _update_value_digest(digest, fp_sol[:n])
    digest.update(b"\xff")
    _update_value_digest(digest, metrics)
    return digest.hexdigest()


def _inventory_sha256(dataset, data_root: Path) -> str:
    digest = hashlib.sha256()
    for path_string in dataset.all_files:
        path = Path(path_string)
        relative = _canonical_relative(path, data_root)
        digest.update(f"{relative}\0{path.stat().st_size}\n".encode("utf-8"))
    return digest.hexdigest()


def _mib_is_input_compatible(area_target, constraints, hard_target_sol, n: int) -> bool:
    """Whether one shape can satisfy each MIB group within area tolerance.

    ``hard_target_sol`` is consulted only for fixed/preplaced dimensions, which
    are part of the contest input contract.  Free-block golden shapes never
    affect this predicate.
    """
    if constraints is None or len(getattr(constraints, "shape", ())) < 2:
        return True
    if constraints.shape[1] <= 2:
        return True
    groups = constraints[:n, 2]
    max_group = int(groups.max().item()) if groups.numel() else 0
    for group_id in range(1, max_group + 1):
        members = [i for i in range(n) if int(_scalar(groups[i])) == group_id]
        if len(members) < 2:
            continue
        soft_areas = []
        hard_shapes = []
        columns = constraints.shape[1]
        for i in members:
            fixed = columns > 0 and _scalar(constraints[i, 0]) != 0.0
            preplaced = columns > 1 and _scalar(constraints[i, 1]) != 0.0
            if fixed or preplaced:
                hard_shapes.append(
                    (_scalar(hard_target_sol[i, 0]), _scalar(hard_target_sol[i, 1]))
                )
            else:
                soft_areas.append(_scalar(area_target[i]))
        # The official area-tolerance check excludes fixed and preplaced
        # blocks.  Only genuinely soft members constrain the common-shape area
        # interval; hard members constrain its dimensions directly.
        if soft_areas and min(soft_areas) <= 0.0:
            return False
        lower = max((0.99 * area for area in soft_areas), default=None)
        upper = min((1.01 * area for area in soft_areas), default=None)
        if lower is not None and lower > upper + 1e-9:
            return False
        if hard_shapes:
            width, height = hard_shapes[0]
            if width <= 0.0 or height <= 0.0:
                return False
            if any(
                round(other_width, 4) != round(width, 4)
                or round(other_height, 4) != round(height, 4)
                for other_width, other_height in hard_shapes[1:]
            ):
                return False
            if lower is not None:
                shape_area = width * height
                if shape_area < lower - 1e-7 or shape_area > upper + 1e-7:
                    return False
    return True


def _valid_edge_count(rows) -> int:
    if rows is None:
        return 0
    return sum(1 for row in rows if int(_scalar(row[0])) >= 0 and _scalar(row[2]) > 0)


def _case_metadata(sample, sample_index: int, file_path: str, offset: int, n: int):
    area, b2b, p2b, _pins, constraints = sample["input"]
    nc = constraints.shape[1] if constraints is not None and constraints.dim() > 1 else 0

    def count(column, predicate=lambda value: value != 0.0):
        if nc <= column:
            return 0
        return sum(predicate(_scalar(constraints[i, column])) for i in range(n))

    mib_groups = set()
    cluster_groups = set()
    if nc > 2:
        mib_groups = {int(_scalar(constraints[i, 2])) for i in range(n)} - {0}
    if nc > 3:
        cluster_groups = {int(_scalar(constraints[i, 3])) for i in range(n)} - {0}
    return {
        "case_id": f"{file_path}#{offset}",
        "sample_index": sample_index,
        "source_file": file_path,
        "file_offset": offset,
        "block_count": n,
        "fixed_blocks": count(0),
        "preplaced_blocks": count(1),
        "mib_groups": len(mib_groups),
        "cluster_groups": len(cluster_groups),
        "boundary_blocks": count(4),
        "b2b_edges": _valid_edge_count(b2b),
        "p2b_edges": _valid_edge_count(p2b),
        "area_sum": round(sum(_scalar(area[i]) for i in range(n)), 6),
        "input_sha256": _input_sha256(sample),
        "optimizer_target_sha256": _optimizer_target_sha256(sample, n),
        "scoring_label_sha256": _scoring_label_sha256(sample, n),
    }


def build_manifests(
    dataset,
    *,
    data_root: Path,
    min_blocks: int,
    max_blocks: int,
    num_folds: int,
    per_size: int,
    seed: int,
    require_mib_input_compatible: bool,
    case_selection: str,
    offset_seed: int,
    max_files: int | None,
):
    if case_selection not in {"sequential", "hash_one_per_source"}:
        raise ValueError(f"unsupported case_selection: {case_selection}")
    dataset.all_files = sorted(str(Path(path).resolve()) for path in dataset.all_files)
    if hasattr(dataset, "cached_file_idx"):
        dataset.cached_file_idx = -1
    official_floorset_commit = _git_commit(data_root)
    if not official_floorset_commit:
        raise ValueError(
            f"data root must live in the pinned FloorSet git checkout: {data_root}"
        )
    inventory_sha256 = _inventory_sha256(dataset, data_root)
    file_records = []
    for file_index, path_string in enumerate(dataset.all_files):
        path = Path(path_string)
        relative = _canonical_relative(path, data_root)
        digest = hashlib.sha256(f"order:{seed}:{relative}".encode()).digest()
        file_records.append((digest, file_index, relative))
    file_records.sort()
    if max_files is not None:
        file_records = file_records[:max_files]

    buckets = [defaultdict(list) for _ in range(num_folds)]
    files_seen = [set() for _ in range(num_folds)]
    incompatible_rejected = 0
    samples_examined = 0
    for _digest, file_index, relative in file_records:
        fold = _fold_for_file(relative, seed, num_folds)
        if all(len(buckets[fold][n]) >= per_size for n in range(min_blocks, max_blocks + 1)):
            continue
        base = file_index * dataset.layouts_per_file
        used_file = False
        if case_selection == "hash_one_per_source":
            offsets = (
                _hash_offset(relative, offset_seed, int(dataset.layouts_per_file)),
            )
        else:
            offsets = range(dataset.layouts_per_file)
        for offset in offsets:
            sample_index = base + offset
            sample = dataset[sample_index]
            samples_examined += 1
            area, _b2b, _p2b, _pins, constraints = sample["input"]
            _tree, hard_target_sol, _metrics = sample["label"]
            n = int((area != -1).sum().item())
            if n < min_blocks or n > max_blocks or len(buckets[fold][n]) >= per_size:
                continue
            if require_mib_input_compatible and not _mib_is_input_compatible(
                area, constraints, hard_target_sol, n
            ):
                incompatible_rejected += 1
                continue
            buckets[fold][n].append(
                _case_metadata(sample, sample_index, relative, offset, n)
            )
            used_file = True
        if used_file:
            files_seen[fold].add(relative)
        if all(
            len(buckets[fold][n]) >= per_size
            for fold in range(num_folds)
            for n in range(min_blocks, max_blocks + 1)
        ):
            break

    missing = {}
    for fold in range(num_folds):
        for n in range(min_blocks, max_blocks + 1):
            if len(buckets[fold][n]) < per_size:
                missing[f"fold_{fold}:n_{n}"] = per_size - len(buckets[fold][n])
    if missing:
        preview = dict(list(missing.items())[:20])
        raise RuntimeError(f"incomplete fold quotas ({len(missing)} missing buckets): {preview}")

    manifests = []
    for fold in range(num_folds):
        cases = [row for n in range(min_blocks, max_blocks + 1) for row in buckets[fold][n]]
        manifests.append(
            {
                "fold": fold,
                "seed": seed,
                "min_blocks": min_blocks,
                "max_blocks": max_blocks,
                "per_size": per_size,
                "require_mib_input_compatible": require_mib_input_compatible,
                "source_file_count": len(files_seen[fold]),
                "case_count": len(cases),
                "cases": cases,
            }
        )
    return {
        "schema_version": 3,
        "split_unit": "source_file",
        "num_folds": num_folds,
        "dataset": {
            "name": "FloorSet-Lite",
            "official_floorset_commit": official_floorset_commit,
            "loader": "lite_dataset.FloorplanDatasetLite",
            "layouts_per_file": int(dataset.layouts_per_file),
            "source_file_count": len(dataset.all_files),
            "source_inventory_sha256": inventory_sha256,
        },
        "generation": {
            "min_blocks": min_blocks,
            "max_blocks": max_blocks,
            "num_folds": num_folds,
            "per_size": per_size,
            "seed": seed,
            "case_selection": case_selection,
            "offset_seed": (
                offset_seed if case_selection == "hash_one_per_source" else None
            ),
            "max_files": max_files,
            "mib_policy": (
                "input_area_interval_and_hard_target_compatible"
                if require_mib_input_compatible
                else "unfiltered"
            ),
        },
        "samples_examined": samples_examined,
        "mib_input_incompatible_rejected": incompatible_rejected,
        "manifests": manifests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=FLOORSET)
    parser.add_argument("--min-blocks", type=int, default=100)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--per-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument(
        "--case-selection",
        choices=("sequential", "hash_one_per_source"),
        default="sequential",
        help=(
            "sequential scans offsets for the compatibility stratum; "
            "hash_one_per_source chooses one label-blind offset and guarantees "
            "at most one case per source file"
        ),
    )
    parser.add_argument("--offset-seed", type=int, default=20260711)
    parser.add_argument(
        "--allow-mib-incompatible",
        "--allow-mib-corrupt",
        dest="allow_mib_incompatible",
        action="store_true",
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "folds" / "heavy_clean_v1.json",
    )
    args = parser.parse_args()
    if not (1 <= args.num_folds <= 20):
        parser.error("--num-folds must be in [1, 20]")
    if args.min_blocks > args.max_blocks or args.per_size < 1:
        parser.error("invalid block range or per-size quota")

    dataset = FloorplanDatasetLite(str(args.data_root))
    result = build_manifests(
        dataset,
        data_root=args.data_root,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        num_folds=args.num_folds,
        per_size=args.per_size,
        seed=args.seed,
        require_mib_input_compatible=not args.allow_mib_incompatible,
        case_selection=args.case_selection,
        offset_seed=args.offset_seed,
        max_files=args.max_files,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"wrote {args.output}: {result['num_folds']} folds, "
        f"{sum(m['case_count'] for m in result['manifests'])} cases, "
        f"{result['mib_input_incompatible_rejected']} input-incompatible MIB samples rejected"
    )


if __name__ == "__main__":
    main()

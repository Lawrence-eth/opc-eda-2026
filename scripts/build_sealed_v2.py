#!/usr/bin/env python3
"""Freeze dual-stratum sealed heavy panels from sources unused by v1.

Each source contributes two layouts: an input-visible MIB-compatible offset and
an independently hash-ordered raw offset.  Source files are assigned to the
beta/final roles by the same deterministic whole-source fold hash enforced by
the evaluation harness.  Golden metrics and free-block geometry never affect
selection; scoring labels are hashed only to bind the frozen evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOORSET = ROOT / "external" / "FloorSet"
sys.path[:0] = [str(FLOORSET), str(ROOT / "scripts")]

from build_holdout_folds import (  # noqa: E402
    _case_metadata,
    _fold_for_file,
    _git_commit,
    _inventory_sha256,
    _mib_is_input_compatible,
)
from lite_dataset import FloorplanDatasetLite  # noqa: E402
from train_order_model import (  # noqa: E402
    _atomic_write_json,
    _file_sha256,
    load_holdout_sources,
    load_or_build_source_index,
)


SCHEMA_VERSION = 3
ROLE_NAMES = ("beta_sealed", "final_sealed")


def _hash_order(namespace: str, seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{namespace}:{seed}:{value}".encode("utf-8")).digest()


def _offset_order(namespace: str, seed: int, relative: str, count: int):
    return sorted(
        range(count),
        key=lambda offset: _hash_order(namespace, seed, f"{relative}#{offset}"),
    )


def build_sealed_manifest(
    dataset,
    *,
    data_root: Path,
    excluded_manifests: list[Path],
    source_index_cache: Path | None,
    min_blocks: int,
    max_blocks: int,
    sources_per_size_per_role: int,
    seed: int,
    clean_offset_seed: int,
    raw_offset_seed: int,
    progress_every_files: int = 0,
):
    if min_blocks < 1 or max_blocks < min_blocks:
        raise ValueError("invalid block-count range")
    if sources_per_size_per_role < 1:
        raise ValueError("sources-per-size-per-role must be positive")
    excluded = load_holdout_sources(excluded_manifests)
    indexed, index_provenance = load_or_build_source_index(
        dataset,
        data_root=data_root,
        cache_path=source_index_cache,
        progress_every_files=progress_every_files,
    )
    official_commit = _git_commit(data_root)
    if not official_commit:
        raise ValueError("data root must live in the pinned FloorSet checkout")
    inventory_sha256 = _inventory_sha256(dataset, data_root)
    layouts_per_file = int(dataset.layouts_per_file)

    buckets = {
        (fold, n): []
        for fold in range(len(ROLE_NAMES))
        for n in range(min_blocks, max_blocks + 1)
    }
    candidates = [
        source
        for source in indexed
        if min_blocks <= source.block_count <= max_blocks
        and source.relative_path not in excluded.paths
    ]
    candidates.sort(
        key=lambda source: _hash_order("sealed-source", seed, source.relative_path)
    )

    clean_incompatible_offsets = 0
    sources_without_clean_offset = 0
    sources_examined = 0
    for source in candidates:
        fold = _fold_for_file(source.relative_path, seed, len(ROLE_NAMES))
        key = (fold, source.block_count)
        if len(buckets[key]) >= sources_per_size_per_role:
            continue
        sources_examined += 1
        base = source.file_index * layouts_per_file
        clean_offset = None
        for offset in _offset_order(
            "sealed-clean-offset", clean_offset_seed, source.relative_path, layouts_per_file
        ):
            sample = dataset[base + offset]
            area, _b2b, _p2b, _pins, constraints = sample["input"]
            _tree, hard_target_sol, _metrics = sample["label"]
            if _mib_is_input_compatible(
                area, constraints, hard_target_sol, source.block_count
            ):
                clean_offset = offset
                break
            clean_incompatible_offsets += 1
        if clean_offset is None:
            sources_without_clean_offset += 1
            continue
        raw_offsets = _offset_order(
            "sealed-raw-offset", raw_offset_seed, source.relative_path, layouts_per_file
        )
        raw_offset = next(offset for offset in raw_offsets if offset != clean_offset)

        cases = []
        for stratum, offset in (("clean", clean_offset), ("raw", raw_offset)):
            sample_index = base + offset
            sample = dataset[sample_index]
            case = _case_metadata(
                sample,
                sample_index,
                source.relative_path,
                offset,
                source.block_count,
            )
            case["stratum"] = stratum
            cases.append(case)
        buckets[key].append(cases)
        if all(
            len(rows) >= sources_per_size_per_role for rows in buckets.values()
        ):
            break

    missing = {
        f"{ROLE_NAMES[fold]}:n{n}": sources_per_size_per_role - len(buckets[fold, n])
        for fold in range(len(ROLE_NAMES))
        for n in range(min_blocks, max_blocks + 1)
        if len(buckets[fold, n]) < sources_per_size_per_role
    }
    if missing:
        raise RuntimeError(
            f"insufficient unused compatible sources ({len(missing)} buckets): {missing}"
        )

    manifests = []
    selected_sources = set()
    for fold, role in enumerate(ROLE_NAMES):
        cases = []
        role_sources = set()
        for n in range(min_blocks, max_blocks + 1):
            for pair in buckets[fold, n]:
                cases.extend(pair)
                role_sources.add(pair[0]["source_file"])
        if selected_sources & role_sources:
            raise AssertionError("sealed roles share a source file")
        selected_sources.update(role_sources)
        manifests.append(
            {
                "fold": fold,
                "role": role,
                "min_blocks": min_blocks,
                "max_blocks": max_blocks,
                "per_size": 2 * sources_per_size_per_role,
                "sources_per_size": sources_per_size_per_role,
                "source_file_count": len(role_sources),
                "case_count": len(cases),
                "cases": cases,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "split_unit": "source_file",
        "num_folds": len(ROLE_NAMES),
        "dataset": {
            "name": "FloorSet-Lite",
            "official_floorset_commit": official_commit,
            "loader": "lite_dataset.FloorplanDatasetLite",
            "layouts_per_file": layouts_per_file,
            "source_file_count": len(dataset.all_files),
            "source_inventory_sha256": inventory_sha256,
        },
        "generation": {
            "min_blocks": min_blocks,
            "max_blocks": max_blocks,
            "num_folds": len(ROLE_NAMES),
            "per_size": 2 * sources_per_size_per_role,
            "sources_per_size_per_role": sources_per_size_per_role,
            "seed": seed,
            "clean_offset_seed": clean_offset_seed,
            "raw_offset_seed": raw_offset_seed,
            "selection": "unused_source_hash_then_input_visible_clean_plus_label_blind_raw",
            "mib_policy": "clean_input_compatible_and_raw_unfiltered",
            "excluded_manifests": [str(path.relative_to(ROOT)) for path in excluded_manifests],
            "excluded_manifest_sha256s": list(excluded.manifest_sha256s),
            "excluded_source_files": len(excluded.paths),
            "builder_sha256": _file_sha256(Path(__file__)),
            "source_index": index_provenance,
        },
        "sources_examined": sources_examined,
        "sources_without_clean_offset": sources_without_clean_offset,
        "clean_incompatible_offsets_rejected": clean_incompatible_offsets,
        "selected_source_files": len(selected_sources),
        "manifests": manifests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=FLOORSET)
    parser.add_argument(
        "--excluded-manifest",
        type=Path,
        action="append",
        help="existing source holdout; repeat for a union",
    )
    parser.add_argument(
        "--source-index-cache",
        type=Path,
        default=ROOT / "results" / "work" / "order_source_index_v1.json",
    )
    parser.add_argument("--min-blocks", type=int, default=100)
    parser.add_argument("--max-blocks", type=int, default=120)
    parser.add_argument("--sources-per-size-per-role", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--clean-offset-seed", type=int, default=20260714)
    parser.add_argument("--raw-offset-seed", type=int, default=20260715)
    parser.add_argument("--progress-every-files", type=int, default=250)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "folds" / "heavy_sealed_v2.json",
    )
    args = parser.parse_args()
    excluded_manifests = args.excluded_manifest or [
        ROOT / "results" / "folds" / "heavy_clean_v1.json",
        ROOT / "results" / "folds" / "heavy_raw_hash_v1.json",
    ]
    dataset = FloorplanDatasetLite(str(args.data_root))
    result = build_sealed_manifest(
        dataset,
        data_root=args.data_root,
        excluded_manifests=excluded_manifests,
        source_index_cache=args.source_index_cache,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        sources_per_size_per_role=args.sources_per_size_per_role,
        seed=args.seed,
        clean_offset_seed=args.clean_offset_seed,
        raw_offset_seed=args.raw_offset_seed,
        progress_every_files=args.progress_every_files,
    )
    _atomic_write_json(args.output, result)
    print(
        f"wrote {args.output}: {result['selected_source_files']} unused sources, "
        f"{sum(row['case_count'] for row in result['manifests'])} bound cases"
    )


if __name__ == "__main__":
    main()

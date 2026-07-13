import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fold_for_file(relative_path, seed, num_folds):
    digest = hashlib.sha256(f"{seed}:{relative_path}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % num_folds


def test_sealed_v2_is_source_disjoint_stratified_and_bound_to_v1():
    path = ROOT / "results" / "folds" / "heavy_sealed_v2.json"
    sealed = json.loads(path.read_text())
    excluded_paths = [
        ROOT / "results" / "folds" / "heavy_clean_v1.json",
        ROOT / "results" / "folds" / "heavy_raw_hash_v1.json",
    ]
    assert sealed["schema_version"] == 3
    assert sealed["split_unit"] == "source_file"
    assert sealed["selected_source_files"] == 210
    assert sealed["generation"]["excluded_manifest_sha256s"] == [
        _sha256(path) for path in excluded_paths
    ]

    excluded_sources = set()
    for excluded_path in excluded_paths:
        data = json.loads(excluded_path.read_text())
        for manifest in data["manifests"]:
            excluded_sources.update(case["source_file"] for case in manifest["cases"])

    selected_sources = set()
    seed = sealed["generation"]["seed"]
    for fold, manifest in enumerate(sealed["manifests"]):
        assert manifest["fold"] == fold
        assert manifest["source_file_count"] == 105
        assert manifest["case_count"] == 210
        assert Counter(case["block_count"] for case in manifest["cases"]) == Counter(
            {n: 10 for n in range(100, 121)}
        )
        assert Counter(case["stratum"] for case in manifest["cases"]) == Counter(
            clean=105, raw=105
        )
        by_source = defaultdict(list)
        for case in manifest["cases"]:
            by_source[case["source_file"]].append(case)
        role_sources = set(by_source)
        assert not role_sources & excluded_sources
        assert not role_sources & selected_sources
        assert all(
            _fold_for_file(source, seed, sealed["num_folds"]) == fold
            for source in role_sources
        )
        assert all(
            {case["stratum"] for case in cases} == {"clean", "raw"}
            and len({case["file_offset"] for case in cases}) == 2
            for cases in by_source.values()
        )
        selected_sources.update(role_sources)
    assert len(selected_sources) == 210

# Oracle reconstruction evidence

`dual_parent_oracle_clean_fold0_v1.json` is the compact, hash-bound result of
the research-only dual-parent representation gate. It uses the pre-existing
source-disjoint `heavy_clean_v1` fold 0; it does not open the sealed-v2 beta or
final panels.

Reproduce it from a local FloorSet checkout with:

```bash
python scripts/audit_dual_parent_oracle.py \
  --data-root /path/to/FloorSet \
  --manifest results/folds/heavy_clean_v1.json \
  --fold 0 \
  --output results/oracle/dual_parent_oracle_clean_fold0_v1.json
```

The gate passed all 105 layouts (11,550 blocks): geometry and official metrics
match golden exactly, all 11,445 B*-tree edges hold, and the decoded RF=1 score
is identical to golden at `1.1077270456502948`. The JSON binds the manifest,
official evaluator, decoder, and audit script by SHA-256.

## Phase-1 dual-parent cache

`scripts/build_dual_parent_cache.py` converts that representation into
schema-versioned numeric research shards without implementing or selecting a
neural model. A production build excludes the union of the clean, raw-hash,
and sealed-v2 source-file manifests by default, then makes deterministic,
source-disjoint train/development/calibration partitions. Run it only from a
clean committed tree:

```bash
.venv/bin/python scripts/build_dual_parent_cache.py build \
  --data-root external/FloorSet \
  --output /tmp/dual-parent-cache-v3 \
  --source-index-cache /tmp/floorset-source-index-v1.json \
  --progress-every-sources 100

.venv/bin/python scripts/build_dual_parent_cache.py validate \
  --cache-dir /tmp/dual-parent-cache-v3 \
  --data-root external/FloorSet \
  --holdout-manifest results/folds/heavy_clean_v1.json \
  --holdout-manifest results/folds/heavy_raw_hash_v1.json \
  --holdout-manifest results/folds/heavy_sealed_v2.json \
  --expected-manifest-sha256 <digest-reported-by-build>
```

The output path must not already exist. Schema v3 requires `--data-root` to be
the root of the exact pinned FloorSet Git checkout and executes
`FloorplanDatasetLite` only from the verified loader bytes at that root; the
programmatic build API accepts no caller-supplied dataset. It binds the
checkout's commit, tree, loader module/class, and official-source manifest. The
builder writes deterministic NPZ members into a sibling staging directory,
validates every shard with `allow_pickle=False`, and
publishes with Linux `renameat2(RENAME_NOREPLACE)`, so a concurrently created
target is never replaced. Source paths and layout offsets occur only in
`manifest.json`; the manifest also binds source bytes, input/tree/golden
payloads, code, the pinned FloorSet revision, holdout manifests, and shard
descriptors. Raw layouts with internally inconsistent golden MIB dimensions
retain topology labels but receive per-block shape-supervision masks.

Full validation with `--data-root` requires the externally retained manifest
SHA-256. It reloads every recorded source file and raw offset through that same
verified dataset class, then recomputes the input, hard-target, tree, golden
geometry, and golden metrics hashes. Manifest-only consumers must not treat a
structural validation without `--data-root` as training authorization.

For a deliberately incomplete real-data smoke, add a small `--max-sources`,
`--max-layouts-per-source 1`, and `--allow-partial-partitions`. Such a cache is
marked partial in its manifest and is not a production training corpus.

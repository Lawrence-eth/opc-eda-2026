# Structured predictor evidence

`structured_linear_v1` is the first input-only learned dual-parent baseline.
It is a **rejected research candidate**, not part of v32 and not eligible for a
production merge.

## Contract

- Uses the existing 60 inference-visible permutation-equivariant block
  features plus nine direct inference-visible pair relations: normalized B2B
  weight, B2B presence, MIB/cluster equality, relative area, boundary
  compatibility, shared-pin similarity, and preplaced-anchor similarity.
- Numeric group identifiers are used only for equality. Source path, file
  offset, worker, instance, generator, PRNG, and seed never enter a model
  matrix.
- Training excluded the union of `heavy_clean_v1`, `heavy_raw_hash_v1`, and
  `heavy_sealed_v2`: 741 whole source files. The sealed manifest was used only
  to exclude source names; sealed layouts, labels, and scores were never opened
  by either evaluation harness.
- A deterministic projection creates a valid binary B*-tree. Learned vertical
  supports are restricted to positive predicted x-overlap. The skyline mode
  instead places each block on the highest already-placed x-overlapping block,
  which is overlap-free by construction when the predicted structure is
  compatible with all exact preplaced anchors.
- Any artifact error, decoder error, hard infeasibility, low confidence, or
  caller validation failure returns v32. The calibrated confidence threshold
  is `1.0` because no internally held-out layout was exactly predicted.

## Evidence and verdict

The model trained on 4,172 layouts from 1,043 sources. A separate internal
partition contains 464 layouts / 51,048 blocks from 116 sources. Internal
accuracy was 46.71% shape, 100% root, 9.22% horizontal edge, and 28.07%
vertical support/floor, with 0/464 exact layouts.

| Panel / mode | Raw candidate feasible | Exact layouts | Offline wins vs v32 | Model increment (median) | Gated result |
|---|---:|---:|---:|---:|---:|
| Public / learned support | 0/100 | 0 | 0 | 0.337 s | v32 `1.615379`, 100/100 |
| Public / skyline support | 37/100 | 0 | 0 | 0.336 s | v32 `1.615379`, 100/100 |
| Clean-v1 fold 0 / learned support | 0/105 | 0 | 0 | 0.626 s | v32 `1.790581`, 105/105 |
| Clean-v1 fold 0 / skyline support | 18/105 | 0 | 0 | 0.634 s | v32 `1.790581`, 105/105 |

All 87 clean learned-mode decoder failures and 87/105 clean skyline failures
were exact preplaced-anchor conflicts; the public counts were 62 and 63.
Skyline removed ordinary overlaps but cannot repair a wrongly predicted
horizontal structure that fixes multiple preplaced blocks to incompatible
relative coordinates. Candidate-only public skyline RF=1 was `9.729509`; its
37 feasible candidates still produced zero offline wins. The promotion verdict
is therefore **reject and keep v32**.

Evidence:

- `structured_linear_v1_public.json`
- `structured_linear_v1_clean_fold0.json`
- `../models/structured_linear_v1.json`

Reproduce training after all three source-exclusion manifests are present:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/train_structured_predictor.py \
  --source-index-cache results/work/order_source_index_v1.json \
  --layouts-per-source 4 --shard-layouts 32 \
  --output /tmp/structured_linear_v1.json
```

Reproduce the non-sealed evaluations:

```bash
.venv/bin/python scripts/evaluate_structured_predictor.py \
  --manifest results/folds/heavy_clean_v1.json --fold 0 \
  --model results/models/structured_linear_v1.json \
  --output /tmp/structured_clean_fold0.json

.venv/bin/python scripts/evaluate_structured_public.py \
  --model results/models/structured_linear_v1.json \
  --output /tmp/structured_public.json
```

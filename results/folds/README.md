# File-disjoint evaluation contract

The beta campaign uses two complementary heavy-layout panels. Neither is a
surrogate for the hidden leaderboard; together they expose different failure
regimes without source-file leakage.

| Manifest | Cases / sources | Selection | Intended use |
|---|---:|---|---|
| `heavy_clean_v1.json` | 525 / 522 | Input-visible MIB compatibility; five cases per n=100–120 per fold | Primary quality/decode development stratum |
| `heavy_raw_hash_v1.json` | 525 / 525 | One label-blind SHA-256-selected offset per source; no MIB filtering | Raw-data robustness and covariate-shift stratum |

The clean panel is deliberately not called representative: 522/525 cases are
offset 0 because the supplied training data's later configurations usually
contain input-incompatible MIB membership (official issue #12). The raw panel
uses all 112 offsets (only 6/525 are offset 0). Its absolute score can include
unavoidable corrupt-MIB penalties, so candidate decisions use paired deltas,
hard feasibility, HPWL/area, and excess-soft attribution rather than treating
its absolute cost as a clean target.

Pinned v31 baselines are 525/525 hard-feasible in both regimes: pooled RF=1
score **1.778134** clean and **1.848364** raw. The raw supplied golden layouts
violate MIB in 519/525 cases.

## Frozen roles

| Folds | Role | Permitted use |
|---|---|---|
| 0–2 | Development | Train/ablate ideas and estimate paired quality |
| 3 | Calibration | Freeze selector confidence/abstention thresholds |
| 4 | Sealed beta audit | Incumbent baseline only until the documented beta freeze |

The roles apply to both panels. Public validation is report-only. Repeatedly
using fold 4 to choose ideas would turn it into another public set.

## Manifest guarantees

- Schema 3 resolves by `source_file + file_offset`; `sample_index` is only a
  diagnostic.
- Every case binds the base input, derived fixed/preplaced optimizer targets,
  and the exact scoring layout/metrics with separate SHA-256 digests.
- The evaluator verifies the pinned FloorSet commit, ordered source inventory,
  fold hash, quota, metadata, all three identities, and exact solver-output
  schema before official scoring.
- Whole `.th` files belong to one fold, preventing nearby layouts from leaking
  across folds. The raw panel additionally uses at most one case per source.
- Comparison is fail-closed on manifests, case identities, evaluator/harness,
  runtime/oracle policy, and official checkout state, and hashes every input
  artifact.

The pinned-official primary-selector audit on clean folds 0–3 covers 420 cases
and 3,665 options: 412 oracle matches, zero false accepts, eight missed local
wins, and 0.000202 weighted regret. New candidate families must be re-audited
because they change the tradeoff distribution.

Manifest SHA-256:

- clean: `48ecda41bb642caa67d2e617ff9e467816a0392d6a68a0a91c38cf2e5f847895`
- raw: `9b4ff6a36e1945718411a83045f598228c2b301fdfa22340e33c297da9ac41ec`

Both bind FloorSet commit `aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa`
and source-inventory SHA-256
`c1984cc01d159dbd1a88a90f12340c0a3cc26998c2c498e2854092d3ad53d725`.

## Reproduction

```bash
.venv/bin/python scripts/build_holdout_folds.py
.venv/bin/python scripts/build_holdout_folds.py \
  --allow-mib-incompatible --case-selection hash_one_per_source \
  --output results/folds/heavy_raw_hash_v1.json

for regime in clean raw; do
  manifest="results/folds/heavy_${regime}_v1.json"
  [ "$regime" = raw ] && manifest=results/folds/heavy_raw_hash_v1.json
  for fold in 0 1 2 3 4; do
    .venv/bin/python scripts/evaluate_training_holdout.py \
      --fold-manifest "$manifest" --fold "$fold" \
      --output "results/folds/v31_${regime}_fold${fold}.json"
  done
done

.venv/bin/python scripts/compare_fold_results.py \
  --baseline results/folds/v31_clean_fold{0..4}.json \
  --candidate results/folds/v31_clean_fold{0..4}.json \
  --manifest results/folds/heavy_clean_v1.json \
  --output results/folds/v31_clean_summary.json
```

For a real candidate, compare folds 0–2 first, calibrate once on fold 3, and
leave fold 4 untouched until the documented freeze decision.

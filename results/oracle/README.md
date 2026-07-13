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

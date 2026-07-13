# Learned model artifacts

`order_ridge_v4.json` is the reproducible compact topology-guidance baseline.
It is a research artifact, not part of the submitted v32 solver.

## v4 contract

- 60 input-only, block-permutation-equivariant features from
  `contest_solution/learned_order.py`.
- MIB/cluster identifiers are equality keys only; raw numeric IDs never enter
  the model.
- Training excludes the union of both frozen heavy panels: 531 unique source
  `.th` files from `heavy_clean_v1.json` and `heavy_raw_hash_v1.json`.
- Remaining files are deterministically selected and split by whole source,
  stratified by block count.
- The artifact binds both manifests, exact selected source bytes, trainer and
  feature source, schema, message-step count, normalization, coefficient
  widths, and a canonical payload SHA-256. Inference rejects any drift.
- Inference uses only stdlib arithmetic; NumPy/PyTorch are training-time tools.

Training corpus: 332,592 blocks / 3,048 layouts. Validation corpus: 68,184
blocks / 624 layouts, with every n=100–120 present. Validation center MAE is
0.07938/0.08004 (x/y); pairwise inversion is 0.10109/0.08484. Prediction ties
count as half an inversion, so a constant predictor cannot appear perfect.

Artifact SHA-256:
`4058105b4fee368c8a7293ba74ff462e652dadee675f08b8dcc91a4a6786e19a`.

Reproduce byte-for-byte with:

```bash
.venv/bin/python scripts/train_order_model.py \
  --output results/models/order_ridge_v4.json
```

Most supplied nonzero-offset training layouts have input-incompatible MIB
membership. v4 is therefore the unmasked baseline, not the final model. The
next ablation must compare masked-all, compatible-only MIB features, and a
hybrid clean MIB head on decoded contest score. Coordinate accuracy alone is
never promotion evidence.

# Learned model artifacts

## Final-fidelity policy promotion

`order_v5b_final_fidelity_v2.json` is the promoted, provenance-complete policy.
It binds the raw evidence under
`results/research/order_v5b_final_fidelity_v2/`, the reducer and audit
harnesses, and the rejection-only public confirmation. The older
`order_v5b_final_fidelity_provisional_v2.json` is retained only as a historical
preregistration; it must not be supplied to the finalizer or relabeled as
complete evidence.

A promotable policy is produced in three separate, non-live steps:

1. `scripts/derive_redundant_slot_map.py` consumes exactly the frozen clean/raw
   fold 0–2 development matrix plus both fold-3 confirmation panels. Its
   schema-2 artifact binds the clean solver commit, all live component hashes,
   exact frozen case identities, and both audit harnesses. Confirmation is
   rejection-only.
2. `scripts/audit_public_slot_fidelity.py` consumes those exact artifact bytes
   and audits exactly one public case for every confirmed mapping without
   computing public golden cost. Its `config.slot_map_sha256` is the byte-level
   link to the derivation artifact.
3. `scripts/finalize_learned_policy.py` verifies both complete schemas,
   current harness bytes, clean solver snapshots with identical live component
   bindings, mapped sizes, width factors, cases, counts, and recomputed
   preservation verdicts. Non-live audit commits may differ and are recorded
   explicitly; the six deployable module hashes may not differ.
   It emits `final_rejection_only_public_audit_complete` only after constructing
   the final map by set subtraction. Public evidence can add abstentions; it
   cannot add a size or retune a width factor.

The final artifact records the SHA-256 of both input files and the finalizer,
plus `public_rejected_sizes`, the final `replacement_wf_by_size`, and the full
heavy-size complement in `abstain_sizes`. The generator refuses to overwrite
an existing output unless `--overwrite` is explicit. Generate into
`results/work/<campaign>/` first; replacing the tracked policy and changing
live solver bytes are separate promotion actions after review and evaluation.
The v2 policy was promoted without changing live solver bytes because its final
map exactly matched the preregistered inline map.

## Full-heavy v5 ablations

`order_ridge_v5_heavy.json` and `order_ridge_v5b_clean_raw.json` fix the v4
selection bug: block count is indexed and filtered before `--max-files`, so
all **1,159** eligible n=100–120 sources remaining after the v1 and sealed-v2
exclusions are used.  The whole-source stratified split contains 985 training
and 174 validation sources; no source crosses a partition or holdout.

Both artifacts use the same eight-layout-per-source budget: 7,880 training
layouts / 866,544 blocks and 1,392 validation layouts / 153,088 blocks.

- v5 is the raw-hash ablation. Only 66 training and 15 validation layouts are
  input-compatible with their MIB annotation; incompatible layouts mask every
  MIB feature channel. Validation inversion is 0.09492/0.07505 (x/y).
- v5b reserves one deterministically found input-compatible layout, then fills
  the budget with label-blind hash-selected raw offsets. It finds clean offsets
  for 974/985 training and 170/174 validation sources; the remaining 15 sources
  have no compatible offset among all 112 layouts. Validation inversion is
  0.09526/0.07635. Although aggregate coordinate accuracy is slightly worse
  than v5, v5b is the decoded research candidate because it materially reduces
  the clean-panel domain mismatch.

Both exclude the union of v1 and `heavy_sealed_v2.json` (**741 sources**) and
bind the exact source bytes, source-index payload, manifests, feature/trainer
implementations, layout-selection policy, and canonical model payload.

Artifact SHA-256:

- v5: `476c58e22e79aa48251d8d8f9e542140fb5839727e6e8d04d0552b73d5e76602`
- v5b: `09c936e726efc61c6f43058f7aaa21daf84a6a8a35924b8a9345bddb2edfb0ff`

Reproduce v5b byte-for-byte with:

```bash
.venv/bin/python scripts/train_order_model.py \
  --output /tmp/order_ridge_v5b_clean_raw.json
```

The deployed research module is generated deterministically from v5b:

```bash
.venv/bin/python scripts/export_order_model_module.py
```

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

# Script catalog

This directory contains the maintained evaluation and research tooling for the
FloorSet ICCAD 2026 entry. Run commands from the repository root unless a
command says otherwise. “Supported” means the tool participates in the current
workflow; it does not mean that every research output is suitable for release.

## Environment and safety

- Use `.venv/bin/python`, with the dependencies from the official contest
  requirements installed. [`setup_and_evaluate.sh`](setup_and_evaluate.sh) can
  create that environment, but it also updates `.venv`, copies solver sources
  into the official checkout, downloads validation data when needed, and runs a
  full evaluation.
- The official checkout is expected at
  `external/FloorSet`, including `external/FloorSet/iccad2026contest`, and must
  be at the commit bound by
  [`results/release_manifest.json`](../results/release_manifest.json).
- Official-evaluator validation needs the checkout and its validation data.
  The manifest-only release gate works in a clean clone once the tagged package
  asset is restored; only its optional optimizer-sync check needs the checkout.
  Holdout, training, retrieval, and fuzz workflows additionally need the
  extracted `external/FloorSet/floorset_lite` training shards.
- The packaged executable is AMD64 Linux. Run binary fuzz and wrapper parity on
  AMD64, or pass a configured x86/QEMU launcher explicitly on another
  architecture. Emulated startup time is not contest-runtime evidence.
- Most generators accept `--output` or `--out`. During experiments, write to a
  unique path under `results/work/<date-or-branch>/`, or use `/tmp` when the
  output is intentionally disposable. Do not overwrite the incumbent result,
  frozen fold manifests, model artifact, or release manifest until the
  corresponding promotion gate has passed.
- Use each supported Python tool's `--help` for its exact interface. The three
  deprecated tools at the end of this document are exceptions and should not be
  used as release gates.

These routine checks are read-only with respect to tracked source and result
artifacts:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_official_sources.py
.venv/bin/python scripts/audit_submission_package.py \
  --release-manifest results/release_manifest.json
.venv/bin/python scripts/check_public_release.py
.venv/bin/python scripts/audit_results.py \
  results/integrated_v32.json --expected-cases 100 --require-positions
.venv/bin/python scripts/analyze_results.py results/integrated_v32.json --top 20
.venv/bin/python scripts/compare_results.py \
  results/integrated_v31.json results/integrated_v32.json
.venv/bin/python scripts/check_position_parity.py \
  results/<source-run>.json results/<packaged-wrapper-run>.json
```

## Supported release and evaluation tools

| Tool | Role | Important notes |
|---|---|---|
| [`check_official_sources.py`](check_official_sources.py) | Verify the pinned FloorSet commit/tree/files, tracked wrapper and document extracts, and optional original Drive downloads without network access. | CI requires the pinned checkout; pass `--materials-dir` to recheck locally retained PDFs. |
| [`audit_submission_package.py`](audit_submission_package.py) | Reject unsafe archives, wrong-architecture or too-new ELF payloads, wrapper/source drift, real-torch leakage, and optionally smoke-run the binary. | Pass `--release-manifest` for a frozen asset and `--smoke` on AMD64. New builds require notices. |
| [`package_mode_tournament.py`](package_mode_tournament.py) | Report-only package construction and timing | Build `off`, `replacement`, `additive`, and `additive_first_pass` packages from one explicit committed tree without editing the source worktree, then benchmark the audited binaries through the exact organizer wrapper in a balanced Williams sequence. This does not change or select the production default. |
| [`setup_and_evaluate.sh`](setup_and_evaluate.sh) | Bootstrap the pinned official checkout, run tests, run the official 100-case evaluator, save positions, and audit the result. | Mutates `.venv` and the optimizer copy under `external/FloorSet`; set `OUTPUT=/tmp/opc-eda-evaluation.json` to avoid replacing a repository artifact. |
| [`preflight_official_solver.py`](preflight_official_solver.py) | Import and behavior-check the registry-copied solver before official evaluation. | `setup_and_evaluate.sh` runs it with real Torch; it rejects model-integrity drift, incomplete replacement coverage, unexpected abstentions, and a broken safe-MIB repair path. |
| [`solver_components.py`](solver_components.py) | Define the authoritative live solver-module set used by copy, synchronization, and holdout-provenance workflows. | Add every new live implementation module here, together with any generated deployment artifact it imports. |
| [`check_public_release.py`](check_public_release.py) | Fail-closed release gate for manifest hashes, source and package provenance, result integrity, score limits, public-safe text, and optional optimizer-copy parity. | The no-argument command consumes [`results/release_manifest.json`](../results/release_manifest.json). Run it after building the exact archive intended for release. |
| [`audit_results.py`](audit_results.py) | Validate evaluator JSON structure, summaries, finite metrics, case IDs, feasibility, and optionally every saved rectangle. | Run before comparing or publishing any full result. `--require-positions` is expected for a release artifact. |
| [`compare_results.py`](compare_results.py) | Compare a public full-result candidate with a baseline and fail on infeasibility, missing cases, or a non-improvement. | This is the RF=1 public-quality gate; it is not a holdout uncertainty test and does not infer the leaderboard runtime denominator. |
| [`check_position_parity.py`](check_position_parity.py) | Require exact saved rectangle and non-runtime quality parity between source and packaged-wrapper evaluator runs. | Defaults to a complete 100-case panel, audits both inputs, and compares coordinates as IEEE-754 binary64 values; both runs must save positions. |
| [`analyze_results.py`](analyze_results.py) | Rank score contributors, summarize block-count bands, attribute soft violations, and optionally write diagnostic sidecars. | Always pass the intended result explicitly; the historical default is not the incumbent. Writing enriched diagnostics requires the official contest directory. |
| [`fuzz_binary.py`](fuzz_binary.py) | Exercise the packaged executable end to end on training cases through the exact JSON protocol. | Preferred robustness gate for the shipped artifact. It requires training shards and a native AMD64 binary or an explicit compatible launcher. |

Package construction and wrapper details intentionally live beside the package
sources: see [`packaging/build_submission.sh`](../packaging/build_submission.sh)
and [`packaging/README_SUBMISSION.md`](../packaging/README_SUBMISSION.md). Building
the archive is a mutating operation; follow it with wrapper position parity,
`audit_results.py`, and `check_public_release.py` as described in
[`HANDOFF.md`](../HANDOFF.md).

### Four-mode package timing (report only)

The package tournament never edits `contest_solution/`. Its build command
requires an explicit Git revision containing the running orchestrator and the
mode-independent package self-test, exports that revision with `git archive`,
and performs all mode substitutions in disposable copies. The substitution
gate requires exactly one byte-for-byte default assignment and checks the
entire copied tree before each build; `replacement` is a byte-identical
control. Output and temporary workspace paths must be outside this repository.

Building all four pinned-container packages is intentionally expensive and is
not a routine test:

```bash
.venv/bin/python scripts/package_mode_tournament.py build \
  --commit "$(git rev-parse HEAD)" \
  --output-dir /tmp/opc-four-mode-packages
```

Every archive is audited with notices and `--smoke`, including the packaged
learned-model/replacement and safe-MIB module probes. The source fallback and
compiled executable also run the read-only `--report-default-mode` attestation;
their byte-exact JSON reports must agree with the package treatment. The
schema-2 build manifest binds the source commit/tree, exact patch, complete
live-registry and package-support inventories, every archived-source hash,
binary/package/wrapper descriptors, build and audit logs, and all harnesses.
Legacy schema-1 tournament manifests are deliberately rejected because they
do not contain those bindings. Before timing, every recorded commit, tree,
Git-archive, tooling, registry/support source, and patched optimizer hash is
re-derived from a clean checkout of the exact build commit; the running
orchestrator must also be byte-identical to that commit.

Timing requires explicit public case IDs so a full campaign cannot start by
accident. Each case first receives one balanced four-mode warmup pass whose
measurements are retained but discarded. One measured cycle is the complete
four-row Williams design—16 wrapper/binary executions per case—and the first
row rotates by case and cycle. Each run invokes the pinned official evaluator
on the package's verbatim organizer `op_wrapper.py` in an allowlisted,
thread-pinned environment; `MY_OPT_BIN`, `PYTHONPATH`, solver debug variables,
emulator variables, and Python bytecode writes are absent. The public timing
command requires the data root to be the exact Git-verified official checkout.
The timing gate rehashes and re-audits all packages first, rejects
infeasibility, non-positive block dimensions, wrong dataset block counts,
neutral-RF cost/summary drift, or nondeterministic repeated positions, and
writes raw official result files plus a hash-bound summary atomically:

```bash
.venv/bin/python scripts/package_mode_tournament.py time \
  --build-manifest /tmp/opc-four-mode-packages/build_manifest.json \
  --case-id 79 --case-id 81 --case-id 99 --cycles 1 \
  --output-dir /tmp/opc-four-mode-timing
```

Run timing only on the native AMD64 machine whose measurements are intended as
evidence. The attestation requires agreeing Python-platform and `uname`
architecture, recognized x86 CPU vendor/flags, and no ARM/QEMU contradiction;
ordinary native GitHub AMD64 virtualization is allowed. CPU identity, process
affinity, frequency governors, load averages, and pre/post drift are recorded.
After the final run, the tool rehashes every selected input and label, the
evaluator, organizer wrappers, packages, binaries, official-source checker,
source manifest, and build manifest. It also inventories and rehashes every
regular file in each extracted package, including the complete PyInstaller
`_internal` runtime tree, and rejects links or special files. It then reruns
official-source integrity.

The schema-3 timing report recomputes the exact feasible contest formula from each
mode/case's median wrapper runtime for hypothetical field medians of 0.25,
0.5, 1, 2, and 3 seconds, including the 9.999999 cap and `exp(n/12)` case
weights. It reports deltas against both `off` and `replacement`, per-scenario
winners, nondominated modes, and the all-scenario winner intersection. A
subset is explicitly named
`partial_panel_exp_n_over_12_weighted_cost_not_official_total_score`; only the
exact 100-case 21..120 panel is labeled an official total-score scenario. The
four modes remain report-only benchmark treatments; policy promotion is
governed by the holdout selector and its closed calibration gate.

## Supported fold, model, and research tools

The frozen panel contract and exact reproduction commands are in
[`results/folds/README.md`](../results/folds/README.md). Do not regenerate a
frozen manifest merely to run an experiment: evaluate the existing manifest
and write the candidate result to a new path.

| Tool | Category | Role |
|---|---|---|
| [`evaluate_public_mode.py`](evaluate_public_mode.py) | Public ablation | Run one explicit learned-policy mode through the pinned official public evaluator without changing live solver bytes. It verifies official-source provenance, checks solver hashes before/after, records the applied in-memory configuration, and refuses to overwrite output by default; repeat `--test-id` for bounded panels. |
| [`derive_redundant_slot_map.py`](derive_redundant_slot_map.py) | Learned-policy derivation | Derive candidate paid-slot replacements from the frozen clean/raw fold 0–2 matrix, then use both fold-3 panels only to reject mappings. The schema-2 output binds exact case identities, harnesses, solver components, and the clean solver commit. |
| [`audit_public_slot_fidelity.py`](audit_public_slot_fidelity.py) | Public structural audit | Rerun every still-mapped public heavy case with its preregistered paid slot removed. It computes no public golden cost and can report only preservation or rejection; its output hashes the exact schema-2 slot-map bytes. |
| [`finalize_learned_policy.py`](finalize_learned_policy.py) | Learned-policy promotion | Fail-closed composition gate for a schema-2 derivation and its exact public audit. It verifies harnesses, clean snapshots with identical live-module hashes, one matching case per mapped size, and every recomputed verdict; it can only subtract rejected sizes. It writes a provenance-bound final artifact atomically and refuses overwrite by default. |
| [`build_holdout_folds.py`](build_holdout_folds.py) | Fold construction | Build deterministic, source-file-disjoint heavy-layout manifests with input-visible MIB compatibility or label-blind hash selection. Manifest generation is an intentional campaign-level action, not routine evaluation. |
| [`evaluate_training_holdout.py`](evaluate_training_holdout.py) | Fold evaluation | Score the submitted solver on a manifest fold with the pinned official evaluator, pristine inputs, strict output validation, and complete provenance. RF=1 is deliberately neutral; measured wall time is recorded separately. |
| [`compare_fold_results.py`](compare_fold_results.py) | Promotion statistics | Compare matched fold artifacts fail-closed and estimate paired uncertainty by source-cluster bootstrap. Use development folds first, calibration once, and the sealed fold only at the documented freeze. |
| [`audit_candidate_selector.py`](audit_candidate_selector.py) | Selector research | Measure deployable selector regret against an offline golden-baseline oracle without exposing that oracle to solver inference. Re-run it whenever candidate families or selector behavior change. |
| [`train_order_model.py`](train_order_model.py) | Model training | Train the deterministic compact order model while excluding the union of frozen source-file holdouts. The current artifact contract is documented in [`results/models/README.md`](../results/models/README.md). |
| [`dissect_eval.py`](dissect_eval.py) | Engine diagnostics | Evaluate the dissection engine on selected public cases with official metric semantics and compare it with the historical locked baseline. Useful for fast engine iteration, but not a full release gate. |
| [`mine_golden.py`](mine_golden.py) | Structural research | Recompute public-validation golden geometry and soft-constraint statistics with official semantics. Its output informs hypotheses; it is not hidden-set evidence. |
| [`match_validation_in_training.py`](match_validation_in_training.py) | Dataset research | Scan training shards for validation area-signature matches to test the retrieval hypothesis. This can be I/O intensive and its signature is a screening heuristic, not proof of layout identity. |
| [`sp_labels.py`](sp_labels.py) | Research library | Provide sequence-pair label extraction and packing helpers. It is imported and tested as a module; it has no command-line workflow. |

Safe smoke and comparison examples use existing frozen inputs and disposable
outputs:

```bash
# Engine-only public smoke; prints metrics and writes nothing.
.venv/bin/python scripts/dissect_eval.py --cases 95,97,99

# Bounded official public ablation; writes a new provenance-bound artifact.
.venv/bin/python scripts/evaluate_public_mode.py \
  --learned-mode replacement --test-id 99 \
  --output results/work/<campaign>/public-replacement-case99.json

# Finalize only after deriving a full schema-2 artifact and auditing those
# exact bytes. These placeholders denote the frozen six-plus-two audit panel.
.venv/bin/python scripts/derive_redundant_slot_map.py \
  <clean-fold0> <clean-fold1> <clean-fold2> \
  <raw-fold0> <raw-fold1> <raw-fold2> \
  --confirmation-audits <clean-fold3> <raw-fold3> \
  --output results/work/<campaign>/derived-slot-map.json
.venv/bin/python scripts/audit_public_slot_fidelity.py \
  --slot-map results/work/<campaign>/derived-slot-map.json \
  --output results/work/<campaign>/public-slot-audit.json
.venv/bin/python scripts/finalize_learned_policy.py \
  --slot-map results/work/<campaign>/derived-slot-map.json \
  --public-audit results/work/<campaign>/public-slot-audit.json \
  --output results/work/<campaign>/final-policy.json

# One bounded selector-audit smoke on the existing clean panel.
.venv/bin/python scripts/audit_candidate_selector.py \
  --fold-manifest results/folds/heavy_clean_v1.json \
  --fold 0 --max-cases 5 --output /tmp/opc-eda-selector-audit.json

# Reproduce a model without replacing the tracked artifact.
.venv/bin/python scripts/train_order_model.py \
  --output /tmp/opc-eda-order-model.json

# Compare the already-recorded five-fold clean incumbent and v32 candidate.
.venv/bin/python scripts/compare_fold_results.py \
  --baseline results/folds/v31_clean_fold{0..4}.json \
  --candidate results/folds/v32_reuse_p1_clean_fold{0..4}.json \
  --manifest results/folds/heavy_clean_v1.json
```

## Deprecated legacy tools

The files below are retained only for historical reproducibility. Do not use
their output to promote a solver, update the release manifest, or make a
submission decision.

### `benchmark_ml_score.py`

[`benchmark_ml_score.py`](benchmark_ml_score.py) is deprecated.

- It hard-codes the obsolete `/workspace/eda/FloorSet` checkout, so it cannot
  even import the evaluator in the repository-standard layout.
- Its `--model` argument is only recorded in JSON. The `ml_available` flag is
  initialized to false and never enabled, so the advertised ML comparison
  branch never runs.
- It relies on private evaluator and optimizer hooks and has no source-disjoint
  holdout or provenance contract.

Use [`train_order_model.py`](train_order_model.py) to train a model,
[`evaluate_training_holdout.py`](evaluate_training_holdout.py) plus
[`compare_fold_results.py`](compare_fold_results.py) to measure decoded held-out
quality, and [`audit_candidate_selector.py`](audit_candidate_selector.py) to
measure deployment-time selector regret.

### `n9_robustness.py`

[`n9_robustness.py`](n9_robustness.py) is a deprecated v9-era in-process
harness.

- It calls `MyOptimizer.solve()` directly and uses a hand-written feasibility
  checker, so it does not exercise the packaged executable, wrapper JSON,
  stubs, startup, or the complete official evaluator path.
- `--timeout` is accepted and reported as an intended limit but is never
  enforced.
- It always writes `results/n9_robustness.json`, making accidental replacement
  of the historical artifact easy.

Use [`fuzz_binary.py`](fuzz_binary.py) for end-to-end package robustness. Use
[`evaluate_training_holdout.py`](evaluate_training_holdout.py) when official
metric semantics, fold identity, and provenance are required.

### `score_real.py`

[`score_real.py`](score_real.py) is a deprecated exploratory runtime estimator.

- It applies one assumed median to every test case, while the official
  denominator is the cross-submission median for each individual test case and
  is unavailable locally. Its “self” mode uses the candidate's own per-block
  median and is not an official-score estimate.
- It compares independently recorded timings without paired-run drift control,
  silently defaults missing metric fields, and does not apply the official
  feasible-cost cap of `10 - 1e-6`.
- It has no `argparse` interface (`--help` is treated as a filename) and accepts
  only one displayed baseline.

There is no local tool that can recover the unknown official runtime
denominators. Use the official evaluator's neutral RF=1 result with
[`audit_results.py`](audit_results.py) and [`compare_results.py`](compare_results.py)
for quality, then use immediate paired control/candidate runs and per-case
median timings for runtime scenarios. The current formula and keep/revert gate
are documented in [`HANDOFF.md`](../HANDOFF.md), and
[`results/v32_runtime_summary.json`](../results/v32_runtime_summary.json) is the
current machine-readable evidence format.

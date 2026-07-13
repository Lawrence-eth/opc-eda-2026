# Results and evidence catalog

This directory contains curated evidence for the optimizer, not a general
experiment dump. The release source of truth is `release_manifest.json`; it
binds the solver commit and source hashes, public result, official FloorSet
revision, and submission archive digest. Lower contest score is better.

Large pre-cleanup scratch outputs and abandoned-worktree patches are indexed
in [`ARCHIVE_INDEX.md`](ARCHIVE_INDEX.md) and stored as a checksum-bound
private release asset rather than committed here.

## Current release: v32

The frozen pre-beta release is tag `v32-prebeta-20260711` (repository commit
`21973d62759442751715eb827cf05769193a7ba9`). Its manifest binds solver commit
`d8f2c19bd6d4c4d27f2ad1f90c923c204e7f1e66`.

| Artifact | Purpose |
|---|---|
| `release_manifest.json` | Machine-checkable v32 release contract and hashes |
| `integrated_v32.json` | Canonical in-process public evaluation: 1.6153787745353914, 100/100 feasible |
| `wrapper_v32_amd64.json` | Official wrapper plus packaged AMD64 executable parity run; the score matches the canonical result |
| `folds/v32_reuse_p1_{clean,raw}_all5_summary.json` | Pooled five-fold promotion evidence and source-cluster bootstrap |
| `v32_runtime_summary.json` | Paired A-B-B-A timing analysis and runtime-adjusted scores |
| `v32_reuse_main_runtime_r{1,2}.json.gz` | Hash-bound compressed candidate timing inputs used by the timing summary |

The wrapper run was performed through an AMD64 container on a non-AMD64 host.
Its positions and quality score are parity evidence; its container/QEMU
wall-clock timings are not performance measurements. Use the paired native
timing artifacts for runtime decisions.

### Recover and verify the exact package

A fresh clone does not contain the generated archive. Download the asset
attached to the exact tag, then check both its SHA-256 and the complete release
contract:

```bash
git clone https://github.com/Lawrence-eth/opc-eda-2026.git
cd opc-eda-2026
git checkout --detach v32-prebeta-20260711

mkdir -p submission
gh release download v32-prebeta-20260711 \
  --repo Lawrence-eth/opc-eda-2026 \
  --pattern iccad2026_submission.tar.gz \
  --dir submission

printf '%s  %s\n' \
  72d8fc5b6c4831a6af3547bacc16f19c800f1991b413500efbe467db8aec72c3 \
  submission/iccad2026_submission.tar.gz | sha256sum --check -
python3 scripts/check_public_release.py
```

The tagged release page is
<https://github.com/Lawrence-eth/opc-eda-2026/releases/tag/v32-prebeta-20260711>.
Do not substitute an archive from a different tag, even if its filename is the
same.

## Rollback and regression anchors

- `integrated_v31.json` is the canonical rollback result: 1.6166380547746888,
  100/100 feasible. `wrapper_v31.json` is its package/wrapper parity evidence.
- `folds/v31_{clean,raw}_fold*.json` and
  `folds/v31_{clean,raw}_summary.json` freeze the baseline used for v32's heavy
  panel comparisons.
- `v31_runtime_control_r{5,6}.json.gz` are the compressed control measurements
  paired with the v32 timing runs.
- `v9_locked.json` is the 2.718225068193903 sprint-5 baseline and remains a
  solver fallback/regression anchor. It predates the later `integrated_v*`
  milestone namespace and must not be inferred from `integrated_v9.json`.

These files are reference points. Re-running an older source tree must write a
new scratch artifact; it must not overwrite a frozen anchor.

## Source-disjoint clean/raw evidence

`folds/README.md` defines the evaluation contract, leakage controls, fold
roles, and reproduction commands. The main artifact families are:

- `folds/heavy_clean_v1.json`: 525-case input-compatible MIB panel across five
  file-disjoint folds.
- `folds/heavy_raw_hash_v1.json`: 525-case label-blind, one-offset-per-source
  robustness panel.
- `folds/v32_reuse_p1_{clean,raw}_fold*.json`: per-fold candidate results.
- `folds/v32_reuse_p1_{clean,raw}_all5_summary.json`: final pooled comparisons.
  v32 improves v31 from 1.778134 to 1.767565 on clean and from 1.848364 to
  1.834588 on raw, with 1,050/1,050 candidate cases feasible.
- `folds/v32_reuse_p1_{clean,raw}_fold4_summary.json`: separately recorded
  sealed-fold audit.

Per-fold files are retained because the pooled summaries bind their paths,
sizes, and SHA-256 digests. Never edit a panel manifest or result in place;
version the contract or candidate name instead.

## Golden, diagnostics, and retrieval

These artifacts explain the problem and support error analysis. They are not
independent promotion panels.

| Artifact | Contents and intended use |
|---|---|
| `golden_scored.json` | Official-score decomposition of the 100 supplied golden validation layouts |
| `golden_structure.json` | Mined geometry, support, grouping, boundary, MIB, and utilization structure; includes an exact check against `golden_scored.json` |
| `enriched_diagnostics.json` | Per-case v31 residual attribution and soft-constraint ledger |
| `retrieval_scan.json` | Exhaustive 1,008,000-sample signature scan showing no validation layout hit in training |
| `training_holdout*_mib_clean.json` | Earlier MIB-compatible training holdout studies |
| `n9_robustness.json` | Historical small-layout robustness evidence |

Public validation and golden-derived artifacts are report/diagnostic evidence.
Candidate promotion should be based on the file-disjoint folds, feasibility,
paired deltas, and runtime-adjusted gates documented elsewhere in the repo.

## Learned-model provenance

`models/order_ridge_v4.json` is a compact, reproducible topology-guidance
research model. `models/README.md` records its feature contract, whole-source
split, panel exclusions, training statistics, reproduction command, and
artifact SHA-256. The artifact additionally binds its own training inputs and
code provenance. It is not deployed by the v32 submission, so it is evidence
for future ablations rather than part of the current release package.

## Historical result families

The retained history is organized by naming convention instead of one index
entry per experiment:

- `integrated_v*.json` records promoted public-set milestones. v32 is current;
  v31 is rollback; earlier versions are trajectory and regression history.
- `wrapper_v*.json` records packaged-launcher parity checkpoints for promoted
  versions. Only a matching release manifest/tag makes a wrapper result a
  release artifact.
- `tuned*_official_full.json` preserves the pre-integrated tuning trajectory.
- selector audits and baseline/candidate files under `folds/` preserve the
  statistical decision trail for the current evaluation contract.

Historical files can explain a decision, but they must not be mistaken for the
current incumbent. Start at `release_manifest.json`, not the largest version
number found by a glob.

## Retention and scratch policy

Use these classes when adding, reviewing, or eventually archiving results:

| Class | Meaning | Policy |
|---|---|---|
| R0 — release | Manifest-bound result, package parity, and release summaries | Immutable; keep in the tagged commit and keep the package on the tagged release |
| R1 — rollback | Last verified incumbent and locked solver baselines | Keep until at least one later release survives beta/hidden feedback |
| R2 — reproducibility | Fold manifests/results, paired timing inputs, golden/retrieval evidence, model provenance | Keep while cited by active code, summaries, or decision records |
| R3 — history | Superseded integrated, wrapper, and tuning milestones | Keep representative milestones; archive only with an index and preserved provenance |
| S — scratch | Probes, traces, temporary reruns, plots, and failed variants | Write under `results/work/`; never commit by default |

New experiments should write to `results/work/<date-or-branch>/`, not the
`results/` root. That directory is ignored for every file type. Promote an
artifact only after it has a stable name, passes the relevant audit, and is
cited by a summary or decision record; because result formats are ignored by
default, promotion is an explicit `git add -f <path>` action. Do not use
`*_tmp` as a permanent evidence class, and do not commit raw duplicates when a
deterministic compressed input plus digest is sufficient.

Before deleting or moving any tracked result, search all documentation,
manifests, summaries, and scripts for references. Release-bound artifacts and
anything transitively hash-bound by them are never cleanup candidates.

# Learned-policy tournament v1

This directory is the portable evidence bundle for choosing the submitted
learned-order policy. The frozen outcome is **`replacement`**; the other three
modes remain report-only benchmark treatments. All selection scores use the
official cost at neutral runtime factor `RF=1`. Packaged runtime is evaluated
separately and cannot silently change this policy.

## Evidence status and chronology

- Solver snapshot: `84cd1e0fb95b7441555064783494a54119b79e71`
- Official FloorSet snapshot: `aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa`
- Clean manifest SHA-256:
  `48ecda41bb642caa67d2e617ff9e467816a0392d6a68a0a91c38cf2e5f847895`
- Raw manifest SHA-256:
  `9b4ff6a36e1945718411a83045f598228c2b301fdfa22340e33c297da9ac41ec`
- Development comparisons completed at 2026-07-14 02:37 UTC.
- The first fail-closed development ledger was written at 02:41 UTC.
- Selector commit `755e0ae` froze the exact policy and thresholds before the
  scored fold-3 runs began at 02:44 UTC.
- Fold-3 confirmation and the first final ledger completed by 02:51 UTC.
- Selector hardening commit `a176b59` added partition, training-exclusion,
  portability, infeasible-candidate, and replay-provenance checks without
  changing any threshold or the selected mode.
- Sealed-protocol commit `df61ab2` froze the fold-4 gates and one-finalist-only
  composition path before any fold-4 result was evaluated.

The chronology does **not** prove a repository preregistration before the
development results existed. The `0.002` worst-suite rule is therefore a
deliberately conservative post-hoc tail-risk policy, frozen before scored
confirmation, not a statistically preregistered cutoff. In addition, the
replacement slot map was derived on folds 0--2 and had already received a
label-blind structural rejection audit on fold 3. Fold 3 is confirmation on a
structurally exposed panel, not wholly unseen validation. No policy or gate may
change after this record. Fold 4 remains sealed for one final, genuinely
untouched `off` versus `replacement` check.

## Decision

Development uses 315 clean and 315 raw cases per mode across folds 0--2. Every
artifact is fully feasible.

| Mode | Clean score | Clean wins/losses | Raw score | Raw wins/losses | Decision |
|---|---:|---:|---:|---:|---|
| `off` | 1.749181734 | -- | 1.819204655 | -- | Baseline |
| `replacement` | 1.743956314 | 23 / 0 | 1.802014499 | 47 / 0 | Sole policy passer |
| `additive` | 1.735325266 | 53 / 0 | 1.791480502 | 74 / 3 | Tail-risk rejection |
| `additive_first_pass` | 1.735556373 | 47 / 0 | 1.798647827 | 62 / 3 | Tail-risk rejection |

Both additive modes passed every other gate. They were rejected only because
their raw one-case-per-block-count worst sampled deltas were `+0.003252362`
and `+0.002830395`, respectively, above the frozen `+0.002` policy limit.
Replacement had zero development losses on both panels.

Fold-3 confirmation evaluated only the development finalist:

| Panel | Cases feasible | Off score | Replacement score | Delta | Wins/losses |
|---|---:|---:|---:|---:|---:|
| Clean | 210 / 210 | 1.803334515 | 1.796691795 | -0.006642721 | 15 / 1 |
| Raw | 210 / 210 | 1.845309258 | 1.836188450 | -0.009120808 | 12 / 0 |

The one clean regression stayed within every frozen tail gate: worst weighted
contribution `0.000055638`, regression CVaR5 `0.000009273`, and worst sampled
21-size-suite delta `0.000278191`.

The report-only 100-case public source panel was also 100/100 feasible for all
four modes:

| Mode | Official RF=1 score | Delta vs off | Wins/losses vs off |
|---|---:|---:|---:|
| `off` | 1.615181467 | -- | -- |
| `replacement` | 1.611094635 | -0.004086832 | 2 / 0 |
| `additive` | 1.610652242 | -0.004529225 | 3 / 0 |
| `additive_first_pass` | 1.611485708 | -0.003695759 | 3 / 0 |

Public results are report-only and were not used to rewrite the frozen policy.
Their absolute paths record the execution host; selection artifacts and all
paths they dereference are repository-relative.

## Integrity contract

`scripts/select_policy_tournament.py` does not trust comparison summaries. It
strictly decodes and rehashes all raw artifacts, reconstructs official RF=1
costs and summaries, replays every 30,000-sample source-cluster bootstrap and
pseudo-suite comparison, then applies the frozen gates. Schema v2 additionally:

- proves source-file disjointness across all folds in each panel;
- requires every overlapping clean/raw source to have the same fold ID;
- binds the deployed v5b model to the exact three-manifest, 741-source training
  exclusion union;
- records the committed clean selector tree and canonical replay environment;
- records an infeasible challenger as failed without blocking safe modes; and
- permits only `1e-15` absolute last-bit float drift during replay.

The final selector hashes are inserted only after replay from the clean,
committed evidence base:

- `development_selector.json`:
  `1f53d0ae17c55efdb6ddb9fee8310411a1dfc76707a5c3d5fc8eaefe754dc290`
- `final_selector.json`:
  `6f4dd39cf16448e712ca1335f1ad4e02f33b8d68bd60a156fb8b756a0e509259`

## Directory layout

- `development/holdouts/`: 24 raw artifacts (`2 panels x 4 modes x 3 folds`).
- `development/comparisons/`: six repository-relative comparisons.
- `calibration/holdouts/`: fold-3 `off` and `replacement` artifacts.
- `calibration/comparisons/`: two fold-3 comparisons.
- `public/source/`: four raw, report-only full public source evaluations.
- `development_selector.json`: pre-confirmation selector ledger.
- `final_selector.json`: fold-3 composition ledger and frozen policy.

The raw evaluator timings are diagnostic only. Authoritative runtime evidence
must come from audited packaged binaries on native AMD64 through the organizer's
exact `op_wrapper.py`.

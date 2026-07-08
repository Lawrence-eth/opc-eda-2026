# Agent onboarding — ICCAD 2026 Problem C (FloorSet Challenge)

**The only goal is to WIN this contest.** Time and effort are not constraints
(operator directive, 2026-07-07). A verified, submittable entry always exists —
never regress it; improve on top of it behind gates.

## Read in this order

1. `HANDOFF.md` — **complete handoff**: current verified state, architecture,
   workflows, ranked open leads. Start here.
2. `docs/CAMPAIGN_GOLDEN.md` — the active campaign: evidence, milestones (G1…),
   and measured gates.
3. `SUBMISSION_PLAN.md` — the submission mechanics (organizer format, packaging,
   verification matrix, rebuild gate). The packaged entry is the safe floor.
4. `MASTER_PLAYBOOK.md` — the strategy history: 8 quality approaches tried and
   why each failed. **Its dead-end list (Part II.6) is binding** — do not retry
   those. Its "v9 is final" conclusion is superseded by the campaign doc.
5. `docs/SUMMARY.md` — presentation-level overview.

## Hard rules (from the operator + verified evidence)

- **Never regress HEAD.** The committed solver (shelf + dissection portfolio)
  is the floor: **1.8074 official, 100/100 feasible**, deterministic. New ideas
  integrate behind the per-case feasibility-gated selection in `solve()`; keep
  a change only if it wins runtime-adjusted at median ∈ {1,2,3}s
  (`HANDOFF.md` §5.3).
- **Feasibility is sacred.** One infeasible case = cost 10 ≈ catastrophic.
  Hard constraints: no overlaps (>1e-6), soft-block area within ±1% (symmetric),
  fixed dims exact (1e-4), preplaced (x,y,w,h) exact (1e-4). Cluster abutment
  requires EXACT shared coordinates (evaluator uses shapely union, no epsilon).
- **Judge on runtime-adjusted score** (`scripts/score_real.py`), not the local
  RF=1 number. The runtime floor (≤0.305× field median → hard 0.7×) is banked
  by the current package; a slower-but-better engine must beat it at
  median ∈ {1,2,3}s.
- **Every experiment gets a hypothesis line in `PLAN_EXECUTION_LOG.md` before
  running, and a verdict after.** Failures are logged, not hidden.
- Before ANY resubmission: run the rebuild gate in `SUBMISSION_PLAN.md` §7
  (packaged binary must reproduce the committed result bit-for-bit and pass the
  fuzz).

## Environment (this VM)

- Fresh clone bootstrap: `python3 -m venv .venv && .venv/bin/pip install torch
  --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install
  numpy shapely matplotlib tqdm requests pytest pyinstaller`
- Official framework: `external/FloorSet/` (clone of IntelLabs/FloorSet;
  gitignored). Validation data auto-downloads (~15MB). Training data
  (1M samples, ~6.6GB tar): `LiteTensorData_v2.tar.gz` extracted to
  `external/FloorSet/floorset_lite/` — already present on this VM.
- Evaluate: `cd external/FloorSet/iccad2026contest && PYTHONPATH=..
  ../../../.venv/bin/python iccad2026_evaluate.py --evaluate my_optimizer.py`
  (copy the optimizer in first; see SUBMISSION_PLAN.md §7 for the packaged path).

## Map

| Path | What |
|---|---|
| `contest_solution/my_optimizer.py` | THE solver (shelf + dissection portfolio + selector) |
| `contest_solution/dissect.py` | exact-area dissection engine (the campaign's core) |
| `packaging/` | submission package sources (torch-free executable; see SUBMISSION_PLAN.md) |
| `scripts/` | evaluation/analysis/audit/fuzz tooling (each has a docstring) |
| `results/integrated_v10.json` | the CURRENT verified result (1.8074) |
| `results/v9_locked.json` | pre-campaign locked result (2.7182) |
| `results/golden_scored.json` | golden layouts scored by the official evaluator (per case) |
| `results/retrieval_scan.json` | proof that validation ∉ training (1M scanned, 0 hits) |
| `docs/extracted/` | contest problem statement + Q&A text |
| `docs/archive/` | superseded plans (history only) |

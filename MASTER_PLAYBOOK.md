# FloorSet ICCAD-2026 — MASTER PLAYBOOK (self-navigating, multi-week)

**Author:** planning agent (Opus 4.8). **Supersedes** `NEXT_PLAN.md`, `SPRINT5_PLAN.md`, `SPRINT6_PLAN.md` (kept for history).
**Audience:** executing agent running 24/7 with unlimited compute/tokens. **This document is designed so you do NOT need a new plan every hour** — it encodes the decision logic. Navigate it yourself via Part III. Only stop for a re-plan when an **Escalation Trigger (Part VI)** fires.

---
---

# ⏰ OVERNIGHT AUTONOMOUS RUN — COMPLETE (round 1 done 2026-05-30; reusable protocol below)

> **Round-1 outcome:** ran the Q0→Q11 queue; built the ENGINE infra (correct, fast); score essentially unchanged (~2.718) because **local repair-based search cannot escape the shelf local optimum** (Part 0 UPDATE 3). Disciplined, gated, no busywork — exactly right. **Conclusion = global plateau (E4): all light/medium methods are now exhausted. The committed next move is IV.P2 (topological SA), upgraded below into a de-risked milestone build.** To re-arm for a future overnight run, set this header back to ACTIVE and follow the rules/queue — but the QUEUE is now the IV.P2 milestones M1→M6, not Q0→Q11.

**Keep making real progress; don't halt prematurely — but the goal is PROGRESS, not motion.** The operator is away ~7 hours. Work the ranked queue below top-to-bottom; when you finish one valuable item, start the next. Do NOT stop to "await instructions" while positive-EV work remains. **Equally: do NOT manufacture busywork** — no trivial parameter nudges, no re-testing dead-ends, no experiments without a hypothesis, just to look busy.

### Overnight rules (override normal behavior for this window)
0. **VALUE GUARD (most important).** Every experiment must have a **one-line hypothesis logged BEFORE running** it ("expect Δscore from X because Y"). If you cannot state a plausible mechanism by which it improves the runtime-adjusted score (or is necessary groundwork/diagnostic for something that will), **do not run it.** Forbidden as busywork: re-running anything in II.6 dead-ends, tweaking greedy packers (Part 0 UPDATE 2), trivial constant nudges, or re-confirming already-known results. **Quality of work > quantity of commits.**
1. **No premature halting / no asking — but graceful pause beats busywork.** If you hit a normal **Escalation Trigger (Part VI)**, do NOT stop: log it under `## OVERNIGHT BLOCKERS` and move to the next queue item / Standing Backlog (Part V). **Only if you genuinely exhaust ALL positive-EV work** (entire queue + backlog + P2 groundwork all done or blocked) is it correct to **stop and write a clear `## OVERNIGHT — POSITIVE-EV WORK EXHAUSTED` note** rather than invent meaningless jobs. That graceful pause is acceptable; churning is not.
2. **Always-committed, monotone progress.** Every change is **best-of gated against `sprint5_v9`** via `_true_contest_cost`, so HEAD can never regress. After every experiment: commit (code if kept, or revert code + commit the log entry if not). The repo must be feasible & committed at all times.
3. **Keep + revert rule (no deliberation):** keep a change iff runtime-adjusted total (`score_real.py`, median 1–3s) strictly improves AND feasibility = 100/100. Else `git checkout -- contest_solution/my_optimizer.py` and move on. **Do NOT revert a change merely for being slower than 0.18s** — slower-but-at-floor (≤~0.5s) with better quality is the goal (I.3).
4. **Time-box dead avenues:** if a single sub-step shows no path to improvement after ~45 min, log it and advance. Do not loop on greedy packers (Part 0 UPDATE 2).
5. **Running report:** maintain a `## OVERNIGHT PROGRESS` section at the TOP of `PLAN_EXECUTION_LOG.md`, one line per step: `<time> <step> — <result> — kept/reverted`. This is the operator's morning summary.
6. **Never leave cores idle:** the persistent pool / parallel chains should be running during long evals; if waiting, start the next independent step.
7. **Re-ground each cycle (anti-drift — protects quality over a long session).** At the START of every new experiment, re-read **Part 0**, **II.6 dead-ends**, and the **`## OVERNIGHT PROGRESS`** log before acting, so context compaction can't make you repeat a dead-end or lose a nuance. Correctness before speed (verify a move is feasible & monotone before optimizing it). If two consecutive experiments both regress or you notice you're unsure why HEAD is what it is, **pause and re-read the log** rather than pressing on — a confused cycle is worse than a missed one.
8. **Quality-first guarantee:** HEAD is best-of gated, so it can only improve or stay equal — a bad experiment costs compute, never the result. Prefer one careful, hypothesis-driven, correctly-validated experiment over three rushed ones.

### Overnight ORDERED QUEUE (do top-to-bottom; each item is atomic & committable)
> Primary thrust = build **IV.ENGINE** incrementally (each move added & gated separately, so progress is safe & monotone). Then tune, then fallbacks. This is >7h of work; you will not run out.

- **Q0 (≤45 min, hard cap): shape lever validity check.** In `_skyline_pack` (~3592) replace the naive `score = th` with **fit-to-gap** (pick the aspect whose width best matches the current skyline-valley width / target row width; fall back to golden-prior aspect ≈1.45). Re-measure standalone util on cases 99/97/95. Log the numbers. **Regardless of result, advance to Q1** (the same fit-to-gap logic is reused by the ENGINE reshape move). Do not integrate the packer.
- **Q1: ENGINE Step 1 — verify correctness baseline.** Confirm `_correctness_first_polish` (~3654) is monotone + feasible on 99/97/95 (full-recompute greedy). Fix if not. Commit.
- **Q2: ENGINE Step 2 — incremental cost** + `assert incremental≈full` debug every N moves; benchmark moves/s (target ≥10⁴/s @ n=120). Commit.
- **Q3: ENGINE Step 3a — relocate-to-centroid move** (HPWL lever, ripple-repair overlaps). Full eval + `score_real.py`. Keep/revert per rule. Commit + log.
- **Q4: ENGINE Step 3b — row/slab inward compaction move** (AREA lever; re-touch boundary blocks after). Full eval. Keep/revert. Commit + log.
- **Q5: ENGINE Step 3c — soft-block reshape-to-fill move** (SHAPE lever, ±1% area, fit-to-gap from Q0, MIB group together; applied to feasible layout so it can't break feasibility). Full eval. Keep/revert. Commit + log.
- **Q6: ENGINE Step 3d — unequal swap + local re-legalize, and cluster rigid-move-to-centroid.** Full eval. Keep/revert. Commit + log.
- **Q7: ENGINE Step 4 — SA tail** (exact incremental cost, true `exp(2·vrel)`, geometric cooling) to escape minima after greedy descent. Tune temp/cooling on cases 99/95. Full eval. Keep/revert. Commit + log.
- **Q8: ENGINE Step 5 — K parallel chains** (different seeds/move-weights) on the persistent pool within the free budget (~0.3–0.5s); keep best by `_true_contest_cost`. Full eval. Commit + log.
- **Q9: budget/chain sweep** — find K and time-budget giving best runtime-adjusted at median∈{1,2,3}s; commit best as `s6_engine`.
- **Q10: move-weight tuning via held-out folds** (split the 100; tune on train folds, select on held-out — never the full 100). Commit best generalizing weights.
- **Q11+ (Standing Backlog, Part V):** boundary-ring constructor; connectivity-aware packing order with cluster super-blocks; per-net hub weighting; MIB shared-shape selection; extended golden mining (cluster topology / ring structure). Each gated & committed.
- **If you reach here with time left:** begin **IV.P2 groundwork** (numba skeleton for a B*-tree/sequence-pair packer + obstacle-handling prototype on case 99, proving exact feasibility) — research track, do NOT wire into `solve()` yet. Keep going; do not stop.

### Morning handoff (when operator returns)
Leave: HEAD committed & 100/100; the `## OVERNIGHT PROGRESS` log filled; the best `s6_*` result tagged; `score_real.py` table for the new best vs sprint5_v9 at all medians; and any `## OVERNIGHT BLOCKERS` noted for re-planning.

---
---

> How to use this file: Read Part I (invariants) and Part II (facts) once. Then operate the **decision tree in Part III** — it always tells you the single next action based on your latest result. Part IV gives the deep implementation spec for whatever track Part III sends you to. Part V is the backlog for idle compute. Part VI says when to come back to the planner. Part VII is the standing protocol.

---
---

# PART 0 — CURRENT STATE & CRITICAL PATH (updated 2026-05-29, after Sprint 6 round 1)

**HEAD = sprint5_v9-equivalent: local 2.7182, 0.18s avg, 100/100. This is the best committed solution and is hard to beat because it sits at the runtime floor.** Sprint 6 round 1 tested P0, P1.A, P1.B — all reverted. The reverts were CORRECT and the evidence has converged on one root cause. Read this before navigating Part III.

### What Sprint 6 round 1 proved (3 experiments, 1 conclusion)
1. **P0 fast portfolio (construction-only members, persistent pool): reverted.** Persistent pool worked, but quality only reached **2.670** vs v9's 2.718 (a mere 1.8% gain) at 2.3× the runtime (0.41s). Runtime-adjusted, **v9 dominates at every median up to ~5s** (verified via `score_real.py`). **Correction to an earlier planning claim:** portfolio_v1's 2.40 quality came from its *per-member SA*, NOT from member diversity. **Diversity alone is not a quality lever.** Do not pursue construction-only portfolios for quality.
2. **P1.A shape-aware packing: reverted — 15/100 feasible.** Giving soft blocks non-square aspect ratios makes the **shelf packer produce overlaps** (its fixed-row-height model can't absorb wider/shorter blocks). The shape lever (the biggest area opportunity, II.2) is real but **the shelf packer physically cannot exploit it.**
3. **P1.B correctness-first polish (full-recompute greedy): reverted.** Regression-proof and feasible (good — the discipline works), but **full true-cost recompute per move is too slow** (+0.16s for ~nil gain) and the moves are still weak.

### The converged conclusion → THE CRITICAL PATH
**The shelf packer is the quality ceiling, proven three independent ways** (can't do shapes; diversity around it doesn't help; polishing its output is too slow for the gain). Patching it is exhausted. **The next move is not optional and not cheap: replace the packer.**

➡️ **CRITICAL PATH = build a skyline/tetris packer with integrated shape (aspect-ratio) selection** — see the new **Part IV.PACKER** spec below. This is the substrate that (a) unlocks the 52%→~96% utilization win (the single biggest quality lever), (b) is feasible *by construction* (skyline never overlaps), and (c) is fast (O(n log n)). It is a lighter, lower-risk first step than full topological SA (P2) and likely captures most of the area win. Everything else (incremental polish P1.B, topological SA P2) layers on top of it.

### Revised navigation (overrides Part III until the packer lands)
You are at **STATE: PACKER**. Do Part IV.PACKER. Gate: prototype must hit **>0.70 utilization on cases 99/97/95 standalone** before integration; integrated result must beat v9 runtime-adjusted at median∈{1,2,3}s, 100/100. On pass → resume Part III at STATE 2 (P2) with the new packer as substrate. Do NOT spend more runtime to win (v9 is at the floor); win on QUALITY at v9-level runtime (~0.2–0.4s).

### Standing correction to strategy
Quality comes from **a better packer + a fast optimizer**, not from portfolio breadth and not from spending more wall-clock. Target: match v9's runtime (~0.2–0.4s, at floor) while cutting hpwl_gap/area_gap. Stop re-testing portfolio-diversity and shape-on-shelf (both are now dead-ends, II.6).

---

### UPDATE 2 (2026-05-29, after the skyline-packer round) — STOP TWEAKING GREEDY PACKERS

**What happened:** the skyline packer was built but only reached util 0.46–0.59 (< the 0.70 gate) and was reverted. Two reasons it is NOT valid evidence against the shape lever:
1. **The shape heuristic is naive.** `_skyline_pack` (line ~3592) scores shapes by `score = th` → it makes *every* block maximally wide (5:1). Golden uses a **mix** (median 1.45) chosen to **fit each contour gap**. "Make everything wide" is not shape-aware packing and packs badly.
2. **Greedy packing is structurally capped** (~0.5–0.7 for mixed rectangles under boundary+cluster+obstacle constraints). It **cannot** reach golden's 0.96 by tuning. **Therefore: stop building/tuning greedy packers (shelf, skyline, BLF). That avenue is exhausted.** (`_skyline_*` code remains in the file but is un-wired; leave it or delete it — do not keep tweaking it.)

**Meta-warning:** Sprints 4–6 made the committed *quality* WORSE (2.47→2.72), "winning" only on the low-median runtime bet. We are not closing the 2×-golden gap. The repeated pattern — build a light variant, see it isn't instantly faster than the 0.18s baseline, revert — must stop. **The win requires committing to ONE real optimization engine and giving it hours, not reverting v1 for being slow.** It is best-of gated, so it cannot regress; there is no reason to revert it.

### UPDATE 3 (2026-05-30, after the overnight ENGINE run) — PLATEAU CONFIRMED → COMMIT TO TOPOLOGICAL SA

**The overnight run closed the last light/medium avenue.** The ENGINE (incremental cost + correctness-first repair polish) was built correctly and is fast, but improved the score only microscopically because **local search cannot escape the shelf-packed local optimum**: blocks are too tightly packed for relocation moves, ripple-repair breaks feasibility, and there was **zero gain on the 116–120 band (34% of score)**. 

**We have now exhaustively proven the ceiling** — none of these break ~2.7 / ~0.52 util: greedy packing (shelf/skyline ≤0.59 util), shape-on-greedy (capped + the lever is real but greedy can't use it), portfolio diversity (2.67), and local repair search (stuck in the shelf optimum). **This is Escalation Trigger E4 (global plateau).** Continuing any of the above is forbidden busywork.

**THE decision: commit fully to IV.P2 — topological-representation SA (sequence-pair / B*-tree).** It is the only remaining lever and the academically-proven method for this exact problem (soft+hard modules, ≤ few-hundred blocks, min area+wirelength): **every state is already a tight, overlap-free packing**, so there is no shelf structure to be trapped in and no legalization step to destroy quality. Expect this to be a multi-day build; it will NOT beat v9 until substantially complete — **that is expected; do not revert partial progress** (see the anti-revert rule in the upgraded IV.P2). Reusable from the night: the incremental cost evaluator and soft-violation counters (use them as the SA objective).

### REVISED CRITICAL PATH (supersedes IV.PACKER as the next action)
Build the **strong repair-based optimizer** on top of the existing, working, regression-proof `_correctness_first_polish` (line ~3654). This is higher-probability-of-success than a from-scratch B*-tree and attacks hpwl + area + shape together. Spec = new **Part IV.ENGINE** below. Do this next; keep topological-SA (IV.P2) as the follow-on only if ENGINE plateaus.
- **First (1–2h) validity check before the big build:** fix the shape heuristic to **fit-to-gap** (choose the aspect whose width matches the current skyline valley / remaining row width, not max width) and re-measure standalone util on 99/97/95. This finally tests the shape lever honestly. Whatever the result, then proceed to IV.ENGINE (the polish reshape-move uses the same fit-to-gap logic on an already-feasible layout, so it can't break feasibility).
- **Do NOT** spend more than ~2h on the standalone packer check; the durable win is IV.ENGINE, not the constructive packer.

---
---

# PART I — INVARIANTS (these never change; everything is judged against them)

### I.0 GUIDING PRINCIPLES (operator's standing orders — apply ALWAYS, not just overnight)
1. **Winning first.** The only goal is a winning-quality, 100%-feasible submission. Ship a change only if it genuinely improves the winning metric (runtime-adjusted total, median 1–3s). HEAD is best-of gated and always a valid submission.
2. **Never rush.** Correctness before speed; validate properly (full eval + `score_real.py`); if a result is partial, uncertain, or unverified, do NOT ship it — keep iterating or pause. A slower-but-correct path beats a fast-but-broken one.
3. **No meaningless work.** Every experiment needs a one-line hypothesis with a plausible path to improving the score. Forbidden: re-testing II.6 dead-ends, trivial nudges, busywork to look productive. If positive-EV work is genuinely exhausted, **pause and report** — a graceful stop beats churn.

### I.0a TIMELINE & PACING (set 2026-05-30)
- **Presentation checkpoint: ~2 weeks out (≈2026-06-13).** Real submission deadline: ~1 month after that (≈2026-07-13).
- **There is ample time — do NOT rush; correctness/quality over speed-of-delivery.** This is why we attempt the hard full floorplanner rather than settling.
- **Phase plan:**
  - **Weeks 1–2 (→ presentation): land the constraint-aware global floorplanner's QUALITY** (boundary-edge encoding, cluster super-blocks, MIB shared shape, per-block aspect in moves). Dev-budget speed is fine for now. Goal by presentation: a **dev-budget result that beats v9 on quality (lower hpwl_gap + area_gap) at 100% feasibility**, even if not yet fast. v9 remains the safe presentation fallback.
  - **Weeks 3–6 (→ real deadline): speed + polish.** numba inner loop, warm-start from v9, parallel chains → get runtime-adjusted win; then de-overfit (cross-validate on folds), robustness (fuzz on training set), push util→0.9, optional ML warm-start. Final winning submission.
- **Always keep HEAD = a valid, committed, 100%-feasible entry** (currently v9) so a submittable solution exists at every moment.

### I.1 Mission
Win the ICCAD-2026 FloorSet contest (Problem C). The contest ranks submissions by total cost on a **hidden** 100-case set (same 21–120 block range, different instances). Minimize the **runtime-adjusted** total cost while remaining **100% feasible**.

### I.2 The objective you are actually optimizing
Per case (`iccad2026_evaluate.py:306`, `ALPHA=0.5 BETA=2.0 GAMMA=0.3 M=10`):
```
cost_i = (1 + 0.5·(max(0,hpwl_gap) + max(0,area_gap))) · exp(2·V_rel) · max(0.7, (rt_i/median)^0.3)
       = 10.0 if infeasible
Total  = Σ_i cost_i · exp(n_i/12) / Σ_i exp(n_j/12)         (n≥100 ≈ 79% of total weight)
hpwl_gap = max(0,(hpwl−hpwl_base)/hpwl_base);  area_gap = max(0,(bbox−area_base)/area_base)
V_rel    = soft_violations / n_soft   (boundary + Σ(cluster components−1) + Σ(MIB distinct shapes−1))
```
Baselines (`results/_baselines.json`) are the **golden** hpwl/bbox; gaps are clamped ≥0 (you cannot score below golden on quality).

### I.3 Runtime is first-class — the floor logic (memorize)
`max(0.7,(rt/median)^0.3)`:

| rt/median | 0.10 | 0.305 | 0.5 | 1 | 2 | 4 | 7 | 15 |
|---|---|---|---|---|---|---|---|---|
| mult | 0.70 | **0.70** | 0.81 | 1.00 | 1.23 | 1.52 | 1.79 | 2.25 |

- **At rt ≤ 0.305·median you hit the hard 0.7 floor** (a flat 30% cost cut, unbeatable). Below the floor, **more speed buys nothing**.
- Above median, penalty is **uncapped**.
- **RULE (absolute): never trade quality for speed once you are at the floor.** Sprint 5's mistake. Spend the "free budget" (rt up to ~0.3·median) entirely on quality.
- The local evaluator hard-codes `rt/median=1.0`, so the local score **hides runtime**. Judge with `scripts/score_real.py` (Part VII) across median∈{0.5,1,2,3,5}s.

### I.4 Sacred rules
1. **Feasibility 100/100 is non-negotiable.** One infeasible case = cost 10 ≈ catastrophic. Every candidate passes the hard gate: no overlap (tol 1e-6), every soft-block area within ±1% of target, fixed dims exact, preplaced (x,y,w,h) exact. Never move a preplaced block.
2. **Judge only on the runtime-adjusted total** (Part VII), not the local RF=1 number. Keep a change iff it improves the runtime-adjusted total at median∈{1,2,3}s AND stays 100/100.
3. **De-overfit:** the ranking set is a *different* 100. Tune **count-agnostic** params; select hyperparameters on **held-out folds** of the 100, never on the full 100. Per-count tuned tables do not transfer.
4. **Never regress HEAD.** Always leave the repo committed, 100/100 feasible, with the best runtime-adjusted config. Use best-of gating (true-cost) so new methods can only help.
5. **Append every experiment to `PLAN_EXECUTION_LOG.md`** (Part VII template), including failures + why.

---
---

# PART II — KNOWLEDGE BASE (verified facts; do not re-derive)

### II.1 Where the score lives
n≥100 ≈ **79%** of total; n≥116 alone ≈ 34%. n≤80 ≈ 4% combined. **Optimize the big cases; ignore small-n micro-tuning.** Top weighted cases: 99,98,97,96,95,89 (debug here).

### II.2 Quality gap vs golden (the headroom — this is the prize)
Current best per dominant band (n≥100): hpwl_gap ≈ 1.12–1.37, area_gap ≈ 0.82–1.08. Meaning we are **~2.1× golden wirelength and ~1.8–2.0× golden bbox area.**
- **GOLDEN PACKS AT 96.6% area utilization (Σblock_area/bbox). WE PACK AT ~52%.** Closing to 74% util ⇒ area_gap 0.87→0.30 ⇒ quality factor −0.28. Closing further approaches golden.
- **GOLDEN USES SOFT-BLOCK ASPECT RATIOS UP TO 3:1** (median 1.45, p90 ≈ 2.5). At **constant area** (±1% allowed). The **evaluator does NOT check aspect ratio — only area** — so shape is a *free* variable, bounded only by your own choice. We under-use it.
- **Implication:** the area gap is mostly a **shape-aware tight-packing problem**, and it is the single biggest, most certain quality lever. Tighter packing also reduces HPWL (blocks closer). Aspect change does NOT change a block's center, so it has **no HPWL downside** — it only helps area + boundary-touch.

### II.3 Constraint reality (all 100 cases)
Every case has fixed + preplaced + MIB + cluster + boundary. Big-case (n≥100) profile: fixed 7–17, **preplaced 1–9 (present in all 21 big cases — fixed obstacles)**, MIB 1 group, cluster 3–4 groups, boundary 26–37 blocks (~30% must touch a bbox edge → forces a perimeter ring), b2b edges 661–7056. V_rel already ~0.10 (near floor) → **protect soft, don't chase it.**

### II.4 Runtime status
sprint5_v9 (current best): n≥100 ≈ 0.46s/case (at floor for median≥1.5s) but **quality regressed** (hpwl 1.37/area 1.08). It wastes the free quality budget. baseline quadratic_v1: 1.47s, quality 1.17/0.87. portfolio_v1: 6.96s (too slow), quality 1.12/0.82.

### II.5 Architecture & key line numbers (`contest_solution/my_optimizer.py`, ~3741 lines)
```
_worker_solve            :26    module-level pool worker (re-instantiates MyOptimizer)
solve                    :68    builds portfolio, **creates a NEW pool PER CASE (bug, fix in P0)**, true-cost select
_build_portfolio         :88    currently only 2 configs (shrunk due to pool overhead)
_construct_layout(shelf) :197   degree-ordering shelf packer + refine + SA  (the quality workhorse)
_choose_dimensions       :1896  picks (w,h) — **shape lever lives here; under-exploited (see II.2)**
_pack_interior_units     :356   shelf row packer
_refine_free_block_shifts:1093
_selection_cost (proxy)  :2000
_true_contest_cost       :2056  EXACT cost, feasibility-gated — use for all selection/gating (correct)
_soft_violation_count    :2069  matches evaluator numerator
_sa_post_optimization    :2521  SLOW (full HPWL recompute/move over ≤7000 edges) + WEAK (equal-area swaps, overlap-reject, frozen soft factor)
_analytical_construct    :2749  QP/contour path (loses to shelf at legalization)
_abacus_construct        :2874
_analytical_global_placement(QP/CG):3484   (linear QP; collapses without spreading)
_analytical_legalize     :3632
```

### II.6 Consolidated DEAD-ENDS (do NOT retry; from all sprints)
BFS ordering; multi-start SA as built; force-directed refinement under hard overlap-reject; centroid sorting; equal-area/overlap-reject position swaps; compaction-to-origin; **analytical x-ordering without cluster super-blocks** (41–58 soft viols); **linear QP without spreading**; QP+relaxation hybrid; real-cost in the *tuned* variant loop; 200 relaxation sweeps; **per-case process-pool creation** (self-inflicted overhead); **portfolio of heavy-SA members** (runtime blowup); **density-spread QP fed only into the losing analytical path**; **incremental-HPWL local search built incrementally-first** (broke, 83 worsened — must be correctness-first); **aspect-ratio fitting bolted onto the weak SA** (no effect — but see II.2: aspect IS a top lever when done in a real packer); **sacrificing quality for sub-floor speed**. **[Sprint 6] construction-only portfolio for quality** (diversity ≠ quality; only 2.67 vs 2.72); **non-square shapes on the shelf packer** (15/100 feasible — shelf row model can't absorb them; needs the Part IV.PACKER skyline packer); **full-recompute polish at scale** (too slow per move — must use incremental cost from a verified-correct base).

---
---

# PART III — DECISION TREE (your navigator; consult after every experiment)

> Determine your current STATE from the latest committed result + log, then take the indicated ACTION. Each phase has a hard ACCEPTANCE GATE and explicit on-pass / on-fail branches. Spec for each phase is in Part IV.

### STATE 0 — start here (P0 not yet committed)
**ACTION:** Implement **P0 (fast persistent-pool portfolio)** [Part IV.P0].
**GATE:** runtime-adjusted total beats sprint5_v9 at *every* median∈{0.5,1,2,3,5}s AND beats baseline at median≤3s; 100/100; n≥100 hpwl_gap≤1.12 & area_gap≤0.82 (≥ portfolio_v1 quality).
- **PASS →** commit as new baseline `s6_p0`; go STATE 1.
- **FAIL (quality recovered but still slower than floor) →** reduce per-member work / increase parallel breadth; re-measure. If pool won't persist (fork/torch issue), use the fallback in IV.P0.5.
- **FAIL (quality not recovered even with full portfolio) →** the construction-only members are too weak; add a single short shared polish (current SA) on top-2 winners; if still short, go STATE 2 early (the polish itself is the lever).

### STATE 1 — P0 banked; attack the quality ceiling
Run **P3.1 golden-mining priors** [IV.P3.1] first (cheap, ~1 day) to set concrete shape/util targets, THEN:
**ACTION:** Implement **P1 (shape-aware tight packing + correctness-first legalization polish)** [Part IV.P1]. This is the highest-EV quality work (II.2).
**GATE:** beats `s6_p0` runtime-adjusted at median∈{1,2,3}s; 100/100; **n≥100 area_gap drops below 0.7** (utilization >0.59) as the first milestone, then chase <0.5.
- **PASS →** commit `s6_p1`; go STATE 2.
- **FAIL (polish regresses or breaks feasibility) →** you violated correctness-first. Revert to full-recompute hill-climb with feasibility asserts (IV.P1.1). Do NOT proceed to incremental cost until monotone improvement is proven on cases 99/97/95.
- **PARTIAL (area improves, hpwl stuck) →** add connectivity-aware relocation moves (IV.P1.2 move 1) and re-seed packing order from QP centroids; if hpwl still stuck, go STATE 2 (topological SA is the structural HPWL fix).

### STATE 2 — P1 banked (or P1 plateaued); go for the structural breakthrough
**ACTION:** Implement **P2 (topological-representation SA: B*-tree/sequence-pair with shape curves + fixed obstacles)** [Part IV.P2]. Prototype obstacle handling on test 99 FIRST and prove exact feasibility before scaling.
**GATE:** beats `s6_p1` runtime-adjusted at median∈{1,2,3}s; 100/100.
- **PASS →** commit `s6_p2`; go STATE 3.
- **FAIL (obstacle feasibility intractable) →** demote P2 to a *packing subroutine* inside P1 for obstacle-free sub-regions; return to STATE 1 backlog (Part V) and STATE 3.
- **FAIL (feasible but not better) →** check the numba inner loop speed and seeding (must seed from P0/P1, not random); if still not better after schedule tuning, freeze P2 and go STATE 3.

### STATE 3 — analytic pipeline matured; pursue the moonshot + harden
Run **P3.2 (learned seeding)** [IV.P3.2] as a parallel research track AND **CONTINUOUS hardening** [IV.C].
**GATE for shipping P3.2:** beats the analytic pipeline on a **held-out fold**, runtime-adjusted, 100/100.
- Always keep the best analytic pipeline as HEAD; only swap in the learned seeder if it wins held-out.
- When all of P0–P3 are committed and gates plateau (no track improves runtime-adjusted total by >0.5% in its last full pass), **fire Escalation Trigger E4 (Part VI)** for a fresh strategic plan.

### Idle / blocked at any state
If a track is mid-build or waiting, pull the next item from the **Experiment Backlog (Part V)** (ranked by EV). Never leave the 48 cores idle.

---
---

# PART IV — TRACK SPECS (deep implementation detail)

## IV.P0 — Fast persistent-pool portfolio  *(STATE 0)*

**Goal:** portfolio_v1-level quality (≥) at ~0.3–0.5s/case (at/near floor). Beats sprint5_v9 at all medians.

**P0.1 Persistent pool.** Module-level lazy `ProcessPoolExecutor(max_workers=min(48,cpu))` with **fork**, created on first `solve()`, reused for all 100 cases, closed via `atexit`. Removes per-case fork/import overhead (the real cause of "portfolio too costly"). Keep a serial fallback if pool init throws.

**P0.2 Real portfolio (construction-only members, ≤0.15s each for n=120).** K≈16–32: shelf×row_factor{0.8,0.9,1.0,1.1} × {boundary-cluster on/off}; analytical(QP+contour); abacus; a few tie-break RNG seeds. **No per-member SA.** Select best by `_true_contest_cost` (:2056).

**P0.3 One shared polish.** Run the existing `_sa_post_optimization` (or P1 engine when ready) ONLY on the top-1/top-2 winners, hard time-budgeted so total wall-clock stays ≤ ~0.3·median_guess (target n≥100 ≈ 0.3–0.5s). Best-of-gated.

**P0.4 Budget calibration.** Use `score_real.py` to confirm n≥100 sits at/near the 0.7 floor for median∈{1,2,3}s; spend all remaining "free" time on portfolio breadth + polish, none beyond the floor.

**P0.5 Fallback if fork+torch unstable:** use `multiprocessing.get_context('fork')`; if a worker imports CUDA, force CPU in workers (`CUDA_VISIBLE_DEVICES=""`); if still unstable, run K members **serially in-process but SA-free** (each ≤0.05–0.1s) — even 16 serial SA-free constructions fit in ~0.3–0.5s for n=120.

## IV.P1 — Shape-aware tight packing + correctness-first legalization polish  *(STATE 1; highest quality EV)*

This track has two coupled sub-levers. **Lever A (AREA, highest-confidence per II.2): shape-aware tight packing.** **Lever B (HPWL): connectivity-aware legalization-repair moves.** Build A first (certain win), then B.

**P1.A — Shape-aware tight packing (target util 0.96 like golden).**
- Rework dimension choice (`_choose_dimensions` :1896) and the packer so soft-block (w,h) is chosen **to tile tightly**, not near-square. Allowed shapes = any (w,h) with `|w·h−target|/target ≤ 0.01`; aspect is otherwise free (golden uses ≤3:1; you may exceed since the evaluator doesn't check aspect, but start with ≤3:1 as a sane prior and test relaxing it).
- Use a **skyline/shelf packer with shape selection**: when placing a block, pick the aspect ratio (within ±1%) that best fills the current gap (match remaining row height, or fill a column to the contour) — a 1-D strip-packing-with-resizable-items step.
- Add a **post-pack shape+compaction sweep:** for each soft block, try aspect ratios that (i) close the contour gap above/beside it, (ii) let a boundary block reach its edge, (iii) let a cluster member abut its mate; accept by exact `_true_contest_cost`, re-legalizing locally. (This is the Sprint-4 aspect idea, but now inside a packer that can exploit it.)
- **MIB group:** reshape all members together (must stay identical (w,h)).
- **Acceptance milestone:** n≥100 utilization >0.59 (area_gap<0.7), then >0.74 (area_gap<0.3).

**P1.B — Correctness-first legalization-aware polish (fixes the broken Thrust B).**
1. **Correctness first (regression-proof):** greedy hill-climb on a copy of the best layout; propose move → apply → **recompute FULL exact `_true_contest_cost`** → accept iff strictly lower, else revert. Monotone ⇒ cannot regress. After every accept, **assert hard feasibility** (overlap/area/preplaced). Validate true-cost improvement + feasibility on cases 99,97,95 before wiring in (best-of gated).
2. **Repair-not-reject moves** (preserve hard constraints): (a) relocate block toward connectivity centroid / into a gap, ripple-push the few overlappers along min-displacement axis; (b) unequal swap + local re-legalize; (c) row/slab inward shift then re-touch boundary; (d) aspect reshape (P1.A); (e) cluster rigid-move toward centroid.
3. **Only then add incremental cost** (per-net HPWL via incident edges using adjacency near :3200; bbox via tracked extremes; soft via per-cluster components / per-MIB shape sets) — **with a debug assert `incremental≈full` every N moves.** Greedy sweeps → short SA tail (exact incremental cost, true `exp(2·V_rel)`, not frozen). Run a few parallel chains on the pool; keep best.

## IV.ENGINE — Strong repair-based optimizer (the committed heavyweight; build this next)  *(supersedes IV.PACKER)*

**Premise:** every constructive packer (shelf/skyline/contour) is feasible-but-capped. The win is a strong *optimizer* that starts from any feasible layout (use the existing shelf-SA result as seed) and drives hpwl+area+soft down with **repair-based moves** under the **exact** contest cost. Build on `_correctness_first_polish` (line ~3654), which already works and is regression-proof. Do it in this strict order so it never breaks (the lesson from the broken Sprint-5 attempt):

**Step 1 — Correctness baseline (already done; verify):** greedy hill-climb, FULL `_true_contest_cost` recompute per move, accept iff strictly lower, assert feasibility after each accept. Confirm monotone + feasible on cases 99/97/95.

**Step 2 — Make it fast (incremental cost):** maintain running hpwl (per-incident-edge re-sum using adjacency already built at ~3672), bbox via tracked extremes, soft via per-cluster component counts + per-MIB shape sets + per-block boundary checks; recompute the exact `(1+0.5(hg+ag))·exp(2·vrel)` from these. **Keep a debug mode asserting `incremental ≈ full` every N moves** — this is the guardrail the earlier broken attempt lacked. Target ≥10⁴ moves/s for n=120.

**Step 3 — Strong moves (each preserves hard constraints; relocate-and-REPAIR, never reject):**
- **Relocate-to-centroid:** move block i toward its connectivity centroid; ripple-push the few overlappers along the min-displacement axis. (HPWL lever.)
- **Row/slab inward compaction:** slide a row/column inward to shrink bbox, re-touch boundary blocks to the new extremes. (AREA lever.)
- **Soft-block reshape-to-fill (±1% area):** retune (w,h) — fit-to-gap (match the local free-space width/height), to (i) fill a contour gap, (ii) let a boundary block reach its edge (kills a soft viol), (iii) abut a cluster mate. MIB group reshapes together. (SHAPE/AREA lever — this is how we exploit the golden aspect insight WITHOUT a new packer, on an already-feasible layout so feasibility can't break.)
- **Unequal swap + local re-legalize; cluster rigid-move toward centroid.**

**Step 4 — Use the free runtime budget + cores:** we are under the floor (v9 = 0.18s; floor allows ~0.3·median ≈ 0.3–0.6s). Run **K parallel chains** (different seeds/move-weights) on the persistent pool within that budget; keep best by `_true_contest_cost`. Greedy descent first, then short SA tail (exact incremental cost, true `exp(2·vrel)`) to escape minima.

**Gate:** beats sprint5_v9 runtime-adjusted at median∈{1,2,3}s, 100/100, at ≤~0.5s/case. **Do NOT revert for being slower than 0.18s — slower-but-at-floor with better quality is the goal (I.3).** Tag `s6_engine`. Expect hpwl_gap AND area_gap to drop (first sub-1.2 area, then chase sub-0.7).

## IV.PACKER — Skyline/tetris packer with shape selection  *(DEMOTED — greedy packing is capped; see Part 0 UPDATE 2; do not keep tuning)*

**Why:** the shelf packer is the proven ceiling (Part 0). A skyline packer is feasible-by-construction, fast (O(n log n)), and is the only substrate that can exploit the shape lever (II.2: golden = 96% util via aspect ≤3:1; we are at 52%).

**Core data structure — skyline (a.k.a. contour/Bottom-Left-Fill):**
- Maintain the upper contour of placed blocks as a list of horizontal segments `(x_start, x_end, height)`.
- To place a block of footprint (w,h): scan candidate x-positions (segment left edges and preplaced-obstacle edges); for each, the landing y = max contour height over [x, x+w]; choose the (x) that minimizes a placement score (primary: resulting max-contour-height / wasted area beneath; tie-break: proximity to the block's connectivity-centroid x for HPWL). Update the contour. This **never overlaps** (y is always above the contour).
- Preplaced blocks: initialize the contour to include them as fixed obstacles (raise the contour over their footprint to their top); movable blocks then pack around them automatically. This is how obstacles are handled cleanly — no special cases.

**Shape selection (the lever):** when placing a soft block, do NOT use a fixed (w,h). Enumerate a small set of aspect ratios r∈{1.0,1.3,1.6,2.0,2.5,3.0} (and their transposes), each giving (w,h)=(√(area·r), √(area/r)) renormalized so `|w·h−target|/target ≤ 0.01`. Pick the (r, orientation) that best fills the current contour notch — e.g. width matching a valley, or the shape that minimizes contour-height increase. This is "strip packing with resizable items." MIB group: pick ONE shared shape for all members. Boundary blocks: bias shape/placement so the block can touch its required edge.

**Ordering:** try several orders as cheap portfolio members (we have the pool): (a) decreasing area (classic BLF), (b) decreasing max-side, (c) connectivity-centroid sweep with clusters as super-blocks (super-blocks mandatory — bare analytical x-order is a dead-end). Pack clusters as contiguous macro-blocks (reuse `_cluster_local_pack`).

**Aspect ratio is free re HPWL:** changing (w,h) does not move a block's center, so it has zero HPWL cost — it is a pure area/boundary lever. You may even exceed 3:1 (evaluator doesn't check aspect; backlog item 2) — test it.

**De-risked build & validation (avoid the try→revert trap):**
1. Build the packer as a STANDALONE function. **First milestone: on cases 99, 97, 95, measure raw utilization directly** (Σarea/bbox). Gate: **>0.70** (vs current ~0.52) before any integration. If <0.70, iterate the placement score / shape set in isolation — do NOT touch `solve()` yet.
2. Assert feasibility (no overlap, areas within ±1%, preplaced untouched) inside the packer.
3. Only after the standalone util gate passes: add it as a portfolio path (`'path':'skyline'`) selected by `_true_contest_cost`; it can only help (best-of gated).
4. Then layer the P1.B incremental polish on its output, and use it as the P2 seed.

**Acceptance:** beats v9 runtime-adjusted at median∈{1,2,3}s, 100/100, at ~v9 runtime (≤~0.4s). Expect the area_gap to be the first big mover. Tag `s6_packer`.

## IV.P2 — Topological-representation SA  *(THE COMMITTED PRIMARY; de-risked milestone build)*

**This is now the whole game (Part 0 UPDATE 3).** Topological SA has no legalization step — **every state is a valid, tight, overlap-free packing** — so it escapes the shelf optimum that capped everything else, and is the proven route to near-golden area+wirelength.

**CRITICAL ANTI-REVERT RULE (read first):** this is a multi-day build that will NOT beat sprint5_v9 until M6. **Do NOT gate intermediate milestones (M1–M5) against the full contest score, and do NOT revert them for being slow or worse than v9.** Develop the SA as a **separate module / new code paths** (e.g. `_sp_pack`, `_sp_sa`), leaving `solve()` on sprint5_v9 so HEAD stays the best committed result throughout. Each milestone is validated on its OWN intrinsic metric below. Only at M6 do you wire it into `solve()` behind the `_true_contest_cost` best-of gate and compare runtime-adjusted vs v9. Commit each milestone (the module is dormant, so committing it can't regress HEAD).

**Representation:** start with **sequence-pair** (two permutations; positions via longest-path on horizontal/vertical constraint graphs — cleaner for adding positional/obstacle constraints than B*-tree). B*-tree is an acceptable alternative. **Inner packing loop in numba `@njit`** (pure Python is too slow for 10⁴–10⁵ perturbations; this is mandatory for the contest runtime budget). Soft blocks: discrete per-block aspect set within ±1% area chosen in the move set (exploits II.2 — the shape lever the greedy packers couldn't use).

**Milestones (each individually committable; validate on the stated intrinsic metric, NOT the full score):**
- **M1 — Packer correctness (movable-only, ignore obstacles).** Implement SP encode/decode + longest-path packing. *Gate:* on case 99's movable blocks, produces a **valid non-overlapping packing**; assert no overlaps. (No score yet.)
- **M2 — PROOF-OF-CONCEPT. ✅ DONE (2026-05-30): util = 0.828 on case 99 movable-only (7849 moves/30s dev budget) — PASS, method VALIDATED.** SP-SA decisively beats the 0.59 greedy ceiling. Full build (M3→M6) is justified. **Decision: CONTINUE, do not pivot to ML.** Do not spend more time optimizing movable-only util — the real challenges are the constraints (M3/M4) and speed (M5).
- **M3 — Fixed obstacles.** Add preplaced rectangles. Pragmatic approach: SA cost includes a large obstacle-overlap penalty (annealed); **reject any final state that overlaps an obstacle**. *Gate:* exact feasibility (no movable–obstacle overlap) on all 21 big cases; util still >0.75.
- **M4 — Soft constraints in the objective.** Add boundary (per-block edge touch), cluster (component−1), MIB (distinct-shape−1) using the existing incremental counters as the SA cost (true `exp(2·V_rel)`). *Gate:* V_rel ≤ v9's (~0.10) while util stays high.
- **M5 — Speed + parallelism.** numba-optimize the inner loop; **parallel-temper short chains across the 48 cores**, seed one chain from the v9/shelf layout (decode coords→SP). *Gate:* wall-clock ≤ ~0.5s/case for n=120 at the util/soft of M4.
- **M6 — Integrate + gate.** Wire into `solve()` behind `_true_contest_cost` best-of (so v9 is the floor). Full eval + `score_real.py`. *Gate:* beats v9 runtime-adjusted at median∈{1,2,3}s, 100/100. Tag `s7_sp`. **This is the make-or-break for winning quality.**

**Execution notes for M3–M6 (post-M2-pass, 2026-05-30):**
- **Expect util to ERODE as you add constraints — do NOT panic or revert when it does.** Movable-only = 0.828; adding obstacles (M3) fragments space, and the soft constraints (M4) — especially ~30% boundary blocks forming a perimeter frame + cluster abutment — will pull integrated util down to perhaps ~0.70–0.78. That is still a massive win over 0.52. Target: keep integrated util **>0.70** and area_gap **<0.35**; measure the erosion at each step.
- **Speed (M5) is now the main risk.** M2 took 30s for ~7800 moves (pure Python); the contest needs ~0.5s. Path to ~60×: (a) **numba `@njit`** the longest-path/packing inner loop (mandatory), (b) **warm-start the SA from the v9 shelf layout** (decode coords→sequence-pair) so it *refines* instead of starting cold → far fewer moves needed, (c) **parallel chains across the 48 cores**, keep best. Warm-start is the highest-leverage of the three — prototype it early in M5.
- **M6 gate sanity:** with area_gap ~0.25–0.35 (from ~0.87) the quality factor should drop enough that even at ~0.5s/case (vs v9's 0.18s) it wins runtime-adjusted at median≥1s. Verify with `score_real.py`; if it only wins at median≥2s, push M5 speed harder.
- **ML (IV.P3) is no longer a pivot — it becomes a *future enhancement*:** a learned model that predicts a good initial sequence-pair would cut the moves M5 needs (faster) — revisit after M6 ships.

### M4 BARRIER + RESOLUTION (2026-05-30) — DECOMPOSE: interior SP-SA + keep v9's frame

**What happened:** M3 (obstacles) = 18/20 feasible, util 0.705. M4 (soft constraints in a free SP-SA) **failed**: V_rel = 0.712 (vs v9's 0.10), util fell to 0.667. Root cause: a *free* SP-SA optimizes area only and has no way to make boundary blocks touch edges, cluster members abut, or MIB shapes match — so it shreds the soft constraints (exp(2·0.712)=4.2× penalty swamps any area gain).

**This is NOT a dead end — it tells us WHERE to apply the SA.** The two engines are each good at half the problem: v9's explicit logic satisfies soft constraints (frame via `_place_boundary_items`, contiguous clusters via `_cluster_local_pack`, MIB shapes via `_choose_dimensions`) → V_rel 0.10; SP-SA packs tightly → util 0.83. **Combine them — this is also literally golden's structure (perimeter ring + tight interior).**

**RESOLUTION = surgical hybrid (the new M4'; do this instead of a free full-SA):**
1. **Keep v9's pipeline intact** — boundary frame, preplaced handling, cluster super-blocks, MIB shape logic all stay (they already give V_rel≈0.10 and 100% feasibility).
2. **Replace ONLY the interior packer** (`_pack_interior_units`, the part that gives ~0.52 util) **with the SP-SA**, packing the non-boundary/non-cluster interior blocks tightly into the free space inside the frame, treating the frame + preplaced as fixed obstacles.
3. **Clusters → super-blocks in the SP** (reuse `_cluster_local_pack`) so abutment is automatic; **MIB → one shared shape** for the group. These are free structural wins, not new SA moves.
4. **Per-case best-of gate against v9 (guaranteed safety):** if SP-SA loses or is infeasible on a case, keep v9's result. ⇒ **cannot regress and is always 100% feasible** — this also fixes M3's 2 infeasible cases automatically. The win is wherever interior-tightness helps (the high-weight big cases).

*Gate (M4'):* runtime-adjusted total beats v9 at median∈{1,2,3}s with V_rel ≤ ~0.12 and 100/100 feasible (dev-budget speed first; M5 makes it fast). **Do NOT pivot to ML** — the SA works for what it's for; we were just pointing it at the whole problem instead of the interior.

### M4' HYBRID OUTCOME (2026-05-30) — interior-only is futile; the FRAME + SHAPES are the real levers

**Result:** interior-only SP-SA gave local 2.701 (vs v9 2.718, ~0.6%) at 0.69s (3.8× slower) → **worse runtime-adjusted at every median; reverted.** Critically, **area_gap did NOT move (1.18 → 1.18 on the 81–100 band).**

**Decisive lesson:** tightening the *interior* cannot reduce the bounding box, because the bbox is governed by the **boundary frame + overall block shapes**, not the interior. So all three of {greedy, interior-only SP-SA} are now proven incapable of closing the area gap. The two real, still-unexploited levers (both confirmed by golden mining) are:
1. **SHAPE** — golden uses aspect ratios med 1.45 / up to 3:1; we use ~1.0. Free variable (evaluator only checks area). We have NEVER successfully exploited it (shelf packer overlaps with non-square; SP-SA breaks soft constraints).
2. **FRAME/global arrangement** — the perimeter blocks set the bbox; they must be reshaped + ordered to form a tight frame, which is exactly the constraint-laden part.

**⇒ The only remaining path to a winning-tier score is a FULL constraint-aware global floorplanner** — SP-SA (or B*-tree) that optimizes the *whole* layout (frame + interior), with: boundary blocks encoded/penalized to their required edges, clusters as super-blocks, MIB shared shape, and **per-block aspect-ratio in the move set** (the shape lever). This is the hard build the earlier note called "complex/high-risk" — it is now confirmed as the necessary one; the shortcuts are exhausted. Build as a dormant module, milestone-gated, per-case best-of vs v9 (so HEAD stays safe). **Boundary-edge encoding is the key technical risk — prototype it on one case first.** If after a genuine effort it can't reach a runtime-adjusted win, the fallbacks are: ship v9 (valid, beats baseline) or pivot to ML (IV.P3).

**If M6 wins:** strip overfit tuning, cross-validate (IV.C), then push util→0.9 with better SA schedules / shape curves.

## IV.P3 — Data-driven (contest's stated theme; fast inference ⇒ floor-friendly)

**P3.1 Mine golden (DONE once, extend as needed; do early — informs P0/P1/P2).** `scripts/mine_golden.py`. Known so far: util 0.966, aspect ≤3:1 (med 1.45). Extend to: per-cluster shape/topology, perimeter-ring layout pattern, per-net wirelength distribution, how golden places preplaced relative to clusters. Turn findings into concrete priors (preferred aspect set, target density, ring builder).

**P3.2 Learned seeding (offline training, A100 available; STATE 3).** Train a small **GNN** on (area targets + b2b/p2b connectivity + constraints) → per-block (position, aspect) and/or ordering/sequence-pair; supervise on golden (+ optional RL with the evaluator reward near :1155). Inference = one forward pass (ms ⇒ floor) → seed → P1/P2 polish. **Generalization risk:** augment heavily, **select on held-out fold**, ship only if it beats analytic pipeline held-out. Keep model small for fast inference.

## IV.C — CONTINUOUS hardening (run throughout once P0 is in)
- **Cross-validated tuning:** k-fold over the 100; tune count-agnostic params on train folds, select on held-out. Large sweeps OK (unlimited compute) but **never select on the full 100.**
- **Median-robustness:** prefer configs Pareto-good across median∈{1,2,3,5}s.
- **Feasibility fuzzing:** run on FloorSet Lite *training* instances (not just the 100) to catch hidden-set feasibility edge-cases.
- **Housekeeping:** delete stale `results/summary.json`; keep result JSONs tagged; commit each win.

---
---

# PART V — EXPERIMENT BACKLOG (ranked by EV; pull when idle/blocked)

1. **Shape-aware packing prototype** on case 99 in isolation: measure best achievable utilization with aspect∈{1:1..3:1} vs current. (Validates IV.P1.A ceiling.) *(highest EV)*
2. **Relax aspect beyond 3:1** experiment: does aspect up to 5:1 or 8:1 pack tighter / touch more boundaries without HPWL cost? (Evaluator allows it.) Keep only if runtime-adjusted improves held-out.
3. **Connectivity-aware packing order** (seed shelf/skyline order from QP centroid x, clusters as super-blocks) — careful: analytical x-order w/o super-blocks is a dead-end; super-blocks are mandatory.
4. **Boundary-ring constructor:** build the ~30% boundary blocks as an explicit perimeter frame (golden structure) then pack the interior — measure soft + area.
5. **Polish move-weight tuning** via held-out folds.
6. **Per-net weighting:** check if hub nets (high-degree) dominate HPWL; prioritize their endpoints in placement.
7. **MIB-aware shaping:** pick the single MIB-group shape that best packs across all members.
8. **Parallel-tempering schedule sweep** for P2.
9. **Warm-start P2 from P1** vs random — quantify.
10. **GNN feature ablations** (P3.2).

---
---

# PART VI — ESCALATION TRIGGERS (the ONLY times to stop for a re-plan)

Fire one of these → write the situation to `PLAN_EXECUTION_LOG.md` and request a new plan. Otherwise keep navigating Part III autonomously.
- **E1 — Feasibility cannot be maintained:** a track can't stay 100/100 on the validation set despite the gate (e.g., P2 obstacle handling fundamentally infeasible). 
- **E2 — Metric/rule surprise:** the evaluator/contest reveals behavior contradicting Part I (e.g., a hard runtime cap, aspect-ratio constraint, batch eval, or the median assumption proven wrong by an official leaderboard).
- **E3 — A track wins big and opens new questions** the playbook doesn't cover (e.g., learned seeder beats everything and you need a training-data/eval strategy).
- **E4 — Global plateau:** all of P0–P3 committed and no track improves the runtime-adjusted total by >0.5% in its last full pass. (Time for a fundamentally new idea.)
- **E5 — Contradiction with a stated fact in Part II** (e.g., golden util turns out not to be reachable due to a missed constraint).
Do NOT escalate for ordinary failed experiments — those are expected; log them and pull the next backlog item.

---
---

# PART VII — STANDING PROTOCOL

### Run + judge (every change)
```bash
cd /home/ubuntu/EDA
python3 -c "import ast; ast.parse(open('contest_solution/my_optimizer.py').read())"          # syntax gate
cp contest_solution/my_optimizer.py external/FloorSet/iccad2026contest/
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. /usr/bin/python3 iccad2026_evaluate.py --evaluate my_optimizer.py --output /home/ubuntu/EDA/results/<tag>.json
cd /home/ubuntu/EDA
python3 scripts/analyze_results.py results/<tag>.json --top 20
python3 scripts/score_real.py results/<tag>.json results/<current_best>.json results/quadratic_v1.json
```
- Single-case debug: `--test-id <id> --verbose` (use 99,98,97,95,89).
- **KEEP a change iff** runtime-adjusted total (median 1–3s) strictly improves AND feasibility 100/100. Else revert.
- Commit each win: `git add -A && git commit -m "..."` ending with the Co-Authored-By line below.

### `scripts/score_real.py` (must exist from Sprint 5; if missing, build it)
Reads result JSON(s) with per-case {runtime, hpwl_gap, area_gap, violations_relative, block_count}; recomputes weighted Total under median∈{0.5,1,2,3,5}s using the I.2 formula; prints a side-by-side table for all JSONs passed. This is the scoreboard.

### Log template (append to `PLAN_EXECUTION_LOG.md` per experiment)
```
### <date> <tag> — <one-line change>
- Runtime-adjusted Total @ median {0.5,1,2,3,5}s: a / b / c / d / e   (vs prev-best: …)
- Local RF=1 Total: X | Feasible: N/100 | runtime sum/max: Ss / Ss
- n≥100 hpwl_gap: … area_gap: … (utilization …) V_rel: …
- Verdict: ✅ kept / ❌ reverted — why
- New dead-ends (if any): …
```

### Commit / PR footers
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

## One-paragraph north star (re-read when lost)
We are ~2× golden on both wirelength and area, and golden proves **96% packing via aggressive aspect-ratio use is achievable** — that's the prize. Bank a robust win first (P0: fast portfolio at the runtime floor), then close the area gap with **shape-aware tight packing** (P1.A, highest-confidence lever) and the HPWL gap with **legalization-repair local search** (P1.B, correctness-first), then break the structural ceiling with **topological SA** (P2) and finally the **learned seeder** (P3). Judge everything on the runtime-adjusted scoreboard, keep 100% feasible, de-overfit on held-out folds, and only stop for the Part VI triggers.

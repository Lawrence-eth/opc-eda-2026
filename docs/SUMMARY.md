# FloorSet ICCAD-2026 (Problem C) — Project Summary & Presentation Guide

> Single-source overview: problem, solution, results, methodology, honest
> assessment. Each section ≈ one slide. Numbers verified against
> `results/integrated_v10.json` (current) and `results/v9_locked.json`
> (pre-campaign baseline). Updated 2026-07-07.

---

## 1. The Problem (1 slide)

**ICCAD-2026 FloorSet Challenge, Problem C — Data-Driven SoC Floorplanning.**

Given a chip with **21–120 rectangular blocks**, place them on a 2-D canvas to
minimize area and wire length while obeying placement rules.

**Inputs per case:** block area targets (soft blocks: any aspect at that area,
±1%), block-to-block and pin-to-block connectivity, and constraints — *fixed*
dims, *preplaced* immovable rectangles, *MIB* groups (identical shapes),
*cluster* groups (must abut), *boundary* blocks (~30% must touch a die edge).

**Output:** (x, y, w, h) per block — no overlaps, all hard constraints met.

## 2. How It's Scored (1 slide — the two twists)

```
cost = (1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, (rt/median)^0.3)
     = 10 if infeasible;   Total = exp(n/12)-weighted avg (n≥100 ≈ 79%)
```
- Gaps are measured against **golden** (near-optimal ground-truth) layouts and
  clamped ≥ 0; violations are exponentially penalized.
- **Twist 1 — runtime is a first-class term**: ≥3× faster than the field's
  median earns a hard 30% discount; slower than median is penalized uncapped.
- **Twist 2 — the executable IS the timed unit**: official evaluation spawns
  the submitted binary once per case (organizers' `op_wrapper.py`); process
  startup counts. A torch-bundling binary pays seconds per case.

## 3. The Key Insight (1 slide — the breakthrough)

Mining the golden data showed: **golden layouts are near-perfect tessellations**
— utilization ≈ 0.97, soft-block areas equal targets *exactly*, ~3% whitespace
from a few floaters. Every previously tried method (shelf/skyline packers,
sequence-pair SA, ML prediction) packed *rigid* rectangles and capped at
0.5–0.8 utilization, or broke the cluster-abutment rule when packing tight.

**So we stopped packing and started *dissecting*:** build the layout as
full-width exact-fill rows where each soft block's dimensions are *derived*
from the structure (w = area/height). Then:
- utilization ≈ 1 in unobstructed regions — **by construction**;
- overlap-free — by construction;
- **clusters tile contiguous regions ⇒ abutment satisfied by construction**
  (the exact property every packer lost);
- boundary demands become structure: bottom/top bands, row-end slots,
  vertical edge stacks; preplaced rectangles are carved around; MIB groups
  get identical slots.
- HPWL is handled by *ordering*: barycenter iteration over connectivity with
  absolute pin anchors, two construction passes.

## 4. Results (1 slide)

| Metric | Pre-campaign (v9) | **Current** |
|---|---|---|
| Official score (RF=1) | 2.7182 | **1.8074** (−33%) |
| Feasible | 100/100 | **100/100** |
| Runtime | 0.18s avg | **0.18s avg** (same speed) |
| Runtime-adjusted @ median 1s | 2.110 | **1.353** (−36%) |
| Packaged binary, official command | — | 1.807413, 0 position diffs, 0.198s incl. spawn |

Calibration: golden-equivalent play = 1.108 (RF=1) / 0.776 (at the runtime
floor); theoretical bound 0.70. Golden itself violates soft constraints on
90/100 cases — even perfection isn't 1.0.

## 5. Verification story (1 slide — trust the numbers)

- Official evaluator, all 100 validation cases, reproduced bit-identically
  across machines (2-core VM vs 48-core dev box).
- The shipped PyInstaller binary re-verified through the organizers' exact
  command after every engine change: identical positions, 400/400
  training-instance feasibility fuzz.
- 51 regression tests + result audit + release gate on every commit; every
  experiment (including 4 reverted ones) logged with verdicts.

## 6. Methodology / rigor (1 slide)

- Evidence before code: golden structure mined from the dataset (tessellation
  property, B*-tree label semantics), retrieval hypothesis **disproven** by
  scanning all 1,008,000 training instances (0 validation matches).
- Every change judged runtime-adjusted at median ∈ {1,2,3}s — two experiments
  that improved raw quality were **reverted** for losing runtime-adjusted.
- Feasibility protected by a per-case, feasibility-gated best-of selector;
  two real bugs (selector gap-clamping, preplaced-moving retouch) were caught
  by the full-100 gates before they could ship.

## 7. Honest assessment / limitations (1 slide)

- **Strengths:** 100% feasible, deterministic, fast (runtime-floor-friendly
  even through the per-case-spawn harness), structurally sound constraint
  handling, 33% quality improvement banked with zero runtime cost.
- **Remaining gap:** hpwl_gap 0.86 / area_gap 0.25 on the heavy cases —
  ~0.7 runtime-adjusted points above golden-equivalent play. Ranked leads
  with evidence: `HANDOFF.md` §6.
- **Risk:** field median runtime unknown; if the field is slow, dormant
  budgeted-search machinery can be re-enabled to spend the headroom.

## 8. Reproduce (backup slide)

```bash
# bootstrap: HANDOFF.md §5 (venv + FloorSet + auto-downloaded validation data)
cp contest_solution/my_optimizer.py contest_solution/dissect.py \
   contest_solution/sequence_pair_sa.py external/FloorSet/iccad2026contest/
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. ../../../.venv/bin/python iccad2026_evaluate.py --evaluate my_optimizer.py
# -> Total Score: 1.8074, Feasible: 100
python -m pytest                                    # 51/51
python scripts/check_public_release.py              # PASS
```

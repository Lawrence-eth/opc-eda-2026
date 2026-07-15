# SUBMISSION PLAN — ICCAD 2026 Problem C (FloorSet Challenge)

**Status date: 2026-07-11. Operator directive: the only goal is to win; beta and
final freeze dates remain hard operational constraints.** This document covers the submission *mechanics*: the organizer
format, the verified package, and the rebuild gate. The active plan for winning
score improvements is **`docs/WINNING_PLAN.md`** — the package documented here
is the safe floor that campaign work must never regress.

---

## 1. What changed since the repo was last touched (2026-06-01)

The organizers published three documents (2026-06-16/18) that **change the submission format**:

| Doc | Requirement |
|---|---|
| Submission Guidelines (06-16) | Single `.tar.gz` with an **optimizer executable** (PyInstaller recommended), requirements.txt, helper files, optional README. Source code accepted only as fallback. **No Docker.** |
| Q&A (06-18), Q7–Q11 | Official evaluation runs `python iccad2026_evaluate.py --evaluate op_wrapper.py`; the organizers' `op_wrapper.py` **spawns the submitted executable once per test case** (JSON over stdin/stdout, `subprocess.run(..., timeout=60)`). I/O schema = `solve()` schema of `optimizer_template.py`. |
| Q&A (06-18), Q2/Q3 | Eval host: Debian 13, Python 3.13.14, PyTorch 2.12.0+cu130, 48-core Icelake Xeon, A100 80GB, **128 GB RAM** (guidelines PDF says 250 GB — assume 128), **no internet**, cases evaluated **sequentially**, multiprocessing allowed within a case. |

**Strategic consequence:** runtime is measured around `solve()` — for an executable
submission that includes **process spawn + imports + JSON I/O every case**. A binary
that bundles torch pays seconds of import per case and forfeits the 0.7× runtime
floor that v9's whole strategy banks on. The package therefore had to be torch-free.

## 2. Verification history (through 2026-07-11)

| Check | Result |
|---|---|
| Official evaluation of `my_optimizer.py` (100 validation cases) | **2.718225, 100/100 feasible** — every per-case cost **bit-identical** to `results/v9_locked.json`, despite different hardware → determinism confirmed |
| Test suite / result audit / release gate | 167/167 pass / PASS / PASS |
| Golden layouts scored by the official evaluator | feasible 100/100, but **90/100 have soft violations** (mean V_rel 0.051, max 0.148) → golden itself scores **1.1079** @RF=1 |
| Retrieval hypothesis (are validation instances in the 1M training set?) | **All 1,008,000 training samples scanned: 0 hits** (area-multiset signature). Train/val disjoint ⇒ hidden test almost certainly disjoint ⇒ golden **cannot be retrieved**, only computed |
| Torch-free port equivalence (python shim via op_wrapper) | 2.718225, 100/100, **0/100 position differences** vs in-process run |
| **Packaged binary** via the official command (`--evaluate op_wrapper.py`) | **2.718225, 100/100, 0 position differences**; runtime avg 0.244s/case, n≥100 avg 0.475s, max 0.759s (79× headroom under the 60s wrapper timeout) |
| Binary fuzz on 400 random training instances (exact wrapper protocol) | **400/400 hard-feasible**, avg 0.205s, max 0.705s |
| **Re-verified after CAMPAIGN_GOLDEN G4 engine** (2026-07-07): wrapper + rebuilt binary | **2.120411, 100/100, 0 position diffs** vs in-process; avg 0.33s/case incl. spawn (max 0.96s); fuzz 400/400 feasible (max 1.15s) |
| **Historical (2026-07-09, integrated v29)**: wrapper + rebuilt binary | **1.618110, 100/100, 0 position diffs** vs `results/integrated_v29.json`; avg 0.218s/case incl. spawn (p95 0.407s, max 0.628s); fuzz 400/400 feasible |
| **CURRENT (2026-07-11, integrated v32)** | **1.615379, 100/100**; AMD64 Debian 13/Python 3.13 package has exact official-wrapper parity (all 28,200 position scalars and every quality metric match) |

**2026-07-10 release blocker fixed:** the earlier archive was built
on an ARM64 host even though the evaluation machine is an Intel Xeon.
`op_wrapper.py` does not fall back after an executable-format error, so that
archive was not submittable. `build_submission.sh` now builds through a
digest-pinned AMD64 Debian 13/Python 3.13 container on every host by default
and rejects any
artifact whose ELF machine is not AMD x86-64. The current archive passes the
guard; only a package produced by this build may be uploaded.
Current hardened archive SHA-256:
`72d8fc5b6c4831a6af3547bacc16f19c800f1991b413500efbe467db8aec72c3`.
The exact archive is preserved as the `iccad2026_submission.tar.gz` asset on
the GitHub pre-release
[`v32-prebeta-20260711`](https://github.com/Lawrence-eth/opc-eda-2026/releases/tag/v32-prebeta-20260711).
This matters because `submission/` is generated and intentionally gitignored;
a fresh clone must download that asset (or rebuild it) before running the
manifest-bound release gate.

Fixed during packaging: `torch_stub.py` originally aliased `float`, shadowing the
builtin inside the stub (all cases silently fell back → score 9.98). Caught by the
100-case equivalence gate; the gate is mandatory before any future rebuild.

## 3. The submission package (built + verified)

```
packaging/                  (tracked sources)
├── solver_main.py          executable entry: JSON stdin → solve() → JSON stdout;
│                           crash-proof always-feasible fallback layout
│                           (bundle also carries contest_solution/dissect.py)
├── torch_stub.py           list-backed mini-Tensor (exact ops my_optimizer uses)
├── eval_stub.py            FloorplanOptimizer base + 3 metric helpers (verbatim, torch-free)
├── op_wrapper.py           organizers' wrapper, verbatim (the interface contract)
├── README_SUBMISSION.md    package README for the organizers
└── build_submission.sh     one-shot build → submission/iccad2026_submission.tar.gz

submission/ (generated, gitignored)
├── dist/my_optimizer/      PyInstaller --onedir, 53 MB, NO real torch
└── iccad2026_submission.tar.gz   (23 MB: dist + op_wrapper + README + source fallback)
```

Design decisions:
- **`--onedir`** (not `--onefile`): no per-case self-extraction; measured spawn ≈0.11s.
- **Torch-free**: op_wrapper delivers plain JSON lists; `my_optimizer.py` is used
  **unmodified** — stubs satisfy its imports. Equivalence proven, not assumed.
- **Never crash**: any internal failure emits a conservative feasible layout
  (preplaced exact, fixed dims exact, sqrt-area squares in disjoint shelf rows).
  One crashed case = cost 10 ≈ +0.25 weighted on a big case; the fallback costs ≈9.6
  only on that case but stays feasible.
- **Source fallback** included in the tar.gz per guidelines (pure stdlib Python).

## 4. Answer: "is the best solution possible to be found?"

**A best solution exists mathematically, but a proof of the global optimum for
the full 120-block mixed discrete/nonlinear problem is not realistic inside the
contest runtime. A winning solution is still a credible engineering goal; it
must be established against the leaderboard, not assumed from the public set.**

- **Score floor**: quality gaps are clamped at 0 vs golden baselines, so perfect play =
  match golden HPWL/area, zero-out soft violations where possible, hit the runtime floor.
  Golden itself carries violations (90/100 cases), so realistic "perfect" =
  **≈0.776** (golden-equivalent at the floor); the absolute bound is 0.70.
- **Retrieval is dead**: 1M-sample scan, zero overlap with validation.
  Reverse-engineering the generator is explicitly disqualifying (spec footnote 6).
- **Computing golden-quality layouts**: the repo's 8-approach history (shelf/skyline,
  portfolio, ENGINE local search, SP-SA ×5 variants, ML POC) all failed to beat v9's
  runtime-adjusted score; best util 0.74 vs golden 0.97 always at the cost of the
  cluster-abutment rule. Closing that gap is an industrial-solver build, not a 6-day task.
- **Leverage table** (weighted, from v9's verified per-case results):

| Scenario | RF=1 | at 0.7 floor |
|---|---|---|
| v9 as-is | 2.718 | **1.903** |
| v9 + area_gap→0 | 2.077 | 1.454 |
| v9 + hpwl_gap→0 | 1.887 | 1.321 |
| v9 + V_rel→0 | 2.180 | 1.526 |
| both gaps→0 (keep V_rel) | 1.246 | 0.872 |
| golden-equivalent | 1.108 | 0.776 |
| theoretical bound | 1.000 | 0.700 |

**Decision (updated per operator directive):** the verified, feasibility-proof,
runtime-floor entry is **banked as the rollback floor**; the pursuit of
golden-quality scores continues under **`docs/WINNING_PLAN.md`**, with the
dissection engine as one safe decoder. Promotion depends on deployable selector
regret and runtime as well as offline candidate headroom.

## 5. Path to submission

**P0 — submission logistics (owner: Lawrence; needed from the contest account)**
1. Confirm the upload channel and team credentials on the ICCAD contest site
   (http://iccad-contest.org); track any format updates in the FloorSet repo/Q&A.
2. When submitting: upload the current `submission/iccad2026_submission.tar.gz`
   (rebuild via §7 first if the optimizer changed). After upload, re-download and
   re-run the package once (`--evaluate op_wrapper.py`) as a checksum-level sanity pass.

**P1 — final hardening (done unless noted)**
1. ✅ v32 package built + 100-case target-arch equivalence verified through the
   organizers' wrapper. The v31 target binary also passed 100/100 random
   training instances under QEMU; timings are nonrepresentative. The latest
   400-case native binary fuzz is from v29, and v31's low-size path passed
   140/140 MIB-clean held-out cases.
2. ✅ Cross-hardware determinism (2-core VM reproduces 48-core results exactly).
3. ✅ Build is now guarded to AMD x86-64 on Debian 13/Python 3.13 and rejects
   a wrong-architecture ELF before packaging.
4. Optional: run `scripts/fuzz_binary.py` on an AMD64 host (or through a
   configured QEMU launcher here) for 2,000+ instances overnight.

**P2 — quality improvements (ACTIVE)**
- Run as `docs/WINNING_PLAN.md`. Every candidate must pass the sealed
  generalization, selection-regret, tail-risk, feasibility, and runtime-scenario
  gates before the §7 rebuild gate. Historical dead ends bind the exact failed
  implementations, not materially different learned or structured methods.

**P3 — presentation** — `docs/SUMMARY.md` is current; refresh from the campaign's
leverage table and progress log when the engine lands.

## 6. Risk register

| Risk | Assessment | Mitigation |
|---|---|---|
| Field median runtime < 0.8s (floor not reached) | Unlikely: every executable submission pays spawn per case; ML entries pay torch/model-load per case (seconds) | Even at RF=1 exactly, 1.6154 with 100/100 is a sound entry |
| Hidden-set infeasibility | Historical 5,000-instance solve() fuzz + v29 400-instance native binary fuzz had 0 failures; v32 has exact wrapper parity and 1,050/1,050 feasible tournament evaluations; v31 also has 100/100 target-binary QEMU fuzz and a 140/140 clean holdout; crash-proof fallback remains | Run a larger current-engine fuzz natively before final upload |
| Eval-host incompatibility (glibc/arch) | Prior ARM64 artifact was invalid; build now targets AMD x86-64 on Debian 13 and verifies the ELF header | Never upload an archive that has not passed the guarded build and target-host smoke test |
| Organizers run source instead of binary | Source fallback in package is the same solver, stdlib-only | README documents both paths |
| Wrapper protocol drift (organizers change op_wrapper) | We ship their exact wrapper + a schema-conform binary | Monitor the FloorSet repo/Q&A until deadline |

## 7. Rebuild & re-verify (the gate that must pass before ANY resubmission)

To restore and audit the unchanged frozen package in a fresh clone:

```bash
mkdir -p submission
gh release download v32-prebeta-20260711 --pattern iccad2026_submission.tar.gz \
    --dir submission
echo "72d8fc5b6c4831a6af3547bacc16f19c800f1991b413500efbe467db8aec72c3  submission/iccad2026_submission.tar.gz" \
    | sha256sum --check -
python3 scripts/check_public_release.py
python3 scripts/audit_submission_package.py \
  --release-manifest results/release_manifest.json \
  --require-notices --smoke
```

After any optimizer change, build and prove the new package instead:

```bash
bash packaging/build_submission.sh
cp -r submission/dist packaging/op_wrapper.py external/FloorSet/iccad2026contest/
cd external/FloorSet/iccad2026contest
PYTHONPATH=.. ../../../.venv/bin/python iccad2026_evaluate.py --evaluate op_wrapper.py \
    --output ../../../results/wrapper_check.json
# REQUIRED: Total Score 1.615379, Feasible 100, and 0 position diffs vs results/integrated_v32.json
cd ../../..
# On an AMD64 host (or with --binary set to a configured x86 launcher):
.venv/bin/python scripts/fuzz_binary.py --num 400   # REQUIRED: 0 failures
```

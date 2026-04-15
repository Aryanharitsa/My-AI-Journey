# Gold-10 v3 — Complete Experiment Report

**Date:** April 13, 2026
**Model:** openai/gpt-oss-120b (MXFP4, vLLM 0.19.0)
**Hardware:** 1× NVIDIA A100-SXM4-80GB (RunPod)
**vLLM:** `--enforce-eager --kv-cache-dtype auto --max-model-len 65536 --served-model-name openai/gpt-oss-120b`
**Solver:** AIMO3Solver44 (44/50-score reference solver) with entropy-weighted voting, N=8, early stop at 4/8 consensus

---

## 1. Problem Set

10 problems designed to target GPT-OSS-120B failure modes. Answers independently TIR-verified before scoring.

| ID | Problem Summary | Domain | Tier | Expected | Contrastive | LeaP |
|---|---|---|---|---|---|---|
| P1 | Sophie 4×4 grid paths (4 move types, no revisit) | combinatorics | medium | 147456 | No | No |
| P2 | n! divisors with non-decreasing consecutive gaps, n∈[3,200] | number_theory | medium | 2 | No | No |
| P3 | Partition {1..12} into 4 groups of 3, no consecutive in group | combinatorics | tough | 1721 | No | No |
| P4 | f(f(f(n)))=n+3, strictly increasing, sum f(1)+...+f(30) | algebra | tough | 495 | Yes | No |
| P5 | Non-touching rectangle pairs in 6×6 grid | combinatorics | tough | 28420 | No | No |
| P6 | Largest prime p<10000: non-Wieferich AND non-Mirimanoff | number_theory | tough | 9973 | Yes | No |
| P7 | Routh's theorem: trisectors, 21·X/Y | geometry | tough | 3 | Yes | No |
| P8 | Circular arrangements of 1-8, adjacent products ≤ 25 | combinatorics | tough | 192 | Yes | No |
| P9 | Snake merge on circle, 20 snakes, compute 3E | combinatorics | wall | 798 | Yes | Yes |
| P10 | Hamiltonian paths in 4×4 grid | combinatorics | wall | 552 | No | Yes |

---

## 2. Master Scorecard

| Problem | Tier | Expected | **B0 Baseline** | **B1 PACER** | **B3 GenSelect** | **B4 LeaP** |
|---|---|---|---|---|---|---|
| P1 | medium | 147456 | **N** pred=0 | **N** | **N** | **N** (copy) |
| P2 | medium | 2 | **Y** pred=2 | **Y** | **Y** gs=2 | **Y** (copy) |
| P3 | tough | 1721 | **Y** pred=1721 | **Y** | **Y** gs=1721 | **Y** (copy) |
| P4 | tough | 495 | **Y** pred=495 | **Y** | **Y** gs=495 | **Y** (copy) |
| P5 | tough | 28420 | **Y** pred=28420 | **Y** | **Y** gs=28420 | **Y** (copy) |
| P6 | tough | 9973 | **Y** pred=9973 | **Y** | **Y** gs=9973 | **Y** (copy) |
| P7 | tough | 3 | **Y** pred=3 | **Y** | **Y** gs=3 | **Y** (copy) |
| P8 | tough | 192 | **Y** pred=192 | **Y** | **Y** gs=192 | **Y** (copy) |
| P9 | wall | 798 | **Y** pred=798 | **Y** | **Y** (copy) | **N** (DAMAGED) |
| P10 | wall | 552 | **N** pred=276 | **N** | **N** gs=276 | **N** |
| **TOTAL** | | | **8/10** | **8/10** | **8/10** | **7/10** |

---

## 3. Batch 0 — Baseline: Per-Sample Detail

### B0 Candidate Answers (all 8 attempts per problem)

| Problem | Expected | Selected | Vote Distribution | Early Stop | Time |
|---|---|---|---|---|---|
| P1 | 147456 | 0 | **{}** — 0/8 produced any answer | N/A | 751s |
| P2 | 2 | 2 | {2: 4} | Yes (4/8) | 800s |
| P3 | 1721 | 1721 | {1721: 4} | Yes (4/8) | 104s |
| P4 | 495 | 495 | {495: 4} | Yes (4/8) | 166s |
| P5 | 28420 | 28420 | {28420: 4} | Yes (4/8) | 193s |
| P6 | 9973 | 9973 | {9973: 4} | Yes (4/8) | 70s |
| P7 | 3 | 3 | **{3: 4, 1470: 1}** — split vote | Yes (4/8) | 848s |
| P8 | 192 | 192 | {192: 4} | Yes (4/8) | 232s |
| P9 | 798 | 798 | **{798: 1}** — only 1 sample correct | Yes (trivially) | 901s |
| P10 | 552 | 276 | **{276: 4, 552: 3}** — correct answer in pool, lost vote | Yes (4/8 on wrong) | 365s |

**Total B0 wall time:** 4430s (73.8 min)

### B0 Per-Sample Detail — Key Problems

**P1 (FAIL — generation miss):**
All 8 samples ran to budget (751s) without producing a `\boxed{}` answer. The problem requires DFS enumeration of paths on a 4×4 grid with 4 move types — the model attempted computation but couldn't complete it within the context window.

**P7 (PASS — split vote):**
- 4 samples → 3 (correct, Routh's theorem via coordinates)
- 1 sample → 1470 (wrong, likely confused Routh with a different formula)
- 3 samples → early-stopped before answering
- Early stop triggered at 4/8 on answer 3.

**P9 (PASS — razor-thin margin):**
- 1 sample → 798 (correct, recognized the snake merge as random walk coalescence)
- 7 samples → None (failed to produce answer)
- The single correct sample won by default (only answer in pool).
- **This is the most fragile correct answer in the set.** On another run, 0/8 might get it.

**P10 (FAIL — early stop suppressed correct answer):**
- 4 samples → 276 (wrong — likely counted Hamiltonian paths starting from one corner only, or divided by 2)
- 3 samples → 552 (correct — full count including all starting vertices)
- 1 sample → early-stopped before answering
- Early stop triggered at 4/8 on 276. **The 3 correct samples arrived too late.**
- With N=8 and early_stop=4, the first answer to reach 4 votes wins. The wrong answer happened to converge first.

---

## 4. Batch 1 — PACER: Revision Layer

PACER fires when: vote margin < 4, OR top answer ∈ {0,1,2,3}, OR mean entropy > 2.0.

| Problem | PACER Fired? | Reason | Pre-PACER | Post-PACER | Rescue? | Damage? |
|---|---|---|---|---|---|---|
| P1 | Yes | No valid answers to revise | N (0) | N (0) | — | — |
| P2 | Yes | Entropy triggered | Y (2) | Y (2) | No | No |
| P3 | No | 4/4 consensus | Y | Y | — | — |
| P4 | No | 4/4 consensus | Y | Y | — | — |
| P5 | No | 4/4 consensus | Y | Y | — | — |
| P6 | No | 4/4 consensus | Y | Y | — | — |
| P7 | Yes | Split vote {3:4, 1470:1} | Y (3) | Y (3) | No | No |
| P8 | No | 4/4 consensus | Y | Y | — | — |
| P9 | Yes | Only 1 valid answer | Y (798) | Y (798) | No | No |
| **P10** | **No** | **4/4 consensus on 276** | **N (276)** | **N (276)** | **—** | **—** |

**Critical finding:** PACER did not fire on P10 because early stopping produced a 4/4 consensus on 276. From PACER's perspective, the vote was decisive. The 3 correct samples (552) that were suppressed by early stopping were invisible to PACER.

**PACER revision outcomes (when fired):**
- P2: All 4 samples confirmed their answer (2). No flips.
- P7: 4 samples confirmed 3, 1 sample confirmed 1470. No flips.
- P9: 1 sample confirmed 798. No flips.

**Verdict:** PACER never disagrees with the solver's consensus. It confirms, it doesn't challenge. **0 rescues, 0 damages.**

---

## 5. Batch 3 — GenSelect: Comparative Selection

GenSelect shows all 8 candidates to the model and asks "which is correct?" — repeated 16 times with permuted candidate ordering.

| Problem | B0 Selected | GenSelect Selected | 16 Trials Agreement | Rescue? | Damage? |
|---|---|---|---|---|---|
| P1 | 0 | 0 (skipped, <2 candidates) | — | — | — |
| P2 | 2 | 2 | unanimous | No | No |
| P3 | 1721 | 1721 | unanimous | No | No |
| P4 | 495 | 495 | unanimous | No | No |
| P5 | 28420 | 28420 | unanimous | No | No |
| P6 | 9973 | 9973 | unanimous | No | No |
| P7 | 3 | 3 | unanimous | No | No |
| P8 | 192 | 192 | unanimous | No | No |
| P9 | 798 | 798 (skipped, <2 candidates) | — | — | — |
| **P10** | **276** | **276** | **16/16 picked 276** | **No** | **No** |

**P10 GenSelect analysis:** All 16 selection trials, across different candidate orderings, picked 276 over 552. The model CANNOT distinguish the correct answer from the wrong one by reasoning about the candidates. Both 276 and 552 are plausible (552 = 2×276), and the model's "judge" persona agrees with the wrong majority.

**P9 was skipped** because only 1 candidate had a valid answer — GenSelect needs ≥2 to compare.

**Verdict:** GenSelect is a rubber stamp for B0's majority vote. **0 rescues, 0 damages.**

---

## 6. Batch 4 — LeaP: Peer Exchange (P9, P10 only)

LeaP runs 3 rounds of generation with peer summary exchange between attempts. Other problems copy B0.

| Problem | B0 | LeaP Rounds | LeaP Result | Change |
|---|---|---|---|---|
| P9 | Y (798, 1/8) | 3 | **N** | **DAMAGED** — peer exchange killed the lone correct sample |
| P10 | N (276, 4/8) | 2 | **N** | Still wrong |

**P9 LeaP failure mechanism:** In B0, one sample found 798 while 7 failed. In LeaP, after round 1, the 7 failing samples shared peer summaries showing no answer or wrong approaches. The one correct sample was overwhelmed by the wrong peer context in rounds 2-3 and either changed its answer or failed to produce one.

**P10 LeaP:** 2 rounds, still converged on the wrong answer. Peer exchange reinforced 276.

**Verdict:** LeaP is net negative (−1). Peer exchange amplifies the majority opinion, which hurts when the majority is wrong. **0 rescues, 1 damage.**

---

## 7. Cross-Batch Comparison Matrix

| Problem | B0 | B1 PACER | B3 GenSelect | B4 LeaP | Best |
|---|---|---|---|---|---|
| P1 | N | N | N | N | All fail |
| P2 | Y | Y | Y | Y | All agree |
| P3 | Y | Y | Y | Y | All agree |
| P4 | Y | Y | Y | Y | All agree |
| P5 | Y | Y | Y | Y | All agree |
| P6 | Y | Y | Y | Y | All agree |
| P7 | Y | Y | Y | Y | All agree |
| P8 | Y | Y | Y | Y | All agree |
| P9 | Y | Y | Y | **N** | B0/B1/B3 |
| P10 | N | N | N | N | All fail |
| **Score** | **8** | **8** | **8** | **7** | **8** |

**Net deltas vs B0:**
- B1 PACER: +0
- B3 GenSelect: +0
- B4 LeaP: **−1**

---

## 8. Failure Class Analysis

### F5: Generation Miss (P1)
All 8 samples failed to produce any answer. The problem requires exhaustive DFS enumeration of paths on a 4×4 grid with 4 non-standard move types. The model attempts code but the state space is too large for the Jupyter timeout (6s per execution). **No post-processing technique can fix this** — the answer never enters the candidate pool.

### F1: Attractor Trap (P10)
4/8 samples converge on 276 (wrong), 3/8 find 552 (correct). The wrong answer is a natural "half-count" trap — counting Hamiltonian paths from one endpoint instead of all endpoints, or dividing by a symmetry that doesn't apply. Early stopping at 4/8 locks in 276 before the 3 correct samples can shift the vote. Every post-processing technique (PACER, GenSelect, LeaP) reinforces the wrong majority.

### Thin Margin (P9)
Only 1/8 samples solves P9 correctly. The problem requires recognizing an abstract random walk coalescence structure. On another run, 0/8 might solve it — this is pure sampling variance. LeaP's peer exchange actively killed this fragile correct sample.

---

## 9. Stacking Recommendation

| Technique | Verdict | Reasoning |
|---|---|---|
| **B1 PACER** | **SKIP** | 0 rescues, 0 damages. Confirms consensus but never challenges it. |
| **B3 GenSelect** | **SKIP** | 0 rescues, 0 damages. Rubber-stamps majority vote on every problem including wrong ones. |
| **B4 LeaP** | **EXCLUDE** | 0 rescues, 1 damage. Peer exchange amplifies wrong majorities and kills fragile correct minorities. |
| **B0 Baseline** | **USE AS-IS** | The 44-score solver's entropy-weighted voting is already near-optimal for the techniques tested. |

---

## 10. Implications for Competition

### The P10 Lesson: Early Stopping Costs Accuracy
The correct answer (552) appeared in 3/8 samples but lost to 276 (4/8) because early stopping triggered first. **Disabling early stopping** on problems where the vote is close (e.g., 4-3 split) could rescue problems like P10. The tradeoff: more wall time per problem, fewer problems attempted in 5 hours.

### The P9 Lesson: More Samples Help, Post-Processing Doesn't
P9 was solved by 1/8 samples. At N=16 or N=32, the probability of at least one correct sample increases. This is the *only* lever that demonstrably helps on wall-tier problems: **more independent samples, not smarter aggregation.**

### The Fundamental Constraint
Every post-processing technique tested (PACER, GenSelect, LeaP) operates on the candidate pool produced by the solver. If the pool doesn't contain the correct answer (P1), or the wrong answer has a majority in the pool (P10), no aggregation method can fix it. The frontier is **generation quality and quantity**, not selection.

### Score Projection
- Gold-10 v3 baseline: 8/10
- Estimated public equivalence: ~40/50 (matching the arxiv paper's mean of 39.7)
- Maximum achievable with current solver: ~44/50 (arxiv paper's observed max)
- Gap to 47/50: requires solving 3 additional "generation miss" or "attractor trap" problems — no tested technique bridges this gap.

---

## 11. Artifacts

All experiment traces and logs were generated on RunPod pod (154.54.102.49:19425, now terminated).

**Local copies available:**
- `gold10/gold10_problems.jsonl` — v3 problem set (10 problems, TIR-verified)
- `gold10/gold10_runner.py` — 5-batch experiment runner (~800 lines)
- `gold10/gold10_analyze.py` — cross-batch analysis generator

**Pod-only (lost with pod termination):**
- `gold10_runs/batch0_baseline/` — full per-sample traces
- `gold10_runs/batch1_pacer/` — PACER revision traces
- `gold10_runs/batch3_genselect/` — 16-trial selection traces
- `gold10_runs/batch4_leap/` — peer exchange round traces
- `gold10_analysis.md` — auto-generated analysis

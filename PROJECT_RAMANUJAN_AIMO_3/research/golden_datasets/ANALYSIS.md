# Golden Datasets: Calibration Analysis

**Dates:** April 12-15, 2026
**Model:** GPT-OSS-120B (MXFP4, vLLM 0.19.0)
**Solver:** AIMO3Solver44, entropy-weighted voting, N=8, early stop at 4/8

---

## Purpose

A golden dataset is a small, curated problem set used to evaluate pipeline changes before submission. The problems need to be hard enough that the solver doesn't score 100%, but not so hard that every intervention looks equally useless. Building a well-calibrated golden dataset turned out to be its own research problem.

---

## Evolution

### Gold-10 v1: AIME-style problems

**10 problems. Score: 10/10.**

The first version drew from AIME 2025-2026 problems at the #11-15 difficulty level (the hardest problems on each paper). GPT-OSS-120B solved all of them with N=8 majority voting. Even the marginal problems (AIME i_15 at 3/8 correct samples, AIME ii_13 at 2/8) were saved by majority vote.

**Lesson:** AIME-hard is trivial for GPT-OSS-120B with TIR. The model's training distribution includes extensive AIME-level competition math, and tool use lets it verify combinatorial counts that pure reasoning might get wrong. This difficulty tier provides no discrimination between pipeline configurations.

### Gold-10 v2: Mixed difficulty

**10 problems. Score: 9/10.**

Added some IMO shortlist problems and non-standard competition problems. One problem was consistently missed, but the other 9 remained trivially solved.

**Lesson:** Still too easy. A 9/10 baseline means interventions can only show a +1 improvement at best, and a -1 regression is the only negative signal. The dynamic range is too narrow for meaningful ablation.

### Gold-10 v3: Failure-class targeting

**10 problems. Score: 8/10.**

Problems were selected to target specific GPT-OSS-120B failure modes identified during Reference10 evaluation:

| ID | Problem | Domain | Tier | Score |
|---|---|---|---|---|
| P1 | Sophie 4x4 grid paths | combinatorics | medium | FAIL (generation miss) |
| P2 | n! divisors with gaps | number theory | medium | PASS |
| P3 | Partition {1..12} no consecutive | combinatorics | tough | PASS |
| P4 | f(f(f(n)))=n+3 functional | algebra | tough | PASS |
| P5 | Non-touching rectangle pairs 6x6 | combinatorics | tough | PASS |
| P6 | Largest non-Wieferich non-Mirimanoff prime | number theory | tough | PASS |
| P7 | Routh's theorem trisectors | geometry | tough | PASS (split vote 4:1) |
| P8 | Circular arrangement products | combinatorics | tough | PASS |
| P9 | Snake merge on circle | combinatorics | wall | PASS (1/8 margin) |
| P10 | Hamiltonian paths 4x4 | combinatorics | wall | FAIL (attractor trap) |

This version hit the target: 2 failures from different failure classes (generation miss on P1, F1 attractor trap on P10), 1 razor-thin pass (P9 with only 1/8 correct samples), and 7 clean passes. The 8/10 score gave enough room for both positive and negative signals from interventions.

**Key finding from v3 experiments:** All four post-processing techniques tested (PACER, GenSelect, LeaP, baseline) scored either 8/10 or 7/10 on this set. LeaP was the only one to move the score, and it moved it downward (-1 by damaging P9). The v3 set was good for detecting damage but still couldn't show positive signal because the 2 failed problems were genuine walls.

### Gold-10 v4: Discriminative subset

**7 problems (trimmed from v3). Score: 5/7.**

Removed the 3 easiest problems (those with 4/4 early-stop consensus) to concentrate the set on problems where interventions had a chance of making a difference. The remaining 7 included 2 borderline problems, 3 tough problems with occasional split votes, and 2 wall problems.

5/7 was the best calibration achieved: close enough to 50% that both improvements and regressions would show clearly in the score. The 2 wall problems served as negative controls (nothing should improve them), while the borderline problems provided the signal.

---

## What the Progression Taught Us

The 4-version progression from 10/10 to 5/7 exposed a non-obvious truth about GPT-OSS-120B: the model's capability frontier is sharp. Problems are either well within its ability (8/8 consensus, no variance across runs) or firmly outside it (0/8, no intervention helps). The narrow band of borderline problems -- those in the 2/8 to 6/8 range where sampling luck and prompt engineering actually matter -- is smaller than expected.

For Reference10's 10 problems, the distribution was roughly 5 easy (>80% solve rate), 2 borderline (50% solve rate), and 3 wall (0% solve rate). Extrapolating to the public 50-problem set: approximately 15 easy, 20 medium, 10 hard, and 5 wall. The competition score is determined almost entirely by which of the 10 hard problems fall on the correct side of the borderline -- which is dominated by sampling variance, not inference-time tricks.

Building the golden dataset taught us where to focus: not on the walls (immovable) or the easy problems (already solved), but on the narrow borderline where contrastive prompting, sample count, and early-stopping thresholds can shift 1-2 problems.

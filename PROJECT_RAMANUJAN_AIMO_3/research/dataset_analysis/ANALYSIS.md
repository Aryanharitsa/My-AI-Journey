# Reference10 Dataset Analysis

**Model:** GPT-OSS-120B (MXFP4, vLLM 0.19.0)
**Solver:** AIMO3Solver44, N=8 majority vote with entropy weighting, early stop at 4/8
**Evaluation runs:** Multiple runs across April 2-15, 2026

---

## Overview

Reference10 is the official 10-problem calibration set used throughout Project Ramanujan development. Problems were drawn from the AIMO3 competition problem pool with known ground-truth answers. The set was used to validate every pipeline change before Kaggle submission.

---

## Per-Problem Breakdown

### Tier 1: Consistently Solved (>80% solve rate)

Five problems fall in this tier. The model solves them with N=8 majority voting on every run, typically with 4/8 or better consensus.

These problems span combinatorics, number theory, algebra, and geometry. They are hard by human competition standards (AIME #11-15 level and above), but GPT-OSS-120B's combination of reasoning capability and tool-integrated verification handles them reliably. The early-stop mechanism triggers at 4/8 on these problems, saving compute budget for harder problems.

These problems provide no diagnostic value for pipeline optimization. They serve as regression detectors -- if a pipeline change drops any of these, something is fundamentally broken.

### Tier 2: Borderline (50% solve rate)

**Problem 0e644e (answer: 336)**
Solved roughly half the time across runs. In the Diversity experiment with 16 samples, contrastive prompting produced 2/8 correct answers while normal sampling produced 0/4 in one batch, but normal sampling has produced correct answers in other runs. The problem sits right at the model's capability boundary -- it knows the relevant mathematics but doesn't reliably find the correct solution path.

**Problem a295e9 (answer: 520)**
Similar profile to 0e644e. Normal sampling gets 2/4 correct in some runs, contrastive gets 2/8. The solve rate varies between 25% and 50% per sample depending on the run. Majority vote at N=8 usually selects the correct answer when the pool contains it, but the pool sometimes contains zero correct samples.

These two problems are the intervention targets. Contrastive prompting, higher sample counts, and careful early-stopping thresholds can shift whether these problems land correct or incorrect on a given run. They represent the 2-point swing that separates a good submission from a great one.

### Tier 3: Knowledge Walls (0% solve rate)

**Problem 641659 (answer: 57,447)**
Fibonacci sequences combined with incircle tangent length geometry. Zero correct answers across all runs, all sample counts, all interventions. The model lacks the specific technique for connecting Fibonacci recurrences to geometric incircle properties. In 16-sample experiments, it produced zero candidate answers -- not wrong answers, but no answers at all. The model recognizes the problem is hard and runs out of context window trying to find an approach.

**Problem 86e8e5 (answer: 8,687)**
n-Norwegian numbers at scale 3^{2025!}. Zero solve rate. The problem requires number-theoretic machinery at a scale the model hasn't encountered in training. Multiple routing strategies were tested (tool-first forcing, direct witness search) without success. The model attempts modular arithmetic approaches but cannot handle the astronomical scale.

**Problem dd7f5e (answer: 160)**
Shifty functions -- an abstract functional algebra problem. Historical solve rate was 0% until the Diversity experiment, where a single normal sample (1 out of 16) produced the correct answer 160. This was a pure sampling lottery win at T=1.0 -- no intervention caused it. The problem sits at the extreme edge of the model's capability: the technique exists somewhere in its training distribution but surfaces in fewer than 1 in 16 samples.

The dd7f5e result is notable because it shows the wall is not perfectly binary. At N=16, the model produced 1 correct sample. At N=32, the expected count would be roughly 2. But at N=8 (the competition budget), the probability of getting at least one correct sample is only about 40%. In competition conditions, this problem is functionally a wall.

---

## Distribution Summary

| Tier | Count | Solve Rate | Intervention Response |
|---|---|---|---|
| Consistently solved | 5 | >80% per sample | None needed; regression detector only |
| Borderline | 2 | ~50% per sample | Contrastive prompting helps; sample count matters |
| Knowledge wall | 3 | 0-6% per sample | Nothing tested works |

---

## Extrapolation to Public-50

The Reference10 distribution maps approximately to the full 50-problem competition set:

| Difficulty | Ref10 Count | Estimated Public-50 Count |
|---|---|---|
| Easy (trivially solved) | 5 | ~15 |
| Medium (high solve rate) | 0 | ~20 |
| Hard (borderline) | 2 | ~10 |
| Wall (0% solve rate) | 3 | ~5 |

The out-of-box baseline scored ~20/50, which captures most of the easy problems and some medium problems with default sampling. The final v63 submission scored 36/50, capturing nearly all easy and medium problems plus some hard borderline problems. The gap between 36 and the theoretical ceiling of ~45 consists of borderline-to-hard problems where sampling variance dominates and wall problems where no current technique helps.

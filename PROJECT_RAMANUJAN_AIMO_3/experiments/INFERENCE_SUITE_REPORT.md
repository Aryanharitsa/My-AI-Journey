# Inference Suite Experiment Report

**Date:** April 10-12, 2026
**Model:** openai/gpt-oss-120b (MXFP4, vLLM 0.19.0)
**Hardware:** 1x NVIDIA A100-SXM4-80GB (RunPod)
**vLLM config:** `--enforce-eager --kv-cache-dtype auto --max-model-len 32768`

---

## 1. Experiment Overview

Three experiment rounds were conducted to evaluate inference-time interventions for improving math problem-solving accuracy:

| Experiment | Problems | Samples/Problem | Configs Tested | Total Attempts |
|---|---|---|---|---|
| **AIME Baseline** | 7 (AIME 2026 #11-15) | 8 | 1 (baseline only) | 56 |
| **Hard 4-Run** | 7 (IMO-level) | 8 | 4 (baseline, prefill, contrastive, combined) | 224 |
| **Diversity 16-Sample** | 8 (unsolved problems) | 16 | 3 sample types per batch | 128 |

**Interventions tested:**
- **Prefill seeding:** Injecting analysis-channel text to steer the model's reasoning trajectory
- **Contrastive prompting:** Developer prompt asking model to "consider both a correct approach and one common wrong approach"
- **Temperature diversification:** Mixed temperatures (0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2) across 8 samples
- **Combined (all three):** Prefill + contrastive + temperature diversification
- **Entropy-weighted voting:** `w = 1 + 1/(entropy + 0.1)` for answer aggregation
- **Extreme temperatures:** T=0.3 (cold) and T=1.5 (hot)
- **Forced verification:** "Solve twice with different methods, compare"
- **Meta-reasoning:** Ask model for 3 approaches + failure modes before solving

---

## 2. AIME Baseline (7 Problems, N=8)

**Result: 7/7 correct.**

| Problem | Domain | Expected | Predicted | Correct Samples | Time |
|---|---|---|---|---|---|
| aime_2026_i_11 | Combinatorics | 896 | 896 | 7/8 | 426s |
| aime_2026_i_12 | Geometry | 161 | 161 | 7/8 | 241s |
| aime_2026_i_13 | Number Theory | 39 | 39 | **8/8** | 293s |
| aime_2026_i_15 | Combinatorics | 83 | 83 | 3/8 | 504s |
| aime_2026_ii_12 | Geometry | 223 | 223 | **8/8** | 328s |
| aime_2026_ii_13 | Combinatorics | 107 | 107 | 2/8 | 583s |
| aime_2026_ii_15 | Number Theory | 393 | 393 | **8/8** | 60s |

**Key observations:**
- AIME #11-15 problems (the hardest on each paper) are fully solvable by gpt-oss-120b with N=8 and simple majority voting.
- Three problems had unanimous agreement (8/8). Two problems were marginal: `i_15` (3/8) and `ii_13` (2/8) — majority vote barely saved these.
- These problems were **too easy** to show differentiation between interventions, confirming the need for harder test problems.

---

## 3. Hard Problem 4-Run Experiment

### 3.1 Problem Set

Seven verified hard problems at IMO/TST level, requiring deep algorithmic knowledge:

| Problem | Domain | Answer | Difficulty |
|---|---|---|---|
| discordant_perms_12 | Combinatorics | 21,599,745 | Permanent of 12x12 matrix |
| tournament_kings | Graph Theory | 110,048 | Tournament enumeration (2^21) |
| petersen_surjective | Algebra | 7,232,400 | Chromatic polynomial + inclusion-exclusion |
| square_product_subsets | Number Theory | 2,047 | GF(2) linear algebra |
| node_kayles | Game Theory | 851 | Sprague-Grundy theory |
| lattice_coprime | Number Theory | 2,394 | Quadratic residues + CRT |
| discordant_perms_14_mod | Combinatorics | 89,847 | Permanent mod prime |

### 3.2 Results: 4 Configs x 7 Problems

**All four configs: 4/7 correct. Identical problem-level outcomes.**

| Problem | Baseline | Prefill | Contrastive | Combined |
|---|---|---|---|---|
| discordant_perms_12 | 0/8 | 0/8 | 0/8 | 0/8 |
| tournament_kings | 0/8 | 0/8 | 0/8 | 0/8 |
| petersen_surjective | 0/8 | 0/8 | 0/8 | 0/8 |
| **square_product_subsets** | **8/8** | **8/8** | **8/8** | **8/8** |
| **node_kayles** | **8/8** | **8/8** | **8/8** | **8/8** |
| **lattice_coprime** | **8/8** | **8/8** | **8/8** | **8/8** |
| **discordant_perms_14_mod** | **4/8** | **4/8** | **7/8** | **4/8** |

### 3.3 Per-Sample Analysis: discordant_perms_14_mod

This was the **only problem** showing differentiation between configs:

| Config | Correct Samples | Answers | Tools Used |
|---|---|---|---|
| Baseline | 4/8 | [89847, 96229, 89847, 11973, 89847, 31783, 88944, 89847] | 25 |
| Prefill | 4/8 | [59810, 89847, 39959, 41968, 54309, 89847, 89847, 89847] | 22 |
| **Contrastive** | **7/8** | [89847, 89847, 89847, 89847, 89847, 70036, 89847, 89847] | 36 |
| Combined | 4/8 | [89847, 89847, 8814, 89847, 39957, 89847, 29473, 40003] | 25 |

**Finding:** Contrastive prompting improved sample accuracy from 50% to 87.5% on this borderline problem. It used more tool rounds (36 vs 25), suggesting the "consider wrong approaches" prompt encouraged more verification compute.

### 3.4 Aggregate Metrics

| Metric | Baseline | Prefill | Contrastive | Combined |
|---|---|---|---|---|
| Correct | 4/7 | 4/7 | 4/7 | 4/7 |
| Pool correct | 4/7 | 4/7 | 4/7 | 4/7 |
| Selection accuracy | 100% | 100% | 100% | 100% |
| Avg reasoning tokens/sample | 2,749 | 2,540 | **3,180** | 2,990 |
| Avg time/problem | 184s | 196s | 202s | 201s |
| Unanimity (8/8 agree) | 3/7 | 3/7 | 3/7 | 3/7 |
| Wrong unanimity | 0/7 | 0/7 | 0/7 | 0/7 |

**Finding:** Contrastive uses ~15% more reasoning tokens than baseline. This is the cost of the "consider wrong approaches" instruction — more thinking, which helps on borderline problems but doesn't break through knowledge walls.

---

## 4. Diversity Experiment (16 Samples, Unsolved Problems)

### 4.1 Problem Set

Eight problems the model has never or rarely solved, drawn from ref10 and the hard suite:

| Problem | Source | Expected | Historical Solve Rate |
|---|---|---|---|
| 0e644e | ref10 | 336 | ~50% (sometimes) |
| 641659 | ref10 | 57,447 | 0% (never) |
| 86e8e5 | ref10 | 8,687 | 0% (never) |
| a295e9 | ref10 | 520 | ~50% (sometimes) |
| dd7f5e | ref10 | 160 | 0% (never) |
| hard_discordant_perms_12 | hard suite | 21,599,745 | 0% (never) |
| hard_tournament_kings | hard suite | 110,048 | 0% (never) |
| hard_petersen_surjective | hard suite | 7,232,400 | 0% (never) |

### 4.2 Sample Split (16 per problem)

| Indices | Type | Config |
|---|---|---|
| 0-3 | **Normal** | Vanilla T=1.0, no tricks |
| 4-11 | **Contrastive** | "Consider wrong approaches" T=1.0 |
| 12 | **Wild: extreme_hot** | T=1.5, "unconventional approach" |
| 13 | **Wild: extreme_cold** | T=0.3, "extreme precision" |
| 14 | **Wild: forced_verify** | T=1.0, "solve twice, compare" |
| 15 | **Wild: decompose_first** | T=1.0, structural prefill |

### 4.3 Batch 1 Results (Entropy-Weighted Voting)

| Problem | Expected | Majority Vote | Pool Hit? | Unique Answers | Which Type Worked? |
|---|---|---|---|---|---|
| **0e644e** | 336 | **336** | **YES** | 1 | Contrastive only (2/8) |
| 641659 | 57,447 | None | NO | 0 | Nothing (0/16) |
| 86e8e5 | 8,687 | None | NO | 0 | Nothing (0/16) |
| **a295e9** | 520 | **520** | **YES** | 1 | Normal (2/4) + Contrastive (2/8) |
| **dd7f5e** | 160 | **160** | **YES** | 1 | **Normal only (1/4)** |
| hard_discordant_perms_12 | 21,599,745 | None | NO | 0 | Nothing (0/16) |
| hard_tournament_kings | 110,048 | None | NO | 0 | Nothing (0/16) |
| hard_petersen_surjective | 7,232,400 | None | NO | 0 | Nothing (0/16) |

**Score: 3/8 correct.**

### 4.4 Per-Type Breakdown (All 8 Problems)

| Sample Type | Total Correct Answers | Total Samples | Hit Rate |
|---|---|---|---|
| **Normal** | 5 | 32 | 15.6% |
| **Contrastive** | 4 | 64 | 6.3% |
| Wild: extreme_hot (T=1.5) | 0 | 8 | 0% |
| Wild: extreme_cold (T=0.3) | 0 | 8 | 0% |
| Wild: forced_verify | 0 | 8 | 0% |
| Wild: decompose_first | 0 | 8 | 0% |

### 4.5 The dd7f5e Breakthrough

Problem `dd7f5e` (shifty functions, answer=160) had a **0% historical solve rate** across all prior runs. In this experiment:

- **Normal sample #2** produced `160` (correct) — the only correct answer out of 16 attempts
- All 8 contrastive samples: None
- All 4 wild samples: None

This is a pure **sampling lottery win**: at T=1.0 with enough samples, the model occasionally hits the right reasoning path. No intervention caused this — it was vanilla sampling diversity.

### 4.6 Batch 3 (Meta-Reasoning) — Partial

One problem completed before cancellation:

| Problem | Result | Meta-Reasoning Output |
|---|---|---|
| 0e644e | **MISS** (0/16) | 7,534 chars of approach analysis generated but inference failed |

The meta-reasoning step generated a detailed analysis of 3 approaches (coordinate geometry, trigonometric, algebraic) with failure modes for each, but the actual solving samples produced no correct answers — worse than batch 1 where contrastive got 2/8. The overhead of the meta step appears to eat into the inference budget.

---

## 5. Key Findings

### 5.1 Contrastive Prompting

**Verdict: Marginal positive on borderline problems, negative on easy problems, zero effect on hard-wall problems.**

| Evidence | Direction | Magnitude |
|---|---|---|
| discordant_perms_14_mod: 4/8 → 7/8 correct samples | Positive | +37.5pp sample accuracy |
| 0e644e: only contrastive produced correct answers (2/8) | Positive | Enabled solve that normal missed |
| a295e9: contrastive 2/8 vs normal 2/4 (lower rate) | **Negative** | -25pp per-sample rate |
| dd7f5e: contrastive 0/8 vs normal 1/4 | **Negative** | Missed what normal caught |
| 5 hard-wall problems: 0/8 across all configs | Zero | No effect on knowledge gaps |
| Token cost: +15% more reasoning tokens | Cost | More compute per sample |

**Mechanism:** The "consider wrong approaches" instruction activates verification and comparison reasoning, which helps when the model can solve the problem but is error-prone. It hurts when the model needs to explore freely — the contrastive overhead constrains exploration.

### 5.2 Prefill Seeding

**Verdict: Ineffective. No positive signal across any experiment.**

- Hard 4-run: identical 4/7 to baseline on every problem
- Sample-level on discordant_perms_14_mod: 4/8 (same as baseline), with prefilled samples performing no better than non-prefilled
- On AIME i_11 (from the prefill run before it was killed): 1/8 correct vs baseline's 7/8 — prefill caused massive timeouts by steering toward brute-force exploration

**Root cause:** RL-trained reasoning models develop fixed reasoning templates (RAGEN-2 "template collapse"). Generic prefill seeds either get ignored or push the model off its learned optimal trajectory. Problem-specific prefills might work but can't be generated without prior knowledge of the answer.

### 5.3 Temperature Diversification

**Verdict: No measurable effect.**

The combined config included mixed temperatures (0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2) but scored identically to baseline on every problem. The diversity experiment's extreme temperatures (T=0.3 and T=1.5) produced **zero correct answers** across all problems.

**Root cause:** At T=1.0, gpt-oss-120b already exhibits sufficient stochastic diversity (pairwise error correlation = -0.258 per the arxiv paper). Lowering temperature reduces exploration. Raising it degrades answer quality.

### 5.4 Wild Techniques

**Verdict: Total failure. 0 correct answers across 32 attempts (4 types x 8 problems).**

| Technique | Correct | Assessment |
|---|---|---|
| T=1.5 "unconventional" | 0/8 | Too hot — degraded reasoning |
| T=0.3 "extreme precision" | 0/8 | Too cold — reduced exploration |
| "Solve twice, compare" | 0/8 | Doubled compute without benefit |
| Structural decomposition prefill | 0/8 | Same prefill failure mode |

### 5.5 Entropy-Weighted Voting

**Assessment: Could not be properly evaluated.** The vLLM `/v1/completions` endpoint does not return per-token log-probabilities in the format needed to compute entropy. The entropy-weighted vote fell back to majority voting with neutral weights. A proper evaluation requires either:
- Using the `/v1/chat/completions` endpoint with `logprobs=True`
- Or computing entropy from the Harmony token IDs directly

This remains the **highest-value unimplemented optimization**, based on the 43/50 public notebook that uses it.

### 5.6 Knowledge Wall Classification

Problems fall into three tiers with no overlap:

| Tier | Behavior | Examples | Intervention Effect |
|---|---|---|---|
| **Solvable** (8/8 unanimous) | Model knows the algorithm, code verifies | square_products, node_kayles, lattice_coprime, AIME i_13/ii_12/ii_15 | None needed |
| **Borderline** (2-7/8 correct) | Model sometimes finds the path | discordant_14_mod, AIME i_15/ii_13, 0e644e, a295e9 | **Contrastive helps here** |
| **Knowledge wall** (0/16) | Model cannot even produce a candidate answer | 641659, 86e8e5, tournament_kings, discordant_12, petersen | **Nothing helps** |

---

## 6. Implications for Competition Strategy

### 6.1 Score Ceiling

Based on all experiments, the interventions tested can move scores by **at most +1 to +2 problems** (from borderline → correct via contrastive). Knowledge-wall problems are immovable without fundamentally new capabilities (fine-tuning, specialized tool libraries, or model changes).

This aligns with the arxiv paper finding: gpt-oss-120b mean = 39.7/50, max = 44/50 across 13 runs. The variance is dominated by which borderline problems happen to land correct via sampling, not by inference-time tricks.

### 6.2 Recommended Submission Config

```
Temperature: 1.0
Samples: 8
Reasoning effort: high
TIR: enabled (model chooses when)
Contrastive prompt: ON (helps borderline, negligible cost)
Prefill: OFF (no benefit, risk of harm)
Temperature diversification: OFF (no benefit)
Voting: Simple majority (entropy needs logprobs integration)
Early stopping: 4/8 consensus → move to next problem
```

### 6.3 Version 56 Score (20/50) — Likely Causes

The submission scored 20/50 against a target of 35+. Possible root causes not yet diagnosed:
- **Budgeting:** 5-hour wall clock for 50 problems with N=8 = 360s/problem average. If early problems consumed disproportionate time, later problems got starved.
- **Throughput:** A100-SXM4-80GB with max-model-len=32768 can only handle ~8 concurrent requests. If the notebook used a larger context or lower GPU utilization, throughput drops.
- **Context truncation:** Our experiments used max-model-len=32768 (down from 65536) due to VRAM constraints. This may have caused reasoning truncation on harder problems.
- **Answer extraction:** If the finalization/extraction pipeline fails, correct reasoning produces `None` → fallback to 0.

---

## 7. Files and Artifacts

```
baseline_kaggle_submission_fix5_inference_suite/
├── test_suite_7.jsonl              # 7 AIME hard problems
├── test_suite_7_hard.jsonl         # 7 IMO-level problems
├── unsolved_8.jsonl                # 8 unsolved problems for diversity experiment
├── test_suite_runs_aime/           # AIME baseline results
├── test_suite_runs_hard/           # Hard 4-run results (baseline/prefill/contrastive/combined)
├── diversity_runs/                 # 16-sample diversity results
│   ├── entropy/                    # Batch 1: full 8 problems
│   ├── no_entropy/                 # Batch 2: 1 problem (cancelled)
│   └── meta/                       # Batch 3: 1 problem (cancelled)
├── analysis_hard.md                # Auto-generated analysis
├── decisions_hard.md               # Auto-generated decisions
├── vllm_server.log                 # vLLM server log
├── hard_experiment_output.log      # 4-run experiment log
└── diversity_output.log            # Diversity experiment log
```

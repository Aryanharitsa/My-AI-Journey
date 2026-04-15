# Prefill Seeding: Findings

**Experiment dates:** April 10-12, 2026
**Model:** GPT-OSS-120B (MXFP4, vLLM 0.19.0)
**Hardware:** 1x A100-SXM4-80GB

---

## Technique

Prefill seeding injects text into the analysis channel before the model begins its own reasoning. The idea: a short directive at the start of the chain-of-thought nudges the model toward a productive reasoning strategy without consuming a significant portion of the token budget.

Three seed types were tested across the Hard 4-Run experiment (7 IMO-level problems, N=8) and the Diversity experiment (8 unsolved problems, N=16 with mixed sample types).

---

## Seeds Tested

### Seed 1: "Try small cases"

A directive to build intuition by testing small values before attempting a general solution.

**Result: Catastrophic.** On AIME i_11 (combinatorics, answer 896), baseline scored 7/8 correct. The "try small cases" prefill dropped it to 1/8. The seed pushed the model into brute-force enumeration loops -- it started generating cases for n=1, n=2, n=3 and never reached the general formula before exhausting the context window. What should have been a 426-second problem turned into a timeout.

This is the worst-case failure mode for prefill: the seed is locally reasonable (small cases *are* a valid mathematical technique) but globally destructive (the model follows the directive so literally that it never reaches the actual solution).

### Seed 2: "Structural insight"

A directive to identify the mathematical structure before computing.

**Result: No effect.** On discordant_perms_14_mod (the only problem showing any intervention sensitivity), structural insight prefill scored 4/8 -- identical to baseline. Across all 7 Hard 4-Run problems, the prefill config matched baseline outcomes on every single problem.

The model appears to simply ignore this type of vague structural guidance. Its own reasoning template already includes structural analysis as an early step, so the seed is redundant.

### Seed 3: "Decompose first"

A directive to break the problem into subproblems before attempting a solution. Tested in the Diversity experiment as one of the "wild" sample types (1 sample per problem, T=1.0, structural decomposition prefill).

**Result: No effect.** Zero correct answers across 8 problems (0/8). The other wild techniques (T=1.5, T=0.3, forced verification) also scored 0/8, suggesting that all single-sample interventions were underpowered, but decomposition-first showed no qualitative difference in the reasoning traces.

---

## Why Prefill Fails on RL-Trained Models

Two lines of research explain the negative results:

### Template collapse (RAGEN-2)

RL-trained reasoning models develop fixed reasoning patterns through reinforcement learning. These templates are learned behaviors that consistently reach correct answers on the training distribution. External text injected into the reasoning stream either gets absorbed into the existing template (the model proceeds as if the seed wasn't there) or disrupts it (the model follows the injected directive literally, breaking its learned optimal trajectory).

On GPT-OSS-120B, the absorption case was dominant. The structural insight and decomposition seeds were simply ignored -- the model's reasoning traces looked functionally identical with and without the seed. The "try small cases" seed was an exception: it was specific enough to override the learned template, but the override was destructive because brute-force enumeration is the wrong strategy for most of these problems.

### First-token mode switching (steering vector research)

Research on steering vectors shows that the first few tokens of a reasoning chain act as a mode switch. The model commits to a broad reasoning strategy very early, and this commitment is robust to perturbation in later tokens. Generic seeds like "start by considering the structure" don't function as mode-switching tokens because they're too abstract -- they don't specify a mathematical strategy concretely enough to redirect the model's initial commitment.

The implication: only seeds that function as actual mode switches (specific enough to override the model's default strategy selection) would affect generation. But such seeds require knowing which mathematical approach is correct for the problem -- a chicken-and-egg problem.

---

## What Might Work (Not Implemented)

Community experiments with DeepSeek R1 reported success with **problem-specific seeds from a critic model**. The approach:

1. A separate (smaller) model analyzes the problem and identifies the likely key mathematical insight.
2. That insight is formatted as a prefill seed for the main solver.
3. The main solver receives a seed that functions as a genuine mode switch because it's mathematically specific.

This was not implemented in Project Ramanujan for three reasons: it requires a second model call per sample (latency cost), the critic model needs to be good enough to identify the right insight (reliability concern), and the approach was reported anecdotally without controlled experiments (evidence quality). It remains the most plausible path to making prefill seeding work on RL-trained reasoning models.

# Contrastive Prompting: Findings

**Experiment dates:** April 10-12, 2026
**Model:** GPT-OSS-120B (MXFP4, vLLM 0.19.0)
**Hardware:** 1x A100-SXM4-80GB

---

## Technique

A developer-level prompt addition asking the model to consider both correct and incorrect approaches before committing to a solution path. The contrastive instruction activates comparison and exclusion reasoning -- the model explicitly identifies what could go wrong, then solves with that awareness.

Tested across two experiment rounds: the Hard 4-Run (7 IMO-level problems, 4 configs, N=8 per config) and the Diversity experiment (8 unsolved problems, N=16 with mixed sample types).

---

## Results

### Hard 4-Run: discordant_perms_14_mod (the only differentiating problem)

| Config | Correct Samples | Sample Accuracy | Tool Rounds |
|---|---|---|---|
| Baseline | 4/8 | 50.0% | 25 |
| Prefill | 4/8 | 50.0% | 22 |
| **Contrastive** | **7/8** | **87.5%** | **36** |
| Combined | 4/8 | 50.0% | 25 |

Contrastive prompting improved sample accuracy from 50% to 87.5% on this single borderline problem. It used 44% more tool rounds (36 vs 25), meaning the model ran more verification code per sample. The remaining 6 problems showed zero differentiation -- 3 were unanimously correct across all configs, 3 were unanimously unsolvable.

### Diversity experiment: 8 unsolved problems, N=16

| Sample Type | Correct Answers | Total Samples | Hit Rate |
|---|---|---|---|
| Normal (T=1.0) | 5 | 32 | 15.6% |
| Contrastive (T=1.0) | 4 | 64 | 6.3% |
| Wild techniques | 0 | 32 | 0% |

At the aggregate level, contrastive had a *lower* per-sample hit rate than normal sampling (6.3% vs 15.6%). This is misleading -- the contrastive pool was 2x larger and targeted harder problems. On individual problems:

- **0e644e** (borderline, answer 336): Contrastive produced 2/8 correct answers; normal produced 0/4. Contrastive enabled a solve that normal sampling missed entirely.
- **a295e9** (borderline, answer 520): Normal got 2/4 (50%), contrastive got 2/8 (25%). Lower per-sample rate, but still contributed correct answers to the vote pool.
- **dd7f5e** (knowledge wall, answer 160): Normal got 1/4, contrastive got 0/8. The overhead of analyzing wrong approaches consumed tokens that normal sampling used for exploration.
- **5 hard-wall problems** (641659, 86e8e5, and 3 hard suite): 0/8 contrastive, 0/4 normal. Zero effect on knowledge gaps.

### Token cost

Contrastive prompting used approximately 15% more reasoning tokens per sample than baseline (3,180 vs 2,749 average across the Hard 4-Run). This is the computational cost of the "consider wrong approaches" instruction -- more thinking per sample, which means fewer samples in a fixed time budget if you're throughput-constrained.

---

## Mechanism

The contrastive instruction works through two channels:

1. **Error awareness.** By explicitly identifying common mistakes before solving, the model avoids the specific failure modes it enumerated. On discordant_perms_14_mod, baseline samples scattered across 6 different wrong answers (96229, 11973, 31783, 88944, 59810, 39959). Contrastive samples converged: 7/8 on the correct 89847, with only one outlier at 70036.

2. **Increased verification compute.** The 44% increase in tool rounds suggests the model runs more code to check its work after identifying what could go wrong. This is productive verification -- not redundant rechecking, but targeted validation against the failure modes it identified.

The technique fails when:
- The problem is a knowledge wall (the model can't identify the right approach to begin with, so contrastive analysis is vacuous).
- The model needs maximum exploration freedom (contrastive overhead constrains the reasoning budget, reducing the chance of stumbling onto a novel path -- as seen on dd7f5e where normal sampling's lucky hit beat contrastive's systematic approach).

---

## Recommendation

Use contrastive prompting selectively, not universally. Best applied to borderline problems where the model already has the mathematical capability but is error-prone. Not useful for problems the model either solves trivially (waste of tokens) or cannot solve at all (waste of tokens for a different reason). In competition settings, applying it to all problems is net neutral to slightly negative due to the token overhead.

# Selector Suite: Evolution of Post-Processing Techniques

**Experiment date:** April 13, 2026
**Model:** GPT-OSS-120B (MXFP4, vLLM 0.19.0)
**Framework:** Gold-10 v3 (10 problems), AIMO3Solver44, N=8, early stop at 4/8
**Evaluation:** Each technique tested as a separate batch on the same problem set

---

## Motivation

After stabilizing the baseline at 8/10 on Gold-10 v3, the question was whether post-generation answer selection could recover the 2 missed problems (P1: generation miss, P10: attractor trap). Four techniques were evaluated: PACER, GenSelect, LeaP, and DeepConf Online.

---

## Technique 1: PACER (Revision Layer)

**Mechanism:** After initial generation and voting, PACER asks the model to revise its answer if the vote is uncertain. Fires when: vote margin < 4, top answer is suspiciously small (0, 1, 2, or 3), or mean entropy > 2.0.

**Results:**

| Problem | PACER Fired? | Trigger | Pre-PACER | Post-PACER | Change |
|---|---|---|---|---|---|
| P1 | Yes | No valid answers | FAIL | FAIL | None |
| P2 | Yes | Entropy | PASS (2) | PASS (2) | None |
| P7 | Yes | Split vote (3:1) | PASS (3) | PASS (3) | None |
| P9 | Yes | Only 1 answer | PASS (798) | PASS (798) | None |
| P10 | **No** | 4/4 consensus on 276 | FAIL (276) | FAIL (276) | None |
| P3-P6, P8 | No | Clean 4/4 consensus | PASS | PASS | None |

**Score: 8/10 (same as baseline).**

PACER never disagreed with the solver's consensus. When it fired (4 out of 10 problems), every sample confirmed its original answer during revision. Zero flips, zero rescues, zero damages.

The critical failure: PACER didn't fire on P10 -- the one problem where revision might have helped. Early stopping produced a 4/4 consensus on 276 (wrong), which looked decisive to PACER's triggering logic. The 3 correct samples at 552 had been suppressed by early stopping and were invisible to the revision layer.

**Verdict:** Confirmation bias in algorithm form. PACER confirms what the solver already decided. It doesn't challenge, it rubber-stamps. 0 rescues, 0 damages.

---

## Technique 2: GenSelect (Comparative Selection)

**Mechanism:** Instead of majority voting, present the model with all candidate answers and their reasoning traces, then ask it to comparatively select the best answer. Tested with 16 permuted selection trials to control for order effects.

**Results:**

| Problem | Expected | Majority Vote | GenSelect Pick | Agree? |
|---|---|---|---|---|
| P1 | 147456 | None (0/8) | None | N/A |
| P2 | 2 | 2 | 2 | Yes |
| P3-P8 | (various) | Correct | Correct | Yes |
| P9 | 798 | 798 | 798 | Yes |
| **P10** | **552** | **276** | **276** | **Yes (both wrong)** |

**Score: 8/10 (same as baseline).**

Across all 16 permuted trials on every problem, GenSelect matched majority vote. It never selected a minority answer, even when that minority answer was correct. On P10, GenSelect consistently picked 276 over 552 -- the wrong answer's reasoning traces looked more complete and more numerous (5 samples vs 3), and the comparative selector weighted both quantity and quality of supporting evidence.

**Verdict:** Majority vote by another name. The model-as-judge cannot distinguish between plausible wrong answers and correct ones when both are computationally valid. All 16 trials rubber-stamped 276 (wrong) over 552 (correct) on P10.

---

## Technique 3: LeaP (Peer Exchange)

**Mechanism:** Run 3 rounds of peer summary sharing between parallel solution attempts. After each round, samples receive summaries of what other samples found, allowing cross-pollination of ideas and error correction.

**Results:**

| Problem | Expected | Baseline | LeaP | Change |
|---|---|---|---|---|
| P1-P8 | (various) | 8 correct | 8 correct | None |
| **P9** | **798** | **PASS (1/8 correct)** | **FAIL** | **-1 (DAMAGED)** |
| P10 | 552 | FAIL (276) | FAIL (276) | None |

**Score: 7/10 (baseline minus 1).**

LeaP actively damaged P9. In baseline, 1 out of 8 samples found the correct answer (798) while the other 7 failed to produce any answer. That single correct sample won by default (only answer in the pool). With LeaP's peer exchange, the 7 wrong/empty samples shared their (incorrect or empty) reasoning summaries with the 1 correct sample. Over 3 rounds, the correct sample's reasoning was contaminated by the majority's confusion. The final vote after peer exchange no longer contained the correct answer.

This is the consensus amplification failure mode. Peer exchange assumes that the majority has useful information to share. When the majority is wrong (or stuck), sharing their state doesn't help the minority -- it drowns it. The 1 correct sample's signal was overwhelmed by 7 peers' noise.

**Verdict:** Dangerous on problems where minority correctness exists. Net -1 vs baseline. The 7 wrong peers overwhelmed the 1 correct sample.

---

## Technique 4: DeepConf Online (Confidence Filtering)

**Mechanism:** Mid-generation termination of low-confidence traces based on sliding-window logprob analysis. Implemented per the paper (Fu et al., Meta AI, 2025): 16 warmup traces, 90th percentile threshold, 2048-token window. Timeline sampled every 512 tokens.

**Result:** On problems where the model had 95%+ consensus (P2 through P8 in Gold-10), every sample's confidence was high from early in generation -- termination never triggered. On knowledge-wall problems (P1, P10), the model was confidently wrong -- high-confidence but incorrect traces wouldn't be caught by confidence-based termination.

**Verdict:** Needs problems in the 50-80% solve range to show signal. Our reference set had too few problems in that intermediate range. The technique is theoretically sound but the problem distribution didn't exercise it. `ENABLE_DEEPCONF` stayed at `"0"`.

---

## Aggregate Conclusion

| Technique | Score | Delta | Rescues | Damages | Mechanism |
|---|---|---|---|---|---|
| Baseline | 8/10 | -- | -- | -- | Entropy-weighted majority vote |
| PACER | 8/10 | +0 | 0 | 0 | Confirms consensus, never challenges it |
| GenSelect | 8/10 | +0 | 0 | 0 | Reproduces majority vote with more compute |
| LeaP | 7/10 | **-1** | 0 | **1** | Amplifies wrong majority, drowns correct minority |
| DeepConf | 8/10 | +0 | 0 | 0 | Never triggered on this problem distribution |

Model-level generation quality dominates all post-processing techniques. The only lever that demonstrably helps on hard problems is more independent samples -- and that's a compute question, not an algorithm question. No post-processing technique improved over baseline. When the correct answer is in the sample pool, majority voting already selects it. When the correct answer is absent or in the minority, no tested selector can recover it.

The implication for competition strategy: invest compute budget in more samples and better generation (contrastive prompting, throughput optimization), not in post-processing selection layers.

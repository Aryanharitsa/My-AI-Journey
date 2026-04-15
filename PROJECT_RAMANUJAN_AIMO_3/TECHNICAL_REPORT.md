# Project Ramanujan: Entropy-Weighted Sampling and Tool-Integrated Reasoning for Competition Mathematics

**Aryan Haritsa**
April 2026

---

## Abstract

We describe Project Ramanujan, a system for solving competition-level mathematics problems under the constraints of the AIMO3 Kaggle competition. The system pairs GPT-OSS-120B (a 116.8B-parameter mixture-of-experts model with 5.1B active parameters) with a vLLM inference server, N=8 parallel sampling with tool-integrated reasoning, and an entropy-weighted voting scheme for answer selection. On the AIMO3 public leaderboard, the system achieves a score of 36/50, with stable performance on easy and medium-difficulty problems and meaningful but inconsistent gains on hard problems.

---

## 1. Problem and Competition Context

The AI Mathematical Olympiad Prize 3 (AIMO3) is a Kaggle competition requiring participants to solve 50 competition-level mathematics problems. Each problem admits a unique integer answer in the range [0, 99999]. Submissions run under a 5-hour wall-clock limit on NVIDIA H100 GPU infrastructure provided by Kaggle. The evaluation metric is simple: the count of exactly correct answers out of 50.

The competition setting imposes several non-trivial engineering constraints. The wall-clock budget means that inference throughput directly trades against sample count per problem. The integer-answer format eliminates partial credit and demands exact numerical extraction from model outputs. The problem difficulty spans roughly four tiers: straightforward competition problems solvable by most capable models, medium-difficulty problems requiring multi-step reasoning, hard problems where even strong models fail intermittently, and a small set of problems that resist all current approaches.

## 2. Base Model

We use GPT-OSS-120B, a 116.8B-parameter mixture-of-experts language model with 5.1B active parameters per forward pass, quantized to MXFP4 precision. The model was selected on the basis of empirical performance: internal evaluations and results reported in arXiv:2603.27844 indicate a roughly 17-percentage-point gap between GPT-OSS-120B and the next-best alternative on competition mathematics benchmarks.

The MoE architecture is critical for our throughput requirements. With only 5.1B active parameters per token, the model fits within the memory and compute envelope of the Kaggle H100 allocation while maintaining reasoning capability competitive with much larger dense models. MXFP4 quantization further reduces memory pressure, enabling the N=8 parallel sampling regime described below.

## 3. Inference Architecture

The inference stack is built on vLLM 0.19.0, serving GPT-OSS-120B through an OpenAI-compatible API endpoint. Key configuration choices:

- **Attention backend**: TRITON_ATTN (via `VLLM_ATTENTION_BACKEND` environment variable). The FlashAttention backend was also evaluated but TRITON_ATTN provided more stable behavior under our quantization and sequence-length regime. Backend naming conventions changed between vLLM 0.10.x and 0.11.x; we verified the enum values against the 0.19.0 source.
- **KV cache**: Automatic management (`gpu_memory_utilization` tuned to leave headroom for 8 concurrent sequences).
- **Eager execution**: Enabled (`enforce_eager=True`) to avoid CUDA graph compilation overhead on the heterogeneous sequence lengths produced by tool-integrated reasoning.
- **Streaming**: All completions are streamed, returning `token_ids` and `logprobs` incrementally. This enables both real-time answer detection and entropy computation without waiting for full sequence generation.

The client-side interface uses the Harmony protocol for message encoding. Harmony wraps the OpenAI chat completion format with structured `SystemContent` messages carrying `ReasoningEffort.HIGH`, tool configuration metadata, and preference prompts that bias the model toward step-by-step derivation and explicit numerical answers.

## 4. Sampling Strategy

Each problem is solved via N=8 independent samples drawn in parallel using a `ThreadPoolExecutor`. The sampling hyperparameters are:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 1.0 | Full diversity; no sharpening |
| top_p | 1.0 | Unrestricted nucleus |
| min_p | 0.02 | Filters only extremely low-probability tokens |
| N | 8 | Throughput-constrained maximum |

Each sample is assigned a deterministic seed for reproducibility. Seeds are derived from the problem index and sample index to ensure that re-runs on identical inputs produce identical outputs, which is essential for debugging and ablation.

**Early stopping.** If 4 out of 8 samples converge to the same answer (simple majority of the full sample set), generation for the remaining samples is terminated. This recovers wall-clock time on easy problems, which is reallocated to harder problems later in the sequence.

**Tool-integrated reasoning (TIR).** Each of the 8 parallel samples has access to a dedicated Jupyter kernel from a pool of 8 persistent kernels. The model can emit Python code blocks during generation, which are intercepted, executed in the corresponding kernel with a 6-second timeout, and the result is fed back into the generation context. The kernels maintain state across multiple code executions within a single sample, enabling iterative computation (e.g., building up a search over candidate values). TIR is the primary mechanism by which the system handles problems requiring exhaustive enumeration or numerical verification.

## 5. Answer Selection

### 5.1 Answer Detection

Answers are detected inline during streaming via `\boxed{}` pattern scanning. The detector maintains a 32-token lookback window (`search_tokens=32`) over the streamed token IDs and triggers on the closing brace of a `\boxed{...}` expression. A final-channel parser runs after generation completes to catch any answers the streaming detector may have missed (e.g., answers split across tool-call boundaries).

### 5.2 Entropy-Weighted Voting

Given N candidate answers, each with an associated entropy value, the final answer is selected by weighted plurality vote. The weight for sample i is:

```
w_i = 1 / max(H_i, 1e-9)
```

where H_i is the Shannon entropy computed from the top-5 token logprobs observed during the streaming generation of that sample's final answer tokens. Low-entropy answers (high model confidence) receive proportionally higher weight. The `1e-9` floor prevents division by zero for deterministic outputs.

This scheme outperforms unweighted majority voting on problems where a minority of samples produce the correct answer with high confidence while the majority produces incorrect answers with lower confidence. On our reference set, entropy-weighted voting recovered approximately 2 additional correct answers compared to unweighted majority.

## 6. Verification Experiments

We conducted several experiments aimed at improving accuracy beyond the base sampling-and-voting pipeline.

**Contrastive prompting.** We prepend a contrastive instruction to the system prompt that asks the model to explicitly consider and rule out common error modes before committing to an answer. On borderline problems (those where the base system answers correctly 30--70% of the time), contrastive prompting improved accuracy by +37.5 percentage points. On wall problems (base accuracy below 20%), the technique had zero effect. This suggests contrastive prompting helps the model avoid attractor traps on problems it can already partially solve, but does not expand the frontier of solvable problems.

**Prefill seeding.** We attempted to seed the model's reasoning by prefilling the assistant turn with a partial solution sketch. This was ineffective: the model's RL-trained generation template appears to collapse when the opening tokens deviate from its expected format, producing incoherent continuations.

**Temperature diversification.** We tested temperature schedules (e.g., drawing half the samples at T=0.7 and half at T=1.2) to increase answer diversity. No measurable effect on accuracy was observed. At N=8, the diversity from T=1.0 already appears sufficient, and the entropy-weighted voting scheme already handles confidence variation.

## 7. Post-Processing Techniques

We evaluated four post-processing techniques from the recent literature, applied after the initial voting stage.

**PACER.** A confidence-based rescue mechanism that re-evaluates low-confidence answers. In our setting, PACER produced 0 rescues across the full problem set. The entropy-weighted voting scheme already captures the signal PACER targets, leaving no low-confidence answers in a recoverable state.

**GenSelect.** A generative re-ranking method that asks the model to select among candidate answers. In practice, GenSelect consistently rubber-stamped the majority-vote winner, adding computational cost without changing any answers.

**LeaP (Learn-then-Apply Prompting).** A technique that extracts solution patterns from solved problems and applies them to unsolved ones. LeaP produced a net score of -1: it damaged one fragile correct answer (a problem where 4/8 samples were correct) without rescuing any incorrect answers. The pattern extraction step appears to introduce noise that disrupts borderline solutions.

**DeepConf Online.** Implemented following the published specification, DeepConf Online uses the model's own confidence calibration to decide whether to accept or revise an answer. Results were mixed: it corrected one answer and damaged another, for a net effect of approximately zero. The technique shows promise but requires careful threshold tuning that we did not complete within the competition timeline.

## 8. Results

The system achieves a public leaderboard score of **36/50** on the AIMO3 competition.

### Score Breakdown by Difficulty Tier

| Tier | Estimated Count | Estimated Accuracy | Contribution |
|------|---------------:|-------------------:|-------------:|
| Easy | ~15 | ~95% | ~14 |
| Medium | ~20 | ~75% | ~15 |
| Hard | ~10 | ~50% | ~5 |
| Wall | ~5 | ~20% | ~1--2 |

These estimates are derived from performance on our reference-10 and gold-10 evaluation subsets. The reference-10 set (10 problems of mixed difficulty) stabilized at 7/10 across multiple runs. The gold-10 v3 set (10 problems biased toward medium and hard) achieved 8/10 at baseline.

The primary variance in score comes from the hard tier: these problems are solved in some runs but not others, depending on the specific samples drawn. The wall problems contribute at most 1--2 correct answers and are not reliably solvable by the current system.

## 9. Failure Mode Taxonomy

Analysis of incorrect answers reveals five recurring failure modes.

**F1: Attractor traps.** The model converges to a specific incorrect answer with high confidence across multiple samples. This is the most common failure mode on medium-difficulty problems. The incorrect answer is typically a plausible intermediate result or a solution to a subtly different problem. Contrastive prompting partially mitigates this mode.

**F2: Framing ambiguity.** The problem statement admits multiple valid interpretations, and the model selects one that does not match the intended interpretation. This is difficult to address without access to the problem setter's intent. It accounts for 1--2 errors in a typical run.

**F3: Long-horizon reasoning.** Problems requiring more than approximately 15 sequential reasoning steps see a sharp drop in accuracy. The model loses track of intermediate results or introduces errors that propagate through the chain. TIR partially mitigates this by offloading numerical computation, but the logical chain itself remains vulnerable.

**F4: Existential search.** Problems requiring the model to find an object whose existence is not obvious (e.g., a specific polynomial with certain properties) are poorly served by the sampling strategy. The model either guesses or attempts a brute-force search that exceeds the computation budget.

**F5: Generation-miss.** In rare cases, the model produces a correct derivation but fails to extract the final numerical answer, or extracts it incorrectly (e.g., off-by-one in modular arithmetic). The `\boxed{}` detection catches most cases, but subtle extraction errors remain.

## 10. Future Work

Several directions remain unexplored or partially explored.

**DeepConf Online with proper calibration.** The mixed results from our DeepConf implementation suggest that the technique has potential but requires problem-difficulty-aware threshold tuning. A calibration phase using held-out competition problems could yield a meaningful improvement.

**Adaptive sample count.** The current system uses a fixed N=8 for all problems. Allocating more samples to hard problems (detected via early-sample disagreement) and fewer to easy problems (detected via early consensus) could improve the accuracy-throughput tradeoff.

**Throughput optimization.** The current pipeline does not fully saturate the H100 compute budget. Profiling indicates that Jupyter kernel round-trips and streaming overhead account for approximately 20% of wall-clock time. Reducing this overhead would allow either larger N or longer generation limits.

**Problem-specific prefill.** While generic prefill seeding failed (Section 6), problem-type-specific prefills (e.g., seeding number theory problems with modular arithmetic setup) might avoid the RL template collapse observed with generic prefills. This requires a problem classifier, which introduces its own error modes.

---

## References

1. arXiv:2603.27844 -- GPT-OSS-120B technical report and benchmark results.
2. vLLM documentation, v0.19.0 -- Inference engine configuration and attention backend specification.
3. AIMO3 Kaggle competition page -- Problem format and evaluation protocol.

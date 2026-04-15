# Project Ramanujan — The 3-Week Sprint

A technical journal of building a competitive AIMO3 submission from scratch in 3 weeks, solo, while most teams had 4-6 months.

---

## Week 0: Context (Late March 2026)

The AI Mathematical Olympiad Progress Prize 3 had been running on Kaggle since November 2025. 50 original math problems, integer answers up to 5 digits, scored by exact match. Hardware: H100 GPUs, 5-hour wall clock for 50 problems. The winning score from AIMO2 was 34/50, and the public leaderboard already showed 43-44/50 from teams using GPT-OSS-120B with basic pipelines.

I entered with ~3 weeks remaining. No prior competition experience with this model or evaluation format. The starting position: an out-of-box GPT-OSS-120B baseline scored roughly 20/50.

The thesis was straightforward: the model has the capability (proven by the 44/50 notebooks), so the gap between 20 and 44 is infrastructure, sampling strategy, and answer selection — engineering, not research.

---

## Week 1: Infrastructure & First Baseline (March 25 - March 31)

### Day 1-3: Getting vLLM to Start

The first 72 hours were entirely infrastructure. GPT-OSS-120B is a 116.8B parameter Mixture-of-Experts model (5.1B active per forward pass) that requires specific vLLM configurations to run. The debugging trail:

1. **FlashInfer fp8 dtype assertion.** vLLM 0.19.0 on A100 threw `ValueError: type fp8e4nv not supported in this architecture`. Root cause: the A100's CUDA compute capability doesn't support the fp8e4nv format that vLLM defaulted to for KV cache. Fix: `--kv-cache-dtype auto` instead of `fp8`.

2. **VLLM_ATTENTION_BACKEND enum name changes.** Between vLLM 0.10.x and 0.11.x+, the valid attention backend names changed. Setting `VLLM_ATTENTION_BACKEND=FLASH_ATTN` on 0.19.0 failed silently. The correct value was `TRITON_ATTN` (auto-selected when not specified).

3. **Startup timing race condition.** The Kaggle notebook launched vLLM as a subprocess and immediately tried to connect. On cold starts, the model needed 60-90 seconds to load weights, but the health check timed out at 30 seconds. Fix: polling loop with progressive backoff, 1500-second startup limit.

4. **`--enforce-eager` requirement.** Without this flag, vLLM attempted CUDA graph compilation that stalled indefinitely on certain GPU configurations. Adding `--enforce-eager` traded ~10% throughput for reliable startup.

### Day 4-6: Harmony Protocol Integration

GPT-OSS-120B uses OpenAI's Harmony message format — a multi-channel protocol with `analysis` (hidden reasoning), `commentary` (visible), and `final` (answer) channels, plus tool-calling via MCP. Integrating this required:

- Understanding `openai_harmony` library's `SystemContent`, `ReasoningEffort`, and `ToolNamespaceConfig` APIs
- Rendering conversations to token IDs via `render_conversation_for_completion()`
- Parsing streaming completions back into messages via `parse_messages_from_completion_tokens()`
- Handling tool calls: detecting `recipient == "python"` in assistant messages, executing code in sandboxed Jupyter kernels, returning tool results

### Day 7: First Baseline Score

With vLLM running reliably and Harmony transport working, the first end-to-end evaluation on Reference10 (10 curated problems): **7/10 correct**. The N=8 majority voting with TIR (tool-integrated reasoning) worked. Three problems were consistently unsolved:
- `641659` (Fibonacci + incircle geometry): 0% solve rate across all runs
- `86e8e5` (n-Norwegian numbers at scale 3^{2025!}): 0% solve rate
- `dd7f5e` (shifty functions, abstract functional algebra): 0% solve rate

**Week 1 score: 7/10 on reference set, estimated ~30/50 on public.**

---

## Week 2: Sampling Strategy & Verification (April 1 - April 7)

### The Regression Lesson (Day 8-9)

Attempted to improve the 7/10 by adding a problem classifier, policy book (domain-specific prompt guidance), and multi-phase sampling. The classifier routed problems to different prompt templates based on detected domain. Result: **5/10** — a 2-point regression.

Root cause: the classifier was wrong on 2 problems (misclassified geometry as combinatorics), and the policy book prompts were too verbose, consuming reasoning tokens on prompt overhead instead of actual problem-solving.

**Lesson learned: additive complexity without controlled ablation is a trap.** Every change to the pipeline needs to be tested in isolation, not stacked.

Recovery: stripped the classifier and policy book, returned to the clean 7/10 baseline. From this point forward, every intervention was tested as a controlled experiment.

### Contrastive Prompting (Day 10-11)

Hypothesis: telling the model to "consider a correct approach and one common wrong approach" would activate comparison reasoning and catch attractor errors.

Experiment: 4-run controlled study on 7 hard problems (AIME #11-15 level).
- Baseline: 4/7 correct
- Prefill seeding: 4/7 (no change)
- Contrastive: 4/7 (no change on problem-level, but 7/8 correct samples on the borderline problem vs 4/8 baseline)
- Combined: 4/7

**Finding: contrastive prompting improved sample accuracy on borderline problems (+37.5pp on discordant_perms_14_mod) but couldn't crack knowledge-wall problems. Adopted for selective use on flagged problems.**

### Entropy-Weighted Voting (Day 11-12)

The 43/50 public notebook used `weight = 1 + 1/(entropy + 0.1)` instead of simple majority vote. Entropy computed from top-5 token logprobs during streaming — low entropy (confident) responses get higher weight.

Implemented this into the pipeline. The key insight from the arxiv paper ("Model Capability Dominates"): at T=1.0, error correlation between samples is already negative (-0.258), meaning samples naturally explore different reasoning paths. Entropy weighting preserves this diversity while upweighting confident answers.

### Reference10 Consistency (Day 12)

Stabilized at 7/10 across multiple runs. The three unsolved problems were genuine knowledge walls — the model didn't have the specific mathematical techniques in its training.

**Week 2 score: 7/10 stable on reference set, classifier and policy book retired.**

---

## Week 3: Optimization & Final Push (April 8 - April 15)

### The 44-Score Solver Integration (Day 13-14)

Discovered that the top public notebooks used a fundamentally different solver architecture: streaming completions with inline `\boxed{}` detection, persistent Jupyter kernel pools (8 sandboxes), and seed-based deterministic sampling. Key differences from my fix5 pipeline:

| Feature | Fix5 Pipeline | 44-Score Solver |
|---|---|---|
| Transport | Non-streaming `/v1/completions` | Streaming with `stream=True` |
| Answer detection | Post-generation regex | Inline detection during stream |
| Tool execution | Process-based sandbox | Persistent Jupyter kernel pool |
| Entropy | Post-hoc computation | Real-time from streaming logprobs |
| Early stopping | None | 4/8 consensus → cancel remaining |

Integrated the 44-score solver as "Option A" alongside the fix5 pipeline as "Option B". Option A scored 8/10 on curated hard problems vs Option B's 7/10.

### Gold-10 Experiment Suite (Day 14-15)

Built a controlled experiment framework to test post-processing techniques. Three versions of problem sets (v1 too easy at 10/10, v2 too easy at 9/10, v3 calibrated at 8/10):

**Gold-10 v3 Results (7 problems):**
| Technique | Score | Delta vs Baseline |
|---|---|---|
| B0 Baseline | 8/10 | — |
| B1 PACER (revision layer) | 8/10 | +0 |
| B3 GenSelect (comparative selection) | 8/10 | +0 |
| B4 LeaP (peer exchange) | 7/10 | **-1** |

Every post-processing technique either matched or damaged baseline accuracy. The model's built-in entropy-weighted voting was already near-optimal.

### Gold-10 v4: The Threshold Experiment (Day 15)

The one problem that could have been rescued: P5 (Hamiltonian paths in 4×4 grid). In baseline, 4/8 samples got 276 (wrong, half-count), 3/8 got 552 (correct). Early stopping at 4/8 locked in the wrong answer.

Hypothesis: raising early-stop threshold to 5/8 would wait for more samples, letting the correct answer accumulate.

**Result: 5/8 threshold made it WORSE.** With all 8 samples, the distribution was {276:5, 552:3}. The wrong answer is simply more probable per sample. This isn't an early-stopping problem — it's a model-level attractor.

### Kaggle Submission (Day 15-16)

Submitted v63 with the 44-score solver, entropy-weighted voting, early stopping at 4/8 consensus, and `--max-num-batched-tokens 512` (later identified as a throughput bottleneck).

**Final public leaderboard score: 36/50.**

Subsequent analysis estimated the 14 missed problems broke down as: 3-4 generation-miss walls, 3-4 attractor traps, 3-4 long-horizon case analysis failures, 1-2 framing ambiguity, 1-2 existential search failures.

---

## What I'd Do With Another Month

1. **DeepConf Online (proper implementation).** Mid-generation termination of low-confidence traces, reinvesting saved tokens into more samples. The paper reports 52-85% token savings — that's 2-3x more samples per problem within the same wall-clock budget.

2. **Problem-specific prefill from a critic model.** Generic prefill seeds failed because RL-trained models resist external steering (template collapse). But problem-specific prefill — where a separate model analyzes the problem and generates a tailored reasoning seed — showed promise in DeepSeek R1 community experiments.

3. **Adaptive sample count.** Instead of fixed N=8, detect problem difficulty from first 2-3 samples and allocate N=16 or N=32 for hard problems. The snake merge problem (P9) was solved by 1/8 — at N=16 the probability of at least one correct sample doubles.

4. **Fix the throughput bottleneck.** `--max-num-batched-tokens 512` in the Kaggle notebook limited concurrent request processing. The reference solver used much higher values. This alone may have cost 5-10 problems that timed out.

5. **Contrastive overlay on attractor-prone problems.** The Hamiltonian path problem (276 vs 552) is a textbook factor-of-2 attractor. A targeted contrastive prompt warning about half-counting could flip 1-2 samples, enough to change the majority vote.

# Lessons Learned

Research findings from 3 weeks of building a competitive AIMO3 submission. Each essay covers a specific failure mode or dead end encountered during development.

---

## 1. The Classifier Retirement

On day 8, I built a two-stage problem classifier. The first stage used regex heuristics to detect domain keywords ("triangle", "modulo", "how many"). The second stage called GPT-OSS-120B itself to produce a structured JSON classification: domain, difficulty tier, code strategy (enumerate, symbolic solve, brute force), and a recommended sample budget. The classifier's output routed problems to specialized prompt templates -- a combinatorics problem got a "enumerate small cases first" directive, geometry got "use coordinate methods", and so on.

The `AutoClassification` dataclass tracked seven fields: domain, problem type, difficulty, code usefulness, key techniques, code strategy, and sample budget. The `ADAPTIVE_CLASSIFIER_PROMPT_TEMPLATE` gave the model six failure buckets to choose from -- brute_force_enumerable, extremal_tight_bound, counting_with_constraints, construction_existence, impossibility_characterization, and strategy_optimization.

It looked principled. It dropped the score from 7/10 to 5/10.

Two misclassifications caused the regression. The classifier tagged a geometry problem as combinatorics, routing it to a "enumerate small cases" prompt that burned reasoning tokens on irrelevant enumeration instead of coordinate methods. A number theory problem got tagged as "construction_existence" and received a prompt that steered toward proof-style reasoning when computation was the correct approach.

The deeper issue: the classifier was trained on AIME-style patterns. AIMO3 problems are deliberately novel -- they mix domains, use non-standard framings, and resist the clean category boundaries that AIME problems respect. A problem about "Fibonacci sequences and incircle tangent lengths" is simultaneously number theory, geometry, and recurrence relations. Forcing it into one bucket loses information.

I retired the classifier after 2 days and set `ENABLE_STAGE2_CLASSIFIER = "0"` as the permanent default. The clean 7/10 baseline with a single universal prompt outperformed every specialized-routing scheme I tested. The lesson is simple: if the model is already strong enough to solve a problem, don't add a gatekeeper that might misroute it. Additive complexity without controlled ablation is a trap.

---

## 2. Unanimous-Wrong Convergence

The most unsettling failure mode is when all N=8 samples agree on the same wrong answer. This isn't a noisy distribution where majority vote saves you -- the vote is unanimous, the model is confident, and it's wrong.

I observed this on two problems in the Gold-10 suite. On a 3xn tiling problem, 8 out of 8 samples returned 8629 instead of the correct answer 1847. On the Hamiltonian paths problem (P10), the distribution was 5/8 returning 276 against 3/8 returning 552. The correct answer was 552 -- the 276 answers had counted paths starting from one corner only, exactly half the true count.

I call these F1 attractor traps. The wrong answer isn't random; it's a natural half-count, a missed symmetry factor, or a boundary case omission. The model reaches it through valid-looking reasoning that misses one combinatorial subtlety. Worse, the model's TIR code often "verifies" the wrong answer because the code correctly implements the wrong mathematical setup.

What doesn't rescue these: raising sample count doesn't help when the wrong answer is more probable per sample. On P10, even with all 8 samples completing, {276: 5, 552: 3} was the distribution -- the wrong answer wins every vote. PACER (the revision layer) didn't fire because it saw a 4/4 consensus on 276 from early stopping, which looked decisive. GenSelect rubber-stamped 276. LeaP made things worse on the related P9 problem, where 7 wrong peers overwhelmed 1 correct sample through peer exchange.

The only intervention that shifts the per-sample probability is contrastive prompting ("consider what could go wrong"), and even that only works when the model already has the mathematical capability. For genuine attractor traps where the wrong answer is structurally more accessible than the correct one, no post-processing technique tested could recover it. The error is baked into generation.

---

## 3. When TIR Rewards Wrong Answers

Tool-integrated reasoning is the core strength of GPT-OSS-120B for math competition problems. The model writes Python code, executes it, reads the output, and incorporates the result into its reasoning. When it works, it catches algebraic errors, verifies combinatorial counts, and grounds abstract reasoning in concrete computation.

When it fails, it fails invisibly. The model formulates an incorrect mathematical setup -- counts paths from one vertex instead of all vertices, applies an inclusion-exclusion formula with a missing term, or uses the wrong recurrence base case. Then it writes code that correctly implements this wrong setup. The code runs, produces a number, and the model treats the computed result as verification that its approach was correct.

On the Hamiltonian paths problem, samples that produced 276 wrote working Python code that enumerated paths correctly -- but only from a single starting corner of the 4x4 grid. The code was bug-free. The math was wrong. The confidence score was high because the model saw code output matching its expected answer.

This creates a paradox: TIR increases accuracy on average (by catching arithmetic mistakes and testing conjectures computationally), but it also increases confidence in wrong answers when the error is in problem formulation rather than execution. The model doesn't have a mechanism to question whether the code is solving the right problem -- only whether the code is running correctly.

The Gold-10 experiments showed this clearly in the per-sample data. Wrong answers with tool verification had higher marker strength scores (2.0 for `boxed_integer` pattern) and higher protocol compliance than wrong answers without tools. The voting system correctly upweighted these confident-but-wrong answers, because by every observable metric, they looked like better candidates.

---

## 4. DeepConf: Measuring the Wrong Thing

DeepConf was the technique I had the highest hopes for. The paper describes two variants: offline (post-hoc confidence filtering from logprobs) and online (mid-generation termination of low-confidence traces, reinvesting saved tokens into additional samples). The paper reports 52-85% token savings from online termination alone.

I implemented both. DeepConf-offline computed a confidence score from a rolling window of token logprobs (configured via `DEEPCONF_WINDOW_FRAC = 0.10` -- the last 10% of generated tokens). The `_deepconf_bonus` function in the voting module mapped this to a score contribution: `max(-1.5, min(1.5, (value + 2.0) * 0.6))`. In practice, on our reference problems, the DeepConf-offline scores were indistinguishable from entropy-weighted voting. Both measured the same underlying signal -- how peaked the model's token distribution was near the answer -- just through different mathematical lenses.

DeepConf-online was the interesting one. I implemented the timeline sampling (every 512 tokens, record the running entropy), the termination threshold, and the reallocation logic. The idea: if a sample's reasoning entropy stays high for too long, kill it early and start a fresh sample with the saved compute budget.

It never triggered. On problems where the model had 95%+ consensus (P2 through P8 in Gold-10), every sample's confidence was high from early in generation. On knowledge-wall problems (P1, P10), the model was confidently wrong -- high-confidence but incorrect traces wouldn't be caught by confidence-based termination. The technique needs problems in the 50-80% solve range where some traces genuinely wander while others lock in. Our reference set had too few problems in that sweet spot.

The `ENABLE_DEEPCONF` config flag stayed at `"0"`. Not because the technique is bad -- it's theoretically sound -- but because it needs a different problem distribution to show its value. With 5 easy problems, 2 borderline, and 3 walls in Reference10, there simply weren't enough problems in the intermediate range where mid-generation termination would save tokens that could be profitably reinvested.

---

## 5. What RL-Trained Reasoning Models Resist

Prefill seeding -- injecting text into the analysis channel before the model begins its own reasoning -- seemed like a cheap way to steer generation. The hypothesis: a short seed like "Start by trying small cases to build intuition" would nudge the model toward productive reasoning paths without consuming much of the token budget.

I tested three seed types in the Hard 4-Run experiment across 7 IMO-level problems with 8 samples each:

1. **"Try small cases"** -- Caused catastrophic timeouts on AIME i_11, dropping from 7/8 correct (baseline) to 1/8. The seed pushed the model into brute-force exploration loops that exhausted the context window before reaching an answer.

2. **"Structural insight" seeds** -- Zero measurable effect. On discordant_perms_14_mod, prefill-seeded samples scored 4/8, identical to baseline.

3. **"Decompose first"** -- Zero measurable effect across all problems, including the diversity experiment's 16-sample runs where a structural decomposition prefill was one of the wild techniques (0/8 correct answers).

The RAGEN-2 paper calls this "template collapse" -- RL-trained models develop fixed reasoning patterns through reinforcement learning that resist external steering. The model's internal chain-of-thought follows a learned template, and injected text either gets absorbed into the template (ignored) or disrupts it (worse performance).

Steering vector research offers a complementary explanation: the first token of reasoning acts as a mode switch. The model commits to a reasoning strategy based on the initial tokens of its chain-of-thought, and this commitment is robust to perturbation. Generic seeds like "try small cases" don't qualify as mode-switching tokens -- they're too vague to redirect the model's learned trajectory.

Community experiments with DeepSeek R1 showed that *problem-specific* seeds from a critic model (one that analyzes the problem and generates a tailored mathematical insight) could shift generation. But this requires a second model call before each sample and prior knowledge of which mathematical insight is relevant -- essentially, you need to partially solve the problem to know what seed to inject. I didn't implement this path, but it remains the most plausible approach to making prefill work.

---

## 6. The Throughput Bottleneck That Cost 16 Points

Version 56 of the Kaggle notebook scored 20/50. Version 63 scored 36/50. The +16 delta came from multiple fixes, but the single largest contributor was a vLLM throughput configuration.

The early notebooks launched vLLM with `--max-num-batched-tokens 512`. This parameter controls how many tokens vLLM processes in a single forward pass across all concurrent requests. At 512, the server could only process roughly 1 sequence at a time at full context length. With N=8 parallel samples per problem and 50 problems in a 5-hour window, the math didn't work: harder problems that needed 4000+ reasoning tokens per sample were timing out because the server couldn't keep up with 8 concurrent generation streams.

The 44/50 reference solver used much higher batched token limits, allowing true parallel processing of all 8 samples simultaneously. After fixing the throughput throttle alongside other improvements (streaming Harmony transport, Jupyter sandbox pools, inline boxed-answer detection, entropy-weighted voting), v63 reached 36/50.

The throughput bottleneck is especially insidious because it manifests as timeout failures, not wrong answers. The model would be halfway through a correct reasoning chain, run out of its per-problem time budget, and produce no answer at all. On Reference10, the easy problems (P2-P8) were fast enough to complete even with throttled throughput. The hard problems -- the ones where the marginal points lived -- were exactly the ones that needed more reasoning tokens and more parallel samples, and those were the ones that timed out.

This is an infrastructure lesson, not a research lesson. The model had the capability. The math worked out on paper. The answers were there in the samples that completed. The bottleneck was a single `--max-num-batched-tokens` flag that limited the inference server's ability to actually deliver those samples within the competition's wall-clock budget.

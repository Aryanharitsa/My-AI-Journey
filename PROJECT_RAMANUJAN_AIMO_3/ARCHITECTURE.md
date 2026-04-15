# Project Ramanujan: System Architecture

## Overview

This document describes the data flow and component interactions of the Project Ramanujan inference system, from problem input to final integer answer. The system is designed to run within the AIMO3 Kaggle competition constraints: 50 problems, 5-hour wall clock, H100 GPU allocation.

---

## End-to-End Data Flow

```
+---------------------+
|  Problem Input      |
|  (text, problem_id) |
+----------+----------+
           |
           v
+---------------------+
|  Pre-seed / Prompt  |
|  Construction       |
|  - Harmony System   |
|    Content encoding  |
|  - ReasoningEffort  |
|    .HIGH             |
|  - Tool config      |
|  - Preference prompt |
|  - Contrastive inst  |
+----------+----------+
           |
           v
+---------------------+     +-------------------------+
|  ThreadPoolExecutor |     |  Seed Assignment        |
|  N=8 parallel jobs  |<----|  seed = f(prob_id, i)   |
+----------+----------+     +-------------------------+
           |
           | (8 concurrent requests)
           v
+---------------------+
|  GPT-OSS-120B       |
|  via vLLM 0.19.0    |
|  - MXFP4 quantized  |
|  - TRITON_ATTN      |
|  - enforce_eager    |
|  - KV cache auto    |
|  - Streaming on     |
+----------+----------+
           |
           | (token_ids + logprobs, streamed)
           v
+---------------------+     +-------------------------+
|  Streaming Loop     |     |  Jupyter Sandbox Pool   |
|  (per sample)       |<--->|  8 persistent kernels   |
|  - Token accumulate |     |  - 6s exec timeout      |
|  - Code block detect|     |  - Stateful Python env  |
|  - \boxed{} scan    |     |  - Result injection     |
|  - Entropy compute  |     +-------------------------+
|  - Logprob capture  |
+----------+----------+
           |
           | (answer_i, entropy_i) x 8
           v
+---------------------+
|  Early Stop Check   |
|  4/8 consensus?     |
|  - Yes: terminate   |
|    remaining samples |
|  - No: wait for all |
+----------+----------+
           |
           v
+---------------------+
|  Entropy-Weighted   |
|  Voting             |
|  w_i = 1/max(H_i,  |
|         1e-9)       |
|  argmax over        |
|  weighted counts    |
+----------+----------+
           |
           v
+---------------------+
|  Fallback: 0        |
|  (if no valid       |
|   answers extracted) |
+----------+----------+
           |
           v
+---------------------+
|  Final Answer       |
|  int in [0, 99999]  |
+---------------------+
```

---

## Component Interaction Diagram

```
+-----------------------------------------------------------------+
|                        Inference Host (H100)                     |
|                                                                  |
|  +------------------+       +-------------------------------+    |
|  |  vLLM Server     |       |  Orchestrator                 |    |
|  |  (port 8000)     |       |  (Python main process)        |    |
|  |                  |       |                               |    |
|  |  GPT-OSS-120B   |<----->|  OpenAI-compatible Client      |    |
|  |  MXFP4          |  HTTP |  - /v1/chat/completions       |    |
|  |  TRITON_ATTN    |  SSE  |  - stream=True                |    |
|  |  enforce_eager  |       |  - logprobs=True, top_logprobs=5|   |
|  |  KV cache auto  |       |                               |    |
|  +------------------+       +-------+-----------------------+    |
|                                     |                            |
|                                     | spawns 8 threads           |
|                                     v                            |
|                             +-------+-----------------------+    |
|                             |  ThreadPoolExecutor (N=8)     |    |
|                             |                               |    |
|                             |  Thread 0  Thread 1  ...  7   |    |
|                             |    |          |          |     |    |
|                             +----+----------+----------+-----+   |
|                                  |          |          |         |
|                     +------------+---+  +---+---+  +--+---+     |
|                     |  Streaming     |  |       |  |      |     |
|                     |  Loop          |  | ...   |  | ...  |     |
|                     |                |  |       |  |      |     |
|                     |  +----------+  |  +-------+  +------+     |
|                     |  | Harmony  |  |                          |
|                     |  | Encoder  |  |                          |
|                     |  +----------+  |                          |
|                     |       |        |                          |
|                     |       v        |                          |
|                     |  +----------+  |     +-----------------+  |
|                     |  | Answer   |  |     | Jupyter Kernel  |  |
|                     |  | Detector |  |     | Pool            |  |
|                     |  | (boxed{} |  |     |                 |  |
|                     |  |  32-tok  |  |<--->| Kernel 0        |  |
|                     |  |  window) |  |     | Kernel 1        |  |
|                     |  +----------+  |     | ...             |  |
|                     |       |        |     | Kernel 7        |  |
|                     |  +----------+  |     |                 |  |
|                     |  | Entropy  |  |     | - 6s timeout    |  |
|                     |  | Tracker  |  |     | - stateful env  |  |
|                     |  | (top-5   |  |     +-----------------+  |
|                     |  |  logprob)|  |                          |
|                     |  +----------+  |                          |
|                     +-------+--------+                          |
|                             |                                    |
|                             | (answer, entropy) per thread       |
|                             v                                    |
|                     +-------+-----------------------+            |
|                     |  Aggregator                   |            |
|                     |  - Early stop (4/8 consensus) |            |
|                     |  - Entropy-weighted vote      |            |
|                     |  - Fallback to 0              |            |
|                     +-------+-----------------------+            |
|                             |                                    |
|                             v                                    |
|                     +-------+-----------+                        |
|                     |  submission.csv   |                        |
|                     |  id, answer       |                        |
|                     +-------------------+                        |
+-----------------------------------------------------------------+
```

---

## Stage Details

### 1. Pre-seed / Prompt Construction

The prompt for each problem is constructed using the Harmony protocol encoding.

- **SystemContent**: Carries `ReasoningEffort.HIGH`, which instructs the model to use extended chain-of-thought reasoning.
- **Tool configuration**: Declares a Python code execution tool, specifying the Jupyter kernel interface. The model is informed that it may emit fenced Python code blocks that will be executed and whose results will be appended to the context.
- **Preference prompt**: A short instruction biasing the model toward explicit step-by-step derivation, intermediate verification, and a final `\boxed{}` answer.
- **Contrastive instruction**: Asks the model to consider common error modes for the problem type before committing to an answer. Applied unconditionally; empirically helpful on borderline problems (+37.5pp) and harmless elsewhere.

The fully assembled prompt is identical across all 8 samples for a given problem. Diversity comes exclusively from the per-sample seed.

### 2. GPT-OSS-120B via vLLM

The vLLM server hosts GPT-OSS-120B with the following runtime configuration:

| Parameter | Value |
|-----------|-------|
| Quantization | MXFP4 |
| Attention backend | TRITON_ATTN |
| Eager mode | Enabled |
| KV cache management | Automatic |
| Max model length | Configured per deployment |
| GPU memory utilization | Tuned to support 8 concurrent sequences |

The server exposes an OpenAI-compatible `/v1/chat/completions` endpoint. All requests use `stream=True` with `logprobs=True` and `top_logprobs=5`. Stop token IDs are configured to include the model's end-of-turn tokens and the `\boxed{}` closing pattern.

### 3. N=8 Parallel Sampling

A `ThreadPoolExecutor` with 8 workers dispatches one request per sample. Each worker:

1. Constructs the request with `seed = f(problem_id, sample_index)` for deterministic sampling.
2. Opens a streaming connection to the vLLM server.
3. Enters the streaming loop (Stage 4).
4. Returns `(answer, entropy)` to the aggregator.

Sampling parameters are fixed across all samples: `temperature=1.0`, `top_p=1.0`, `min_p=0.02`. No per-sample variation in temperature or nucleus parameters.

### 4. Streaming Loop and Tool Execution

The streaming loop processes server-sent events (SSE) from the vLLM streaming endpoint. For each chunk:

1. **Token accumulation**: Raw token IDs are appended to the running sequence. Decoded text is maintained in a buffer.
2. **Code block detection**: The buffer is monitored for fenced Python code blocks (`` ```python ... ``` ``). On detection, the code is dispatched to the sample's dedicated Jupyter kernel.
3. **Jupyter execution**: The kernel executes the code with a 6-second hard timeout. Output (stdout, return values, errors) is serialized and injected back into the generation context as a tool response message. The kernel is persistent: variables, imports, and state carry across multiple code blocks within the same sample.
4. **Answer detection**: A 32-token sliding window over recent token IDs is scanned for the `\boxed{` ... `}` pattern. On match, the enclosed content is parsed as an integer. Multiple `\boxed{}` occurrences are tracked; the last one is used as the sample's answer.
5. **Entropy computation**: For each token, the top-5 logprobs are recorded. The entropy of the final answer region (tokens within and immediately surrounding the `\boxed{}` expression) is computed as the Shannon entropy of the softmax-normalized top-5 logprobs.

A final-channel parser runs after the stream closes to catch any answers that the inline detector missed due to tokenization boundary effects.

### 5. Early Stop

After each sample completes, the aggregator checks whether 4 or more of the completed samples agree on the same answer. If so, remaining in-flight samples for that problem are cancelled (via request cancellation to vLLM), and the agreed-upon answer is accepted without proceeding to entropy-weighted voting. This optimization recovers wall-clock time proportional to the number of easy problems in the set.

### 6. Entropy-Weighted Voting

When early stop does not trigger, all 8 `(answer, entropy)` pairs are collected and voting proceeds:

1. Group samples by answer value.
2. For each group, compute the total weight: `W_group = sum(1 / max(H_i, 1e-9))` for all samples i in the group.
3. Select the group with the highest total weight.
4. Return that group's answer value.

The `1e-9` entropy floor prevents degenerate weights when the model is maximally confident (all probability mass on a single token). In practice, entropy values for correct answers tend to be lower than for incorrect answers, so the weighting scheme upweights high-confidence samples.

### 7. Fallback

If no sample produces a parseable integer answer (all 8 samples fail to emit a valid `\boxed{}` expression, or all extracted values fall outside [0, 99999]), the system returns 0. This case is rare in practice (observed on fewer than 2% of problems) and represents a total failure of the generation and extraction pipeline.

---

## Resource Utilization

| Resource | Allocation |
|----------|-----------|
| GPU | H100 (Kaggle-provided) |
| vLLM server | Single instance, full GPU |
| Concurrent sequences | 8 (one per sample) |
| Jupyter kernels | 8 persistent (one per sample) |
| Python workers | 8 threads in ThreadPoolExecutor |
| Wall clock per problem | ~5--6 minutes average (varies by difficulty) |
| Total wall clock | ~4.5 hours typical (under 5-hour limit) |

---

## Failure Handling

- **vLLM OOM**: If the server runs out of KV cache memory, requests are retried with reduced `max_tokens`. This has not been observed in production with the current `gpu_memory_utilization` setting.
- **Jupyter timeout**: Code blocks exceeding 6 seconds are killed. The kernel remains alive; the execution result is replaced with a timeout error message injected into the generation context. The model typically adapts by simplifying its computation.
- **Malformed answers**: Non-integer or out-of-range values extracted from `\boxed{}` are discarded. The sample is treated as having produced no answer and receives zero weight in voting.
- **Stream interruption**: If the SSE stream drops mid-generation, the partial sequence is processed for any already-detected answer. If none exists, the sample is marked as failed.

# Mamba install attempt 01 — Phase 4 stretch (session 02)

**Pod session:** 2026-04-19, A100-SXM4-80GB, continuing after Phase 3.5 landed.
**Kill-switch policy:** session-02 handoff §4.7 — 30 min install wall-clock, two
differing install failures, `import mamba_ssm` failing after successful install,
checkpoint unavailable, or nDCG@10 on NFCorpus outside ±0.05.

## Environment probe (before any `pip install`)

Per handoff §4.2 — recorded *before* attempting install so a future session
can tell whether the toolchain shifted:

| Component | Value |
|---|---|
| torch | 2.4.1+cu124 |
| CUDA runtime reported by torch | 12.4 |
| cuDNN | 9.1.0 (90100) |
| nvcc | **NOT INSTALLED** — no CUDA dev toolkit on the pod |
| gcc | 11.4.0 (Ubuntu 22.04) |
| Python | 3.11.10 |
| Latest `mamba-ssm` on PyPI | 2.3.1 |
| Latest `causal-conv1d` on PyPI | 1.6.1 |

**Risk flag — nvcc missing.** If pip falls back to source build, the compile
will fail because no `nvcc` to drive the CUDA frontend. Survival depends on
prebuilt wheels matching our torch/CUDA/Python combo. `mamba-ssm` 2.3.1 and
`causal-conv1d` 1.6.1 typically publish wheels for
`torch 2.4 / cuda 12.x / cp311 linux_x86_64` — we'll see at install time.

## HuggingFace checkpoint search

Queried HF Hub for "mamba retriever" and variants. Candidates:

- `MambaRetriever/SPScanner-130m` — official-looking org name, 130M params
  (matches BERT-base scale, so the Phase 4 Pareto comparison is fair).
- `MambaRetriever/SPScanner-1.3b` — same org, 1.3B params (too large for
  this comparison — 10× BERT-base; different Pareto point).
- `lei-ucsd/mamba-retriever-*` — 4 variants, likely author dev repos.

Checkpoint **is** available. §4.7 kill-switch #4 (checkpoint unavailable)
does not fire. Proceeding.

## Install attempt

=== MAMBA INSTALL attempt1 start 2026-04-19T09:02:39+00:00 ===


## Install attempt 1 — outcome

- **Command:** `pip install --no-build-isolation --prefer-binary mamba-ssm causal-conv1d`
- **Wheel fallback:** no prebuilt wheel matched (torch 2.4.1+cu124 / cp311 / linux_x86_64); pip fell back to source build.
- **`nvcc` resolution:** the initial environment probe reported `nvcc: command not found`, but the CUDA toolchain is actually present at `/usr/local/cuda/bin/nvcc` and the source build picked it up. The probe's false negative was a PATH issue, not a missing toolchain. Noted for Session 03: `export PATH=/usr/local/cuda/bin:$PATH` before probing.
- **Compilation progressed through** `selective_scan_*.cu` kernels targeting gencodes `sm_53`, `sm_62`, `sm_70`, `sm_72`, `sm_75`, `sm_80`, `sm_87`, `sm_90` (A100 is `sm_80`). No error observed during the time the install ran.
- **Install was not allowed to complete.** Terminated when the bi-encoder / cross-encoder mismatch (below) made the install moot for Session 02's Phase 4 plan. Session 03 will retry from scratch with explicit version pins — see its §5.4 gate.
- **Process state when terminated:** still in `ninja`-driven CUDA compilation, roughly 10-12 min wall-clock elapsed. Kill-switch #1 (30 min wall-clock) had NOT fired.

## The architectural finding (the actual kill-switch trigger)

While the install ran, inspecting the HuggingFace checkpoint
`MambaRetriever/SPScanner-130m` on the pod revealed a mismatch that
matters more than whether mamba-ssm compiles:

```
=== config.json ===
{
    "d_model": 768,
    "n_layer": 24,
    "ssm_cfg": {"layer": "Mamba2"},
    "attn_layer_idx": [],
    ...
}

=== README.md (excerpt) ===
# Single-Pass Scanner
This repository contains model checkpoint for Single-Pass Scanner
(https://github.com/MambaRetriever/MambaRetriever)
pipeline_tag: question-answering
```

- `pipeline_tag: question-answering`, not `sentence-similarity`.
- "Single-Pass Scanner" is a **cross-encoder** that consumes
  `(query, passage_1, passage_2, ...)` and emits scores, not a bi-encoder
  that produces per-item embeddings.
- Vitruvius's Phase 4 scaffolding (FAISS `IndexFlatIP`, the `Encoder`
  interface returning `(N, dim)` arrays, `bench-sweep`, `profile`) assumes
  a bi-encoder. Dropping SPScanner into the `Encoder` interface requires
  either (a) a different inference path (scoring pairs instead of encoding
  to vectors) or (b) a bi-encoder Mamba checkpoint we don't have access to.

### Why this triggers §4.7 kill-switch #4 in spirit

The letter of trigger #4 is "checkpoint not downloadable from HF and would
require from-scratch training." The SPScanner checkpoint **is**
downloadable — but it is not the kind of model Phase 4 was planned around.
Integrating it would be a Phase-5-sized architectural change. The
session-02 handoff's own framing on Phase 4 was "drop-in a pre-trained
Mamba Retriever bi-encoder." No such bi-encoder exists publicly (as of
2026-04-19); what exists is a cross-encoder from the same paper's authors.

Operator confirmed the kill-switch, deferred the whole from-scratch path
to Phase 5, and handed over an amended Session 03 plan that trains
LSTM / CNN / Mamba-fs bi-encoders from MS MARCO on equal footing rather
than comparing against a pre-trained Mamba bi-encoder that doesn't exist.

## Starting-point for Session 03

1. The CUDA toolchain **is** on the pod at `/usr/local/cuda/bin/nvcc` —
   no need to install it again. Set `PATH` before probing.
2. `pip install mamba-ssm causal-conv1d` falls back to source build for
   torch 2.4.1+cu124 / cp311. The build appeared healthy before
   termination — Session 02 never observed a compilation error.
3. Session 03's §5.4 recommends a fallback to pinned older versions
   (`mamba-ssm==2.2.2 causal-conv1d==1.4.0`) if the latest versions fail;
   Session 02 did not try pinned versions.
4. Expected source-build wall-clock on this pod: estimate **15-25 min**
   based on how far the kernels had gotten in ~10 min before kill.
   Session 03's 45-min total install budget should be enough for one
   attempt with room for one pinned retry.
5. The `MambaRetriever/SPScanner-130m` cross-encoder path remains open
   for a dedicated future session but is **out of scope** for Session 03
   per the amended handoff (from-scratch bi-encoder training only).

## Artifacts from Session 02's Phase 4 attempt

- This notes file — full discovery + decision log.
- The aborted compile left build artifacts under
  `/tmp/pip-install-nsjjx65m/mamba-ssm_*/` on the pod; these are
  disposable and will vanish when the pod is killed.
- No changes to `src/vitruvius/encoders/mamba_encoder.py` — it remains
  a Phase 4/5 stub that raises `NotImplementedError`. The real
  implementation lands in Session 03.

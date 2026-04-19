# Mamba install attempt 02 — Phase 5 training (session 03)

**Pod session:** 2026-04-20, same pod as session 02. Entered via the
amended Session-03 handoff §5.4 install gate (45-min total budget).
**Outcome:** **SUCCESS on attempt 1 (unpinned)**, ~33 minutes wall-clock from
source build. No fallback needed. `mamba-retriever-fs` was included in the
Phase 5 training + Pareto lineup.

## Environment probe (before install)

| Component | Value | Delta vs attempt-01 |
|---|---|---|
| torch | 2.4.1+cu124 | unchanged |
| CUDA (from torch) | 12.4 | unchanged |
| cuDNN | 9.1.0 | unchanged |
| `nvcc` (PATH'd with `export PATH=/usr/local/cuda/bin:$PATH`) | 12.4, `/usr/local/cuda/bin/nvcc` | attempt-01 missed it because PATH wasn't set; fixed here |
| gcc | 11.4.0 | unchanged |
| Python | 3.11.10 | unchanged |
| `mamba-ssm` latest on PyPI | 2.3.1 | unchanged |
| `causal-conv1d` latest on PyPI | 1.6.1 | unchanged |

## Install attempt 1 (unpinned, latest)

```bash
export PATH=/usr/local/cuda/bin:$PATH
pip install --no-build-isolation mamba-ssm causal-conv1d
```

`--no-build-isolation` because torch must be visible to the source build.
pip picked up the latest wheels from PyPI, found no prebuilt wheel matching
`torch 2.4.1+cu124 / cp311 / linux_x86_64`, and fell back to source build
(expected per attempt-01 notes).

### Build timeline

The `mamba-ssm` package compiles ~8 CUDA kernels targeting gencodes
`sm_53, sm_62, sm_70, sm_72, sm_75, sm_80, sm_87, sm_90` — 4-way parallel
via `ninja` + `cicc` at 100% CPU each.

Compiled in this order (approximate, from `/tmp/pip-install-*/mamba-ssm*/build/temp.*/csrc/selective_scan/*.o` mtimes):

1. `selective_scan.o`                         (10:20)
2. `selective_scan_bwd_bf16_complex.o`        (10:23)
3. `selective_scan_bwd_bf16_real.o`           (10:26)
4. `selective_scan_bwd_fp16_complex.o`        (10:29)
5. `selective_scan_bwd_fp16_real.o`           (10:32)
6. `selective_scan_bwd_fp32_complex.o`        (10:35)
7. `selective_scan_bwd_fp32_real.o`           (10:37)
8. `selective_scan_fwd_bf16.o`                (10:39)
9. `selective_scan_fwd_fp16.o`                (…)
10. `selective_scan_fwd_fp32.o`               (…)
    Plus `causal-conv1d` kernels.

Total wall-clock to `Successfully installed`: **~33 min** (10:20 → 10:53 UTC).

### Install success marker

```
Successfully installed causal-conv1d-1.6.1 einops-0.8.2 mamba-ssm-2.3.1 ninja-1.13.0
=== attempt1 done 2026-04-19T10:53:19Z exit=0 ===
```

### Import + GPU forward smoke

```python
import mamba_ssm, causal_conv1d
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm  # newer path
import torch
blk = Mamba2(d_model=384, d_state=128).to("cuda", dtype=torch.float16)
y = blk(torch.randn(2, 32, 384, dtype=torch.float16, device="cuda"))
# -> out shape (2, 32, 384). Success.
```

`MambaRetrieverEncoder` (12-layer Mamba2, d_model=384, d_state=128,
Linear(384→256) projection) instantiates cleanly, counts **23.74M parameters**
(within the handoff's 15-25M target), and returns unit-norm embeddings.

## Attempt 2 (pinned) — NOT USED

Fallback would have been:

```bash
pip install mamba-ssm==2.2.2 causal-conv1d==1.4.0 --no-build-isolation
```

Skipped because attempt 1 succeeded with time to spare inside the 45-min
budget. Noted for reference: the `MambaRetriever/SPScanner-130m` README
prescribes these pinned versions on a `py3.10 / cuda 11.8 / torch 2.1`
stack (which this pod is not). No observed evidence that the pinned
versions would be preferable on `py3.11 / cu12.4 / torch 2.4`.

## What attempt-02 leaves for future sessions

1. The correct install command on this pod/toolchain is
   `export PATH=/usr/local/cuda/bin:$PATH; pip install --no-build-isolation mamba-ssm causal-conv1d`.
   No special version pin required on PyPI's 2.3.1 / 1.6.1 as of 2026-04-19.
2. Expect ~30 min of CPU-only compile. Do not share the pod with anything
   CPU-heavy during that window.
3. `Mamba2` import path is `mamba_ssm.modules.mamba2`. `RMSNorm` is at
   `mamba_ssm.ops.triton.layer_norm` (the old `.layernorm` path still
   exists as a fallback in the encoder wrapper for older installs).
4. Mamba training in this session used AMP (fp16) and did not observe
   instabilities; there was no need to disable mixed precision for the
   Mamba encoder specifically.


## Appendix — DataLoader segfault on Mamba training (session 03 attempt 1)

**Symptom.** Training `mamba-retriever-fs` with the standard pipeline
(`batch_size=64`, `num_workers=2`, AMP fp16) crashed at ~step 5000 of 19452
with:

```
ERROR: Unexpected segmentation fault encountered in worker.
RuntimeError: DataLoader worker (pid(s) NNNN) exited unexpectedly
```

Val loss was descending cleanly before the crash (step 500 → 2.50,
step 5000 → 0.418 — already better than fully-trained LSTM at 0.472).
GPU and main process were healthy; the segfault was isolated to a
DataLoader worker process.

**Suspected cause.** Mamba's `mamba_ssm` package ships Triton-compiled
kernels for `selective_scan`. Triton's JIT cache and shared-memory state
interact badly with `multiprocessing.fork()` — which is exactly what
PyTorch DataLoader uses when `num_workers > 0`. When a DataLoader worker
is forked after the main process has touched Triton state, the worker
inherits an inconsistent view and segfaults on the next access.

LSTM and CNN trained through with `num_workers=2` without issue because
neither uses Triton kernels; only the Mamba2 blocks do.

**Workaround.** Set `num_workers=0` for Mamba training. Tokenization then
runs in the main process (slight CPU overhead, negligible on this pod's
CPU surplus). No architecture or hyperparameter change.

Session 03 attempt 2 used `num_workers=0` and trained all 19452 steps to
completion.

**Checkpoint preserved from attempt 1.** The step-5000 best-val checkpoint
from the interrupted run is archived on pod at
`models/mamba-retriever-fs/interrupted_run_best_step5000.pt` (94 MB,
val_loss 0.418). Not committed. Reference point if a future session asks
whether Mamba converged faster per-step than LSTM — the answer is yes,
at least for the first 5000 steps.

**Starting-point for future Mamba runs.**
- Use `--num-workers 0` on the `vitruvius train` CLI when training any
  Mamba-backed architecture, unless you've verified Triton+fork works on
  the target toolchain.
- If `num_workers=0` is too slow (dataloader becomes the bottleneck), the
  alternatives in order of effort are:
  (a) Pre-tokenize the dataset to numpy arrays off-line and use a
      memory-mapped DataLoader with `num_workers>0` — the workers never
      touch Python-side Triton state.
  (b) Switch DataLoader's multiprocessing context from `fork` to `spawn`
      (`torch.multiprocessing.set_start_method("spawn", force=True)`) —
      spawn starts a fresh interpreter in each worker, bypassing the
      inherited-state bug. Adds ~5s of worker startup.
  (c) Wait for upstream mamba-ssm / Triton to fix fork-safety.

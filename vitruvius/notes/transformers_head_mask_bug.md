# transformers 5.x silently drops `head_mask` — Phase 7 Session 05 runbook

**TL;DR.** `transformers>=5.0` accepts a `head_mask=...` kwarg on
`BertModel.forward` (via `**kwargs`) but silently discards it. The
attention computation never sees the mask. Unit tests that only check
"all-ones ≈ base" and "all-zeros doesn't crash" pass trivially because
both cases reduce to the same computation (no mask applied).

This cost Session 05 **~3 hours of pod compute** (~$5) on a head-ablation
sweep that produced 9 well-formed JSONs with every per-head
`delta_nDCG@10` exactly `0.0`.

## What was going on

1. `transformers >= 5.0` refactored the attention stack. The classical
   eager attention path from `transformers < 5.0` — which respected
   `head_mask` — was replaced by the new functional attention interface.
2. `BertSelfAttention.forward` in 5.x no longer has `head_mask` as a
   named parameter. Nothing in the new `eager_attention_forward` reads
   `head_mask` from kwargs.
3. `BertModel.forward` accepts `**kwargs` and passes them along, so
   calls like `model(..., head_mask=mask)` succeed but the mask never
   reaches anything that uses it.

## Diagnostic

```python
import torch
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").cuda().eval()
enc = tok(["hello world"], padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    ones = torch.ones(6, 12).cuda()
    zeros = torch.zeros(6, 12).cuda()
    r1 = m(**enc, head_mask=ones).last_hidden_state
    r2 = m(**enc, head_mask=zeros).last_hidden_state
print((r1 - r2).abs().max().item())
```

- Under **transformers 5.5.4**, this prints `0.0` (head_mask ignored).
- Under **transformers 4.57.6 + `attn_implementation="eager"`**, this
  prints ~1.2 (head_mask works).

## Fix

1. Pin in `pyproject.toml`:

   ```toml
   "transformers>=4.40,<5.0"
   ```

2. Load models with eager attention (even on 4.x, the default `sdpa`
   does not support `head_mask`):

   ```python
   AutoModel.from_pretrained(name, attn_implementation="eager")
   ```

3. **Required sanity check before any multi-run head-ablation sweep.**
   Run the diagnostic above or equivalent. If the diff is below 1e-4,
   `head_mask` is broken in the current environment — do not launch
   the sweep.

## Unit-test gap

The original Phase 7 tests (`tests/test_pruned_transformer.py`) checked
two invariants:

- `all-ones head_mask produces output bit-exact (1e-5 tol) to base` — ✓
- `all-zeros head_mask does not crash; output is deterministic, finite` — ✓

Both are trivially true when `head_mask` is ignored (the model computes
the same thing either way). The missing invariant:

- `all-ones head_mask produces output that DIFFERS from single-head-zero head_mask` — ✗ (missed)

Adding this check would have caught the bug in ~2 seconds. The Phase 8
shuffle utility will be validated with the analogous "independent
variable actually moves the measurement" check before launching.

## General discipline adopted post-Session 05

> **Before any multi-hour sweep: run one cell end-to-end, inspect the
> output numerically, and verify the independent variable actually
> moves the measurement.**

For Phase 7: independent var = `head_mask`, measurement = `delta_nDCG`.
The 8 completed Phase 7 cells (post-fix) all show nonzero per-head
deltas in the [-0.01, +0.03] range, confirming the mechanism works.

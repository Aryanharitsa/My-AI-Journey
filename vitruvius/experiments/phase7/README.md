# Phase 7 — head pruning artifacts

## Files

- `head_importance/<encoder>__<dataset>.json` (9 files)
    Per-head ablation results. Schema:
    `{encoder, dataset, baseline_nDCG@10, num_layers, num_heads_per_layer, total_heads, per_head_results: [{layer, head, nDCG@10, delta_nDCG@10}], ranked_by_importance: [{layer, head, rank, delta_nDCG@10}], config, hardware, runtime_seconds}`
- `cumulative_pruning/<encoder>__<dataset>.json` (9 files)
    Cumulative-prune-N-least-important curves. Schema:
    `{encoder, dataset, baseline_nDCG@10, ordering, curve: [{heads_pruned, nDCG@10, rel_drop_pct}], thresholds: {heads_prunable_at_5pct_drop, heads_prunable_at_10pct_drop}, ...}`
- `head_stability_analysis.md` — human-readable Spearman table.
- `head_stability.json` — machine-readable version.
- `head_sweep.log` — raw stdout of the full 9-cell importance sweep.
- `cumulative_sweep.log` — raw stdout of the cumulative pruning run.
- `SUMMARY.md` — headline tables + interpretation.

## Reproduce

```bash
python scripts/head_importance_sweep.py \
    --encoders minilm-l6-v2 bert-base gte-small \
    --datasets nfcorpus scifact fiqa \
    --batch-size 128 --top-k 100 --device cuda

python scripts/cumulative_pruning_sweep.py \
    --encoders minilm-l6-v2 bert-base gte-small \
    --datasets nfcorpus scifact fiqa \
    --batch-size 128 --top-k 100 --device cuda

python scripts/phase7_analysis.py
```

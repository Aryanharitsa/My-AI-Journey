# Phase 8 — token-position shuffle artifacts

## Files

- `<encoder>__<dataset>__<mode>.json` (54 files)
    One per (encoder, dataset, shuffle mode). Full metric block,
    per-query results, hardware, config, and the shuffle flags.
- `position_sensitivity.json`
    Machine-readable aggregated `(baseline − shuffled) / baseline`
    table. 54 rows.
- `query_level_sensitivity.parquet` (or `.json` if parquet unavailable)
    Per-query baseline vs shuffled nDCG@10 and hit@10. Used for
    Phase 6 cross-referencing offline.
- `SUMMARY.md` — headline family table + full 18-row pivot +
    methodology + limitations.
- `README.md` — this file.

## Reproduce

```bash
python scripts/shuffle_sweep.py \
    --encoders minilm-l6-v2 bert-base gte-small lstm-retriever conv-retriever mamba-retriever-fs \
    --datasets nfcorpus scifact fiqa \
    --modes docs-shuffled queries-shuffled both-shuffled \
    --split test --batch-size 128 --top-k 100 --device cuda --seed 1729

python scripts/phase8_analysis.py
python scripts/phase8_writeup_gen.py
```

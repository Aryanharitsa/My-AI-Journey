# Position sensitivity per encoder × dataset × shuffle mode

Position sensitivity = `(baseline_nDCG@10 - shuffled_nDCG@10) / baseline_nDCG@10`.

- **0.0** = shuffle-invariant (shuffling tokens doesn't change retrieval quality).
- **1.0** = complete collapse (shuffle reduces nDCG@10 to 0).
- **< 0** would indicate shuffle *helped*, treated as a bug signal.

Three panels, one per BEIR test subset. Three bars per encoder, one per shuffle mode (docs-only, queries-only, both). Shuffle seed = 1729, identical across all encoders for cross-encoder fairness. Content tokens are permuted within each sequence after tokenization but before encoding; special tokens (`[CLS]`, `[SEP]`, `[PAD]`) stay at their original positions.

Source: `experiments/phase8/<encoder>__<dataset>__<mode>.json`.

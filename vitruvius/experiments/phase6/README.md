# Phase 6 — Per-query failure analysis methodology

## Data sources

- `../phase3/*.json` — 9 bench JSONs (3 pre-trained transformers × 3
  BEIR subsets) from Phase 3.
- `../phase5/bench/*.json` — 9 bench JSONs (3 from-scratch encoders × 3
  BEIR subsets) from Phase 5.
- Every JSON preserves a full `per_query_results` payload with
  `nDCG@10`, `hit@10`, `Recall@10`, `ranked_doc_ids` (top-100 doc IDs),
  `relevance_judgments` (qrels restricted to the evaluated query set),
  and `query_text`.

Schema sanity is re-checked at the top of the Phase 6 notebook —
`vitruvius/notebooks/02_phase6_failure_analysis.ipynb` — before any
analysis runs.

## Failure + success thresholds

| label    | rule               | intuition                                     |
|:---------|:-------------------|:----------------------------------------------|
| failure  | `nDCG@10 < 0.1`    | no relevant doc in top-5, or one at rank 6-10 |
| success  | `nDCG@10 > 0.3`    | at least one strongly-ranked relevant doc     |

The two thresholds span a neutral band; queries that land between 0.1
and 0.3 are neither counted as failures nor as successes in the set
arithmetic (universal-wins / universal-losses / transformer-gap /
unique-wins) so the sets are conservative. The threshold choice is a
judgment call; the raw per-query nDCG is in `query_frame.parquet`, so a
reviewer who prefers `hit@10 == False` can recompute the sets in a
line of pandas.

## Tokenization for query-length features

- `bert-base-uncased` WordPiece tokenizer (via 🤗 `transformers`).
  Matches the training-time tokenizer for the from-scratch encoders so
  length quartiles are comparable across the six encoders.
- A whitespace-split alternative was considered and rejected because
  WordPiece's treatment of numbers and dashes is different in a way
  that systematically correlates with failure (see `NUMERIC-ENTITY`
  discussion in `../../analysis/failure_taxonomy.md`).

## Manual labeling

- **Pool.** Strategic stratified sample of 500+ failing queries —
  150 from `conv-retriever × fiqa`, 100 each from
  `{lstm-retriever × scifact, lstm-retriever × fiqa,
  mamba-retriever-fs × scifact}`, plus 75 universal-loss queries and
  75 unique-success queries. Fixed seed 1729. Actual labeled set:
  1,116 (encoder, dataset, query) triples covering 470 distinct
  (dataset, query) pairs.
- **Labeler.** One analyst (the operator). No inter-annotator
  agreement measured; this is disclosed as a limitation in
  `SUMMARY.md`.
- **Procedure.**
  1. Rule-based codes are applied mechanically from the query string
     and the dataset-specific `DOMAIN_LEXICON`. See
     `scripts/phase6_label_taxonomy.py`.
  2. `UNCATEGORIZED` is reserved for queries that trigger no rule; the
     residual is 154 triples (~14% of the labeled set), plausibly
     matching the semantic-mismatch categories (PARAPHRASE, AMBIGUOUS,
     MULTI-HOP) that the handoff's seed list proposed. Those are
     flagged as future work — they require corpus access that Phase 6
     intentionally does not reload.
- **Reproducibility.** `sampled_qids.json` captures the exact query
  IDs selected, so any re-run of the notebook (or an independent
  analyst) reads the same queries.

## Statistical tests

- **Spearman ρ** on per-query nDCG@10 for cross-encoder agreement, per
  dataset. Reported per (encoder-A, encoder-B) pair and summarized in
  `SUMMARY.md` as within-family / across-family means.
- **Pairwise disagreement counts** at the nDCG@10 > 0 threshold —
  queries where A hits but B misses, both hit, both miss. Plotted as
  stacked bars in `figures/disagreement_bars.png`.
- **No per-query hypothesis test was conducted.** The observational
  counts (universal wins/losses, transformer-gap, unique wins) are
  descriptive; no claim of statistical significance is attached.

## Reproduction

From the `vitruvius/` package directory with the venv activated:

```bash
python scripts/phase6_analysis.py           # builds query_frame.parquet + all grids
python scripts/phase6_label_taxonomy.py     # labels the sampled failures
python scripts/phase6_failure_examples.py   # curates failure_examples.md
python scripts/phase6_promote_figures.py    # builds figures/ top-line plots
```

Everything is deterministic: fixed seed 1729 for sampling, no random
shuffles elsewhere. The per-query bench JSONs under `../phase3/` and
`../phase5/bench/` are the upstream source of truth — Phase 6 neither
re-runs models nor re-downloads datasets.

## Files in this directory

| file                         | what it is                                          |
|:-----------------------------|:----------------------------------------------------|
| `SUMMARY.md`                 | reviewer-facing writeup                             |
| `README.md`                  | this file — methodology                             |
| `query_frame.parquet`        | 7,626-row long-form DataFrame                       |
| `zero_ndcg_rates.csv/.md`    | encoder × dataset zero-nDCG grid                    |
| `failure_rates_0p1.csv`      | same at the 0.1 threshold                           |
| `spearman_<ds>.csv`          | 6×6 rank-correlation matrix per dataset             |
| `disagreement_<ds>.csv`      | pairwise hit/miss counts per dataset                |
| `length_bins.csv`            | mean nDCG per (encoder, ds, length quartile)        |
| `length_quartile_bounds.csv` | per-dataset WordPiece quartile cutoffs              |
| `failure_pivot.csv`          | long-form (family, category, count, examples)       |
| `failure_pivot_matrix.csv`   | pivoted category × family counts                    |
| `cross_encoder_summary.csv`  | universal/unique/gap summary per dataset            |
| `cross_encoder_sets/`        | full qid lists per set per dataset                  |
| `sampled_qids.json`          | reproducible sample used for manual labeling        |
| `labeled_queries.csv`        | labeled (encoder, dataset, qid) rows with codes     |
| `figures/`                   | notebook-style plots (stacked bars, heatmap, etc.)  |

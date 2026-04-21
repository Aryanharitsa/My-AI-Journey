# Phase 6 — Per-query failure analysis (70% milestone)

**Date:** 2026-04-21
**Data:** 18 bench JSONs from Phases 3 and 5 (6 encoders × 3 BEIR
subsets), each with the full per-query payload preserved. 7,626 rows in
`query_frame.parquet`; 470 distinct failing queries read for the
taxonomy.

## One-paragraph finding

Pre-training's value concentrates in two specific places, not uniformly:
(1) *long natural-language questions*, where convolutional and recurrent
encoders run out of receptive field on FiQA and fail on 79–90% of
queries, while transformers stay near-flat across all length quartiles;
and (2) *domain jargon* in biomedical and scientific queries, where
`bert-base` recovers the relevant document at rank 1 on SciFact claims
that all three MS MARCO-trained from-scratch encoders miss by rank > 20.
On every other failure category the architecture gap is smaller than
the accuracy gap would suggest — universal-loss queries (152 on FiQA,
32 on SciFact) show that 23.5% of the hardest dataset is architecture-
agnostic hard, and from-scratch encoders claim only 11 unique wins
across all three datasets vs. 127 unique transformer wins.

## Zero-nDCG@10 rates — per encoder × dataset

| encoder            | nfcorpus | scifact | fiqa  |
|:-------------------|---------:|--------:|------:|
| minilm-l6-v2       | 31.0%    | 20.7%   | 34.3% |
| bert-base          | 32.8%    | 26.7%   | 40.9% |
| gte-small          | 29.7%    | 14.0%   | 33.3% |
| lstm-retriever     | 50.2%    | 52.0%   | 78.7% |
| conv-retriever     | 58.5%    | 73.7%   | 90.4% |
| mamba-retriever-fs | 47.7%    | 49.0%   | 77.3% |

The reframing the Phase 6 handoff asked for is real: even pre-trained
transformers evaluated out-of-distribution on BEIR fail on a quarter to
a third of the queries (gte-small's 14.0% on SciFact is the outlier
low, pulled down by BEIR's overlap with GTE's contrastive pre-training
corpus). From-scratch encoders fail on 47–90%. The gap is large, not
infinite — when reviewers read "CNN fails on 90% of FiQA," it helps to
remember that the transformer baseline is not 0%.

See `zero_ndcg_rates.csv` and `zero_ndcg_rates.md` for raw counts; the
0.1-threshold failure grid (marginally different: one relevant doc at
rank 6-10 counts as a success at threshold 0 but a failure at 0.1) is
in `failure_rates_0p1.csv`.

## Cross-encoder agreement (Spearman ρ on per-query nDCG@10)

The agreement structure flips as the dataset gets harder:

| metric                              | nfcorpus | scifact | fiqa  |
|:------------------------------------|---------:|--------:|------:|
| within-transformer mean ρ           | 0.89     | 0.73    | 0.77  |
| within-from-scratch mean ρ          | 0.79     | 0.63    | 0.47  |
| transformer↔from-scratch mean ρ     | 0.69     | 0.49    | 0.33  |
| CNN↔best transformer (gte) ρ        | 0.58     | 0.32    | 0.24  |

On nfcorpus (short medical queries) every encoder ranks queries
similarly (ρ ≥ 0.58), so the architecture difference is mostly one of
*accuracy level*, not *accuracy pattern*. On FiQA the picture is
qualitatively different: CNN and the transformers disagree on which
queries are easy (ρ ≈ 0.24) — the *pattern* of success itself is
architecture-specific. This is the signal that Phase 7 (attention-head
pruning) and Phase 8 (position shuffle) should amplify: the biggest
retrieval-quality divergence in the current Pareto lives in FiQA.

Full matrices in `spearman_<dataset>.csv`.

## Failure taxonomy — architecture family × category

Categories applied to the 1,116 labeled (encoder, dataset, query)
triples covering 470 distinct failing queries; labels are non-exclusive
(a query can carry several). Definitions and sample queries in
`../../analysis/failure_taxonomy.md`.

| category          | transformer | recurrent | convolutional | ssm |
|:------------------|------------:|----------:|--------------:|----:|
| NATURAL-QUESTION  | 152         | 140       | 166           | 70  |
| DOMAIN-TERM       | 73          | 65        | 64            | 48  |
| LEN-SHORT         | 78          | 34        | 40            | 28  |
| MULTI-CONCEPT     | 40          | 58        | 45            | 33  |
| LEN-LONG          | 23          | 58        | 27            | 48  |
| NUMERIC-ENTITY    | 22          | 48        | 29            | 43  |
| NEGATION          | 5           | 20        | 8             | 16  |
| UNCATEGORIZED     | 38          | 47        | 25            | 44  |

Readable column-normalized version: see
`../../figures/failure_by_architecture.png` panel (b).

**What this does NOT say.** The labels attached to a family include its
share of *universally hard* queries, because the sample strategy
over-sampled queries where all six encoders fail (75 queries × 6
encoders = 450 triples). So transformers' 152 NATURAL-QUESTION
failures is not "transformers fail on 152 independent questions" — it
is "when we look at the labeled failure set, 152 of the transformer
rows have the NATURAL-QUESTION label, many shared with the other
families." The pivot is descriptive of the sample, not an incidence
rate over the whole corpus.

## Where pre-training's value concentrates

The query-level set differences tell the clearest story. With success
defined as nDCG@10 > 0.3 and failure as nDCG@10 < 0.1, on each dataset:

|                                    | nfcorpus | scifact | fiqa |
|:-----------------------------------|---------:|--------:|-----:|
| all 6 succeed (universal wins)     | 45       | 62      | 19   |
| all 6 fail (universal losses)      | 85       | 32      | 152  |
| any transformer wins, all 3 FS lose| 38       | 91      | 235  |
| **unique wins (transformers)**     | **26**   | **22**  | **79**|
| **unique wins (from-scratch)**     | **9**    | **1**   | **9** |

The "transformer-gap" row is the most directly interpretable number in
this phase: it counts queries that are *only* accessible to pre-trained
dense retrieval. On SciFact, 30% of queries (91/300) fall in this set;
on FiQA, 36% (235/648) do. These queries are what a reviewer has in
mind when they ask "how much of what you lose going to a from-scratch
encoder is recoverable via more training data vs. structurally
unrecoverable at this scale?" — the answer for this Pareto is 30–36%
of the test set.

Unique-win counts (queries where exactly one encoder succeeds) are
asymmetric: 127 unique transformer wins across the three datasets, 19
unique from-scratch wins. The "route hard queries to the big model and
easy ones to the small one" hypothesis from Session 03's §6.1 is not
supported: no from-scratch encoder uniquely wins on enough queries to
justify that routing (conv-retriever: 1 unique SciFact win, 1 unique
FiQA win; lstm: 0 SciFact, 4 FiQA; mamba: 1 SciFact, 4 FiQA). The
deployment case for small-encoder ensembles would need a larger unique
set than we see here.

Example queries in `../../analysis/failure_examples.md`; per-dataset
qid sets in `cross_encoder_sets/`.

## Query-length sensitivity — the CNN receptive-field story

Mean nDCG@10 by query-length quartile (WordPiece tokens) on the three
datasets is plotted in
`../../figures/query_length_vs_ndcg.png`. Transformers are essentially
flat across all quartiles (nDCG varies by ≤ 0.05 from Q1 to Q4 on
every dataset). Convolutional nDCG drops monotonically with length on
SciFact and FiQA, reaching 0.02 on FiQA Q4 (longest quartile, 17+
WordPiece tokens); LSTM and Mamba drop more gently. FiQA's Q4 bound
(17 tokens) is exactly the threshold at which no single kernel-7
window inside conv-retriever can span the query — the receptive-field
hypothesis from §4c of the handoff predicts this exact shape, and the
plot confirms it.

## Limitations of this analysis

- **Failure threshold is a choice.** `nDCG@10 < 0.1` was picked
  because it corresponds to "no relevant doc in top-5 OR one relevant
  doc at rank 6-10." The public data (`query_frame.parquet`) lets any
  reviewer recompute the analysis at another threshold.
- **Single labeler (the operator).** No inter-annotator agreement.
  Rule-based codes (LEN-*, NATURAL-QUESTION, NUMERIC-ENTITY, NEGATION,
  MULTI-CONCEPT, DOMAIN-TERM) are mechanical and reproducible from
  the query string and `DOMAIN_LEXICON` in
  `scripts/phase6_label_taxonomy.py`. The semantic codes the handoff
  seeded (PARAPHRASE, AMBIGUOUS, MULTI-HOP) are not applied because
  they require corpus access (see next point).
- **No corpus reload.** Handoff rule §6.2 forbids re-downloading
  BEIR; document titles, relevance-passage text, and
  query↔document vocabulary-overlap are therefore out of scope. This
  excludes the PARAPHRASE / MULTI-HOP categories from Phase 6's
  labeling and caps the specificity of the curated failure examples
  (doc titles would have made them more readable).
- **3 BEIR subsets only.** Out-of-domain generalization across the
  wider BEIR benchmark (TREC-COVID, Touché, CQADupStack, …) is not
  verified here. Phases 7 and 8 should reuse the query-frame loader so
  that if a fourth subset is added, nothing else changes.
- **Sampling strategy is stratified, not random.** The 470 labeled
  queries come from targeted strata (conv-fiqa failures, universal
  losses, unique-success pools). This intentionally oversamples
  informative cells; it is not a representative sample of the 7,626-
  row frame. Cross-encoder set counts in the table above *are*
  computed on the full frame; the pivot over architecture × category
  is not.

## Pointers

- [query_frame.parquet](query_frame.parquet) — long-form data, 7,626 rows
- [zero_ndcg_rates.csv](zero_ndcg_rates.csv) + [`zero_ndcg_rates.md`](zero_ndcg_rates.md)
- [spearman_{dataset}.csv](./) — three 6×6 rank-correlation matrices
- [disagreement_{dataset}.csv](./) — pairwise hit/miss at nDCG > 0
- [length_bins.csv](length_bins.csv) + [length_quartile_bounds.csv](length_quartile_bounds.csv)
- [cross_encoder_summary.csv](cross_encoder_summary.csv) + [cross_encoder_sets/](cross_encoder_sets/)
- [failure_pivot.csv](failure_pivot.csv) + [failure_pivot_matrix.csv](failure_pivot_matrix.csv)
- [sampled_qids.json](sampled_qids.json) + [labeled_queries.csv](labeled_queries.csv)
- [../../analysis/failure_taxonomy.md](../../analysis/failure_taxonomy.md)
- [../../analysis/failure_examples.md](../../analysis/failure_examples.md)
- [../../figures/failure_by_architecture.png](../../figures/failure_by_architecture.png) + caption
- [../../figures/query_length_vs_ndcg.png](../../figures/query_length_vs_ndcg.png) + caption
- [../../notebooks/02_phase6_failure_analysis.ipynb](../../notebooks/02_phase6_failure_analysis.ipynb)

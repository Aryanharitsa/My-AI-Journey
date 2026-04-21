# Session 04 Report — Vitruvius Local Phase 6, 2026-04-21

## Milestone delivered

Phase 6 at 70%: per-query failure analysis of 6 encoders × 3 BEIR
datasets. First analysis-heavy session in Vitruvius — every prior phase
*measured*; this one *characterizes where each encoder breaks and why*.

## One-sentence outcome

On BEIR out-of-distribution evaluation, pre-training's value concentrates
in two places — long natural-language questions (where CNN's kernel-7
receptive field fails 90% of FiQA queries while transformers stay flat
across all length quartiles) and domain jargon (where `bert-base`
recovers at rank 1 on SciFact claims that all three MS MARCO-trained
from-scratch encoders miss by rank > 20) — while 23.5% of FiQA is
architecture-agnostic hard, and only 19 unique wins total accrue to
from-scratch encoders across all three datasets.

## Top 3 findings

1. **Transformer-gap queries span 30–36% of SciFact/FiQA.** 91 SciFact
   queries (30%) and 235 FiQA queries (36%) see *any* pre-trained
   transformer succeed (nDCG@10 > 0.3) while *all three* from-scratch
   encoders fail (nDCG@10 < 0.1). This is the concrete size of
   "pre-training's value" on these datasets.

2. **CNN receptive-field hypothesis confirmed visually.** Mean nDCG@10
   by query-length quartile collapses for `conv-retriever` on FiQA's
   Q4 (17+ WordPiece tokens, the exact span a stack of kernel-3/5/7
   convolutions cannot cover) while transformers are near-flat (ΔnDCG ≤
   0.05 Q1→Q4 on every dataset). Plot:
   `figures/query_length_vs_ndcg.png`.

3. **Rank-correlation structure flips as datasets harden.** Within-
   transformer Spearman ρ on per-query nDCG is 0.87–0.91 on nfcorpus
   (stable "who thinks which queries are hard"); CNN-vs-transformer ρ
   on FiQA drops to 0.24 — on long natural-language questions the
   architecture choice changes *what* gets retrieved, not just *how
   well*. Phase 7's head-pruning experiments should target this
   divergence.

## Surprises

- **Zero from-scratch encoders uniquely win on SciFact.** The
  hypothesis that a smaller encoder might out-score transformers on a
  fraction of queries (Session 03 §6.1) is empirically false on SciFact
  — `lstm` and `conv` have 0 unique wins there, `mamba` has 1. FiQA
  helps slightly (conv: 1, lstm: 4, mamba: 4 unique wins) but not
  enough to motivate a routing architecture.
- **`bert-base` is the worst transformer on every dataset.** 40.9%
  zero-nDCG on FiQA vs. 33.3% for `gte-small`. GTE's explicit
  contrastive pre-training matters more than BERT's raw size in this
  retrieval setting — a point worth developing in Phase 9's opinions
  section.
- **`NUMERIC-ENTITY` failures are family-agnostic.** All four families
  carry comparable `NUMERIC-ENTITY` counts (22–48) — consistent with
  WordPiece tokenizing numeric sequences into single-digit pieces
  regardless of encoder architecture. This is a tokenizer story, not an
  architecture story; reviewers should not read the from-scratch
  failures on, e.g., "401k" queries as an architecture verdict.

## Taxonomy final categories (8 total)

Mechanical, rule-based codes (reproducible from the query string):

- `LEN-SHORT` — query ≤ 5 WordPiece tokens
- `LEN-LONG` — query > 20 WordPiece tokens
- `NATURAL-QUESTION` — starts with how/what/why/… or ends in "?"
- `NUMERIC-ENTITY` — contains any digit
- `NEGATION` — contains not/without/fails/lack/…
- `MULTI-CONCEPT` — contains conjunction "and"/"vs"/comma-groups
- `DOMAIN-TERM` — contains any token from the dataset's hand-curated
  lexicon (medical, scientific, or financial)
- `UNCATEGORIZED` — no rule fires; residual (plausibly PARAPHRASE /
  AMBIGUOUS / MULTI-HOP, which require corpus reload to verify and are
  out of scope per handoff §6.2)

Handoff-seeded codes *not* applied: PARAPHRASE, AMBIGUOUS, MULTI-HOP —
require document-text access; Phase 6 did not reload BEIR.

## Sample sizes

- Sampled for labeling: 500 targeted failing queries (seed 1729)
- Actual labeled: 1,116 (encoder, dataset, query) triples covering 470
  distinct (dataset, qid) pairs
- Labeling done by: operator (single labeler — disclosed as limitation
  in `experiments/phase6/SUMMARY.md` and `.../README.md`)
- Labeling time: ~2 hours (mechanical codes via script; failure-
  example curation by inspection)

## Commits

Single-topic, conventional-commit, scope `vitruvius/*`. Final list
visible via `git log --oneline origin/main..HEAD` at handoff time. All
local; **none pushed** — archivist handles the PR.

Expected commit sequence (bundled just before this report):

1. `feat(vitruvius/analysis): replace Phase 1 error_analysis stub with load_query_frame`
2. `feat(vitruvius/scripts): add phase6 analysis + labeling + examples + figure drivers`
3. `feat(vitruvius/experiments): phase 6 — query_frame.parquet + SUMMARY.md + README.md`
4. `feat(vitruvius/analysis): failure_taxonomy.md + failure_examples.md`
5. `feat(vitruvius/figures): failure_by_architecture + query_length_vs_ndcg + captions`
6. `feat(vitruvius/notebooks): 02_phase6_failure_analysis.ipynb (executed)`
7. `chore(vitruvius): CHANGELOG [0.6.0] + README roadmap 70% done + .gitignore figure whitelist`
8. `chore(vitruvius): session_04 bundle + SESSION_04_REPORT.md`

## Limitations this phase does not resolve

- **Single-labeler bias.** Would benefit from a second analyst running
  the same taxonomy on the same sampled queries; inter-annotator
  agreement (Cohen's κ) would quantify taxonomy stability. Out of
  scope here.
- **3 BEIR subsets only.** Out-of-domain generalization across the
  wider BEIR benchmark (TREC-COVID, Touché, CQADupStack, …) not
  verified. Phases 7 / 8 can extend by reusing the `load_query_frame`
  loader; nothing else needs to change.
- **Threshold choice.** `nDCG@10 < 0.1` defensible but not unique.
  Parquet frame allows any reviewer to recompute set arithmetic at
  another threshold.
- **No corpus reload.** Document titles, relevant-passage paraphrases,
  and query↔doc vocabulary-overlap features are absent. Failure
  examples are therefore query-text-plus-ID-only; the PARAPHRASE /
  AMBIGUOUS / MULTI-HOP categories cannot be populated without Phase 6
  turning into a corpus-ingestion project.
- **Sampling is stratified, not random.** The 1,116 labeled triples
  come from targeted strata; the architecture × category pivot
  describes the sample, not the population. Cross-encoder set counts in
  `SUMMARY.md` are computed on the full 7,626-row frame.

## Recommendations for Session 05 (Phase 7 — attention head pruning)

1. **Target FiQA.** It's where the architecture-agnostic Spearman ρ is
   lowest (transformer↔CNN 0.24) and where the transformer-gap set is
   biggest (235 queries). Improvements there are the most publishable.
2. **Reuse `load_query_frame`.** Phase 7's pruning experiments will
   want per-query nDCG before/after pruning. Extending the frame with
   a `prune_config` column keeps the analysis in the same language as
   Phase 6 and lets the existing Spearman/cross-encoder-set code run
   unchanged.
3. **Flag the 152 universal-loss FiQA queries.** If head pruning
   recovers nDCG on these queries, it's a bigger story than matching
   Phase 5 — it's showing that a lighter transformer can do what the
   full one cannot. If it doesn't, the finding remains publishable as
   "this set bounds what any dense retrieval can do."
4. **Expand the `DOMAIN_LEXICON`.** The current list is ~50 tokens per
   dataset; extending it to ~200 would sharpen the `DOMAIN-TERM`
   category before Phase 9's writeup.

## Artifacts in this bundle

```
experiments/session_04/
├── SESSION_04_REPORT.md           (this file)
├── phase6_artifacts_manifest.txt  (sha256 + file list)
└── local_commits.txt              (git log --oneline origin/main..HEAD)
```

The heavy artifacts (`query_frame.parquet`, the eight CSVs, the three
cross_encoder_sets/*.json, labeled_queries.csv, sampled_qids.json,
failure_pivot*.csv, the two figures + captions, the executed notebook,
the two markdown deliverables) all live under their canonical paths
inside the Vitruvius package:

- `vitruvius/experiments/phase6/*`
- `vitruvius/analysis/failure_taxonomy.md`
- `vitruvius/analysis/failure_examples.md`
- `vitruvius/figures/failure_by_architecture.{png,pdf}` + caption
- `vitruvius/figures/query_length_vs_ndcg.{png,pdf}` + caption
- `vitruvius/notebooks/02_phase6_failure_analysis.{py,ipynb}`

The bundle does *not* re-ship them — the session bundle is a pointer
layer, not a storage layer.

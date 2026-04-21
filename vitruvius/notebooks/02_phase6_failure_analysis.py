# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 6 — Per-query failure analysis
#
# Reproducible walkthrough for the Phase 6 deliverable. Everything this
# notebook does is also implemented in the four `scripts/phase6_*.py` driver
# scripts — they are the canonical entry points. The notebook exists so a
# reviewer can read the analysis top-to-bottom without re-running anything.

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from vitruvius.analysis.error_analysis import (
    DEFAULT_DATASETS,
    DEFAULT_ENCODERS,
    ENCODER_FAMILY,
    FAILURE_THRESHOLD,
    SUCCESS_THRESHOLD,
    decode_parquet_columns,
    load_query_frame,
)

OUT = Path("../experiments/phase6")
FIG = Path("../figures")
sns.set_context("notebook")
sns.set_style("whitegrid")

# %% [markdown]
# ## 1. Gate check
#
# Every one of the 18 bench JSONs must contain a `per_query_results` payload
# with the four required keys. If this cell raises, stop and surface to the
# operator — the information is not recoverable from aggregate metrics.

# %%
required = {"nDCG@10", "ranked_doc_ids", "relevance_judgments", "query_text"}
for pct_dir, encs in [
    ("../experiments/phase3", ["minilm-l6-v2", "bert-base", "gte-small"]),
    ("../experiments/phase5/bench", ["lstm-retriever", "conv-retriever", "mamba-retriever-fs"]),
]:
    for enc in encs:
        for ds in DEFAULT_DATASETS:
            p = Path(pct_dir) / f"{enc}__{ds}.json"
            d = json.load(p.open())
            pq = d["per_query_results"]
            first = pq[next(iter(pq))]
            missing = required - set(first.keys())
            assert not missing, f"{p}: missing {missing}"
print("gate check: PASS — 18 cells, full per-query schema present.")

# %% [markdown]
# ## 2. Load the query-level frame
#
# `load_query_frame` produces one row per (encoder, dataset, query) with
# metrics, features, and the full ranked-list + qrels. 7,626 rows expected
# (323 + 300 + 648 queries × 6 encoders).

# %%
df = decode_parquet_columns(pd.read_parquet(OUT / "query_frame.parquet"))
print("rows:", len(df))
df.head(3)

# %% [markdown]
# Unit test: `n_relevant_docs` must be constant across encoders for the same
# (dataset, query_id) — it is a property of the qrels, not the ranking.

# %%
assert (df.groupby(["dataset", "query_id"])["n_relevant_docs"].nunique() == 1).all()
print("n_relevant_docs invariance: OK")

# %% [markdown]
# ## 3. Zero-nDCG@10 rate grid
#
# Extends Session 03's table with the transformer rates. Even pre-trained
# transformers fail on 14–41% of queries out-of-distribution on BEIR — the
# gap to from-scratch is large, not infinite.

# %%
zero_grid = pd.read_csv(OUT / "zero_ndcg_rates.csv", index_col=0)
(zero_grid * 100).round(1)

# %%
fig, ax = plt.subplots(figsize=(6.8, 4.2))
sns.heatmap(
    zero_grid * 100,
    annot=True,
    fmt=".1f",
    cmap="rocket_r",
    vmin=0,
    vmax=100,
    cbar_kws={"label": "zero-nDCG@10 rate (%)"},
    ax=ax,
)
ax.set_title("Zero-nDCG@10 rates — encoder x dataset")
plt.show()

# %% [markdown]
# ## 4. Cross-encoder agreement (Spearman ρ)
#
# High within-family and moderate across-family agreement on nfcorpus;
# transformer-vs-CNN ρ collapses to 0.24 on FiQA — the *pattern* of which
# queries are easy/hard diverges on long natural-language questions.

# %%
sp = {ds: pd.read_csv(OUT / f"spearman_{ds}.csv", index_col=0) for ds in DEFAULT_DATASETS}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, ds in zip(axes, DEFAULT_DATASETS):
    sns.heatmap(sp[ds], annot=True, fmt=".2f", cmap="vlag", vmin=-0.1, vmax=1.0,
                cbar=False, ax=ax, square=True)
    ax.set_title(f"Spearman ρ — {ds}")
fig.suptitle("Per-query nDCG@10 rank correlation across encoders", y=1.02)
plt.show()

# %% [markdown]
# ## 5. Query length vs. retrieval quality
#
# Quartile cutoffs differ per dataset (nfcorpus: q1=2, q3=7; scifact: q1=15,
# q3=25; fiqa: q1=10, q3=17). Transformers are near-flat in length;
# `conv-retriever` drops sharply on FiQA Q4 — the receptive-field hypothesis
# confirmed.

# %%
pd.read_csv(OUT / "length_quartile_bounds.csv")

# %%
lb = pd.read_csv(OUT / "length_bins.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
palette = {
    "minilm-l6-v2": "#1f77b4",
    "bert-base": "#1f77b4",
    "gte-small": "#1f77b4",
    "lstm-retriever": "#d62728",
    "conv-retriever": "#ff7f0e",
    "mamba-retriever-fs": "#2ca02c",
}
linestyle = {
    "minilm-l6-v2": "-",
    "bert-base": "--",
    "gte-small": ":",
    "lstm-retriever": "-",
    "conv-retriever": "-",
    "mamba-retriever-fs": "-",
}
for ax, ds in zip(axes, DEFAULT_DATASETS):
    sub = lb[lb.dataset == ds]
    for enc in DEFAULT_ENCODERS:
        s = sub[sub.encoder == enc].sort_values("length_quartile")
        ax.plot(
            s["length_quartile"].astype(str),
            s["mean_ndcg"],
            marker="o",
            color=palette[enc],
            linestyle=linestyle[enc],
            linewidth=2.2,
            label=enc,
        )
    ax.set_title(ds)
    ax.set_xlabel("query length quartile (WordPiece)")
    ax.tick_params(axis="x", rotation=25)
axes[0].set_ylabel("mean nDCG@10")
axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
plt.show()

# %% [markdown]
# ## 6. Failure taxonomy — architecture family × category
#
# Labels applied to 1,116 (encoder, dataset, query) triples covering 470
# distinct failing queries. Categories are non-exclusive; a single query can
# carry several.

# %%
pivot = pd.read_csv(OUT / "failure_pivot_matrix.csv", index_col=0)
pivot

# %%
order = ["LEN-SHORT", "LEN-LONG", "NATURAL-QUESTION", "NUMERIC-ENTITY",
         "NEGATION", "MULTI-CONCEPT", "DOMAIN-TERM", "UNCATEGORIZED"]
pivot_ordered = pivot.reindex([c for c in order if c in pivot.index])
families = ["transformer", "recurrent", "convolutional", "ssm"]
pivot_ordered = pivot_ordered[families]
frac = pivot_ordered / pivot_ordered.sum(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.05, 1]})
sns.heatmap(pivot_ordered, annot=True, fmt="d", cmap="rocket_r",
            cbar_kws={"label": "count"}, ax=axes[0])
axes[0].set_title("(a) Count — labeled failures per (family, category)")
sns.heatmap(frac, annot=True, fmt=".2f", cmap="rocket_r",
            vmin=0, vmax=frac.values.max(),
            cbar_kws={"label": "share of family's labeled failures"}, ax=axes[1])
axes[1].set_title("(b) Fraction — column-normalized")
fig.suptitle("Failure categories by architecture family", y=1.02)
plt.show()

# %% [markdown]
# ## 7. Cross-encoder wins and losses
#
# Universal wins (all 6 succeed) are small on FiQA — only 19 queries are
# easy for every encoder. Universal losses (all 6 fail) make up 23.5% of
# FiQA, bounding what any dense-retrieval fix can recover.

# %%
xsum = pd.read_csv(OUT / "cross_encoder_summary.csv", index_col=0)
xsum

# %% [markdown]
# The "transformer-gap" column is the most publishable number in the phase:
# queries where *any* pre-trained transformer succeeds but *all three*
# from-scratch encoders fail. 30–36% of FiQA/SciFact is in this set — that
# is the concrete measure of where pre-training's value concentrates.

# %%
for ds in DEFAULT_DATASETS:
    sets = json.load(open(OUT / "cross_encoder_sets" / f"{ds}.json"))
    print(f"{ds:>9} transformer-gap queries: {sets['transformer_gap_queries']['n']}")

# %% [markdown]
# ## 8. Six curated failure examples
#
# See `../analysis/failure_examples.md` for the reviewer-visible version.
# The notebook just surfaces the raw picks so the reader can verify.

# %%
examples_md = Path("../analysis/failure_examples.md").read_text()
print(examples_md[:4000])
print("...")

# %% [markdown]
# ## 9. Limitations reminder
#
# - Failure threshold `nDCG@10 < 0.1` is a choice (documented in
#   `experiments/phase6/SUMMARY.md`).
# - Single labeler; no inter-annotator agreement.
# - No corpus reload → `PARAPHRASE` / `AMBIGUOUS` / `MULTI-HOP` categories
#   are out of scope.
# - 3 BEIR subsets only.
# - Sampling is stratified, not random; architecture × category pivot is
#   descriptive of the sample, not an incidence rate over the whole frame.

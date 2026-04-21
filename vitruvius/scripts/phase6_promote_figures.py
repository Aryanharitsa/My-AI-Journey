"""Phase 6 — build the two top-line figures and their captions.

Promotes the architecture-family-vs-failure-category heatmap and the
query-length-vs-nDCG line plot into ``vitruvius/figures/`` alongside
captions that mirror the ``pareto_v2_caption.md`` style.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vitruvius.analysis.error_analysis import DEFAULT_DATASETS, DEFAULT_ENCODERS

OUT = Path("experiments/phase6")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)


def plot_failure_by_architecture() -> None:
    mat = pd.read_csv(OUT / "failure_pivot_matrix.csv", index_col=0)
    # drop UNCATEGORIZED from the main pane; show the five core categories plus two
    order = [
        "LEN-SHORT",
        "LEN-LONG",
        "NATURAL-QUESTION",
        "NUMERIC-ENTITY",
        "NEGATION",
        "MULTI-CONCEPT",
        "DOMAIN-TERM",
        "UNCATEGORIZED",
    ]
    mat = mat.reindex([c for c in order if c in mat.index])
    families = ["transformer", "recurrent", "convolutional", "ssm"]
    mat = mat[families]

    # normalize each family column to fraction of its total labeled failures
    frac = mat / mat.sum(axis=0)

    sns.set_context("notebook")
    sns.set_style("white")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.05, 1]})

    sns.heatmap(
        mat,
        annot=True,
        fmt="d",
        cmap="rocket_r",
        cbar_kws={"label": "count"},
        ax=axes[0],
    )
    axes[0].set_title("(a) Count — labeled failures per (family, category)")
    axes[0].set_xlabel("architecture family")
    axes[0].set_ylabel("failure category")

    sns.heatmap(
        frac,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        vmin=0,
        vmax=frac.values.max(),
        cbar_kws={"label": "share of family's labeled failures"},
        ax=axes[1],
    )
    axes[1].set_title("(b) Fraction — same counts, column-normalized")
    axes[1].set_xlabel("architecture family")
    axes[1].set_ylabel("")

    fig.suptitle("Phase 6 — failure categories by architecture family", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "failure_by_architecture.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIG / "failure_by_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_query_length_vs_ndcg() -> None:
    lb = pd.read_csv(OUT / "length_bins.csv")
    sns.set_context("notebook")
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
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
        ax.set_ylabel("mean nDCG@10" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0, None)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.suptitle("Query length vs. retrieval quality (3 BEIR subsets)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "query_length_vs_ndcg.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIG / "query_length_vs_ndcg.pdf", bbox_inches="tight")
    plt.close(fig)


def write_captions() -> None:
    (FIG / "failure_by_architecture_caption.md").write_text(
        """# failure_by_architecture — caption

Per-query failures (nDCG@10 < 0.1) from the Phase 6 labeled sample (470
distinct queries, 1,116 (encoder, dataset, query) triples) assigned to the
eight-category failure taxonomy defined in
`vitruvius/analysis/failure_taxonomy.md`. Left panel (a) shows raw counts;
right panel (b) column-normalizes each architecture family so the reader
can compare *distribution of failure types* rather than absolute counts.
Labels are non-exclusive, so column sums are not the number of queries.

**Reading aid.** The cells worth staring at:

- `NATURAL-QUESTION` dominates the convolutional and recurrent columns —
  FiQA's full-sentence question style is the modal failure for both.
- `LEN-SHORT` is the single largest transformer failure, consistent with
  the high zero-nDCG rate on nfcorpus where queries are 2–5 tokens.
- `DOMAIN-TERM` loads roughly evenly across all four families — a reminder
  that jargon coverage is a shared weakness of both web-pre-trained and
  MS MARCO-trained encoders on biomedical / scientific corpora.
- `LEN-LONG` spares transformers but hits recurrent/SSM models hardest.

Sampling procedure and threshold choice documented in
`vitruvius/experiments/phase6/README.md`.
""",
    )
    (FIG / "query_length_vs_ndcg_caption.md").write_text(
        """# query_length_vs_ndcg — caption

Mean nDCG@10 as a function of query-length quartile (BERT WordPiece tokens)
for the six encoders across the three BEIR subsets. Quartile bounds differ
per dataset (nfcorpus: q1=2, q3=7; scifact: q1=15, q3=25; fiqa: q1=10,
q3=17) — see `experiments/phase6/length_quartile_bounds.csv`. Pre-trained
transformers (blue family) are near-flat in length, confirming that
attention-based aggregation handles query-length variation; convolutional
and recurrent models degrade visibly toward the longest quartile on
scifact and fiqa, supporting the receptive-field hypothesis from §4c of
the Phase 6 handoff.

FiQA's right-most quartile is the cleanest demonstration of the CNN
receptive-field ceiling: the `conv-retriever` line drops below all other
encoders as query length crosses ~17 tokens — the point at which no single
kernel-7 window can span the query.
""",
    )


def main() -> None:
    plot_failure_by_architecture()
    plot_query_length_vs_ndcg()
    write_captions()
    print("wrote figures + captions to", FIG)


if __name__ == "__main__":
    main()

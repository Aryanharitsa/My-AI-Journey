"""Phase 6 — per-query failure analysis driver.

Produces:
  experiments/phase6/query_frame.parquet       (built by error_analysis.py)
  experiments/phase6/zero_ndcg_rates.csv       zero-nDCG@10 rate grid
  experiments/phase6/zero_ndcg_rates.md        markdown-rendered grid
  experiments/phase6/spearman_<dataset>.csv    per-dataset 6x6 rank correlation
  experiments/phase6/disagreement_<dataset>.csv pairwise wins/misses (nDCG>0)
  experiments/phase6/length_bins.csv           nDCG@10 by dataset x encoder x quartile
  experiments/phase6/cross_encoder_sets/<ds>.json  universal/unique/gap sets
  experiments/phase6/sampled_qids.json         reproducible manual-read sample
  experiments/phase6/figures/*.png             plots embedded in notebook later

Run from repo root:
    python scripts/phase6_analysis.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from vitruvius.analysis.error_analysis import (
    DEFAULT_DATASETS,
    DEFAULT_ENCODERS,
    ENCODER_FAMILY,
    FAILURE_THRESHOLD,
    SUCCESS_THRESHOLD,
    decode_parquet_columns,
    load_query_frame,
)

OUT = Path("experiments/phase6")
FIG_OUT = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)
(OUT / "cross_encoder_sets").mkdir(parents=True, exist_ok=True)

ENCODER_ORDER = list(DEFAULT_ENCODERS)  # transformers first, then recurrent/conv/ssm


def build_or_load_frame() -> pd.DataFrame:
    pq = OUT / "query_frame.parquet"
    if pq.exists():
        return decode_parquet_columns(pd.read_parquet(pq))
    df = load_query_frame(root=".", stringify_dicts_for_parquet=True)
    df.to_parquet(pq, index=False)
    # return the in-memory dict/list version for downstream use
    return load_query_frame(root=".")


def zero_ndcg_grid(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.assign(is_zero=lambda r: r["nDCG@10"] == 0.0)
        .groupby(["encoder", "dataset"])
        .agg(zero_rate=("is_zero", "mean"), n=("is_zero", "size"), zero_n=("is_zero", "sum"))
        .reset_index()
    )
    pivot = g.pivot(index="encoder", columns="dataset", values="zero_rate").loc[
        ENCODER_ORDER, list(DEFAULT_DATASETS)
    ]
    pivot.to_csv(OUT / "zero_ndcg_rates.csv", float_format="%.4f")
    with (OUT / "zero_ndcg_rates.md").open("w") as fh:
        fh.write("# Zero-nDCG@10 rates (per encoder x dataset)\n\n")
        fh.write("Values are the fraction of evaluation queries for which the\n")
        fh.write("encoder's top-10 list contains no qrels-relevant document.\n\n")
        fh.write(pivot.map(lambda x: f"{100*x:.1f}%").to_markdown() + "\n\n")
        fh.write("Raw counts (failures / total):\n\n")
        counts = g.assign(cell=lambda r: r["zero_n"].astype(str) + "/" + r["n"].astype(str))
        counts_p = counts.pivot(index="encoder", columns="dataset", values="cell").loc[
            ENCODER_ORDER, list(DEFAULT_DATASETS)
        ]
        fh.write(counts_p.to_markdown() + "\n")
    return pivot


def failure_rate_grid(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["encoder", "dataset"])
        .agg(failure_rate=("is_failure", "mean"), mean_ndcg=("nDCG@10", "mean"))
        .reset_index()
    )
    pivot = g.pivot(index="encoder", columns="dataset", values="failure_rate").loc[
        ENCODER_ORDER, list(DEFAULT_DATASETS)
    ]
    pivot.to_csv(OUT / "failure_rates_0p1.csv", float_format="%.4f")
    return pivot


def spearman_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ds in DEFAULT_DATASETS:
        sub = df[df.dataset == ds]
        wide = sub.pivot(index="query_id", columns="encoder", values="nDCG@10")[ENCODER_ORDER]
        n = len(ENCODER_ORDER)
        mat = np.zeros((n, n))
        for i, a in enumerate(ENCODER_ORDER):
            for j, b in enumerate(ENCODER_ORDER):
                if i <= j:
                    rho, _ = spearmanr(wide[a], wide[b])
                    mat[i, j] = rho
                    mat[j, i] = rho
        cm = pd.DataFrame(mat, index=ENCODER_ORDER, columns=ENCODER_ORDER)
        cm.to_csv(OUT / f"spearman_{ds}.csv", float_format="%.3f")
        out[ds] = cm
    return out


def disagreement_counts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ds in DEFAULT_DATASETS:
        sub = df[df.dataset == ds]
        hit = sub.pivot(index="query_id", columns="encoder", values="nDCG@10")[ENCODER_ORDER] > 0
        rows = []
        for a, b in combinations(ENCODER_ORDER, 2):
            both_hit = int((hit[a] & hit[b]).sum())
            only_a = int((hit[a] & ~hit[b]).sum())
            only_b = int((~hit[a] & hit[b]).sum())
            both_miss = int((~hit[a] & ~hit[b]).sum())
            rows.append(
                {
                    "pair": f"{a} vs {b}",
                    "a": a,
                    "b": b,
                    "both_hit": both_hit,
                    "only_a": only_a,
                    "only_b": only_b,
                    "both_miss": both_miss,
                }
            )
        out[ds] = pd.DataFrame(rows)
        out[ds].to_csv(OUT / f"disagreement_{ds}.csv", index=False)
    return out


def length_bins(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for ds in DEFAULT_DATASETS:
        sub = df[df.dataset == ds].copy()
        sub["length_quartile"] = pd.qcut(
            sub["query_length_tokens"],
            q=4,
            labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"],
            duplicates="drop",
        )
        agg = (
            sub.groupby(["dataset", "encoder", "length_quartile"], observed=True)
            .agg(mean_ndcg=("nDCG@10", "mean"), n=("nDCG@10", "size"))
            .reset_index()
        )
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "length_bins.csv", index=False, float_format="%.4f")
    return out


def length_quartile_bounds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DEFAULT_DATASETS:
        sub = df[(df.dataset == ds) & (df.encoder == ENCODER_ORDER[0])]
        q = np.quantile(sub["query_length_tokens"], [0.25, 0.5, 0.75, 1.0])
        rows.append({"dataset": ds, "q1": q[0], "q2_median": q[1], "q3": q[2], "max": q[3]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "length_quartile_bounds.csv", index=False)
    return out


def cross_encoder_sets(df: pd.DataFrame) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    transformer_encs = [e for e, f in ENCODER_FAMILY.items() if f == "transformer"]
    from_scratch_encs = [e for e, f in ENCODER_FAMILY.items() if f != "transformer"]

    for ds in DEFAULT_DATASETS:
        sub = df[df.dataset == ds]
        success = sub.pivot(index="query_id", columns="encoder", values="nDCG@10") > SUCCESS_THRESHOLD
        failure = sub.pivot(index="query_id", columns="encoder", values="nDCG@10") < FAILURE_THRESHOLD
        success = success[ENCODER_ORDER]
        failure = failure[ENCODER_ORDER]

        universal_wins = sorted(success.index[success.all(axis=1)].tolist())
        universal_losses = sorted(failure.index[failure.all(axis=1)].tolist())
        unique_wins: dict[str, list[str]] = {}
        for enc in ENCODER_ORDER:
            others = [e for e in ENCODER_ORDER if e != enc]
            only_this = success[enc] & (~success[others]).all(axis=1)
            unique_wins[enc] = sorted(only_this.index[only_this].tolist())
        any_transformer_success = success[transformer_encs].any(axis=1)
        all_fs_failure = failure[from_scratch_encs].all(axis=1)
        transformer_gap_qids = sorted(
            success.index[any_transformer_success & all_fs_failure].tolist()
        )
        payload = {
            "dataset": ds,
            "n_queries": int(len(success)),
            "success_threshold": SUCCESS_THRESHOLD,
            "failure_threshold": FAILURE_THRESHOLD,
            "universal_wins": {"n": len(universal_wins), "qids": universal_wins},
            "universal_losses": {"n": len(universal_losses), "qids": universal_losses},
            "unique_wins": {enc: {"n": len(v), "qids": v} for enc, v in unique_wins.items()},
            "transformer_gap_queries": {
                "n": len(transformer_gap_qids),
                "qids": transformer_gap_qids,
            },
        }
        with (OUT / "cross_encoder_sets" / f"{ds}.json").open("w") as fh:
            json.dump(payload, fh, indent=2)
        summary[ds] = {
            "universal_wins": len(universal_wins),
            "universal_losses": len(universal_losses),
            "transformer_gap": len(transformer_gap_qids),
            **{f"unique_{e}": len(v) for e, v in unique_wins.items()},
        }
    pd.DataFrame(summary).T.to_csv(OUT / "cross_encoder_summary.csv")
    return summary


def sample_qids_for_manual_read(df: pd.DataFrame) -> dict[str, list]:
    rng = np.random.default_rng(1729)
    samples: dict[str, list] = {}

    def _pick(subset: pd.DataFrame, k: int) -> list[str]:
        ids = sorted(subset["query_id"].unique().tolist())
        if len(ids) <= k:
            return ids
        idx = rng.choice(len(ids), size=k, replace=False)
        return sorted([ids[i] for i in idx])

    failing = df[df.is_failure]

    samples["conv_fiqa"] = _pick(
        failing[(failing.encoder == "conv-retriever") & (failing.dataset == "fiqa")], 150
    )
    samples["lstm_scifact"] = _pick(
        failing[(failing.encoder == "lstm-retriever") & (failing.dataset == "scifact")], 100
    )
    samples["lstm_fiqa"] = _pick(
        failing[(failing.encoder == "lstm-retriever") & (failing.dataset == "fiqa")], 100
    )
    samples["mamba_scifact"] = _pick(
        failing[(failing.encoder == "mamba-retriever-fs") & (failing.dataset == "scifact")], 100
    )

    # universal & unique-success sets across all datasets at failure/success thresholds
    universal_losses: list[tuple[str, str]] = []
    unique_success: list[tuple[str, str, str]] = []
    for ds in DEFAULT_DATASETS:
        sub = df[df.dataset == ds]
        fail_wide = sub.pivot(index="query_id", columns="encoder", values="nDCG@10") < FAILURE_THRESHOLD
        succ_wide = sub.pivot(index="query_id", columns="encoder", values="nDCG@10") > SUCCESS_THRESHOLD
        fail_wide = fail_wide[ENCODER_ORDER]
        succ_wide = succ_wide[ENCODER_ORDER]
        for q in fail_wide.index[fail_wide.all(axis=1)]:
            universal_losses.append((ds, q))
        for enc in ENCODER_ORDER:
            others = [e for e in ENCODER_ORDER if e != enc]
            only_this = succ_wide[enc] & (~succ_wide[others]).all(axis=1)
            for q in only_this.index[only_this]:
                unique_success.append((ds, q, enc))

    samples["universal_losses_pool"] = universal_losses
    samples["unique_success_pool"] = unique_success

    ul_pool = universal_losses
    us_pool = unique_success
    k_ul = min(75, len(ul_pool))
    k_us = min(75, len(us_pool))
    if len(ul_pool) > k_ul:
        idx = rng.choice(len(ul_pool), size=k_ul, replace=False)
        samples["universal_losses_sample"] = sorted([ul_pool[i] for i in idx])
    else:
        samples["universal_losses_sample"] = sorted(ul_pool)
    if len(us_pool) > k_us:
        idx = rng.choice(len(us_pool), size=k_us, replace=False)
        samples["unique_success_sample"] = sorted([us_pool[i] for i in idx])
    else:
        samples["unique_success_sample"] = sorted(us_pool)

    with (OUT / "sampled_qids.json").open("w") as fh:
        json.dump(samples, fh, indent=2)
    return samples


def make_figures(
    df: pd.DataFrame,
    zero_grid: pd.DataFrame,
    length_tbl: pd.DataFrame,
    disagreement: dict[str, pd.DataFrame],
    spearman: dict[str, pd.DataFrame],
) -> None:
    sns.set_context("notebook")
    sns.set_style("whitegrid")

    # zero-nDCG heatmap
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
    ax.set_xlabel("dataset")
    ax.set_ylabel("encoder")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "zero_ndcg_heatmap.png", dpi=150)
    plt.close(fig)

    # spearman matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, ds in zip(axes, DEFAULT_DATASETS):
        sns.heatmap(
            spearman[ds],
            annot=True,
            fmt=".2f",
            cmap="vlag",
            vmin=-0.1,
            vmax=1.0,
            cbar=False,
            ax=ax,
            square=True,
        )
        ax.set_title(f"Spearman ρ — {ds}")
    fig.suptitle("Per-query nDCG@10 rank correlation across encoders", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "spearman_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # query-length vs nDCG lines
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, ds in zip(axes, DEFAULT_DATASETS):
        sub = length_tbl[length_tbl.dataset == ds]
        for enc in ENCODER_ORDER:
            s = sub[sub.encoder == enc].sort_values("length_quartile")
            ax.plot(
                s["length_quartile"].astype(str),
                s["mean_ndcg"],
                marker="o",
                label=enc,
            )
        ax.set_title(ds)
        ax.set_xlabel("query length quartile (WordPiece tokens)")
        ax.set_ylabel("mean nDCG@10" if ds == DEFAULT_DATASETS[0] else "")
        ax.tick_params(axis="x", rotation=20)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("Retrieval quality vs. query length", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "query_length_vs_ndcg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # disagreement stacked bars
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    for ax, ds in zip(axes, DEFAULT_DATASETS):
        d = disagreement[ds].copy()
        d = d.set_index("pair")[["both_hit", "only_a", "only_b", "both_miss"]]
        d.plot(kind="bar", stacked=True, ax=ax, width=0.8, colormap="tab20c")
        ax.set_title(ds)
        ax.set_ylabel("query count" if ds == DEFAULT_DATASETS[0] else "")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=70)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Pairwise encoder disagreement at nDCG@10 > 0", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "disagreement_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build_or_load_frame()
    print(f"[frame] rows={len(df):,}")

    zero = zero_ndcg_grid(df)
    print("[zero_ndcg_rates]\n", zero.to_string(float_format=lambda x: f"{100*x:.1f}%"))

    fail = failure_rate_grid(df)
    print("[failure_rates_<0.1]\n", fail.to_string(float_format=lambda x: f"{100*x:.1f}%"))

    lb = length_bins(df)
    qb = length_quartile_bounds(df)
    print("[length_quartile_bounds]\n", qb.to_string(index=False))

    sp = spearman_matrices(df)
    dis = disagreement_counts(df)
    xsets = cross_encoder_sets(df)
    print("[cross_encoder_summary]")
    for ds, s in xsets.items():
        print(" ", ds, s)

    samples = sample_qids_for_manual_read(df)
    print(
        "[sampled_qids] conv_fiqa=",
        len(samples["conv_fiqa"]),
        "lstm_scifact=",
        len(samples["lstm_scifact"]),
        "lstm_fiqa=",
        len(samples["lstm_fiqa"]),
        "mamba_scifact=",
        len(samples["mamba_scifact"]),
        "universal_losses=",
        len(samples["universal_losses_sample"]),
        "unique_success=",
        len(samples["unique_success_sample"]),
    )

    make_figures(df, zero, lb, dis, sp)
    print("[figures] written to", FIG_OUT)


if __name__ == "__main__":
    main()

"""Phase 7 analysis: Spearman cross-dataset head-stability + figures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ENCODERS = ["minilm-l6-v2", "bert-base", "gte-small"]
DATASETS = ["nfcorpus", "scifact", "fiqa"]
FAMILY_COLORS = {
    "minilm-l6-v2": "#1f77b4",
    "bert-base": "#2a6aa8",
    "gte-small": "#5fa6d6",
}


def _load_importance(d: Path, enc: str, ds: str) -> dict | None:
    p = d / f"{enc}__{ds}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load_cumulative(d: Path, enc: str, ds: str) -> dict | None:
    p = d / f"{enc}__{ds}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _importance_matrix(artifact: dict) -> np.ndarray:
    L = artifact["num_layers"]
    H = artifact["num_heads_per_layer"]
    m = np.zeros((L, H), dtype=np.float32)
    for r in artifact["per_head_results"]:
        m[r["layer"], r["head"]] = r["delta_nDCG@10"]
    return m


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(x))
        return ranks
    ra = _rank(a)
    rb = _rank(b)
    cov = np.mean(ra * rb) - np.mean(ra) * np.mean(rb)
    return float(cov / (np.std(ra) * np.std(rb) + 1e-12))


def stability_analysis(imp_dir: Path) -> dict:
    out = {"per_encoder": {}}
    for enc in ENCODERS:
        mats = {}
        for ds in DATASETS:
            art = _load_importance(imp_dir, enc, ds)
            if art is not None:
                mats[ds] = _importance_matrix(art).flatten()
        if len(mats) < 2:
            continue
        pairs = {}
        dss = list(mats.keys())
        for i in range(len(dss)):
            for j in range(i + 1, len(dss)):
                rho = spearman(mats[dss[i]], mats[dss[j]])
                pairs[f"rho({dss[i]},{dss[j]})"] = round(rho, 4)
        out["per_encoder"][enc] = {
            "num_heads": len(next(iter(mats.values()))),
            "pairwise_spearman": pairs,
            "mean_rho": round(float(np.mean(list(pairs.values()))), 4),
        }
    return out


def heatmap_for_encoder(imp_dir: Path, enc: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arts = {ds: _load_importance(imp_dir, enc, ds) for ds in DATASETS}
    arts = {k: v for k, v in arts.items() if v is not None}
    if not arts:
        return
    fig, axes = plt.subplots(1, len(arts), figsize=(5 * len(arts), 4),
                             constrained_layout=True)
    if len(arts) == 1:
        axes = [axes]
    all_deltas = np.concatenate([_importance_matrix(a).flatten() for a in arts.values()])
    vmin, vmax = float(all_deltas.min()), float(all_deltas.max())
    for ax, (ds, art) in zip(axes, arts.items()):
        m = _importance_matrix(art)
        im = ax.imshow(m, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        ax.set_xlabel("head index")
        ax.set_ylabel("layer")
        ax.set_title(f"{ds}\n(baseline nDCG@10 = {art['baseline_nDCG@10']:.4f})")
        ax.set_xticks(range(art["num_heads_per_layer"]))
        ax.set_yticks(range(art["num_layers"]))
        flat = [(r["layer"], r["head"], r["delta_nDCG@10"]) for r in art["per_head_results"]]
        flat.sort(key=lambda x: x[2], reverse=True)
        for layer, head, d in flat[:5]:
            ax.plot(head, layer, marker="*", markersize=14, color="white",
                    markeredgecolor="black", markeredgewidth=1)
    fig.suptitle(f"{enc} — per-head Δ nDCG@10 when ablated (red = more essential)",
                 fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Δ nDCG@10 when head ablated")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def by_layer_figure(imp_dir: Path, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ENCODERS), figsize=(5 * len(ENCODERS), 4),
                             constrained_layout=True)
    for ax, enc in zip(axes, ENCODERS):
        arts = [_load_importance(imp_dir, enc, ds) for ds in DATASETS]
        arts = [a for a in arts if a is not None]
        if not arts:
            continue
        L = arts[0]["num_layers"]
        per_layer = []
        labels = []
        for layer in range(L):
            vals = []
            for a in arts:
                for r in a["per_head_results"]:
                    if r["layer"] == layer:
                        vals.append(r["delta_nDCG@10"])
            per_layer.append(vals)
            labels.append(f"L{layer}")
        bp = ax.boxplot(per_layer, labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(FAMILY_COLORS[enc])
            patch.set_alpha(0.7)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_title(enc)
        ax.set_xlabel("layer")
        ax.set_ylabel("Δ nDCG@10 when head ablated")
    fig.suptitle("Per-layer head-importance distribution (combined over 3 BEIR subsets)",
                 fontsize=11)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def cumulative_figure(cum_dir: Path, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4),
                             constrained_layout=True, sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for enc in ENCODERS:
            art = _load_cumulative(cum_dir, enc, ds)
            if art is None:
                continue
            xs = [pt["heads_pruned"] / art["total_heads"] for pt in art["curve"]]
            ys = [pt["nDCG@10"] / art["baseline_nDCG@10"] for pt in art["curve"]]
            ax.plot(xs, ys, marker="o", label=enc, color=FAMILY_COLORS[enc], linewidth=2)
            t5 = art["thresholds"]["heads_prunable_at_5pct_drop"]
            if t5 is not None:
                ax.plot(t5 / art["total_heads"], 0.95, marker="X", markersize=10,
                        color=FAMILY_COLORS[enc], markeredgecolor="black")
        ax.axhline(0.95, color="gray", linestyle="--", alpha=0.6, label="95% baseline")
        ax.axhline(0.90, color="gray", linestyle=":", alpha=0.6, label="90% baseline")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("fraction of heads pruned")
        ax.set_ylabel("nDCG@10 relative to baseline")
        ax.set_title(ds)
        if ds == DATASETS[0]:
            ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("Cumulative pruning — nDCG@10 relative to baseline vs fraction of heads pruned",
                 fontsize=11)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--imp-dir", type=Path, default=Path("experiments/phase7/head_importance"))
    ap.add_argument("--cum-dir", type=Path, default=Path("experiments/phase7/cumulative_pruning"))
    ap.add_argument("--figures-dir", type=Path, default=Path("figures"))
    ap.add_argument("--analysis-out", type=Path,
                    default=Path("experiments/phase7/head_stability_analysis.md"))
    ap.add_argument("--stability-json-out", type=Path,
                    default=Path("experiments/phase7/head_stability.json"))
    args = ap.parse_args(argv)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_out.parent.mkdir(parents=True, exist_ok=True)

    stab = stability_analysis(args.imp_dir)
    args.stability_json_out.write_text(json.dumps(stab, indent=2, sort_keys=True))

    md = ["# Phase 7 — Cross-dataset head-importance stability", ""]
    md.append("Spearman rank correlation between per-head importance vectors "
              "(flattened across (layer, head)) across the 3 BEIR subsets. "
              "High ρ means the same heads matter on different datasets — "
              "evidence for a **universal retrieval-essential head set** per "
              "encoder that a structured-pruning recipe could target. Low ρ "
              "means important heads are domain-specific.")
    md.append("")
    md.append("| Encoder | ρ(nfcorpus,scifact) | ρ(nfcorpus,fiqa) | ρ(scifact,fiqa) | Mean ρ |")
    md.append("|---|---:|---:|---:|---:|")
    for enc, d in stab["per_encoder"].items():
        pairs = d["pairwise_spearman"]
        md.append(
            f"| `{enc}` | {pairs.get('rho(nfcorpus,scifact)', '—')} | "
            f"{pairs.get('rho(nfcorpus,fiqa)', '—')} | "
            f"{pairs.get('rho(scifact,fiqa)', '—')} | **{d['mean_rho']}** |"
        )
    md.append("")
    md.append("## Interpretation")
    md.append("")
    for enc, d in stab["per_encoder"].items():
        r = d["mean_rho"]
        if r >= 0.7:
            kind = "**high**"
            verdict = ("head-importance transfers strongly across domains; a small, "
                       "stable head-essential set is a candidate for a structured "
                       "pruning recipe.")
        elif r >= 0.4:
            kind = "**moderate**"
            verdict = ("head-importance partially transfers; some universal essential "
                       "heads exist but cumulative pruning fractions should be set "
                       "conservatively per-domain.")
        else:
            kind = "**low**"
            verdict = ("head-importance is largely domain-specific; a universal "
                       "pruning recipe would under-serve at least one dataset.")
        md.append(f"- `{enc}` (mean ρ = {r}): {kind} cross-dataset stability — {verdict}")
    md.append("")
    args.analysis_out.write_text("\n".join(md) + "\n")

    for enc in ENCODERS:
        heatmap_for_encoder(args.imp_dir, enc,
                            args.figures_dir / f"head_importance_heatmap_{enc}.png")
        (args.figures_dir / f"head_importance_heatmap_{enc}_caption.md").write_text(
            f"# Head-importance heatmap — `{enc}`\n\n"
            f"Three panels, one per BEIR test subset. Each cell = Δ nDCG@10 when "
            f"that (layer, head) is ablated (all others left intact). Redder = "
            f"more essential. White stars mark top-5 most-important heads per "
            f"panel. Color scale shared across panels for direct cross-dataset "
            f"comparison.\n\n"
            f"Source: `experiments/phase7/head_importance/{enc}__*.json`.\n"
        )

    by_layer_figure(args.imp_dir, args.figures_dir / "head_importance_by_layer.png")
    (args.figures_dir / "head_importance_by_layer_caption.md").write_text(
        "# Per-layer head-importance distribution\n\n"
        "Box plot per (encoder, layer): distribution of Δ nDCG@10 across the "
        "heads in that layer, combined over the 3 BEIR subsets. Tests the "
        "intuition that early layers (surface features) are more prunable "
        "than late layers (task-specific).\n\n"
        "Source: `experiments/phase7/head_importance/`.\n"
    )

    cumulative_figure(args.cum_dir, args.figures_dir / "cumulative_pruning_curves.png")
    (args.figures_dir / "cumulative_pruning_curves_caption.md").write_text(
        "# Cumulative pruning curves\n\n"
        "Three panels, one per BEIR test subset. X-axis: fraction of heads "
        "pruned. Y-axis: nDCG@10 relative to baseline. Each line = one encoder. "
        "The 0.95 / 0.90 horizontal reference lines correspond to 5% / 10% "
        "relative drop; each encoder's X marker annotates the 5%-drop crossing.\n\n"
        "Ordering: single-head-importance ascending (least-important first). "
        "This is NOT the true cumulative-optimal set (heads can compensate); "
        "a Taylor-saliency or iterative-greedy baseline would likely push "
        "these curves higher. Flagged per handoff §7.4 as a limitation.\n\n"
        "Source: `experiments/phase7/cumulative_pruning/`.\n"
    )

    print(json.dumps({
        "stability_md": str(args.analysis_out),
        "stability_json": str(args.stability_json_out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

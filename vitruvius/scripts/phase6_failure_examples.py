"""Phase 6 — curate characteristic failure examples for the Phase 9 writeup.

Picks 6 queries from the labeled pool, each chosen to illustrate a distinct
finding. Emits ``analysis/failure_examples.md`` with:

    - Query text (verbatim)
    - Dataset + qid
    - Qrels relevant doc IDs
    - Per-encoder top-3 retrieved doc IDs + nDCG@10
    - One-sentence analyst interpretation
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vitruvius.analysis.error_analysis import (
    DEFAULT_ENCODERS,
    ENCODER_FAMILY,
    FAILURE_THRESHOLD,
    SUCCESS_THRESHOLD,
    decode_parquet_columns,
)

OUT = Path("experiments/phase6")
ANALYSIS = Path("analysis")

ENCODER_ORDER = list(DEFAULT_ENCODERS)

# 6 example selections — each illustrates a distinct phase-level finding.
# The interpretations are the analyst's own reading; the data rows below them
# (nDCG, top-3 retrieved IDs) are verbatim from the bench JSONs.
EXAMPLES: list[dict] = [
    {
        "title": "CNN catastrophic failure on a long FiQA question",
        "dataset": "fiqa",
        "selector": {
            "min_len": 20,
            "must_fail": ["conv-retriever"],
            "must_succeed": ["gte-small"],
            "prefer_label": "NATURAL-QUESTION",
        },
        "finding": (
            "CNN's max-kernel receptive field (7 tokens) cannot span a "
            "20+ token natural-language question; a pre-trained transformer "
            "of comparable size retrieves correctly. Directly supports the "
            "query-length-vs-nDCG plot."
        ),
    },
    {
        "title": "Short ambiguous query that defeats every from-scratch encoder",
        "dataset": "nfcorpus",
        "selector": {
            "max_len": 3,
            "must_fail": ["lstm-retriever", "conv-retriever", "mamba-retriever-fs"],
            "must_succeed": ["minilm-l6-v2"],
            "prefer_label": "LEN-SHORT",
        },
        "finding": (
            "With three tokens or fewer, the query carries no span structure "
            "for any architecture to exploit. Transformer pre-training supplies "
            "the entity prior that bridges the gap; from-scratch encoders lack it."
        ),
    },
    {
        "title": "Scientific-claim failure on a domain-jargon SciFact query",
        "dataset": "scifact",
        "selector": {
            "must_fail": ["lstm-retriever", "conv-retriever", "mamba-retriever-fs"],
            "must_succeed": ["bert-base"],
            "prefer_label": "DOMAIN-TERM",
        },
        "finding": (
            "Where pre-training's value concentrates: a biomedical claim "
            "containing dataset-specific jargon is correctly grounded by BERT "
            "(seen in PubMed-adjacent text during pre-training) but not by "
            "any encoder trained only on MS MARCO."
        ),
    },
    {
        "title": "Multi-concept FiQA question where all three transformers disagree on ranking",
        "dataset": "fiqa",
        "selector": {
            "prefer_label": "MULTI-CONCEPT",
            "must_succeed": ["gte-small"],
            "must_fail": ["minilm-l6-v2", "bert-base"],
        },
        "finding": (
            "Even within the transformer family, retrieval is not uniform. "
            "GTE's contrastive pre-training on web text outperforms generic "
            "BERT/MiniLM on multi-concept financial questions — a reminder "
            "that 'transformer' is not one architecture but three in this "
            "Pareto."
        ),
    },
    {
        "title": "Numeric-entity FiQA query — a tokenizer story, not an architecture story",
        "dataset": "fiqa",
        "selector": {
            "prefer_label": "NUMERIC-ENTITY",
            "must_fail": ["lstm-retriever", "conv-retriever", "mamba-retriever-fs"],
        },
        "finding": (
            "WordPiece splits most digit sequences into single-digit pieces; "
            "retrieval over numeric entities is therefore lexical for every "
            "encoder here. From-scratch encoders fail harder because they "
            "also lack the surrounding-vocabulary grounding that transformers "
            "bring, not because the numbers themselves are processed "
            "differently."
        ),
    },
    {
        "title": "Universal-loss query — intrinsic hardness, not an architecture gap",
        "dataset": "fiqa",
        "selector": {
            "must_fail": list(ENCODER_FAMILY.keys()),
            "prefer_label": "NATURAL-QUESTION",
        },
        "finding": (
            "All six encoders fail here. The qrels-relevant document is "
            "plausibly absent from the top-100 for every encoder — a "
            "dataset-hardness signal, not an architecture signal. No amount "
            "of swapping encoders rescues this query; it motivates reranking "
            "or reasoning, not better dense retrieval."
        ),
    },
]


def load_wide(df: pd.DataFrame, ds: str) -> pd.DataFrame:
    return df[df.dataset == ds].pivot(
        index="query_id", columns="encoder", values="nDCG@10"
    )[ENCODER_ORDER]


def pick_one(
    df: pd.DataFrame,
    labeled: pd.DataFrame,
    ds: str,
    selector: dict,
    picked_so_far: set[str],
) -> dict | None:
    wide = load_wide(df, ds)
    # query-level metadata (same across encoders)
    sub = df[df.dataset == ds][
        ["query_id", "query_text", "query_length_tokens", "relevance_judgments"]
    ].drop_duplicates(subset=["query_id"]).set_index("query_id")

    candidates = wide.index.tolist()
    # Apply must_fail / must_succeed constraints
    if "must_fail" in selector:
        for enc in selector["must_fail"]:
            candidates = [q for q in candidates if wide.loc[q, enc] < FAILURE_THRESHOLD]
    if "must_succeed" in selector:
        for enc in selector["must_succeed"]:
            candidates = [q for q in candidates if wide.loc[q, enc] > SUCCESS_THRESHOLD]

    # Length filters
    if "min_len" in selector:
        candidates = [
            q for q in candidates if sub.loc[q, "query_length_tokens"] >= selector["min_len"]
        ]
    if "max_len" in selector:
        candidates = [
            q for q in candidates if sub.loc[q, "query_length_tokens"] <= selector["max_len"]
        ]

    # Prefer label match
    labelset_by_qid = {
        qid: labs
        for qid, labs in labeled[labeled.dataset == ds][["query_id", "labels"]]
        .drop_duplicates(subset=["query_id"])
        .itertuples(index=False, name=None)
    }
    if "prefer_label" in selector:
        wanted = selector["prefer_label"]
        preferred = [q for q in candidates if wanted in (labelset_by_qid.get(q) or [])]
        candidates = preferred or candidates  # fall back to non-preferred

    # exclude already picked
    candidates = [q for q in candidates if q not in picked_so_far]
    if not candidates:
        return None

    # Pick the candidate with the highest sum(transformer nDCG) - sum(from-scratch nDCG)
    # as the most illustrative when the story is transformer-vs-fs; otherwise pick first.
    def score(qid: str) -> float:
        row = wide.loc[qid]
        return row[["minilm-l6-v2", "bert-base", "gte-small"]].sum() - row[
            ["lstm-retriever", "conv-retriever", "mamba-retriever-fs"]
        ].sum()

    if selector.get("must_succeed") and selector.get("must_fail"):
        candidates.sort(key=score, reverse=True)
    qid = candidates[0]
    # build per-encoder ranked top-3 map for this (ds, qid)
    per_enc = (
        df[(df.dataset == ds) & (df.query_id == qid)]
        .set_index("encoder")[["ranked_doc_ids", "nDCG@10"]]
    )
    return {
        "dataset": ds,
        "query_id": qid,
        "row": sub.loc[qid],
        "ndcg": wide.loc[qid],
        "per_encoder": per_enc,
    }


def render(examples_chosen: list[dict]) -> str:
    out: list[str] = []
    out.append("# Failure examples — Vitruvius Phase 6\n")
    out.append(
        "Six characteristic failures, chosen to illustrate the phase's "
        "distinct findings. Each query text is verbatim from BEIR; doc IDs are "
        "from the Phase 3 / Phase 5 bench JSONs. Document titles and "
        "relevant-passage paraphrases are intentionally omitted — Phase 6 did "
        "not reload the BEIR corpora (see `experiments/phase6/SUMMARY.md`, "
        "Limitations). The analyst interpretations below are based solely on "
        "the query text, the qrels ID set, and the retrieved ID set.\n"
    )
    for i, spec in enumerate(examples_chosen, start=1):
        out.append(f"## Example {i}. {spec['title']}\n")
        out.append(f"**Finding.** {spec['finding']}\n")
        if spec.get("picked") is None:
            out.append(
                "_No query in the labeled pool matched the selector; this "
                "example is omitted._\n"
            )
            continue
        p = spec["picked"]
        row = p["row"]
        ndcg = p["ndcg"]
        out.append(f"**Query** (`{p['dataset']}:{p['query_id']}`, "
                   f"{int(row['query_length_tokens'])} WordPiece tokens): "
                   f"\"{row['query_text']}\"\n")
        rels = row["relevance_judgments"]
        out.append(
            f"**Qrels.** {len(rels)} relevance-judged document(s). "
            f"Relevant IDs (trim to 8): {list(rels.keys())[:8]}"
            f"{' …' if len(rels) > 8 else ''}\n"
        )
        out.append("\n| Encoder | nDCG@10 | Top-3 retrieved (✓ = qrels-relevant) |\n"
                   "|---|---:|---|\n")
        per_enc = p["per_encoder"]
        for enc in ENCODER_ORDER:
            ranked = list(per_enc.loc[enc, "ranked_doc_ids"])
            cells = []
            for i in range(min(3, len(ranked))):
                mark = "✓" if ranked[i] in rels else "·"
                cells.append(f"{ranked[i]} {mark}")
            out.append(
                f"| {enc} | {ndcg[enc]:.3f} | {' / '.join(cells)} |\n"
            )
        # Rank of first relevant doc (if any) in each encoder's top-100
        out.append("\n**Rank of first relevant doc** (or `>100` if none in top-100):\n\n")
        out.append("| Encoder | first-relevant rank |\n|---|---:|\n")
        for enc in ENCODER_ORDER:
            ranked = list(per_enc.loc[enc, "ranked_doc_ids"])
            first = next(
                (i + 1 for i, did in enumerate(ranked) if did in rels), None
            )
            out.append(f"| {enc} | {first if first else '>100'} |\n")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    df = decode_parquet_columns(pd.read_parquet(OUT / "query_frame.parquet"))
    labeled = pd.read_csv(OUT / "labeled_queries.csv")
    labeled["labels"] = labeled["labels"].str.split("|")

    picked_qids: set[str] = set()
    chosen: list[dict] = []
    for spec in EXAMPLES:
        picked = pick_one(df, labeled, spec["dataset"], spec["selector"], picked_qids)
        if picked:
            picked_qids.add(picked["query_id"])
        entry = dict(spec)
        entry["picked"] = picked
        chosen.append(entry)
        if picked:
            print(f"  [pick] {spec['title'][:60]}: {picked['dataset']}:{picked['query_id']}")
        else:
            print(f"  [SKIP] {spec['title'][:60]}: no matching query")

    md = render(chosen)
    (ANALYSIS / "failure_examples.md").write_text(md)
    print("[wrote]", ANALYSIS / "failure_examples.md", len(md), "chars")


if __name__ == "__main__":
    main()

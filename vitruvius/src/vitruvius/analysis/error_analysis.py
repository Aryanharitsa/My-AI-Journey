"""Per-query failure analysis for Phase 6.

Reads the ``per_query_results`` payload preserved by Phase 3 and Phase 5
bench runs and materializes a long-form DataFrame, one row per
(encoder, dataset, query), with query-level metrics and features.
"""
from __future__ import annotations

import json
from ast import literal_eval
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

ENCODER_FAMILY: dict[str, str] = {
    "minilm-l6-v2": "transformer",
    "bert-base": "transformer",
    "gte-small": "transformer",
    "lstm-retriever": "recurrent",
    "conv-retriever": "convolutional",
    "mamba-retriever-fs": "ssm",
}

DEFAULT_ENCODERS: tuple[str, ...] = tuple(ENCODER_FAMILY.keys())
DEFAULT_DATASETS: tuple[str, ...] = ("nfcorpus", "scifact", "fiqa")
DEFAULT_BENCH_DIRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("experiments/phase3", ("minilm-l6-v2", "bert-base", "gte-small")),
    ("experiments/phase5/bench", ("lstm-retriever", "conv-retriever", "mamba-retriever-fs")),
)

FAILURE_THRESHOLD = 0.1
SUCCESS_THRESHOLD = 0.3


@lru_cache(maxsize=1)
def _bert_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-uncased")


def _count_wordpiece(text: str) -> int:
    return len(_bert_tokenizer().tokenize(text))


def _coerce_ranked(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        parsed = literal_eval(raw)
        return [str(x) for x in parsed]
    raise TypeError(f"unexpected ranked_doc_ids type: {type(raw).__name__}")


def _coerce_rels(raw) -> dict[str, int]:
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, str):
        parsed = literal_eval(raw)
        return {str(k): int(v) for k, v in parsed.items()}
    raise TypeError(f"unexpected relevance_judgments type: {type(raw).__name__}")


@dataclass(frozen=True)
class CellRef:
    encoder: str
    dataset: str
    path: Path


def discover_cells(
    bench_dirs: Iterable[tuple[str, Iterable[str]]] = DEFAULT_BENCH_DIRS,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    root: Path | str = ".",
) -> list[CellRef]:
    root = Path(root)
    cells: list[CellRef] = []
    for pct_dir, encs in bench_dirs:
        for enc in encs:
            for ds in datasets:
                p = root / pct_dir / f"{enc}__{ds}.json"
                if not p.exists():
                    raise FileNotFoundError(p)
                cells.append(CellRef(encoder=enc, dataset=ds, path=p))
    return cells


def _row_from_query(
    encoder: str,
    dataset: str,
    qid: str,
    payload: dict,
) -> dict:
    ranked = _coerce_ranked(payload["ranked_doc_ids"])
    rels = _coerce_rels(payload["relevance_judgments"])
    query_text = payload["query_text"]
    ndcg = float(payload["nDCG@10"])
    hit = bool(payload.get("hit@10", any(d in rels for d in ranked[:10])))

    def _rel(i: int) -> bool:
        return i < len(ranked) and ranked[i] in rels

    return {
        "encoder": encoder,
        "encoder_family": ENCODER_FAMILY[encoder],
        "dataset": dataset,
        "query_id": str(qid),
        "query_text": query_text,
        "query_length_tokens": _count_wordpiece(query_text),
        "query_length_chars": len(query_text),
        "nDCG@10": ndcg,
        "hit@10": hit,
        "Recall@10": float(payload.get("Recall@10", float("nan"))),
        "ranked_doc_ids": ranked,
        "relevance_judgments": rels,
        "n_relevant_docs": len(rels),
        "top1_is_relevant": _rel(0),
        "top3_is_relevant": any(_rel(i) for i in range(3)),
        "is_failure": ndcg < FAILURE_THRESHOLD,
        "is_success": ndcg > SUCCESS_THRESHOLD,
    }


def load_query_frame(
    bench_dirs: list[tuple[str, list[str]]] | None = None,
    encoders: list[str] | None = None,
    datasets: list[str] | None = None,
    root: Path | str = ".",
    *,
    stringify_dicts_for_parquet: bool = False,
) -> pd.DataFrame:
    """Build the long-form (encoder, dataset, query) DataFrame.

    Columns:
      encoder, encoder_family, dataset, query_id, query_text,
      query_length_tokens, query_length_chars,
      nDCG@10, hit@10, Recall@10,
      ranked_doc_ids (list), relevance_judgments (dict),
      n_relevant_docs, top1_is_relevant, top3_is_relevant,
      is_failure, is_success
    """
    if bench_dirs is None:
        bench_dirs = [(pct, list(encs)) for pct, encs in DEFAULT_BENCH_DIRS]
    if encoders is None:
        encoders = list(DEFAULT_ENCODERS)
    if datasets is None:
        datasets = list(DEFAULT_DATASETS)

    rows: list[dict] = []
    for pct_dir, encs in bench_dirs:
        for enc in encs:
            if enc not in encoders:
                continue
            for ds in datasets:
                p = Path(root) / pct_dir / f"{enc}__{ds}.json"
                with p.open() as fh:
                    blob = json.load(fh)
                pq = blob["per_query_results"]
                for qid, payload in pq.items():
                    rows.append(_row_from_query(enc, ds, qid, payload))
    frame = pd.DataFrame(rows)
    if stringify_dicts_for_parquet:
        # pyarrow infers a single struct schema across dict cells, which collapses
        # per-query relevance_judgments into a shared key set. Serialize to JSON
        # strings to preserve per-row contents.
        frame["relevance_judgments"] = frame["relevance_judgments"].apply(json.dumps)
        frame["ranked_doc_ids"] = frame["ranked_doc_ids"].apply(json.dumps)
    return frame


def decode_parquet_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Undo ``stringify_dicts_for_parquet`` — call after ``pd.read_parquet``."""
    out = df.copy()
    for col, want_type in (("relevance_judgments", dict), ("ranked_doc_ids", list)):
        sample = out[col].iloc[0]
        if isinstance(sample, str):
            out[col] = out[col].apply(json.loads)
        elif isinstance(sample, want_type):
            continue
        else:
            raise TypeError(f"{col}: unexpected parquet encoding ({type(sample).__name__})")
    return out


__all__ = [
    "ENCODER_FAMILY",
    "FAILURE_THRESHOLD",
    "SUCCESS_THRESHOLD",
    "CellRef",
    "discover_cells",
    "load_query_frame",
    "decode_parquet_columns",
]

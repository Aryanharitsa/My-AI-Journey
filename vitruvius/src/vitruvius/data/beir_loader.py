"""Wrapper around the BEIR GenericDataLoader.

Phase 1: do not auto-download. If the dataset isn't present on disk, raise a
clear error pointing the operator at ``scripts/download_beir.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vitruvius.utils.logging import get_logger

_log = get_logger(__name__)

DEFAULT_BEIR_ROOT = Path("data/beir")
SUPPORTED_DATASETS = ("nfcorpus", "scifact", "fiqa")


@dataclass
class BEIRSplit:
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    name: str
    split: str


def load_beir(
    dataset: str,
    split: str = "test",
    root: Path | str = DEFAULT_BEIR_ROOT,
) -> BEIRSplit:
    """Load a BEIR dataset split. Raises FileNotFoundError if not present.

    The directory layout mirrors what ``scripts/download_beir.py`` produces:
        <root>/<dataset>/{corpus.jsonl, queries.jsonl, qrels/<split>.tsv}
    """
    root = Path(root)
    dataset_dir = root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"BEIR dataset {dataset!r} not found at {dataset_dir}. "
            f"Run: python scripts/download_beir.py --datasets {dataset}"
        )

    try:
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as e:
        raise ImportError(
            "beir is required to load real BEIR data. Install with `uv pip install beir`."
        ) from e

    _log.info("beir.load dataset=%s split=%s root=%s", dataset, split, dataset_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_dir)).load(split=split)
    _log.info(
        "beir.loaded dataset=%s split=%s n_corpus=%d n_queries=%d n_qrels=%d",
        dataset, split, len(corpus), len(queries), len(qrels),
    )
    return BEIRSplit(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        name=dataset,
        split=split,
    )

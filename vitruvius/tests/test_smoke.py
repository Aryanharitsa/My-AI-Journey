"""End-to-end CPU smoke test on synthetic data.

Runs the same path the ``vitruvius smoke --no-encoder`` CLI does so we never
need to load a real model just to verify wiring.
"""
from __future__ import annotations

import importlib

from vitruvius.cli import _hash_embed, main
from vitruvius.data.synthetic import make_corpus
from vitruvius.evaluation.faiss_index import IndexWrapper
from vitruvius.evaluation.retrieval_metrics import evaluate

EXPECTED_MODULES = [
    "vitruvius",
    "vitruvius.cli",
    "vitruvius.config",
    "vitruvius.utils.device",
    "vitruvius.utils.seed",
    "vitruvius.utils.logging",
    "vitruvius.encoders",
    "vitruvius.encoders.base",
    "vitruvius.encoders.minilm_encoder",
    "vitruvius.encoders.bert_encoder",
    "vitruvius.encoders.gte_encoder",
    "vitruvius.encoders.mamba_encoder",
    "vitruvius.encoders.lstm_encoder",
    "vitruvius.encoders.conv_encoder",
    "vitruvius.data",
    "vitruvius.data.synthetic",
    "vitruvius.data.beir_loader",
    "vitruvius.evaluation",
    "vitruvius.evaluation.retrieval_metrics",
    "vitruvius.evaluation.faiss_index",
    "vitruvius.evaluation.latency_profiler",
    "vitruvius.analysis",
    "vitruvius.analysis.error_analysis",
    "vitruvius.analysis.attention_pruning",
    "vitruvius.analysis.position_sensitivity",
]


def test_every_module_imports():
    for m in EXPECTED_MODULES:
        importlib.import_module(m)


def test_synthetic_pipeline_end_to_end():
    corpus = make_corpus(seed=1729)
    docids = list(corpus.corpus.keys())
    qids = list(corpus.queries.keys())

    doc_texts = [f"{d['title']} {d['text']}" for d in corpus.corpus.values()]
    doc_emb = _hash_embed(doc_texts, dim=256)
    q_emb = _hash_embed(list(corpus.queries.values()), dim=256)

    index = IndexWrapper(dim=256)
    index.add(doc_emb, docids)
    scores, retrieved = index.search(q_emb, top_k=10)

    run = {
        qids[i]: [(retrieved[i][j], float(scores[i][j])) for j in range(len(retrieved[i]))]
        for i in range(len(qids))
    }
    metrics = evaluate(corpus.qrels, run, ks=(1, 5, 10))

    # Hash-bag at dim=256 has enough collision-free room that topic-word
    # overlap dominates noise — random baseline is ~0.2, observed ~0.6.
    assert metrics["nDCG@10"] > 0.45, metrics


def test_cli_smoke_exits_zero():
    rc = main(["smoke", "--cpu", "--no-encoder"])
    assert rc == 0

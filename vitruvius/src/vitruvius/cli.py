"""Command-line entry for Project Vitruvius.

Subcommands implemented in Phase 1:
    smoke     — synthetic-data end-to-end CPU sanity run

Subcommands stubbed (return non-zero) until later phases:
    bench     — Phase 2+ (BEIR retrieval)
    profile   — Phase 3+ (latency profiling)
    shuffle   — Phase 8  (position sensitivity)
    prune     — Phase 7  (attention head pruning)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from vitruvius import __version__
from vitruvius.utils.logging import get_logger
from vitruvius.utils.seed import set_seed

_log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vitruvius",
        description=f"Dense retrieval encoder architecture study (v{__version__})",
    )
    p.add_argument("--version", action="version", version=f"vitruvius {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke", help="run the synthetic-data smoke test")
    p_smoke.add_argument("--cpu", action="store_true", help="force CPU (smoke default)")
    p_smoke.add_argument("--seed", type=int, default=1729)
    p_smoke.add_argument("--no-encoder", action="store_true",
                         help="skip MiniLM load; use a hash-based stand-in embedding")

    p_bench = sub.add_parser("bench", help="benchmark an encoder on a BEIR dataset")
    p_bench.add_argument("--encoder", required=True)
    p_bench.add_argument("--dataset", required=True)
    p_bench.add_argument("--split", default="test")
    p_bench.add_argument("--batch-size", type=int, default=32)
    p_bench.add_argument("--top-k", type=int, default=100)
    p_bench.add_argument("--limit", type=int, default=None)
    p_bench.add_argument("--output", type=str, default=None)
    p_bench.add_argument("--device", default="auto")

    p_prof = sub.add_parser("profile", help="latency-only profile of an encoder")
    p_prof.add_argument("--encoder", required=True)
    p_prof.add_argument("--batch-sizes", default="1,8,32")
    p_prof.add_argument("--device", default="auto")

    p_shuf = sub.add_parser("shuffle", help="position-sensitivity probe")
    p_shuf.add_argument("--encoder", required=True)
    p_shuf.add_argument("--dataset", required=True)

    p_prune = sub.add_parser("prune", help="attention head pruning probe")
    p_prune.add_argument("--encoder", required=True)
    p_prune.add_argument("--dataset", required=True)

    return p


def _hash_embed(texts: list[str], dim: int = 256) -> np.ndarray:  # noqa: F821
    """Cheap deterministic embedding via word-hash bag-of-features.

    Used in the smoke test when no real model is loaded so the synthetic
    corpus still produces a non-degenerate ranking signal.
    """
    import hashlib

    import numpy as np

    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def _cmd_smoke(args: argparse.Namespace) -> int:

    from vitruvius.data.synthetic import make_corpus
    from vitruvius.evaluation.faiss_index import IndexWrapper
    from vitruvius.evaluation.retrieval_metrics import evaluate

    set_seed(args.seed)

    _log.info("smoke.start mode=%s", "no-encoder" if args.no_encoder else "auto")
    corpus = make_corpus(seed=args.seed)
    _log.info("smoke.corpus n_queries=%d n_docs=%d",
              len(corpus.queries), len(corpus.corpus))

    if args.no_encoder:
        dim = 256
        encoder_name = "hash-bag-of-features"
        doc_texts = [f"{d['title']} {d['text']}" for d in corpus.corpus.values()]
        doc_emb = _hash_embed(doc_texts, dim=dim)
        q_emb = _hash_embed(list(corpus.queries.values()), dim=dim)
    else:
        try:
            from vitruvius.encoders import get_encoder

            enc = get_encoder("minilm-l6-v2", device="cpu")
            dim = enc.embedding_dim
            encoder_name = enc.name
            doc_texts = [f"{d['title']} {d['text']}" for d in corpus.corpus.values()]
            doc_emb = enc.encode_documents(doc_texts, batch_size=32)
            q_emb = enc.encode_queries(list(corpus.queries.values()), batch_size=32)
        except Exception as e:
            _log.warning("smoke.encoder_failed err=%r — falling back to hash bag", e)
            dim = 256
            encoder_name = "hash-bag-of-features"
            doc_texts = [f"{d['title']} {d['text']}" for d in corpus.corpus.values()]
            doc_emb = _hash_embed(doc_texts, dim=dim)
            q_emb = _hash_embed(list(corpus.queries.values()), dim=dim)

    index = IndexWrapper(dim=dim)
    docids = list(corpus.corpus.keys())
    index.add(doc_emb, docids)

    qids = list(corpus.queries.keys())
    scores, retrieved = index.search(q_emb, top_k=10)
    run = {
        qids[i]: [(retrieved[i][j], float(scores[i][j])) for j in range(len(retrieved[i]))]
        for i in range(len(qids))
    }

    metrics = evaluate(corpus.qrels, run, ks=(1, 5, 10))

    summary = {
        "encoder": encoder_name,
        "embedding_dim": int(dim),
        "n_queries": len(qids),
        "n_docs": len(docids),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    _log.info("smoke.done status=ok")
    return 0


def _not_yet(phase: str) -> int:
    _log.error("not implemented yet — see %s", phase)
    return 2


def _cmd_bench(args: argparse.Namespace) -> int:
    return _not_yet("Phase 2 (10% milestone)")


def _cmd_profile(args: argparse.Namespace) -> int:
    return _not_yet("Phase 3.5 (latency profiler)")


def _cmd_shuffle(args: argparse.Namespace) -> int:
    return _not_yet("Phase 8 (position sensitivity)")


def _cmd_prune(args: argparse.Namespace) -> int:
    return _not_yet("Phase 7 (attention head pruning)")


_DISPATCH = {
    "smoke": _cmd_smoke,
    "bench": _cmd_bench,
    "profile": _cmd_profile,
    "shuffle": _cmd_shuffle,
    "prune": _cmd_prune,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _DISPATCH[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

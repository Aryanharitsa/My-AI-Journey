"""Thin wrapper around ``faiss.IndexFlatIP`` for inner-product search."""
from __future__ import annotations

import platform

import numpy as np

from vitruvius.utils.logging import get_logger

_log = get_logger(__name__)
_FAISS_CONFIGURED = False


def _configure_faiss(faiss_module) -> None:
    """Apply per-platform threading workarounds. Idempotent."""
    global _FAISS_CONFIGURED
    if _FAISS_CONFIGURED:
        return
    if platform.system() == "Darwin":
        # Avoid the macOS faiss-cpu / torch libomp collision (segfaults on search).
        faiss_module.omp_set_num_threads(1)
        _log.info("faiss.config platform=Darwin threads=1 (libomp workaround)")
    _FAISS_CONFIGURED = True


class IndexWrapper:
    def __init__(self, dim: int):
        import faiss

        _configure_faiss(faiss)
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._docids: list[str] = []
        _log.info("faiss.init dim=%d kind=IndexFlatIP", dim)

    def add(self, doc_embeddings: np.ndarray, docids: list[str]) -> None:
        if doc_embeddings.ndim != 2 or doc_embeddings.shape[1] != self._dim:
            raise ValueError(
                f"expected (n, {self._dim}) got {doc_embeddings.shape}"
            )
        if len(docids) != doc_embeddings.shape[0]:
            raise ValueError("docids length must match embedding count")
        emb = np.ascontiguousarray(doc_embeddings, dtype=np.float32)
        self._index.add(emb)
        self._docids.extend(docids)

    def search(
        self, query_embeddings: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, list[list[str]]]:
        emb = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        scores, indices = self._index.search(emb, top_k)
        docids: list[list[str]] = []
        for row in indices:
            docids.append([self._docids[i] if 0 <= i < len(self._docids) else "" for i in row])
        return scores, docids

    def __len__(self) -> int:
        return self._index.ntotal

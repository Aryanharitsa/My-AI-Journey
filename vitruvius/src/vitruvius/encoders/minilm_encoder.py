from __future__ import annotations

import numpy as np

from vitruvius.encoders.base import Encoder
from vitruvius.utils.device import pick_device
from vitruvius.utils.logging import get_logger

_log = get_logger(__name__)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class MiniLMEncoder(Encoder):
    """Real sentence-transformers wrapper for all-MiniLM-L6-v2.

    Used as the Phase 1 reference encoder and as the Phase 2 BEIR sanity model.
    """

    similarity = "cosine"

    def __init__(self, device: str | None = None, model_id: str = MODEL_ID):
        from sentence_transformers import SentenceTransformer

        self._name = "minilm-l6-v2"
        self._embedding_dim = EMBEDDING_DIM
        self._device = pick_device(device)
        _log.info("encoder.load name=%s model_id=%s device=%s",
                  self._name, model_id, self._device)
        self._model = SentenceTransformer(model_id, device=str(self._device))

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        emb = self._model.encode(
            queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        emb = self._model.encode(
            documents,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)

    def to(self, device):  # type: ignore[override]
        super().to(device)
        self._model = self._model.to(str(self._device))
        return self

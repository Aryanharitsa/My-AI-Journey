from __future__ import annotations

import numpy as np

from vitruvius.encoders.base import Encoder
from vitruvius.utils.device import pick_device
from vitruvius.utils.logging import get_logger

_log = get_logger(__name__)

MODEL_ID = "thenlper/gte-small"
EMBEDDING_DIM = 384


class GTEEncoder(Encoder):
    """Wrapper for thenlper/gte-small. Real impl, not loaded eagerly in tests."""

    def __init__(self, device: str | None = None, model_id: str = MODEL_ID):
        from sentence_transformers import SentenceTransformer

        self._name = "gte-small"
        self._embedding_dim = EMBEDDING_DIM
        self._device = pick_device(device)
        _log.info("encoder.load name=%s model_id=%s device=%s",
                  self._name, model_id, self._device)
        self._model = SentenceTransformer(model_id, device=str(self._device))

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        emb = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(queries, batch_size)

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(documents, batch_size)

    def to(self, device):  # type: ignore[override]
        super().to(device)
        self._model = self._model.to(str(self._device))
        return self

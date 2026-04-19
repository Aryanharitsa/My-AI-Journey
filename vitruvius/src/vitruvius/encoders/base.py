from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch

Similarity = Literal["cosine", "dot"]


class Encoder(ABC):
    """Abstract base for all dense-retrieval encoders.

    Subclasses must populate ``self._name``, ``self._embedding_dim``, and
    ``self._device`` in ``__init__``; implement the two encode methods; and
    declare a ``similarity`` class attribute ("cosine" or "dot") that matches
    the checkpoint's training objective.

    The ``similarity`` attribute is load-bearing: the benchmark harness uses
    it to decide whether to L2-normalize embeddings before FAISS IndexFlatIP.
    Dot-trained checkpoints (e.g. msmarco-bert-base-dot-v5) lose ~0.08 nDCG@10
    on BEIR if forced through cosine.
    """

    _name: str
    _embedding_dim: int
    _device: torch.device

    @property
    @abstractmethod
    def similarity(self) -> Similarity:
        """Training objective — "cosine" (embeddings L2-normalized) or "dot" (raw IP)."""

    @property
    def name(self) -> str:
        return self._name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        """Return a (len(queries), embedding_dim) float32 array."""

    @abstractmethod
    def encode_documents(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        """Return a (len(documents), embedding_dim) float32 array."""

    def to(self, device: str | torch.device) -> Encoder:
        """Move underlying model to the given device. Default is no-op."""
        self._device = torch.device(device) if isinstance(device, str) else device
        return self

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self._name!r}, "
            f"dim={self._embedding_dim}, similarity={self.similarity!r}, device={self._device})"
        )

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class Encoder(ABC):
    """Abstract base for all dense-retrieval encoders.

    Subclasses must populate ``self._name``, ``self._embedding_dim``, and
    ``self._device`` in ``__init__``, and implement the two encode methods.
    """

    _name: str
    _embedding_dim: int
    _device: torch.device

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
        return f"{self.__class__.__name__}(name={self._name!r}, dim={self._embedding_dim}, device={self._device})"

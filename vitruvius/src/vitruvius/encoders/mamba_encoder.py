from __future__ import annotations

import numpy as np
import torch

from vitruvius.encoders.base import Encoder

_PHASE = "Phase 4 — see roadmap (M40, Mamba Retriever integration)"


class MambaEncoder(Encoder):
    """Stub. Real implementation lands in Phase 4."""

    def __init__(self, device: str | None = None):
        self._name = "mamba-stub"
        self._embedding_dim = 0
        self._device = torch.device(device) if device else torch.device("cpu")

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError(_PHASE)

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError(_PHASE)

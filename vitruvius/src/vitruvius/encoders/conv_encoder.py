"""From-scratch 1D-CNN bi-encoder for dense retrieval.

Session-03 Phase 5. Three stacked 1D convs with increasing kernel sizes
(3, 5, 7) to model progressively longer local n-grams. Global max-pool
concatenated with global mean-pool, then a projection head.

Architecture (handoff §5.3.2):
    BERT WordPiece tokenizer (reused, NOT trained)
    Embedding: 128-dim
    Conv1d(128 -> 256, k=3, pad=1) + ReLU
    Conv1d(256 -> 256, k=5, pad=2) + ReLU
    Conv1d(256 -> 256, k=7, pad=3) + ReLU
    Dropout 0.1 between layers
    Global max-pool ⊕ global mean-pool → 512
    Linear(512 -> 256)
    similarity = "cosine"; L2-normalize

Target param count: 5–7M.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from vitruvius.encoders.base import Encoder
from vitruvius.utils.device import pick_device
from vitruvius.utils.logging import get_logger

_log = get_logger(__name__)


VOCAB_SIZE = 30522
EMBED_DIM = 128
CONV_DIM = 256
PROJ_DIM = 256
DROPOUT = 0.1
MAX_SEQ_LEN = 128
DEFAULT_TOKENIZER_ID = "bert-base-uncased"


class ConvRetrieverBody(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        conv_dim: int = CONV_DIM,
        proj_dim: int = PROJ_DIM,
        dropout: float = DROPOUT,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.conv1 = nn.Conv1d(embed_dim, conv_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_dim, conv_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(conv_dim, conv_dim, kernel_size=7, padding=3)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(2 * conv_dim, proj_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)  # (B, L, E)
        x = x.transpose(1, 2)  # (B, E, L) for Conv1d
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.act(self.conv2(x)))
        x = self.act(self.conv3(x))  # (B, C, L)

        # Mask invalid positions before pooling. Broadcast mask to (B, 1, L).
        mask = attention_mask.unsqueeze(1).float()
        # For max-pool: fill invalid positions with a very negative value so
        # they never win.
        x_for_max = x.masked_fill(mask == 0, float("-1e9"))
        mx, _ = x_for_max.max(dim=2)  # (B, C)

        # For mean-pool: zero-out invalid positions, divide by valid count.
        x_for_mean = x * mask
        lens = mask.sum(dim=2).clamp(min=1.0)
        mn = x_for_mean.sum(dim=2) / lens  # (B, C)

        pooled = torch.cat([mx, mn], dim=-1)  # (B, 2C)
        return self.proj(pooled)


class ConvRetrieverEncoder(Encoder):
    """From-scratch 1D-CNN bi-encoder wrapper implementing the Encoder interface."""

    similarity = "cosine"

    def __init__(
        self,
        device: str | None = None,
        checkpoint_path: str | Path | None = None,
        tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    ) -> None:
        from transformers import AutoTokenizer

        self._name = "conv-retriever"
        self._embedding_dim = PROJ_DIM
        self._device = pick_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self._body = ConvRetrieverBody(pad_id=self._tokenizer.pad_token_id or 0)
        n_params = sum(p.numel() for p in self._body.parameters())
        _log.info(
            "encoder.build name=%s params=%.2fM dim=%d device=%s",
            self._name, n_params / 1e6, self._embedding_dim, self._device,
        )
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "body_state_dict" in state:
                state = state["body_state_dict"]
            self._body.load_state_dict(state, strict=True)
            _log.info("encoder.checkpoint_loaded path=%s", checkpoint_path)
        self._body.to(self._device)
        self._body.eval()

    @torch.no_grad()
    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        all_out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self._device)
            attn = enc["attention_mask"].to(self._device)
            out = self._body(input_ids, attn)
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
            all_out.append(out.detach().cpu().to(torch.float32).numpy())
        return np.concatenate(all_out, axis=0) if all_out else np.zeros((0, PROJ_DIM), dtype=np.float32)

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(queries, batch_size)

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(documents, batch_size)

    def to(self, device):  # type: ignore[override]
        super().to(device)
        self._body = self._body.to(self._device)
        return self

    @property
    def body(self) -> ConvRetrieverBody:
        return self._body

    @property
    def tokenizer(self):
        return self._tokenizer


# Alias for registry compatibility.
ConvEncoder = ConvRetrieverEncoder

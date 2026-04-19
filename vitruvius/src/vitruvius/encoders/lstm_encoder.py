"""From-scratch BiLSTM bi-encoder for dense retrieval.

Session-03 Phase 5. Trained on MS MARCO 500k via InfoNCE with in-batch
negatives. See ``src/vitruvius/training/`` for the training loop.

Architecture (handoff §5.3.1):
    BERT WordPiece tokenizer (reused, NOT trained)
    Embedding: 128-dim, vocab 30,522
    2-layer BiLSTM, hidden 256 (→ 512 after bi-concat), dropout 0.1
    Masked mean-pool over valid tokens
    Linear(512 → 256) projection
    similarity = "cosine"; L2-normalize

Target param count: 6–8M. Actual count logged at init.
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
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.1
PROJ_DIM = 256
MAX_SEQ_LEN = 128
DEFAULT_TOKENIZER_ID = "bert-base-uncased"


class LSTMRetrieverBody(nn.Module):
    """Pure-PyTorch body: embedding -> BiLSTM -> masked mean-pool -> projection."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
        proj_dim: int = PROJ_DIM,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(2 * lstm_hidden, proj_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L), attention_mask: (B, L) with 1 on valid tokens.
        x = self.embed(input_ids)
        out, _ = self.lstm(x)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (out * mask).sum(dim=1)
        lens = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lens
        return self.proj(pooled)


class LSTMRetrieverEncoder(Encoder):
    """From-scratch BiLSTM bi-encoder wrapper implementing the Encoder interface."""

    similarity = "cosine"

    def __init__(
        self,
        device: str | None = None,
        checkpoint_path: str | Path | None = None,
        tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    ) -> None:
        from transformers import AutoTokenizer

        self._name = "lstm-retriever"
        self._embedding_dim = PROJ_DIM
        self._device = pick_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self._body = LSTMRetrieverBody(pad_id=self._tokenizer.pad_token_id or 0)
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
    def body(self) -> LSTMRetrieverBody:
        return self._body

    @property
    def tokenizer(self):
        return self._tokenizer


# Backwards-compatible alias for the stub class name the registry knows.
LSTMEncoder = LSTMRetrieverEncoder

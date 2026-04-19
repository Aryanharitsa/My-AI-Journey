"""From-scratch Mamba2 bi-encoder for dense retrieval.

Session-03 Phase 5. Trained on MS MARCO 500k via InfoNCE with in-batch
negatives — identical training regime to LSTM/CNN so the Pareto comparison
is apples-to-apples. The "fs" suffix distinguishes this from any future
pre-trained Mamba retrieval checkpoint (session-02 attempted that path,
hit the cross-encoder SPScanner wall, deferred).

Architecture (handoff §5.3.3):
    BERT WordPiece tokenizer (reused, NOT trained — same as LSTM/CNN)
    Embedding: 384-dim (matches d_model)
    12-layer Mamba2 stack, d_model=384, d_state=128
    Masked mean-pool over valid tokens
    Linear(384 -> 256) projection
    similarity = "cosine"; L2-normalize

Target param count: 15-25M. Actual count logged at init.

Import is lazy so this module is safely importable on pods where mamba-ssm
hasn't been installed — the `Encoder` interface sees `NotImplementedError`
at construction time if the packages are missing.
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
D_MODEL = 384
D_STATE = 128
N_LAYERS = 12
PROJ_DIM = 256
MAX_SEQ_LEN = 128
DEFAULT_TOKENIZER_ID = "bert-base-uncased"


def _have_mamba() -> bool:
    try:
        import causal_conv1d  # noqa: F401
        import mamba_ssm  # noqa: F401
    except Exception:
        return False
    return True


class MambaRetrieverBody(nn.Module):
    """12-layer Mamba2 trunk + masked mean-pool + projection.

    Built from ``mamba_ssm.Mamba2`` blocks directly (not the LM-head wrapper)
    so we can control pooling. RMSNorm applied after the stack.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        d_state: int = D_STATE,
        n_layers: int = N_LAYERS,
        proj_dim: int = PROJ_DIM,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm.modules.mamba2 import Mamba2
            try:
                from mamba_ssm.ops.triton.layer_norm import RMSNorm
            except Exception:
                from mamba_ssm.ops.triton.layernorm import RMSNorm
        except ImportError as e:
            raise NotImplementedError(
                "mamba-ssm / causal-conv1d not installed — see "
                "notes/mamba_install_attempt_02.md for the install runbook."
            ) from e

        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.layers = nn.ModuleList(
            [Mamba2(d_model=d_model, d_state=d_state) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, proj_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer(x)  # residual
        x = self.norm(x)

        mask = attention_mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        lens = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lens
        return self.proj(pooled)


class MambaRetrieverEncoder(Encoder):
    """From-scratch Mamba2 bi-encoder wrapper."""

    similarity = "cosine"

    def __init__(
        self,
        device: str | None = None,
        checkpoint_path: str | Path | None = None,
        tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    ) -> None:
        if not _have_mamba():
            raise NotImplementedError(
                "mamba-ssm / causal-conv1d not installed. Install per "
                "notes/mamba_install_attempt_02.md (or skip and use "
                "lstm-retriever / conv-retriever for a 5-point Pareto)."
            )
        from transformers import AutoTokenizer

        self._name = "mamba-retriever-fs"
        self._embedding_dim = PROJ_DIM
        self._device = pick_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self._body = MambaRetrieverBody(pad_id=self._tokenizer.pad_token_id or 0)
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
    def body(self) -> MambaRetrieverBody:
        return self._body

    @property
    def tokenizer(self):
        return self._tokenizer


# Alias
MambaEncoder = MambaRetrieverEncoder

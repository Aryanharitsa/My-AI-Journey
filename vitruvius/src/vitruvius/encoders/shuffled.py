"""ShuffledEncoder wrapper — applies token-position shuffle before encoding.

Phase 8 wrapper around any existing Encoder. For each text, the wrapper
tokenizes, shuffles content-token positions (if the mode asks), then
forwards through the underlying model's body.

For from-scratch encoders (lstm, conv, mamba) we bypass the wrapper's own
encode pathway and call the body directly with shuffled ids.
For pre-trained transformers we use the underlying AutoModel + pool + norm
path from Phase 7's manual-forward approach (avoids sentence-transformers'
internal tokenize+sort which would undo our shuffle).
"""
from __future__ import annotations

import numpy as np
import torch

from vitruvius.encoders.base import Encoder
from vitruvius.utils.device import pick_device
from vitruvius.utils.logging import get_logger
from vitruvius.utils.shuffle import shuffle_input_ids

_log = get_logger(__name__)


# Pre-trained transformer config (mirrors scripts/head_importance_sweep.py)
_TRANSFORMER_CFG = {
    "minilm-l6-v2": {
        "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
        "pool": "mean", "similarity": "cosine", "max_len": 256,
    },
    "bert-base": {
        "hf_id": "sentence-transformers/msmarco-bert-base-dot-v5",
        "pool": "mean", "similarity": "dot", "max_len": 350,
    },
    "gte-small": {
        "hf_id": "thenlper/gte-small",
        # GTE-small uses MEAN pooling (verified from thenlper/gte-small
        # 1_Pooling/config.json: pooling_mode_mean_tokens=true).
        "pool": "mean", "similarity": "cosine", "max_len": 512,
    },
}


def _mean_pool(last_hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    m = attn.unsqueeze(-1).float()
    return (last_hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def _cls_pool(last_hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    return last_hidden[:, 0, :]


class ShuffledEncoder(Encoder):
    """Wraps an Encoder and applies token-position shuffle before each forward.

    Args:
        base_encoder_name: alias of one of the 6 encoders.
        shuffle_queries: if True, shuffles content positions in query inputs.
        shuffle_docs: if True, shuffles content positions in doc inputs.
        seed: base shuffle seed. Per-sample seed is (seed + position_in_batch).
        device: cuda/cpu.
        checkpoint_path: for from-scratch encoders (lstm-retriever,
            conv-retriever, mamba-retriever-fs), path to the trained .pt body.
    """

    def __init__(
        self,
        base_encoder_name: str,
        shuffle_queries: bool,
        shuffle_docs: bool,
        seed: int,
        device: str | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        self._base_name = base_encoder_name
        self._shuffle_q = bool(shuffle_queries)
        self._shuffle_d = bool(shuffle_docs)
        self._seed = int(seed)
        self._device = pick_device(device)
        self._name = (
            f"shuffled-{base_encoder_name}-"
            f"q{int(self._shuffle_q)}d{int(self._shuffle_d)}"
        )

        if base_encoder_name in _TRANSFORMER_CFG:
            self._kind = "transformer"
            self._load_transformer(base_encoder_name)
        elif base_encoder_name in ("lstm-retriever", "conv-retriever",
                                   "mamba-retriever-fs"):
            self._kind = "fromscratch"
            self._load_fromscratch(base_encoder_name, checkpoint_path)
        else:
            raise ValueError(f"Unknown encoder {base_encoder_name!r}")

    # ---- Pre-trained transformer path --------------------------------
    def _load_transformer(self, alias: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        cfg = _TRANSFORMER_CFG[alias]
        self._similarity = cfg["similarity"]
        self._max_len = cfg["max_len"]
        self._pool_name = cfg["pool"]
        self._tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
        self._model = AutoModel.from_pretrained(cfg["hf_id"]).to(self._device).eval()
        self._embedding_dim = self._model.config.hidden_size
        # All tokenizer special-ids (CLS, SEP, PAD, MASK, unknown)
        self._special_ids = {
            tid for tid in self._tokenizer.all_special_ids if tid is not None
        }

    def _encode_transformer(self, texts: list[str], shuffle: bool,
                            batch_size: int) -> np.ndarray:
        out: list[np.ndarray] = []
        pool_fn = _mean_pool if self._pool_name == "mean" else _cls_pool
        normalize = (self._similarity == "cosine")

        # Tokenize whole set, then iterate batches (so shuffle-positions-within-batch
        # are stable and seed logic is deterministic).
        enc = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self._max_len,
            return_tensors="pt",
        )
        ids_all = enc["input_ids"]
        attn_all = enc["attention_mask"]
        if shuffle:
            ids_all = shuffle_input_ids(ids_all, attn_all, self._special_ids, self._seed)

        n = ids_all.shape[0]
        with torch.no_grad():
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                ids = ids_all[s:e].to(self._device, non_blocking=True)
                am = attn_all[s:e].to(self._device, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    r = self._model(input_ids=ids, attention_mask=am)
                    pooled = pool_fn(r.last_hidden_state, am)
                    if normalize:
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                out.append(pooled.detach().to(torch.float32).cpu().numpy())
        return np.concatenate(out, axis=0)

    # ---- From-scratch encoder path -----------------------------------
    def _load_fromscratch(self, alias: str, checkpoint_path: str | None) -> None:
        from vitruvius.encoders import get_encoder

        self._underlying = get_encoder(
            alias, device=str(self._device),
            **({"checkpoint_path": checkpoint_path} if checkpoint_path else {}),
        )
        self._similarity = self._underlying.similarity
        self._embedding_dim = self._underlying.embedding_dim
        self._tokenizer = self._underlying.tokenizer
        self._body = self._underlying.body
        self._body.eval()
        self._special_ids = {
            tid for tid in self._tokenizer.all_special_ids if tid is not None
        }
        # From-scratch encoders all use BERT tokenizer max_len=128 (matches training)
        self._max_len = 128

    def _encode_fromscratch(self, texts: list[str], shuffle: bool,
                            batch_size: int) -> np.ndarray:
        out: list[np.ndarray] = []
        with torch.no_grad():
            for s in range(0, len(texts), batch_size):
                batch = texts[s:s + batch_size]
                enc = self._tokenizer(
                    batch, padding=True, truncation=True, max_length=self._max_len,
                    return_tensors="pt",
                )
                ids = enc["input_ids"]
                attn = enc["attention_mask"]
                if shuffle:
                    ids = shuffle_input_ids(ids, attn, self._special_ids, self._seed + s)
                ids = ids.to(self._device, non_blocking=True)
                attn = attn.to(self._device, non_blocking=True)
                pooled = self._body(ids, attn)
                if self._similarity == "cosine":
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                out.append(pooled.detach().to(torch.float32).cpu().numpy())
        return np.concatenate(out, axis=0)

    # ---- Encoder interface ------------------------------------------
    @property
    def similarity(self) -> str:  # type: ignore[override]
        return self._similarity

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        shuffle = self._shuffle_q
        if self._kind == "transformer":
            return self._encode_transformer(queries, shuffle, batch_size).astype(
                np.float32, copy=False,
            )
        return self._encode_fromscratch(queries, shuffle, batch_size).astype(
            np.float32, copy=False,
        )

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        shuffle = self._shuffle_d
        if self._kind == "transformer":
            return self._encode_transformer(documents, shuffle, batch_size).astype(
                np.float32, copy=False,
            )
        return self._encode_fromscratch(documents, shuffle, batch_size).astype(
            np.float32, copy=False,
        )

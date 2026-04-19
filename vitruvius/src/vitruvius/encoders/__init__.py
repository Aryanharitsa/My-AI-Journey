from __future__ import annotations

from collections.abc import Callable

from vitruvius.encoders.base import Encoder

# Lazy registry: each value is a zero-arg factory that imports + instantiates
# its encoder. Lazy because heavy deps (sentence-transformers) shouldn't load
# on `import vitruvius`.
_REGISTRY: dict[str, Callable[..., Encoder]] = {}


def _register(name: str, factory: Callable[..., Encoder]) -> None:
    _REGISTRY[name] = factory


def _make_minilm(**kw) -> Encoder:
    from vitruvius.encoders.minilm_encoder import MiniLMEncoder

    return MiniLMEncoder(**kw)


def _make_bert(**kw) -> Encoder:
    from vitruvius.encoders.bert_encoder import BERTEncoder

    return BERTEncoder(**kw)


def _make_gte(**kw) -> Encoder:
    from vitruvius.encoders.gte_encoder import GTEEncoder

    return GTEEncoder(**kw)


def _make_mamba(**kw) -> Encoder:
    from vitruvius.encoders.mamba_encoder import MambaEncoder

    return MambaEncoder(**kw)


def _make_lstm(**kw) -> Encoder:
    from vitruvius.encoders.lstm_encoder import LSTMEncoder

    return LSTMEncoder(**kw)


def _make_conv(**kw) -> Encoder:
    from vitruvius.encoders.conv_encoder import ConvEncoder

    return ConvEncoder(**kw)


_register("minilm-l6-v2", _make_minilm)
_register("bert-base", _make_bert)
_register("gte-small", _make_gte)
_register("mamba-retriever-fs", _make_mamba)
_register("lstm-retriever", _make_lstm)
_register("conv-retriever", _make_conv)


def list_encoders() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_encoder(name: str, device: str = "auto", **kwargs) -> Encoder:
    """Look up and instantiate an encoder by name."""
    if name not in _REGISTRY:
        supported = ", ".join(list_encoders())
        raise ValueError(f"Unknown encoder {name!r}. Supported: {supported}")
    return _REGISTRY[name](device=device, **kwargs)


__all__ = ["Encoder", "get_encoder", "list_encoders"]

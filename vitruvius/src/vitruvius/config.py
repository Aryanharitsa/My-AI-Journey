"""Experiment configs. Pydantic for friendly validation errors."""
from __future__ import annotations

from pydantic import BaseModel, Field


class BenchConfig(BaseModel):
    encoder: str
    dataset: str
    split: str = "test"
    batch_size: int = Field(32, ge=1)
    top_k: int = Field(100, ge=1)
    limit: int | None = None
    output: str | None = None
    device: str = "auto"
    seed: int = 1729


class ProfileConfig(BaseModel):
    encoder: str
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 8, 32])
    n_warmup: int = 10
    n_measure: int = 100
    device: str = "auto"
    seed: int = 1729

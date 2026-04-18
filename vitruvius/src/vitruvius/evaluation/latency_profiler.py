"""Latency profiler.

CUDA path uses ``torch.cuda.Event`` for accurate device-side timing. CPU/MPS
paths fall back to ``time.perf_counter()`` and emit a one-time warning.
"""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence

import torch

from vitruvius.utils.logging import get_logger

_log = get_logger(__name__)


def _measure_cuda(fn: Callable[[int], None], batch_size: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn(batch_size)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # milliseconds


def _measure_wall(fn: Callable[[int], None], batch_size: int) -> float:
    t0 = time.perf_counter()
    fn(batch_size)
    return (time.perf_counter() - t0) * 1000.0  # ms


def profile(
    fn: Callable[[int], None],
    n_warmup: int = 10,
    n_measure: int = 100,
    batch_sizes: Sequence[int] = (1, 8, 32),
    device: str | None = None,
) -> dict[int, dict[str, float]]:
    """Time ``fn(batch_size)`` over several batch sizes.

    Returns ``{batch_size: {"median": .., "p50": .., "p90": .., "p99": ..}}``
    in milliseconds.
    """
    use_cuda = (device == "cuda") or (device is None and torch.cuda.is_available())
    measure = _measure_cuda if use_cuda else _measure_wall
    if not use_cuda:
        _log.warning("latency.profile fallback=perf_counter device=%s", device or "auto")

    out: dict[int, dict[str, float]] = {}
    for bs in batch_sizes:
        for _ in range(n_warmup):
            fn(bs)
        samples = [measure(fn, bs) for _ in range(n_measure)]
        samples.sort()
        out[bs] = {
            "median": statistics.median(samples),
            "p50": samples[int(0.50 * (len(samples) - 1))],
            "p90": samples[int(0.90 * (len(samples) - 1))],
            "p99": samples[int(0.99 * (len(samples) - 1))],
        }
        _log.info(
            "latency.profile batch_size=%d median_ms=%.3f p90_ms=%.3f p99_ms=%.3f",
            bs, out[bs]["median"], out[bs]["p90"], out[bs]["p99"],
        )
    return out

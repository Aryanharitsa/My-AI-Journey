from __future__ import annotations

import math
from typing import Any


def _normalize_logprobs(token_logprobs: list[float] | tuple[float, ...] | None) -> list[float]:
    normalized: list[float] = []
    for value in token_logprobs or ():
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate):
            normalized.append(candidate)
    return normalized


def compute_trace_confidence(
    token_logprobs: list[float] | tuple[float, ...] | None,
    *,
    window_size: int = 64,
    bottom_percentile: float = 0.10,
    tail_tokens: int = 2048,
) -> dict[str, float]:
    normalized = _normalize_logprobs(token_logprobs)
    if not normalized:
        return {
            "group_confidence": -10.0,
            "tail_confidence": -10.0,
            "overall_confidence": -10.0,
        }

    if len(normalized) < window_size:
        average = sum(normalized) / len(normalized)
        rounded = round(average, 6)
        return {
            "group_confidence": rounded,
            "tail_confidence": rounded,
            "overall_confidence": rounded,
        }

    window_scores: list[float] = []
    for start in range(len(normalized) - window_size + 1):
        window = normalized[start : start + window_size]
        window_scores.append(sum(window) / len(window))

    window_scores.sort()
    bottom_k = max(1, int(len(window_scores) * bottom_percentile))
    group_confidence = sum(window_scores[:bottom_k]) / bottom_k

    tail = normalized[-min(tail_tokens, len(normalized)) :]
    tail_confidence = sum(tail) / len(tail)
    overall_confidence = (0.7 * group_confidence) + (0.3 * tail_confidence)

    return {
        "group_confidence": round(group_confidence, 6),
        "tail_confidence": round(tail_confidence, 6),
        "overall_confidence": round(overall_confidence, 6),
    }


def compute_deepconf_score(
    token_logprobs: list[float] | tuple[float, ...] | None,
    *,
    window_frac: float = 0.10,
) -> float | None:
    normalized = _normalize_logprobs(token_logprobs)
    if len(normalized) < 10:
        return None

    window_size = max(1, int(len(normalized) * window_frac))
    if window_size >= len(normalized):
        return round(sum(normalized) / len(normalized), 6)

    best_window_average = min(
        sum(normalized[index : index + window_size]) / window_size
        for index in range(len(normalized) - window_size + 1)
    )
    return round(best_window_average, 6)


def rank_candidates_by_confidence(
    candidates: list[dict[str, Any]],
    *,
    window_frac: float = 0.10,
) -> list[dict[str, Any]]:
    for candidate in candidates:
        normalized = _normalize_logprobs(candidate.get("token_logprobs"))
        if not normalized:
            candidate.setdefault("deepconf_group", None)
            candidate.setdefault("deepconf_tail", None)
            candidate.setdefault("deepconf_overall", None)
            candidate.setdefault("deepconf_score", None)
            continue
        confidence = compute_trace_confidence(normalized)
        candidate["deepconf_group"] = confidence["group_confidence"]
        candidate["deepconf_tail"] = confidence["tail_confidence"]
        candidate["deepconf_overall"] = confidence["overall_confidence"]
        candidate["deepconf_score"] = compute_deepconf_score(normalized, window_frac=window_frac)
    return candidates

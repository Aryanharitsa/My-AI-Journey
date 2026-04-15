from __future__ import annotations

from collections import Counter
from typing import Dict, Optional


def choose_sample_count(
    base_sample_count: int,
    *,
    answer_type: str,
    hard_problem: bool = False,
) -> int:
    sample_count = max(1, base_sample_count)

    if answer_type in {"modular", "expression"}:
        sample_count += 1
    if hard_problem:
        sample_count += 1

    return sample_count


def choose_phase_sample_budget(
    *,
    phase: int,
    agreement_count: int = 0,
    attempts_run: int = 0,
) -> int:
    if phase <= 1:
        return 4
    if phase == 2:
        if attempts_run <= 0:
            return 8
        if agreement_count <= 1:
            return 16
        return 8
    return max(4, attempts_run)


def should_stop_sampling(
    answers: list[str | int | None],
    *,
    min_samples: int = 4,
    consensus_threshold: float = 0.75,
) -> bool:
    if len(answers) < min_samples:
        return False
    counts = Counter(str(answer) for answer in answers if answer is not None)
    if not counts:
        return False
    top_count = counts.most_common(1)[0][1]
    return (top_count / len(answers)) >= consensus_threshold


def should_continue_tool_loop(
    *,
    elapsed_seconds: float,
    per_problem_runtime_seconds: int,
    tool_rounds_used: int,
    max_tool_rounds: int,
) -> bool:
    if tool_rounds_used >= max_tool_rounds:
        return False
    if elapsed_seconds >= per_problem_runtime_seconds:
        return False
    return True


def describe_budget(
    *,
    sample_count: int,
    max_tool_rounds: int,
    per_problem_runtime_seconds: int,
    max_retries: int,
    route_id: Optional[str] = None,
    phase: int | None = None,
) -> Dict[str, int | str | None]:
    return {
        "sample_count": sample_count,
        "max_tool_rounds": max_tool_rounds,
        "per_problem_runtime_seconds": per_problem_runtime_seconds,
        "max_retries": max_retries,
        "route_id": route_id,
        "phase": phase,
    }

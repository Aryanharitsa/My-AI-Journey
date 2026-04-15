from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from typing import Any, Dict, List

import day17_ablation as day17


MARKER_STRENGTH_BY_PATTERN = {
    "final_answer_line": 2.0,
    "answer_line": 1.75,
    "boxed_integer": 2.0,
    "answer_is_phrase": 1.5,
    "fallback_integer_scan": 0.2,
    None: 0.0,
}

PARSE_STRENGTH_BY_ERROR = {
    None: 1.5,
    "no_candidate_found": -1.5,
    "pending_placeholder": -2.0,
    "ambiguous_multiple_candidates": -2.0,
    "negative_not_allowed": -1.0,
    "out_of_range": -1.0,
}


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _default_parse_strength(candidate: Dict[str, Any]) -> float:
    parse_error_type = candidate.get("parse_error_type")
    if candidate.get("is_valid"):
        return PARSE_STRENGTH_BY_ERROR[None]
    return PARSE_STRENGTH_BY_ERROR.get(parse_error_type, -1.0)


def _default_marker_strength(candidate: Dict[str, Any]) -> float:
    return MARKER_STRENGTH_BY_PATTERN.get(candidate.get("matched_pattern"), 0.0)


def _default_protocol_clean(candidate: Dict[str, Any]) -> bool:
    parse_error_type = candidate.get("parse_error_type")
    return bool(candidate.get("is_valid")) and parse_error_type not in {
        "pending_placeholder",
        "ambiguous_multiple_candidates",
    }


def _deepconf_bonus(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return max(-1.5, min(1.5, (float(value) + 2.0) * 0.6))


def _prm_bonus(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return max(-2.0, min(2.0, (float(value) - 0.5) * 4.0))


def _verbalized_confidence_bonus(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return max(-1.5, min(1.5, (float(value) - 5.0) * 0.3))


def score_candidate(candidate: Dict[str, Any], agreement_count: int, *, oracle_mode: bool = True) -> float:
    score = 0.0

    parse_strength = _coerce_float(
        candidate.get("parse_strength"),
        fallback=_default_parse_strength(candidate),
    )
    marker_strength = _coerce_float(
        candidate.get("marker_strength"),
        fallback=_default_marker_strength(candidate),
    )
    protocol_clean = candidate.get("protocol_clean")
    if protocol_clean is None:
        protocol_clean = _default_protocol_clean(candidate)

    score += parse_strength
    score += marker_strength
    score += 1.0 if protocol_clean else -1.0

    if oracle_mode:
        if candidate.get("verification_status") == "exact_match":
            score += 3.0
        elif candidate.get("verification_status") == "mismatch":
            score -= 1.0
    if _is_positive_number(candidate.get("tool_rounds_used", 0)):
        score += 0.5

    tool_verification_success = candidate.get("tool_verification_success")
    if tool_verification_success is True:
        score += 1.0
    elif tool_verification_success is False:
        score -= 1.0

    deepconf_signal = candidate.get("deepconf_score")
    if deepconf_signal is None:
        deepconf_signal = candidate.get("deepconf_overall")
    score += _deepconf_bonus(deepconf_signal)

    independent_verify = candidate.get("independent_verify")
    if independent_verify is True:
        score += 1.5

    score += 1.0 * max(0, agreement_count - 1)
    return score


def _normalize_candidate(candidate: Any) -> Dict[str, Any]:
    if isinstance(candidate, dict):
        return candidate
    return {"parse_error_type": "invalid_candidate_type"}


def _is_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _agreement_key(answer: Any) -> Hashable | None:
    if answer is None or not isinstance(answer, Hashable):
        return None
    return answer


def _competition_candidate_allowed(candidate: Dict[str, Any], *, competition_mode: bool) -> bool:
    if not competition_mode:
        return True
    return candidate.get("competition_answer_eligible") is not False


def _build_answer_groups(
    normalized_candidates: List[Dict[str, Any]],
    scored_entries: List[tuple[int, float, int]],
) -> dict[Hashable, Dict[str, Any]]:
    answer_groups: dict[Hashable, Dict[str, Any]] = {}
    for index, weighted_score, agreement_count in scored_entries:
        candidate = normalized_candidates[index]
        answer_key = _agreement_key(candidate.get("answer"))
        if answer_key is None:
            continue
        group = answer_groups.setdefault(
            answer_key,
            {
                "answer": candidate.get("answer"),
                "total_score": 0.0,
                "member_count": 0,
                "agreement_count": agreement_count,
                "best_candidate_index": index,
                "best_candidate_score": weighted_score,
            },
        )
        group["total_score"] += weighted_score
        group["member_count"] += 1
        group["agreement_count"] = max(group["agreement_count"], agreement_count)
        if (
            weighted_score > group["best_candidate_score"]
            or (
                weighted_score == group["best_candidate_score"]
                and index < group["best_candidate_index"]
            )
        ):
            group["best_candidate_index"] = index
            group["best_candidate_score"] = weighted_score
    return answer_groups


def _rank_answer_groups(answer_groups: dict[Hashable, Dict[str, Any]]) -> list[Dict[str, Any]]:
    ranked_groups = list(answer_groups.values())
    ranked_groups.sort(
        key=lambda group: (
            -float(group["total_score"]),
            -int(group["agreement_count"]),
            -int(group["member_count"]),
            -float(group["best_candidate_score"]),
            int(group["best_candidate_index"]),
        )
    )
    return ranked_groups


def evidence_weighted_vote(
    candidates: List[Any],
    *,
    oracle_mode: bool = True,
    competition_mode: bool = False,
) -> Dict[str, Any]:
    if not candidates:
        return {
            "selected_answer": None,
            "selection_reason": "no_candidates",
            "selected_index": None,
            "scores": [],
        }

    normalized_candidates = [_normalize_candidate(candidate) for candidate in candidates]
    active_indexes = [
        index
        for index, candidate in enumerate(normalized_candidates)
        if _competition_candidate_allowed(candidate, competition_mode=competition_mode)
    ]
    if not active_indexes:
        active_indexes = list(range(len(normalized_candidates)))

    answers = [
        _agreement_key(candidate.get("answer"))
        for index, candidate in enumerate(normalized_candidates)
        if index in active_indexes and _agreement_key(candidate.get("answer")) is not None
    ]
    counts = Counter(answers)

    scored = []
    skipped_indexes = set(range(len(normalized_candidates))) - set(active_indexes)
    for index in active_indexes:
        candidate = normalized_candidates[index]
        answer = candidate.get("answer")
        agreement_key = _agreement_key(answer)
        agreement_count = counts.get(agreement_key, 0) if agreement_key is not None else 0
        score = score_candidate(candidate, agreement_count=agreement_count, oracle_mode=oracle_mode)
        scored.append((index, score, agreement_count))

    # Deterministic order: highest score, then highest agreement, then earliest candidate.
    scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
    selected_index = scored[0][0]
    selected_answer = normalized_candidates[selected_index].get("answer")

    return {
        "selected_answer": selected_answer,
        "selection_reason": "evidence_weighted_vote",
        "selected_index": selected_index,
        "scores": [
            {
                "candidate_index": index,
                "score": (score if index not in skipped_indexes else None),
                "agreement_count": (agreement_count if index not in skipped_indexes else 0),
                "parse_strength": _coerce_float(
                    normalized_candidates[index].get("parse_strength"),
                    fallback=_default_parse_strength(normalized_candidates[index]),
                ),
                "marker_strength": _coerce_float(
                    normalized_candidates[index].get("marker_strength"),
                    fallback=_default_marker_strength(normalized_candidates[index]),
                ),
                "protocol_clean": (
                    normalized_candidates[index].get("protocol_clean")
                    if normalized_candidates[index].get("protocol_clean") is not None
                    else _default_protocol_clean(normalized_candidates[index])
                ),
                "tool_verification_success": normalized_candidates[index].get("tool_verification_success"),
            }
            for index, score, agreement_count in (
                scored + [(index, 0.0, 0) for index in sorted(skipped_indexes)]
            )
        ],
    }


def run2_verified_consensus(candidates: List[Any]) -> Dict[str, Any]:
    normalized_candidates = [_normalize_candidate(candidate) for candidate in candidates]
    counts = Counter(
        _agreement_key(candidate.get("answer"))
        for candidate in normalized_candidates
        if not _is_route_stuck_candidate(candidate)
        and candidate.get("tir_verified")
        and _agreement_key(candidate.get("answer")) is not None
    )
    if not counts:
        return {
            "selected_answer": None,
            "selection_reason": "run2_no_verified_consensus",
            "selected_index": None,
            "scores": [],
        }
    answer, agreement_count = counts.most_common(1)[0]
    if agreement_count < 2:
        return {
            "selected_answer": None,
            "selection_reason": "run2_no_verified_consensus",
            "selected_index": None,
            "scores": [],
        }
    selected_index = next(
        (
            index
            for index, candidate in enumerate(normalized_candidates)
            if not _is_route_stuck_candidate(candidate)
            and candidate.get("tir_verified")
            and candidate.get("answer") == answer
        ),
        None,
    )
    return {
        "selected_answer": answer,
        "selection_reason": "run2_verified_consensus",
        "selected_index": selected_index,
        "scores": [
            {
                "candidate_index": index,
                "score": 0.0,
                "agreement_count": agreement_count if candidate.get("answer") == answer else 0,
            }
            for index, candidate in enumerate(normalized_candidates)
        ],
    }


def _is_route_stuck_candidate(candidate: Dict[str, Any]) -> bool:
    if candidate.get("route_stuck"):
        return True
    tier = candidate.get("extraction_tier")
    return isinstance(tier, str) and tier.upper() == "ROUTE_STUCK"


def _run2_score_components(
    candidate: Dict[str, Any],
    *,
    agreement_count: int,
    oracle_mode: bool,
) -> Dict[str, Any]:
    if day17.simplified_voting_enabled():
        reasoning_tokens_used = int(candidate.get("reasoning_tokens_used") or 0)
        tir_computation_bonus = 0.0
        if _is_positive_number(candidate.get("tool_rounds_used", 0)) and candidate.get("tool_verification_success") is True:
            tir_computation_bonus = 2.0
        reasoning_effort_bonus = 0.0
        if reasoning_tokens_used > day17.MIN_REASONING_THRESHOLD:
            reasoning_effort_bonus = 0.5
        elif reasoning_tokens_used < 5000:
            reasoning_effort_bonus = -1.0
        parse_quality = 1.5 if candidate.get("is_valid") else -1.5
        verbalized_confidence_bonus = _verbalized_confidence_bonus(candidate.get("verbalized_confidence"))
        low_effort_penalty = -2.0 if candidate.get("low_effort_suspect") else 0.0
        component_four_name = "parse_quality"
        component_four_value = parse_quality
        if day17.verbalized_confidence_enabled():
            component_four_name = "verbalized_confidence_bonus"
            component_four_value = verbalized_confidence_bonus
            parse_quality = 0.0
        agreement_bonus = float(agreement_count)
        total = agreement_bonus + tir_computation_bonus + reasoning_effort_bonus + component_four_value + low_effort_penalty
        return {
            "agreement_bonus": agreement_bonus,
            "tir_computation_bonus": tir_computation_bonus,
            "reasoning_effort_bonus": reasoning_effort_bonus,
            "parse_quality": parse_quality,
            "verbalized_confidence_bonus": verbalized_confidence_bonus,
            "component_four_name": component_four_name,
            "component_four_value": component_four_value,
            "low_effort_penalty": low_effort_penalty,
            "weighted_score": total,
            "parse_strength": parse_quality if component_four_name == "parse_quality" else _default_parse_strength(candidate),
            "marker_strength": 0.0,
            "protocol_clean": (
                candidate.get("protocol_clean")
                if candidate.get("protocol_clean") is not None
                else _default_protocol_clean(candidate)
            ),
        }

    parse_strength = _coerce_float(
        candidate.get("parse_strength"),
        fallback=_default_parse_strength(candidate),
    )
    marker_strength = _coerce_float(
        candidate.get("marker_strength"),
        fallback=_default_marker_strength(candidate),
    )
    protocol_clean = candidate.get("protocol_clean")
    if protocol_clean is None:
        protocol_clean = _default_protocol_clean(candidate)

    verification_status = candidate.get("verification_status")
    tool_verification_success = candidate.get("tool_verification_success")
    verification_prompt_judgment = candidate.get("verification_prompt_judgment")
    verification_prompt_pass = bool(candidate.get("verification_prompt_pass"))
    tir_verified = bool(candidate.get("tir_verified"))
    extraction_tier = candidate.get("extraction_tier")
    tool_rounds_used = candidate.get("tool_rounds_used", 0)
    answer = candidate.get("answer")
    deepconf_signal = candidate.get("deepconf_score")
    if deepconf_signal is None:
        deepconf_signal = candidate.get("deepconf_overall")
    independent_verify = candidate.get("independent_verify")
    had_tool_timeout = bool(candidate.get("had_tool_timeout"))
    recovered_after_tool_failure = bool(candidate.get("recovered_after_tool_failure"))
    unsupported_guess_after_failure = bool(candidate.get("unsupported_guess_after_failure"))
    successful_tool_completion_count = int(candidate.get("successful_tool_completion_count") or 0)
    checker_confirmed = bool(candidate.get("checker_confirmed")) or independent_verify is True
    small_case_structure_check_seen = candidate.get("small_case_structure_check_seen")
    direct_witness_validation_seen = candidate.get("direct_witness_validation_seen")
    surrogate_structure_count_only = bool(candidate.get("surrogate_structure_count_only"))

    verification_status_bonus = 0.0
    if oracle_mode:
        if verification_status == "exact_match":
            verification_status_bonus = 3.0
        elif verification_status == "mismatch":
            verification_status_bonus = -1.0

    tool_rounds_bonus = 0.5 if _is_positive_number(tool_rounds_used) else 0.0
    tool_verification_bonus = 0.0
    if tool_verification_success is True:
        tool_verification_bonus = 1.5
    elif tool_verification_success is False:
        tool_verification_bonus = -1.0

    tir_verified_bonus = 1.0 if tir_verified else 0.0
    deepconf_bonus = 0.0 if day17.deepconf_weight_disabled() else _deepconf_bonus(deepconf_signal)
    prm_bonus = _prm_bonus(candidate.get("prm_score"))
    independent_verify_bonus = 0.0
    if independent_verify is True:
        independent_verify_bonus = 0.5
    reasoning_bonus = 0.5 if extraction_tier == "REASONING" and not _is_positive_number(tool_rounds_used) else 0.0
    extraction_confirmation_bonus = 0.5 if extraction_tier == "EXTRACTION" and tool_verification_success is True else 0.0
    verification_prompt_bonus = 0.0
    if verification_prompt_judgment == "PASS" or verification_prompt_pass:
        verification_prompt_bonus = 1.0
    elif verification_prompt_judgment == "FAIL" and not checker_confirmed:
        verification_prompt_bonus = -0.5
    zero_answer_penalty = 0.0
    if answer in {0, "0"} and not (verification_prompt_judgment == "PASS" or verification_prompt_pass):
        zero_answer_penalty = -0.5
        # Stronger penalty for zero answers extracted from weak tiers
        if extraction_tier in {"EXTRACTION", "FAILED", "FORCED_EXTRACTION"}:
            zero_answer_penalty = -2.0
    timeout_penalty = -1.5 if had_tool_timeout and not recovered_after_tool_failure else 0.0
    successful_tool_completion_bonus = 0.25 * min(2, max(0, successful_tool_completion_count))
    checker_confirmed_bonus = 1.0 if checker_confirmed else 0.0
    small_case_structure_penalty = -1.0 if small_case_structure_check_seen is False else 0.0
    direct_witness_validation_bonus = 1.5 if direct_witness_validation_seen is True else 0.0
    surrogate_structure_penalty = -1.0 if surrogate_structure_count_only else 0.0
    verbalized_confidence_bonus = _verbalized_confidence_bonus(candidate.get("verbalized_confidence"))
    low_effort_penalty = -2.0 if candidate.get("low_effort_suspect") else 0.0
    agreement_bonus = 1.0 * max(0, agreement_count - 1)
    length_penalty = _coerce_float(candidate.get("_length_penalty"), 0.0)

    total = (
        parse_strength
        + marker_strength
        + (1.0 if protocol_clean else -1.0)
        + verification_status_bonus
        + tool_rounds_bonus
        + tool_verification_bonus
        + tir_verified_bonus
        + deepconf_bonus
        + prm_bonus
        + independent_verify_bonus
        + reasoning_bonus
        + extraction_confirmation_bonus
        + verification_prompt_bonus
        + zero_answer_penalty
        + timeout_penalty
        + successful_tool_completion_bonus
        + checker_confirmed_bonus
        + small_case_structure_penalty
        + direct_witness_validation_bonus
        + surrogate_structure_penalty
        + verbalized_confidence_bonus
        + low_effort_penalty
        + agreement_bonus
        + length_penalty
    )
    if unsupported_guess_after_failure:
        total *= 0.25
    if surrogate_structure_count_only:
        total *= 0.5

    return {
        "parse_strength": parse_strength,
        "marker_strength": marker_strength,
        "protocol_clean": protocol_clean,
        "verification_status_bonus": verification_status_bonus,
        "tool_rounds_bonus": tool_rounds_bonus,
        "tool_verification_bonus": tool_verification_bonus,
        "tir_verified_bonus": tir_verified_bonus,
        "deepconf_bonus": deepconf_bonus,
        "prm_bonus": prm_bonus,
        "independent_verify_bonus": independent_verify_bonus,
        "reasoning_bonus": reasoning_bonus,
        "extraction_confirmation_bonus": extraction_confirmation_bonus,
        "verification_prompt_bonus": verification_prompt_bonus,
        "zero_answer_penalty": zero_answer_penalty,
        "timeout_penalty": timeout_penalty,
        "successful_tool_completion_bonus": successful_tool_completion_bonus,
        "checker_confirmed_bonus": checker_confirmed_bonus,
        "small_case_structure_penalty": small_case_structure_penalty,
        "direct_witness_validation_bonus": direct_witness_validation_bonus,
        "surrogate_structure_penalty": surrogate_structure_penalty,
        "verbalized_confidence_bonus": verbalized_confidence_bonus,
        "low_effort_penalty": low_effort_penalty,
        "length_penalty": length_penalty,
        "unsupported_guess_after_failure": unsupported_guess_after_failure,
        "agreement_bonus": agreement_bonus,
        "weighted_score": total,
    }


def run2_weighted_vote(
    candidates: List[Any],
    *,
    oracle_mode: bool = False,
    genselect_selected_index: int | None = None,
    prefer_genselect: bool = False,
    competition_mode: bool = False,
) -> Dict[str, Any]:
    if not candidates:
        return {
            "selected_answer": None,
            "selection_reason": "run2_no_candidates",
            "selected_index": None,
            "scores": [],
            "runner_up_index": None,
            "selector_trace": {
                "oracle_mode": oracle_mode,
                "viable_candidate_count": 0,
                "distinct_answer_count": 0,
                "selector_phase_taken": "run2_no_candidates",
                "selected_index": None,
                "runner_up_index": None,
                "candidates": [],
            },
        }

    normalized_candidates = [_normalize_candidate(candidate) for candidate in candidates]
    competition_eligible_indexes = {
        index
        for index, candidate in enumerate(normalized_candidates)
        if _competition_candidate_allowed(candidate, competition_mode=competition_mode)
    }
    if not competition_eligible_indexes:
        competition_eligible_indexes = set(range(len(normalized_candidates)))

    viable_candidates = [
        (index, candidate)
        for index, candidate in enumerate(normalized_candidates)
        if index in competition_eligible_indexes and not _is_route_stuck_candidate(candidate)
    ]
    viable_lengths = sorted(
        int(candidate.get("generation_length_chars") or 0)
        for _index, candidate in viable_candidates
        if int(candidate.get("generation_length_chars") or 0) > 0
    )
    if viable_lengths:
        median_length = viable_lengths[len(viable_lengths) // 2]
        for _index, candidate in viable_candidates:
            generation_length = int(candidate.get("generation_length_chars") or 0)
            if generation_length > 0 and median_length > 0 and (generation_length / median_length) > 2.5:
                candidate["_length_penalty"] = -0.5
            else:
                candidate["_length_penalty"] = 0.0
    if not viable_candidates:
        answer_counts = Counter(
            _agreement_key(candidate.get("answer"))
            for index, candidate in enumerate(normalized_candidates)
            if index in competition_eligible_indexes and _agreement_key(candidate.get("answer")) is not None
        )
        distinct_answer_count = len(
            {
                _agreement_key(candidate.get("answer"))
                for index, candidate in enumerate(normalized_candidates)
                if index in competition_eligible_indexes and _agreement_key(candidate.get("answer")) is not None
            }
        )
        fallback_ranked = []
        fallback_scores_by_index: dict[int, Dict[str, Any]] = {}
        for index, candidate in enumerate(normalized_candidates):
            answer = candidate.get("answer")
            agreement_count = answer_counts.get(_agreement_key(answer), 0) if _agreement_key(answer) is not None else 0
            components = _run2_score_components(
                candidate,
                agreement_count=agreement_count,
                oracle_mode=oracle_mode,
            )
            weighted_score = components["weighted_score"]
            fallback_scores_by_index[index] = {
                "candidate_index": index,
                "score": weighted_score,
                "agreement_count": agreement_count,
                "parse_strength": components["parse_strength"],
                "marker_strength": components["marker_strength"],
                "protocol_clean": components["protocol_clean"],
                "tool_verification_success": candidate.get("tool_verification_success"),
                "tir_verified": candidate.get("tir_verified"),
                "verification_prompt_pass": candidate.get("verification_prompt_pass"),
                "verification_prompt_judgment": candidate.get("verification_prompt_judgment"),
                "verification_status": candidate.get("verification_status"),
                "route_stuck": True,
                "deepconf_overall": candidate.get("deepconf_overall"),
                "deepconf_score": candidate.get("deepconf_score"),
                "prm_score": candidate.get("prm_score"),
                "low_effort_suspect": candidate.get("low_effort_suspect"),
                "verbalized_confidence": candidate.get("verbalized_confidence"),
                "independent_verify": candidate.get("independent_verify"),
                "had_tool_timeout": candidate.get("had_tool_timeout"),
                "unsupported_guess_after_failure": candidate.get("unsupported_guess_after_failure"),
                "checker_confirmed": candidate.get("checker_confirmed"),
                "small_case_structure_check_seen": candidate.get("small_case_structure_check_seen"),
                "score_components": components,
            }
            fallback_ranked.append((index, weighted_score, agreement_count))
        fallback_ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
        fallback_answer_groups = _rank_answer_groups(
            _build_answer_groups(normalized_candidates, fallback_ranked)
        )
        if fallback_answer_groups:
            selected_index = int(fallback_answer_groups[0]["best_candidate_index"])
            runner_up_index = (
                int(fallback_answer_groups[1]["best_candidate_index"])
                if len(fallback_answer_groups) > 1
                else next((index for index, _score, _agreement in fallback_ranked if index != selected_index), None)
            )
        else:
            selected_index = fallback_ranked[0][0]
            runner_up_index = next((index for index, _score, _agreement in fallback_ranked if index != selected_index), None)
        selected_answer = normalized_candidates[selected_index].get("answer")
        return {
            "selected_answer": selected_answer,
            "selection_reason": "fallback_all_routes_stuck_best_candidate",
            "selected_index": selected_index,
            "runner_up_index": runner_up_index,
            "scores": [
                fallback_scores_by_index[index]
                for index in range(len(normalized_candidates))
            ],
            "selector_trace": {
                "oracle_mode": oracle_mode,
                "viable_candidate_count": 0,
                "distinct_answer_count": distinct_answer_count,
                "selector_phase_taken": "fallback_all_routes_stuck_best_candidate",
                "selected_index": selected_index,
                "runner_up_index": runner_up_index,
                "candidates": [
                    {
                        "candidate_index": index,
                        "answer": candidate.get("answer"),
                        "route_stuck": True,
                        "verification_status": candidate.get("verification_status"),
                        "verification_prompt_judgment": candidate.get("verification_prompt_judgment"),
                        "verification_prompt_pass": candidate.get("verification_prompt_pass"),
                        "tir_verified": candidate.get("tir_verified"),
                        "tool_rounds_used": candidate.get("tool_rounds_used", 0),
                        "tool_verification_success": candidate.get("tool_verification_success"),
                        "deepconf_overall": candidate.get("deepconf_overall"),
                        "deepconf_score": candidate.get("deepconf_score"),
                        "prm_score": candidate.get("prm_score"),
                        "low_effort_suspect": candidate.get("low_effort_suspect"),
                        "verbalized_confidence": candidate.get("verbalized_confidence"),
                        "independent_verify": candidate.get("independent_verify"),
                        "had_tool_timeout": candidate.get("had_tool_timeout"),
                        "unsupported_guess_after_failure": candidate.get("unsupported_guess_after_failure"),
                        "checker_confirmed": candidate.get("checker_confirmed"),
                        "small_case_structure_check_seen": candidate.get("small_case_structure_check_seen"),
                        "parse_strength": _coerce_float(
                            candidate.get("parse_strength"),
                            fallback=_default_parse_strength(candidate),
                        ),
                        "marker_strength": _coerce_float(
                            candidate.get("marker_strength"),
                            fallback=_default_marker_strength(candidate),
                        ),
                        "protocol_clean": (
                            candidate.get("protocol_clean")
                            if candidate.get("protocol_clean") is not None
                            else _default_protocol_clean(candidate)
                        ),
                        "agreement_count": fallback_scores_by_index[index]["agreement_count"],
                        "weighted_score": fallback_scores_by_index[index]["score"],
                        "score_components": fallback_scores_by_index[index]["score_components"],
                    }
                    for index, candidate in enumerate(normalized_candidates)
                ],
            },
        }

    answer_counts = Counter(
        _agreement_key(candidate.get("answer"))
        for _index, candidate in viable_candidates
        if _agreement_key(candidate.get("answer")) is not None
    )

    distinct_answer_count = len(
        {
            _agreement_key(candidate.get("answer"))
            for _index, candidate in viable_candidates
            if _agreement_key(candidate.get("answer")) is not None
        }
    )

    scores_by_index: dict[int, Dict[str, Any]] = {}
    ranked = []
    for index, candidate in viable_candidates:
        answer = candidate.get("answer")
        agreement_count = answer_counts.get(_agreement_key(answer), 0) if _agreement_key(answer) is not None else 0
        components = _run2_score_components(
            candidate,
            agreement_count=agreement_count,
            oracle_mode=oracle_mode,
        )
        weighted_score = components["weighted_score"]
        scores_by_index[index] = {
            "candidate_index": index,
            "score": weighted_score,
            "agreement_count": agreement_count,
            "parse_strength": components["parse_strength"],
            "marker_strength": components["marker_strength"],
            "protocol_clean": components["protocol_clean"],
            "tool_verification_success": candidate.get("tool_verification_success"),
            "tir_verified": candidate.get("tir_verified"),
            "verification_prompt_pass": candidate.get("verification_prompt_pass"),
            "verification_prompt_judgment": candidate.get("verification_prompt_judgment"),
            "verification_status": candidate.get("verification_status"),
            "route_stuck": False,
            "deepconf_overall": candidate.get("deepconf_overall"),
            "deepconf_score": candidate.get("deepconf_score"),
            "prm_score": candidate.get("prm_score"),
            "low_effort_suspect": candidate.get("low_effort_suspect"),
            "verbalized_confidence": candidate.get("verbalized_confidence"),
            "independent_verify": candidate.get("independent_verify"),
            "had_tool_timeout": candidate.get("had_tool_timeout"),
            "unsupported_guess_after_failure": candidate.get("unsupported_guess_after_failure"),
            "checker_confirmed": candidate.get("checker_confirmed"),
            "small_case_structure_check_seen": candidate.get("small_case_structure_check_seen"),
            "score_components": components,
        }
        ranked.append((index, weighted_score, agreement_count))

    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    answer_groups = _rank_answer_groups(_build_answer_groups(normalized_candidates, ranked))

    selection_reason = "run2_weighted_vote"
    selected_index = (
        int(answer_groups[0]["best_candidate_index"])
        if answer_groups
        else ranked[0][0]
    )

    if oracle_mode:
        exact_match_indexes = [
            index
            for index, candidate in viable_candidates
            if candidate.get("verification_status") == "exact_match"
        ]
        if exact_match_indexes:
            selected_index = exact_match_indexes[0]
            selection_reason = "oracle_exact_match"

    if selection_reason == "run2_weighted_vote":
        # Only trust verified consensus when the pool is deep enough.
        # A thin pool (e.g. 2 survivors out of 8 ROUTE_STUCK) can lock
        # onto a wrong answer. Require at least 3 viable candidates.
        if len(viable_candidates) >= 3:
            verified_consensus = run2_verified_consensus([candidate for _index, candidate in viable_candidates])
            if verified_consensus["selected_answer"] is not None:
                selected_answer = verified_consensus["selected_answer"]
                selected_index = next(
                    (
                        index
                        for index, candidate in viable_candidates
                        if candidate.get("tir_verified") and candidate.get("answer") == selected_answer
                    ),
                    selected_index,
                )
                selection_reason = "run2_verified_consensus"

    if selection_reason == "run2_weighted_vote" and prefer_genselect and genselect_selected_index is not None:
        selected_candidate = normalized_candidates[genselect_selected_index]
        if genselect_selected_index in competition_eligible_indexes and not _is_route_stuck_candidate(selected_candidate):
            selected_index = genselect_selected_index
            selection_reason = "run2_genselect"

    selected_answer = normalized_candidates[selected_index].get("answer")
    if answer_groups:
        runner_up_index = (
            int(answer_groups[1]["best_candidate_index"])
            if len(answer_groups) > 1
            else next((index for index, _score, _agreement in ranked if index != selected_index), None)
        )
    else:
        runner_up_index = next((index for index, _score, _agreement in ranked if index != selected_index), None)
    score_entries = []
    selector_candidates = []
    for index, candidate in enumerate(normalized_candidates):
        score_entry = scores_by_index.get(index)
        if score_entry is None:
            score_entries.append({
                "candidate_index": index,
                "score": None,
                "agreement_count": 0,
                "route_stuck": _is_route_stuck_candidate(candidate),
                "verification_status": candidate.get("verification_status"),
                "verification_prompt_judgment": candidate.get("verification_prompt_judgment"),
                "verification_prompt_pass": candidate.get("verification_prompt_pass"),
                "tir_verified": candidate.get("tir_verified"),
                "deepconf_overall": candidate.get("deepconf_overall"),
                "deepconf_score": candidate.get("deepconf_score"),
                "prm_score": candidate.get("prm_score"),
                "low_effort_suspect": candidate.get("low_effort_suspect"),
                "verbalized_confidence": candidate.get("verbalized_confidence"),
                "independent_verify": candidate.get("independent_verify"),
                "had_tool_timeout": candidate.get("had_tool_timeout"),
                "unsupported_guess_after_failure": candidate.get("unsupported_guess_after_failure"),
                "checker_confirmed": candidate.get("checker_confirmed"),
                "small_case_structure_check_seen": candidate.get("small_case_structure_check_seen"),
                "score_components": None,
            })
        else:
            score_entries.append(score_entry)

        selector_candidates.append({
            "candidate_index": index,
            "answer": candidate.get("answer"),
            "route_stuck": _is_route_stuck_candidate(candidate),
            "verification_status": candidate.get("verification_status"),
            "verification_prompt_judgment": candidate.get("verification_prompt_judgment"),
            "verification_prompt_pass": candidate.get("verification_prompt_pass"),
            "tir_verified": candidate.get("tir_verified"),
            "tool_rounds_used": candidate.get("tool_rounds_used", 0),
            "tool_verification_success": candidate.get("tool_verification_success"),
            "deepconf_overall": candidate.get("deepconf_overall"),
            "deepconf_score": candidate.get("deepconf_score"),
            "prm_score": candidate.get("prm_score"),
            "low_effort_suspect": candidate.get("low_effort_suspect"),
            "verbalized_confidence": candidate.get("verbalized_confidence"),
            "independent_verify": candidate.get("independent_verify"),
            "had_tool_timeout": candidate.get("had_tool_timeout"),
            "unsupported_guess_after_failure": candidate.get("unsupported_guess_after_failure"),
            "checker_confirmed": candidate.get("checker_confirmed"),
            "small_case_structure_check_seen": candidate.get("small_case_structure_check_seen"),
            "parse_strength": (
                score_entry.get("parse_strength")
                if score_entry is not None
                else _coerce_float(candidate.get("parse_strength"), fallback=_default_parse_strength(candidate))
            ),
            "marker_strength": (
                score_entry.get("marker_strength")
                if score_entry is not None
                else _coerce_float(candidate.get("marker_strength"), fallback=_default_marker_strength(candidate))
            ),
            "protocol_clean": (
                score_entry.get("protocol_clean")
                if score_entry is not None
                else (
                    candidate.get("protocol_clean")
                    if candidate.get("protocol_clean") is not None
                    else _default_protocol_clean(candidate)
                )
            ),
            "agreement_count": score_entry.get("agreement_count", 0) if score_entry is not None else 0,
            "weighted_score": score_entry.get("score") if score_entry is not None else None,
            "score_components": score_entry.get("score_components") if score_entry is not None else None,
        })

    return {
        "selected_answer": selected_answer,
        "selection_reason": selection_reason,
        "selected_index": selected_index,
        "runner_up_index": runner_up_index,
        "scores": score_entries,
        "selector_trace": {
            "oracle_mode": oracle_mode,
            "viable_candidate_count": len(viable_candidates),
            "distinct_answer_count": distinct_answer_count,
            "selector_phase_taken": selection_reason,
            "selected_index": selected_index,
            "runner_up_index": runner_up_index,
            "answer_groups": answer_groups,
            "candidates": selector_candidates,
        },
    }

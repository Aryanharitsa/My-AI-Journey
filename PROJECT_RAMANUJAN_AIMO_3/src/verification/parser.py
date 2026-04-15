import re
from typing import Any, Dict, Optional

ANSWER_MIN = 0
ANSWER_MAX = 99999

FINAL_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:\*\*\s*)?final(?:[_\s]+)answer(?:\s*\*\*)?\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ANSWER_LINE_RE = re.compile(
    r"^\s*(?:\*\*\s*)?answer(?:\s*\*\*)?\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
BOXED_INT_RE = re.compile(r"\\boxed\s*\{\s*\$?\s*(-?[\d,]+)\s*\$?\s*\}")
ANSWER_IS_RE = re.compile(r"(?:the\s+)?answer\s*(?:is|=)\s*\$?\s*(-?[\d,]+)", re.IGNORECASE)
INTEGER_TOKEN_RE = re.compile(r"(?<![\d,])-?(?:\d[\d,]*)(?![\d,])")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]*`")
ALTERNATIVE_SIGNAL_RE = re.compile(r"\bor\b|\beither\b|\bmaybe\b", re.IGNORECASE)

PARSE_REASON_STRONG_MARKER_UNIQUE = "strong_marker_unique"
PARSE_REASON_BOXED_ANSWER_UNIQUE = "boxed_answer_unique"
PARSE_REASON_STRONG_MARKER_REPEATED = "strong_marker_repeated_identical"
PARSE_REASON_STRONG_MARKER_CONFLICT = "strong_marker_conflict"
PARSE_REASON_FALLBACK_LAST_INTEGER = "fallback_last_integer"
PARSE_REASON_AMBIGUOUS_UNSTRUCTURED = "ambiguous_unstructured_numbers"
PARSE_REASON_NO_INTEGER = "no_integer_found"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def detect_answer_type(text: str) -> str:
    lower = text.lower()

    if "integer" in lower or "positive integer" in lower or "natural number" in lower:
        return "integer"
    if "fraction" in lower or "rational" in lower:
        return "fraction"
    if "mod" in lower or "modulo" in lower:
        return "modular"
    if "expression" in lower or "simplify" in lower:
        return "expression"
    return "unknown"


def parse_problem(problem_text: str) -> Dict[str, Any]:
    clean_text = normalize_whitespace(problem_text)

    return {
        "raw_text": problem_text,
        "clean_text": clean_text,
        "answer_type": detect_answer_type(clean_text),
        "length_chars": len(clean_text),
    }


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _extract_integer_tokens(text: str) -> list[str]:
    normalized = text.replace("$", "")
    return [match.group(0).replace(",", "") for match in INTEGER_TOKEN_RE.finditer(normalized)]


def _validation_error(parsed_value: int) -> Optional[str]:
    if parsed_value < ANSWER_MIN:
        return "negative_not_allowed"
    if parsed_value > ANSWER_MAX:
        return "out_of_range"
    return None


def _build_result(
    *,
    parsed_answer: Optional[str],
    is_valid: bool,
    parse_error_type: Optional[str],
    parse_reason: str,
    parse_confidence: str,
    candidates: list[str],
    matched_pattern: Optional[str],
    raw_span: Optional[str],
) -> Dict[str, Any]:
    return {
        "parsed_answer": parsed_answer,
        "is_valid": is_valid,
        "parse_error_type": parse_error_type,
        "parse_reason": parse_reason,
        "parse_confidence": parse_confidence,
        "matched_pattern": matched_pattern,
        "raw_span": raw_span,
        "candidate_answers": candidates,
    }


def _strip_tool_request_blocks(text: str) -> str:
    lines = text.splitlines()
    kept_lines: list[str] = []
    in_tool_block = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        starts_strong_marker = bool(
            re.match(
                r"^(?:\*\*\s*)?(?:final[_\s]*answer|final answer|answer)\b|^(?:the\s+)?answer\s*(?:is|=)",
                lower,
            )
        )
        if in_tool_block:
            if starts_strong_marker or lower.startswith("thought:"):
                in_tool_block = False
            else:
                continue

        if lower.startswith("tool_request:"):
            in_tool_block = True
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines)


def _strip_code_and_tool_content(text: str) -> str:
    without_code_blocks = CODE_BLOCK_RE.sub(" ", text)
    without_inline_code = INLINE_CODE_RE.sub(" ", without_code_blocks)
    return _strip_tool_request_blocks(without_inline_code)


def _extract_strong_markers(text: str) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []

    def _append_line_markers(pattern: re.Pattern[str], marker_kind: str) -> None:
        for match in pattern.finditer(text):
            raw_span = match.group(1).strip()
            integers = _extract_integer_tokens(raw_span)
            if len(integers) > 1 and not ALTERNATIVE_SIGNAL_RE.search(raw_span):
                integers = [integers[0]]
            markers.append(
                {
                    "marker_kind": marker_kind,
                    "matched_pattern": marker_kind,
                    "raw_span": raw_span,
                    "start": match.start(),
                    "candidates": integers,
                    "has_pending": (not integers and "pending" in raw_span.lower()),
                }
            )

    _append_line_markers(FINAL_ANSWER_LINE_RE, "final_answer_line")
    _append_line_markers(ANSWER_LINE_RE, "answer_line")

    for match in BOXED_INT_RE.finditer(text):
        markers.append(
            {
                "marker_kind": "boxed_integer",
                "matched_pattern": "boxed_integer",
                "raw_span": match.group(0),
                "start": match.start(),
                "candidates": [match.group(1)],
                "has_pending": False,
            }
        )

    for match in ANSWER_IS_RE.finditer(text):
        markers.append(
            {
                "marker_kind": "answer_is_phrase",
                "matched_pattern": "answer_is_phrase",
                "raw_span": match.group(0),
                "start": match.start(),
                "candidates": [match.group(1)],
                "has_pending": False,
            }
        )

    markers.sort(key=lambda marker: marker["start"])
    return markers


def parse_final_answer(model_output: str) -> Dict[str, Any]:
    text = model_output or ""
    code_aware_text = _strip_code_and_tool_content(text)

    markers = _extract_strong_markers(code_aware_text)
    if markers:
        pending_only = True
        for marker in markers:
            if marker["candidates"]:
                pending_only = False
                break
        if pending_only and any(marker["has_pending"] for marker in markers):
            return _build_result(
                parsed_answer=None,
                is_valid=False,
                parse_error_type="pending_placeholder",
                parse_reason=PARSE_REASON_NO_INTEGER,
                parse_confidence="none",
                candidates=[],
                matched_pattern=markers[-1]["matched_pattern"],
                raw_span=markers[-1]["raw_span"],
            )

        flattened_candidates = [candidate for marker in markers for candidate in marker["candidates"]]
        multi_candidate_marker = next(
            (
                marker
                for marker in markers
                if len(_dedupe_keep_order(marker["candidates"])) > 1
            ),
            None,
        )
        if multi_candidate_marker is not None:
            return _build_result(
                parsed_answer=None,
                is_valid=False,
                parse_error_type="ambiguous_multiple_candidates",
                parse_reason=PARSE_REASON_STRONG_MARKER_CONFLICT,
                parse_confidence="none",
                candidates=flattened_candidates,
                matched_pattern=multi_candidate_marker["matched_pattern"],
                raw_span=multi_candidate_marker["raw_span"],
            )

        marker_entries: list[dict[str, Any]] = []
        for marker in markers:
            if not marker["candidates"]:
                continue
            value_token = marker["candidates"][-1]
            parsed_value = int(value_token)
            marker_entries.append(
                {
                    "value_token": value_token,
                    "parsed_value": parsed_value,
                    "validation_error": _validation_error(parsed_value),
                    "matched_pattern": marker["matched_pattern"],
                    "raw_span": marker["raw_span"],
                }
            )

        valid_entries = [entry for entry in marker_entries if entry["validation_error"] is None]
        if valid_entries:
            unique_values = _dedupe_keep_order([entry["value_token"] for entry in valid_entries])
            if len(unique_values) > 1:
                return _build_result(
                    parsed_answer=None,
                    is_valid=False,
                    parse_error_type="ambiguous_multiple_candidates",
                    parse_reason=PARSE_REASON_STRONG_MARKER_CONFLICT,
                    parse_confidence="none",
                    candidates=flattened_candidates,
                    matched_pattern="strong_marker_family",
                    raw_span=None,
                )

            selected_entry = valid_entries[-1]
            if len(valid_entries) == 1:
                parse_reason = (
                    PARSE_REASON_BOXED_ANSWER_UNIQUE
                    if selected_entry["matched_pattern"] == "boxed_integer"
                    else PARSE_REASON_STRONG_MARKER_UNIQUE
                )
            else:
                parse_reason = PARSE_REASON_STRONG_MARKER_REPEATED

            return _build_result(
                parsed_answer=str(selected_entry["parsed_value"]),
                is_valid=True,
                parse_error_type=None,
                parse_reason=parse_reason,
                parse_confidence="high",
                candidates=flattened_candidates,
                matched_pattern=selected_entry["matched_pattern"],
                raw_span=selected_entry["raw_span"],
            )

        if marker_entries:
            selected_entry = marker_entries[-1]
            return _build_result(
                parsed_answer=None,
                is_valid=False,
                parse_error_type=selected_entry["validation_error"],
                parse_reason=PARSE_REASON_STRONG_MARKER_UNIQUE,
                parse_confidence="none",
                candidates=flattened_candidates,
                matched_pattern=selected_entry["matched_pattern"],
                raw_span=selected_entry["raw_span"],
            )

    fallback_candidates = _extract_integer_tokens(code_aware_text)
    fallback_confidence = "medium"
    if not fallback_candidates:
        fallback_candidates = _extract_integer_tokens(text)
        fallback_confidence = "low" if fallback_candidates else "none"

    if not fallback_candidates:
        return _build_result(
            parsed_answer=None,
            is_valid=False,
            parse_error_type="no_candidate_found",
            parse_reason=PARSE_REASON_NO_INTEGER,
            parse_confidence="none",
            candidates=[],
            matched_pattern=None,
            raw_span=None,
        )

    unique_fallback = _dedupe_keep_order(fallback_candidates)
    if len(unique_fallback) > 1:
        return _build_result(
            parsed_answer=None,
            is_valid=False,
            parse_error_type="ambiguous_multiple_candidates",
            parse_reason=PARSE_REASON_AMBIGUOUS_UNSTRUCTURED,
            parse_confidence="low",
            candidates=fallback_candidates,
            matched_pattern="fallback_integer_scan",
            raw_span=fallback_candidates[-1],
        )

    selected_value = int(fallback_candidates[-1])
    validation_error = _validation_error(selected_value)
    if validation_error is not None:
        return _build_result(
            parsed_answer=None,
            is_valid=False,
            parse_error_type=validation_error,
            parse_reason=PARSE_REASON_FALLBACK_LAST_INTEGER,
            parse_confidence="none",
            candidates=fallback_candidates,
            matched_pattern="fallback_integer_scan",
            raw_span=fallback_candidates[-1],
        )

    return _build_result(
        parsed_answer=str(selected_value),
        is_valid=True,
        parse_error_type=None,
        parse_reason=PARSE_REASON_FALLBACK_LAST_INTEGER,
        parse_confidence=fallback_confidence,
        candidates=fallback_candidates,
        matched_pattern="fallback_integer_scan",
        raw_span=fallback_candidates[-1],
    )



def parse_final_answer_with_hint(model_output: str, candidate_hint: Optional[str] = None) -> Dict[str, Any]:
    parse_result = parse_final_answer(model_output)
    if parse_result["is_valid"] or candidate_hint is None:
        return parse_result

    candidate_text = str(candidate_hint).strip()
    if not re.fullmatch(r"-?\d+", candidate_text):
        return parse_result

    validation_error = _validation_error(int(candidate_text))
    if validation_error is not None:
        return parse_result

    return _build_result(
        parsed_answer=candidate_text,
        is_valid=True,
        parse_error_type=None,
        parse_reason="adapter_hint",
        parse_confidence="medium",
        candidates=[candidate_text],
        matched_pattern="adapter_hint",
        raw_span=candidate_text,
    )

def extract_final_answer(model_output: str) -> Optional[str]:
    parse_result = parse_final_answer(model_output)
    return parse_result["parsed_answer"] if parse_result["is_valid"] else None


def extract_tool_request(model_output: str) -> Optional[str]:
    match = re.search(
        r"TOOL_REQUEST:\s*\n(.*?)(?:\n(?:FINAL_ANSWER:|THOUGHT:)|\Z)",
        model_output,
        re.DOTALL,
    )
    if match:
        code = match.group(1).strip()
        return code if code else None
    return None

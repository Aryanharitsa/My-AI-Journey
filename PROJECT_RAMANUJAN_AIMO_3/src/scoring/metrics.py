import re
from typing import Optional


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    text = str(answer).strip()
    if re.fullmatch(r"-?\d+", text):
        return str(int(text))
    return text


def is_exact_match(predicted: Optional[str], expected: Optional[str]) -> bool:
    return normalize_answer(predicted) == normalize_answer(expected)

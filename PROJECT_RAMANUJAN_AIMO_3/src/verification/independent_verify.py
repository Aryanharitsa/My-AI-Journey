from __future__ import annotations

import re


VERIFY_PROMPT = """You are given a math problem and a candidate answer.
Write ONLY executable Python code that CHECKS whether the candidate answer
satisfies ALL explicit constraints in the problem.
Do NOT re-derive the answer. Test the candidate directly against the problem statement.

Requirements:
- Output Python code only. No prose. No Markdown fences.
- Restate what you are checking in code comments.
- Use exact arithmetic when possible.
- The code must print exactly one final line of the form:
  print("VERIFIED: True")
  or:
  print("VERIFIED: False")
- If a direct exact checker is hard, write a brute-force or constructive checker.

Problem:
{problem_text}

Candidate answer: {answer}

Write the verification code now."""


VERIFY_RETRY_PROMPT = """Your previous response did not contain usable Python verification code.
Return ONLY executable Python code. No prose. No Markdown fences.

Requirements:
- Use Python only.
- Directly check the candidate answer against the problem statement.
- Prefer brute force or a constructive checker over prose.
- Print exactly one final line:
  print("VERIFIED: True")
  or:
  print("VERIFIED: False")

Problem:
{problem_text}

Candidate answer: {answer}

Return only Python code."""


def build_verification_request(problem_text: str, answer: str | int) -> str:
    return VERIFY_PROMPT.format(problem_text=problem_text, answer=answer)


def build_verification_retry_request(problem_text: str, answer: str | int) -> str:
    return VERIFY_RETRY_PROMPT.format(problem_text=problem_text, answer=answer)


def parse_independent_verification_stdout(stdout: str | None) -> bool | None:
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    match = re.search(r"VERIFIED:\s*(True|False)", stdout, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "true"

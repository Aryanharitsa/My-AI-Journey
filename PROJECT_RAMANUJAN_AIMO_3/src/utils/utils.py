from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.config import LOGS_DIR, ensure_dirs


def now_utc_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str = "run") -> str:
    """Create a short unique run identifier."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short}"


def append_jsonl(log_path: Path, record: Dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file."""
    ensure_dirs()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def default_log_path(filename: str = "runs.jsonl") -> Path:
    """Return default JSONL log path under data/logs/."""
    ensure_dirs()
    return LOGS_DIR / filename


def write_raw_text(output_dir: Path, run_id: str, text: str, suffix: str = ".txt") -> Path:
    """Write raw model output or debug text to disk and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_id}{suffix}"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def make_attempt_record(
    *,
    experiment_id: str,
    problem_id: str,
    model_name: str,
    prompt_version: str,
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    used_python: bool,
    python_success: bool | None,
    python_runtime_s: float | None,
    raw_output_path: str | None,
    parsed_answer: str | None,
    final_answer_normalized: str | None,
    correctness: bool | None,
    latency_s: float | None,
    notes: str = "",
) -> Dict[str, Any]:
    """Create a standardized attempt record for JSONL logging."""
    return {
        "timestamp_utc": now_utc_iso(),
        "experiment_id": experiment_id,
        "problem_id": problem_id,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "seed": seed,
        "decoding": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        "used_python": used_python,
        "python_success": python_success,
        "python_runtime_s": python_runtime_s,
        "raw_output_path": raw_output_path,
        "parsed_answer": parsed_answer,
        "final_answer_normalized": final_answer_normalized,
        "correctness": correctness,
        "latency_s": latency_s,
        "notes": notes,
    }


if __name__ == "__main__":
    # Tiny smoke test
    test_run_id = make_run_id("smoke")
    test_record = make_attempt_record(
        experiment_id="exp000_smoke",
        problem_id="demo_problem",
        model_name="demo_model",
        prompt_version="p_test",
        seed=42,
        temperature=0.2,
        top_p=0.9,
        max_tokens=256,
        used_python=False,
        python_success=None,
        python_runtime_s=None,
        raw_output_path=None,
        parsed_answer="42",
        final_answer_normalized="42",
        correctness=None,
        latency_s=0.01,
        notes="utils smoke test",
    )
    append_jsonl(default_log_path("smoke_runs.jsonl"), test_record)
    print(f"Smoke test log written for run_id={test_run_id}")

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
EVAL_SETS_DIR = DATA_DIR / "eval_sets"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
GPT_OSS_PROTOCOL_AUDIT_DIR = LOGS_DIR / "gpt_oss_protocol_audit"
GPT_OSS_PROTOCOL_RUNS_DIR = LOGS_DIR / "gpt_oss_protocol_runs"


MODEL_BACKEND_TYPE = os.getenv("MODEL_BACKEND_TYPE", os.getenv("MODEL_BACKEND", "arithmetic_debug"))
MODEL_BACKEND = MODEL_BACKEND_TYPE
MODEL_FAMILY = os.getenv("MODEL_FAMILY", "gpt_oss")
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "gpt-oss-20b"))
PROTOCOL_VARIANT = os.getenv("PROTOCOL_VARIANT", "baseline")
ADAPTER_DEBUG = os.getenv("ADAPTER_DEBUG", "0") == "1"
GPT_OSS_MIN_OUTPUT_TOKENS = int(os.getenv("GPT_OSS_MIN_OUTPUT_TOKENS", "4608"))
USE_GPT_OSS_HARMONY = os.getenv("USE_GPT_OSS_HARMONY", "0") == "1"
GPT_OSS_TRANSPORT = os.getenv(
    "GPT_OSS_TRANSPORT",
    "harmony" if USE_GPT_OSS_HARMONY else "responses",
).strip().lower()
USE_RESPONSES_API = os.getenv("USE_RESPONSES_API", "1" if MODEL_FAMILY == "gpt_oss" else "0") == "1"
USE_CHAT_COMPLETIONS_FALLBACK = os.getenv("USE_CHAT_COMPLETIONS_FALLBACK", "0") == "1"
GPT_OSS_EXPECT_HARMONY = os.getenv("GPT_OSS_EXPECT_HARMONY", "1" if MODEL_FAMILY == "gpt_oss" else "0") == "1"
ENABLE_NATIVE_TOOL_CALLS = os.getenv("ENABLE_NATIVE_TOOL_CALLS", "1" if MODEL_FAMILY == "gpt_oss" else "0") == "1"
ENABLE_TEXT_TOOL_REQUEST_FALLBACK = os.getenv("ENABLE_TEXT_TOOL_REQUEST_FALLBACK", "1") == "1"
GPT_OSS_HARMONY_ENABLE_PYTHON_MCP = os.getenv("GPT_OSS_HARMONY_ENABLE_PYTHON_MCP", "1") == "1"
GPT_OSS_HARMONY_ALLOW_TEXT_FALLBACK = os.getenv("GPT_OSS_HARMONY_ALLOW_TEXT_FALLBACK", "0") == "1"
GPT_OSS_HARMONY_TOOL_CALL_LOGIT_BIAS = float(os.getenv("GPT_OSS_HARMONY_TOOL_CALL_LOGIT_BIAS", "5.0"))
ENABLE_IMPLICIT_CODE_EXTRACTION = os.getenv("ENABLE_IMPLICIT_CODE_EXTRACTION", "1") == "1"
GPT_OSS_MAX_MODEL_LEN = int(os.getenv("GPT_OSS_MAX_MODEL_LEN", "32768"))
GPT_OSS_HARMONY_SAFETY_MARGIN = int(os.getenv("GPT_OSS_HARMONY_SAFETY_MARGIN", "512"))
GPT_OSS_MAX_FINALIZATION_CONTINUATIONS = int(os.getenv("GPT_OSS_MAX_FINALIZATION_CONTINUATIONS", "2"))
GPT_OSS_WRITE_PROTOCOL_ARTIFACTS = os.getenv("GPT_OSS_WRITE_PROTOCOL_ARTIFACTS", "0") == "1"
RAMANUJAN_ORACLE_MODE = os.getenv("RAMANUJAN_ORACLE_MODE", "0") == "1"
USE_POLICY_BOOK = os.getenv("USE_POLICY_BOOK", "0") == "1"
ENABLE_STRIPPED_PROMPTS = os.getenv("ENABLE_STRIPPED_PROMPTS", "0") == "1"
_ENABLE_DEEPCONF_ENV = os.getenv("ENABLE_DEEPCONF")
_ENABLE_DEEPCONF_LOGPROBS_ENV = os.getenv("ENABLE_DEEPCONF_LOGPROBS")
if _ENABLE_DEEPCONF_ENV is not None:
    ENABLE_DEEPCONF = _ENABLE_DEEPCONF_ENV == "1"
else:
    ENABLE_DEEPCONF = _ENABLE_DEEPCONF_LOGPROBS_ENV == "1"
ENABLE_DEEPCONF_LOGPROBS = ENABLE_DEEPCONF
DEEPCONF_WINDOW_FRAC = float(os.getenv("DEEPCONF_WINDOW_FRAC", "0.10"))
ENABLE_PRM = os.getenv("ENABLE_PRM", "0") == "1"
PRM_BASE_URL = os.getenv("PRM_BASE_URL", "http://localhost:8002")
PRM_MODEL_NAME = os.getenv("PRM_MODEL_NAME", "Qwen/Qwen2.5-Math-PRM-7B")
PRM_TIMEOUT_SECONDS = int(os.getenv("PRM_TIMEOUT_SECONDS", "15"))
ENABLE_STAGE2_CLASSIFIER = os.getenv("ENABLE_STAGE2_CLASSIFIER", "0") == "1"
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", MODEL_NAME)
ENABLE_ADAPTIVE_CLASSIFIER = os.getenv("ENABLE_ADAPTIVE_CLASSIFIER", "0") == "1"
EXPORT_WRONG_PROBLEM_TRACES = os.getenv("EXPORT_WRONG_PROBLEM_TRACES", "1") == "1"
COMPETITION_MODE = os.getenv("COMPETITION_MODE", "0") == "1"
COMPETITION_TOTAL_BUDGET_SECONDS = int(os.getenv("COMPETITION_TOTAL_BUDGET_SECONDS", "18000"))
COMPETITION_PROBLEMS_TOTAL = int(os.getenv("COMPETITION_PROBLEMS_TOTAL", "50"))
COMPETITION_LAYER1_SAMPLES = int(os.getenv("COMPETITION_LAYER1_SAMPLES", "3"))
COMPETITION_LAYER1_MAX_SECONDS = int(os.getenv("COMPETITION_LAYER1_MAX_SECONDS", "90"))
COMPETITION_LAYER2_SAMPLES = int(os.getenv("COMPETITION_LAYER2_SAMPLES", "8"))
COMPETITION_EARLY_STOP_AGREEMENT = int(os.getenv("COMPETITION_EARLY_STOP_AGREEMENT", "4"))
COMPETITION_ANSWER_MAX = int(os.getenv("COMPETITION_ANSWER_MAX", "99999"))
COMPETITION_FORCE_ALL_SAMPLES = os.getenv("COMPETITION_FORCE_ALL_SAMPLES", "0") == "1"
SAMPLE_PARALLELISM = int(os.getenv("SAMPLE_PARALLELISM", "1"))
COMPETITION_SAMPLE_PARALLELISM = int(
    os.getenv("COMPETITION_SAMPLE_PARALLELISM", str(SAMPLE_PARALLELISM))
)

if MODEL_FAMILY == "gpt_oss":
    if GPT_OSS_TRANSPORT == "harmony" or USE_GPT_OSS_HARMONY:
        GPT_OSS_TRANSPORT = "harmony"
        USE_GPT_OSS_HARMONY = True
        USE_RESPONSES_API = False
        USE_CHAT_COMPLETIONS_FALLBACK = False
    elif GPT_OSS_TRANSPORT == "responses":
        USE_GPT_OSS_HARMONY = False

MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", os.getenv("DEFAULT_MAX_TOOL_ROUNDS", "4")))
ENABLE_TOOL_USE = os.getenv("ENABLE_TOOL_USE", "1") == "1"
TOOL_GRACE_PERIOD_SECONDS = int(os.getenv("TOOL_GRACE_PERIOD_SECONDS", "60"))
FINALIZATION_GRACE_PERIOD_SECONDS = int(os.getenv("FINALIZATION_GRACE_PERIOD_SECONDS", "90"))

# Routing and budget controls
SAMPLE_COUNT = int(os.getenv("SAMPLE_COUNT", os.getenv("DEFAULT_SAMPLE_COUNT", os.getenv("NUM_SAMPLES", "8"))))
NUM_SAMPLES = SAMPLE_COUNT
PER_PROBLEM_MAX_TOKENS = int(os.getenv("PER_PROBLEM_MAX_TOKENS", "1024"))
if MODEL_FAMILY == "gpt_oss":
    PER_PROBLEM_MAX_TOKENS = max(PER_PROBLEM_MAX_TOKENS, GPT_OSS_MIN_OUTPUT_TOKENS)
PER_PROBLEM_MAX_RUNTIME_SECONDS = int(
    os.getenv("PER_PROBLEM_MAX_RUNTIME_SECONDS", os.getenv("DEFAULT_PER_PROBLEM_RUNTIME_SECONDS", "90"))
)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
ROUTE_ID_DEFAULT = os.getenv("ROUTE_ID_DEFAULT", "single_pass")
TOOL_TIMEOUT_SECONDS = int(os.getenv("TOOL_TIMEOUT_SECONDS", "8"))
TOOL_MAX_CODE_CHARS = int(os.getenv("TOOL_MAX_CODE_CHARS", "12000"))
TOOL_MAX_OUTPUT_CHARS = int(os.getenv("TOOL_MAX_OUTPUT_CHARS", "8000"))
SMOKE_MODE_LIMIT = int(os.getenv("SMOKE_MODE_LIMIT", "0"))
EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", str(EVAL_SETS_DIR / "smoke_eval.jsonl"))

MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "MAX_OUTPUT_TOKENS",
        os.getenv("OPENAI_MAX_OUTPUT_TOKENS", os.getenv("OPENAI_MAX_TOKENS", str(PER_PROBLEM_MAX_TOKENS))),
    )
)
if MODEL_FAMILY == "gpt_oss":
    MAX_OUTPUT_TOKENS = max(MAX_OUTPUT_TOKENS, GPT_OSS_MIN_OUTPUT_TOKENS)
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", str(MAX_OUTPUT_TOKENS)))
if MODEL_FAMILY == "gpt_oss":
    OPENAI_MAX_OUTPUT_TOKENS = max(OPENAI_MAX_OUTPUT_TOKENS, GPT_OSS_MIN_OUTPUT_TOKENS)
TEMPERATURE = float(os.getenv("TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0.2")))
TOP_P = float(os.getenv("TOP_P", os.getenv("OPENAI_TOP_P", "0.9")))
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "")
RESPONSES_API_REASONING_EFFORT = os.getenv(
    "RESPONSES_API_REASONING_EFFORT",
    REASONING_EFFORT or ("medium" if MODEL_FAMILY == "gpt_oss" else ""),
)
STOP_SEQUENCES_RAW = os.getenv("STOP_SEQUENCES", "")
try:
    STOP_SEQUENCES = tuple(json.loads(STOP_SEQUENCES_RAW)) if STOP_SEQUENCES_RAW.strip().startswith("[") else tuple(
        item.strip() for item in STOP_SEQUENCES_RAW.split("||") if item.strip()
    )
except json.JSONDecodeError:
    STOP_SEQUENCES = tuple(item.strip() for item in STOP_SEQUENCES_RAW.split("||") if item.strip())

# Real backend settings
REAL_MODEL_API_URL = os.getenv("REAL_MODEL_API_URL", "http://localhost:8000/generate")
REAL_MODEL_TIMEOUT_SECONDS = int(os.getenv("REAL_MODEL_TIMEOUT_SECONDS", "120"))

# OpenAI-compatible (vLLM) backend settings
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000")
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", MODEL_NAME)
OPENAI_REQUEST_TIMEOUT_SECONDS = int(os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "120"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", str(MAX_OUTPUT_TOKENS)))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", str(TEMPERATURE)))
OPENAI_TOP_P = float(os.getenv("OPENAI_TOP_P", str(TOP_P)))

# Long-problem forced tool-first backend settings
LONG_PROBLEM_FORCE_TOOL_FIRST = os.getenv("LONG_PROBLEM_FORCE_TOOL_FIRST", "1") == "1"
TOOL_CALL_OPENAI_BASE_URL = os.getenv("TOOL_CALL_OPENAI_BASE_URL", "http://localhost:8001")
TOOL_CALL_OPENAI_API_KEY = os.getenv("TOOL_CALL_OPENAI_API_KEY", OPENAI_COMPAT_API_KEY)
TOOL_CALL_OPENAI_MODEL_NAME = os.getenv("TOOL_CALL_OPENAI_MODEL_NAME", MODEL_NAME)
TOOL_CALL_REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("TOOL_CALL_REQUEST_TIMEOUT_SECONDS", str(OPENAI_REQUEST_TIMEOUT_SECONDS))
)
TOOL_CALL_MAX_TOKENS = int(os.getenv("TOOL_CALL_MAX_TOKENS", "512"))
TOOL_CALL_TEMPERATURE = float(os.getenv("TOOL_CALL_TEMPERATURE", "0.6"))
TOOL_CALL_TOP_P = float(os.getenv("TOOL_CALL_TOP_P", "0.95"))
TOOL_CALL_FOLLOWUP_TEMPERATURE = float(os.getenv("TOOL_CALL_FOLLOWUP_TEMPERATURE", "0.2"))


@dataclass(frozen=True)
class RunDefaults:
    experiment_id: str = "exp001_baseline_single"
    model_name: str = MODEL_NAME
    prompt_version: str = "p_v1"
    seed: int = 42
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_tokens: int = PER_PROBLEM_MAX_TOKENS


@dataclass(frozen=True)
class RoutingBudgetConfig:
    model_backend: str = MODEL_BACKEND
    model_family: str = MODEL_FAMILY
    model_name: str = MODEL_NAME
    sample_count: int = SAMPLE_COUNT
    max_tool_rounds: int = MAX_TOOL_ROUNDS
    model_timeout_seconds: int = REAL_MODEL_TIMEOUT_SECONDS
    per_problem_max_tokens: int = PER_PROBLEM_MAX_TOKENS
    per_problem_max_runtime_seconds: int = PER_PROBLEM_MAX_RUNTIME_SECONDS
    max_retries: int = MAX_RETRIES
    tool_timeout_seconds: int = TOOL_TIMEOUT_SECONDS
    route_id_default: str = ROUTE_ID_DEFAULT


def get_routing_budget_config() -> RoutingBudgetConfig:
    return RoutingBudgetConfig()


def ensure_dirs() -> None:
    for directory in (
        DATA_DIR,
        LOGS_DIR,
        EVAL_SETS_DIR,
        EXPERIMENTS_DIR,
        SUBMISSIONS_DIR,
        GPT_OSS_PROTOCOL_AUDIT_DIR,
        GPT_OSS_PROTOCOL_RUNS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

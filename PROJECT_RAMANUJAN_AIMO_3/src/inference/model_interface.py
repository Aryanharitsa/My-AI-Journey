from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from adapters.base_adapter import GenerationRequest
from config import (
    ENABLE_DEEPCONF,
    ENABLE_DEEPCONF_LOGPROBS,
    GPT_OSS_HARMONY_ALLOW_TEXT_FALLBACK,
    GPT_OSS_HARMONY_ENABLE_PYTHON_MCP,
    GPT_OSS_HARMONY_SAFETY_MARGIN,
    GPT_OSS_EXPECT_HARMONY,
    GPT_OSS_MAX_MODEL_LEN,
    LONG_PROBLEM_FORCE_TOOL_FIRST,
    MODEL_NAME,
    OPENAI_COMPAT_API_KEY,
    OPENAI_COMPAT_BASE_URL,
    OPENAI_MAX_TOKENS,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
    OPENAI_TEMPERATURE,
    OPENAI_TOP_P,
    REAL_MODEL_API_URL,
    REAL_MODEL_TIMEOUT_SECONDS,
    USE_CHAT_COMPLETIONS_FALLBACK,
    TOOL_CALL_FOLLOWUP_TEMPERATURE,
    TOOL_CALL_MAX_TOKENS,
    TOOL_CALL_OPENAI_API_KEY,
    TOOL_CALL_OPENAI_BASE_URL,
    TOOL_CALL_OPENAI_MODEL_NAME,
    TOOL_CALL_REQUEST_TIMEOUT_SECONDS,
    TOOL_CALL_TEMPERATURE,
    TOOL_CALL_TOP_P,
)
from gpt_oss_replay import (
    make_replay_item,
    normalize_replay_items,
)
from harmony_bridge import (
    HARMONY_RETURN_TOKEN_ID,
    HarmonyIntegrationError,
    normalize_harmony_messages,
    parse_harmony_completion,
    parse_harmony_completion_text,
    render_harmony_prompt,
)


def require_harmony():
    try:
        import openai_harmony as harmony
    except ImportError as exc:
        raise HarmonyIntegrationError("openai-harmony is required for GPT-OSS Harmony transport") from exc
    return harmony


def require_requests():
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for network-backed model transport") from exc
    return requests


def _normalize_structured_output(
    strategy: str,
    content: Any,
    guided_payload: dict[str, Any],
) -> str | None:
    text = str(content or "").strip()
    if not text:
        return None
    if strategy == "guided_regex":
        pattern = guided_payload.get("guided_regex")
        if not isinstance(pattern, str) or not pattern.strip():
            return text
        try:
            match = re.search(pattern, text, flags=re.DOTALL)
        except re.error:
            return None
        if match is None:
            return None
        matched_text = match.group(0).strip()
        return matched_text or None
    if strategy == "guided_json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("final_answer"), str):
            return f"FINAL_ANSWER: {parsed['final_answer']}"
        return None
    return text


class ModelBackend(Protocol):
    def generate_request(self, request: GenerationRequest) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ToolCall:
    id: str | None
    type: str | None
    name: str | None
    arguments_raw: str
    arguments: dict[str, Any] | None
    call_id: str | None = None
    recipient: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "arguments_raw": self.arguments_raw,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "recipient": self.recipient,
        }


@dataclass(frozen=True)
class ModelTurnResult:
    raw_text: str | None
    final_text: str | None
    reasoning_present: bool
    reasoning_content: str | None
    finish_reason: str | None
    usage_prompt_tokens: int | None
    usage_completion_tokens: int | None
    usage_total_tokens: int | None
    tool_calls: tuple[ToolCall, ...]
    raw_response: dict[str, Any]
    endpoint_used: str
    backend_type: str
    raw_has_output_items: bool
    raw_has_output_text: bool = False
    raw_output_item_types: tuple[str, ...] = ()
    raw_output_channels: tuple[str, ...] = ()
    function_call_items_count: int = 0
    mcp_call_items_count: int = 0
    incomplete_details: dict[str, Any] | None = None
    truncation: str | None = None
    final_text_source: str | None = None
    adapter_type: str | None = None
    harmony_enabled: bool = False
    explicit_final_channel_present: bool = False
    explicit_message_channel_present: bool = False
    gpt_oss_replay_items: tuple[dict[str, Any], ...] = ()
    requested_max_tokens: int | None = None
    effective_max_tokens: int | None = None
    prompt_token_estimate: int | None = None
    max_tokens_clipped: bool = False
    max_tokens_clip_reason: str | None = None
    transport_type: str | None = None
    continuation_round_index: int | None = None
    replayed_reasoning_chars: int = 0
    replayed_reasoning_items_count: int = 0
    replayed_tool_calls_count: int = 0
    replayed_encrypted_reasoning_items_count: int = 0
    finalization_status: str | None = None
    finalization_failure_reason: str | None = None
    encrypted_reasoning_include_requested: bool = False
    encrypted_reasoning_include_accepted: bool | None = None
    harmony_completion_class: str | None = None
    harmony_completion_class_source: str | None = None
    harmony_token_ids_present: bool = False
    token_logprobs: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        usage = None
        if any(value is not None for value in (
            self.usage_prompt_tokens,
            self.usage_completion_tokens,
            self.usage_total_tokens,
        )):
            usage = {
                "prompt_tokens": self.usage_prompt_tokens,
                "completion_tokens": self.usage_completion_tokens,
                "total_tokens": self.usage_total_tokens,
            }
        final_text = self.final_text or ""
        raw_text = self.raw_text if self.raw_text is not None else final_text
        return {
            "output_text": final_text,
            "raw_output_text": raw_text,
            "final_text": self.final_text,
            "raw_text": self.raw_text,
            "finish_reason": self.finish_reason,
            "usage": usage,
            "usage_prompt_tokens": self.usage_prompt_tokens,
            "usage_completion_tokens": self.usage_completion_tokens,
            "usage_total_tokens": self.usage_total_tokens,
            "reasoning_present": self.reasoning_present,
            "reasoning_content": self.reasoning_content,
            "tool_calls": [tool_call.as_dict() for tool_call in self.tool_calls],
            "raw_response": self.raw_response,
            "endpoint_used": self.endpoint_used,
            "backend_type": self.backend_type,
            "raw_has_output_items": self.raw_has_output_items,
            "raw_has_output_text": self.raw_has_output_text,
            "raw_output_item_types": list(self.raw_output_item_types),
            "raw_output_channels": list(self.raw_output_channels),
            "function_call_items_count": self.function_call_items_count,
            "mcp_call_items_count": self.mcp_call_items_count,
            "incomplete_details": self.incomplete_details,
            "truncation": self.truncation,
            "final_text_source": self.final_text_source,
            "adapter_type": self.adapter_type or self.backend_type,
            "harmony_enabled": self.harmony_enabled,
            "explicit_final_channel_present": self.explicit_final_channel_present,
            "explicit_message_channel_present": self.explicit_message_channel_present,
            "gpt_oss_replay_items": [dict(item) for item in self.gpt_oss_replay_items],
            "gpt_oss_replay_items_count": len(self.gpt_oss_replay_items),
            "requested_max_tokens": self.requested_max_tokens,
            "effective_max_tokens": self.effective_max_tokens,
            "prompt_token_estimate": self.prompt_token_estimate,
            "max_tokens_clipped": self.max_tokens_clipped,
            "max_tokens_clip_reason": self.max_tokens_clip_reason,
            "transport_type": self.transport_type,
            "continuation_round_index": self.continuation_round_index,
            "replayed_reasoning_chars": self.replayed_reasoning_chars,
            "replayed_reasoning_items_count": self.replayed_reasoning_items_count,
            "replayed_tool_calls_count": self.replayed_tool_calls_count,
            "replayed_encrypted_reasoning_items_count": self.replayed_encrypted_reasoning_items_count,
            "finalization_status": self.finalization_status,
            "finalization_failure_reason": self.finalization_failure_reason,
            "encrypted_reasoning_include_requested": self.encrypted_reasoning_include_requested,
            "encrypted_reasoning_include_accepted": self.encrypted_reasoning_include_accepted,
            "harmony_completion_class": self.harmony_completion_class,
            "harmony_completion_class_source": self.harmony_completion_class_source,
            "harmony_token_ids_present": self.harmony_token_ids_present,
            "token_logprobs": list(self.token_logprobs),
            "token_logprobs_count": len(self.token_logprobs),
            "final_text_present": bool(final_text.strip()),
            "final_text_chars": len(final_text),
            "tool_calls_count": len(self.tool_calls),
        }


def _legacy_request(system_prompt: str, user_prompt: str, *, max_tokens: int, temperature: float, top_p: float) -> GenerationRequest:
    messages: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return GenerationRequest(
        messages=tuple(messages),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def _legacy_prompts_from_messages(messages: tuple[dict[str, Any], ...]) -> tuple[str, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "") or ""
        if role in {"system", "developer"}:
            system_parts.append(content)
        elif role == "assistant":
            user_parts.append(f"Assistant:\n{content}")
        elif role == "tool":
            user_parts.append(f"Tool:\n{content}")
        else:
            user_parts.append(content)
    return "\n\n".join(part for part in system_parts if part.strip()), "\n\n".join(user_parts)


def _normalize_usage_dict(usage: dict[str, Any] | None) -> tuple[int | None, int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None, None
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def _parse_json_arguments(arguments_raw: Any) -> dict[str, Any] | None:
    if not isinstance(arguments_raw, str) or not arguments_raw.strip():
        return None
    try:
        parsed = json.loads(arguments_raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _gpt_oss_budget(
    *,
    requested_max_tokens: int,
    prompt_token_estimate: int | None,
) -> tuple[int, bool, str | None]:
    if prompt_token_estimate is None:
        return requested_max_tokens, False, None
    available_tokens = GPT_OSS_MAX_MODEL_LEN - prompt_token_estimate - GPT_OSS_HARMONY_SAFETY_MARGIN
    effective_max_tokens = max(1, min(requested_max_tokens, available_tokens))
    clipped = effective_max_tokens < requested_max_tokens
    return effective_max_tokens, clipped, ("context_window_clip" if clipped else None)


def _estimate_gpt_oss_prompt_tokens(request: GenerationRequest) -> int | None:
    try:
        rendered = render_harmony_prompt(
            messages=request.messages,
            tools=request.tools,
            reasoning_effort=request.reasoning_effort if GPT_OSS_EXPECT_HARMONY else None,
            enable_python_mcp=GPT_OSS_HARMONY_ENABLE_PYTHON_MCP,
        )
    except Exception:
        return None
    return len(rendered.prompt_tokens)


def _request_replay_metrics(request: GenerationRequest) -> tuple[int, int, int, int]:
    reasoning_chars = 0
    reasoning_items = 0
    tool_call_items = 0
    encrypted_reasoning_items = 0
    for message in request.messages:
        if message.get("role") == "assistant" and message.get("channel") == "analysis":
            content = str(message.get("content", "") or "")
            reasoning_chars += len(content)
            reasoning_items += 1
            if isinstance(message.get("encrypted_content"), str) and message.get("encrypted_content"):
                encrypted_reasoning_items += 1
        if message.get("role") == "assistant" and isinstance(message.get("recipient"), str):
            tool_call_items += 1
    return reasoning_chars, reasoning_items, tool_call_items, encrypted_reasoning_items


def _stream_event_writer(path: str | None):
    if not isinstance(path, str) or not path.strip():
        return lambda event: None

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(event: dict[str, Any]) -> None:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    return _write


class StubModel:
    def generate_request(self, request: GenerationRequest) -> dict[str, Any]:
        return self.generate_with_metadata(*_legacy_prompts_from_messages(request.messages))

    def generate_with_metadata(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        output_text = self.generate(system_prompt, user_prompt)
        return {
            "output_text": output_text,
            "raw_output_text": output_text,
            "finish_reason": None,
            "usage": None,
            "reasoning_content": None,
            "tool_calls": [],
        }

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return """THOUGHT: This is a stubbed model response for pipeline testing.
FINAL_ANSWER: STUB
"""


class ArithmeticDebugModel:
    def generate_request(self, request: GenerationRequest) -> dict[str, Any]:
        for message in reversed(request.messages):
            if message.get("role") == "tool":
                output_text = self._handle_tool_message(str(message.get("content", "") or ""))
                return {
                    "output_text": output_text,
                    "raw_output_text": output_text,
                    "finish_reason": "stop",
                    "usage": None,
                    "reasoning_content": None,
                    "tool_calls": [],
                }
        return self.generate_with_metadata(*_legacy_prompts_from_messages(request.messages))

    def generate_with_metadata(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        output_text = self.generate(system_prompt, user_prompt)
        return {
            "output_text": output_text,
            "raw_output_text": output_text,
            "finish_reason": "stop",
            "usage": None,
            "reasoning_content": None,
            "tool_calls": [],
        }

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        lowered = user_prompt.lower()

        if "tool result:" in lowered:
            return self._handle_followup(user_prompt)

        return self._handle_initial(user_prompt)

    def _extract_problem(self, user_prompt: str) -> str:
        problem_match = re.search(r"Problem:\n(.*?)\n\n(?:Instructions:|Expected answer type:)", user_prompt, re.DOTALL)
        if not problem_match:
            problem_match = re.search(r"Problem:\n(.*?)\n\nYour previous response:", user_prompt, re.DOTALL)
        return problem_match.group(1).strip() if problem_match else user_prompt.strip()

    def _handle_initial(self, user_prompt: str) -> str:
        problem = self._extract_problem(user_prompt)
        text = problem.lower().strip()

        if "sum of the first" in text:
            n = self._extract_first_int(text)
            if n is not None:
                return f"""THOUGHT: I'll use the arithmetic series formula or verify by computation.
TOOL_REQUEST:
n = {n}
print(n * (n + 1) // 2)
FINAL_ANSWER: PENDING
"""

        answer = self._solve(problem)
        return f"""THOUGHT: I can solve this directly.
FINAL_ANSWER: {answer}
"""

    def _handle_followup(self, user_prompt: str) -> str:
        stdout_match = re.search(r"stdout=(.*)", user_prompt)
        stdout = stdout_match.group(1).strip() if stdout_match else ""
        answer = stdout if stdout else "UNKNOWN"
        return f"""THOUGHT: Using the tool result, I can now conclude the answer.
FINAL_ANSWER: {answer}
"""

    def _handle_tool_message(self, tool_content: str) -> str:
        stdout = ""
        error = ""
        try:
            payload = json.loads(tool_content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            stdout = str(payload.get("stdout") or "").strip()
            error = str(payload.get("error") or "").strip()
        answer = stdout if stdout else ("UNKNOWN" if error else "UNKNOWN")
        return f"""THOUGHT: Using the tool result, I can now conclude the answer.
FINAL_ANSWER: {answer}
"""

    def _extract_first_int(self, text: str):
        match = re.search(r"(\d+)", text)
        return int(match.group(1)) if match else None

    def _solve(self, problem: str) -> str:
        text = problem.lower().strip()

        sum_match = re.search(r"sum of the first (\d+) positive integers", text)
        if sum_match:
            n = int(sum_match.group(1))
            return str(n * (n + 1) // 2)

        mult_match = re.search(r"what is (\d+) multiplied by (\d+)", text)
        if mult_match:
            a = int(mult_match.group(1))
            b = int(mult_match.group(2))
            return str(a * b)

        power_match = re.search(r"compute (\d+)\^(\d+)", text)
        if power_match:
            base = int(power_match.group(1))
            exp = int(power_match.group(2))
            return str(base ** exp)

        return "UNKNOWN"


class HttpModel:
    def __init__(self, api_url: str = REAL_MODEL_API_URL, timeout_seconds: int = REAL_MODEL_TIMEOUT_SECONDS):
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds

    def generate_request(self, request: GenerationRequest) -> dict[str, Any]:
        system_prompt, user_prompt = _legacy_prompts_from_messages(request.messages)
        return self.generate_with_metadata(system_prompt, user_prompt)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.generate_with_metadata(system_prompt, user_prompt)["output_text"]

    def generate_with_metadata(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        requests = require_requests()
        payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        response = requests.post(self.api_url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        if "output_text" not in data:
            raise ValueError(f"HTTP model response missing 'output_text': {data}")
        output_text = data["output_text"]
        return {
            "output_text": output_text,
            "raw_output_text": output_text,
            "finish_reason": data.get("finish_reason"),
            "usage": data.get("usage"),
            "reasoning_content": data.get("reasoning_content"),
            "tool_calls": data.get("tool_calls") or [],
        }


class OpenAICompatibleModel:
    def __init__(
        self,
        base_url: str = OPENAI_COMPAT_BASE_URL,
        api_key: str = OPENAI_COMPAT_API_KEY,
        model_name: str = MODEL_NAME,
        timeout_seconds: int = OPENAI_REQUEST_TIMEOUT_SECONDS,
        max_tokens: int = OPENAI_MAX_TOKENS,
        temperature: float = OPENAI_TEMPERATURE,
        top_p: float = OPENAI_TOP_P,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    @property
    def chat_api_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    @property
    def endpoint_used(self) -> str:
        return "/v1/chat/completions"

    @property
    def backend_type(self) -> str:
        return "chat_completions"

    def _build_messages(self, system_prompt: str | None, user_prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        requests = require_requests()
        response = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout_seconds)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise ValueError(f"OpenAI-compatible model returned non-JSON response: {response.text[:300]}") from exc

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = {"model": self.model_name, **payload}
        return self._post_json(self.chat_api_url, payload)

    def _extract_content(self, data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (TypeError, KeyError, IndexError) as exc:
            raise ValueError(f"OpenAI-compatible response missing choices[0].message.content: {data}") from exc
        if content is None:
            return ""
        if not isinstance(content, str):
            raise ValueError("OpenAI-compatible response field choices[0].message.content must be a string")
        return content

    def _extract_finish_reason(self, data: dict[str, Any]) -> str | None:
        try:
            finish_reason = data["choices"][0].get("finish_reason")
        except (TypeError, KeyError, IndexError):
            return None
        return finish_reason if isinstance(finish_reason, str) else None

    def _extract_usage(self, data: dict[str, Any]) -> dict[str, Any] | None:
        usage = data.get("usage")
        return usage if isinstance(usage, dict) else None

    def _extract_reasoning_content(self, data: dict[str, Any]) -> str | None:
        try:
            message = data["choices"][0]["message"]
        except (TypeError, KeyError, IndexError):
            return None
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str):
            return reasoning_content
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str):
            return reasoning
        return None

    def _extract_tool_calls(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            tool_calls = data["choices"][0]["message"].get("tool_calls") or []
        except (TypeError, KeyError, IndexError):
            return []
        parsed_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            function_payload = tool_call.get("function") or {}
            arguments_raw = function_payload.get("arguments", "")
            parsed_calls.append(
                ToolCall(
                    id=tool_call.get("id"),
                    type=tool_call.get("type"),
                    name=function_payload.get("name"),
                    arguments_raw=arguments_raw if isinstance(arguments_raw, str) else "",
                    arguments=_parse_json_arguments(arguments_raw),
                ).as_dict()
            )
        return parsed_calls

    def _truncate_at_canonical_final_line(self, content: str) -> str:
        canonical_patterns = [
            re.compile(r"(?im)^\s*FINAL_ANSWER:\s*-?\d+\s*$"),
            re.compile(r"(?im)^\s*\\boxed\{-?\d+\}\s*$"),
        ]
        for pattern in canonical_patterns:
            match = pattern.search(content)
            if match:
                return content[: match.end()].strip()
        return content

    def _payload_from_request(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": list(request.messages),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.stop_sequences:
            payload["stop"] = list(request.stop_sequences)
        if request.tools:
            payload["tools"] = list(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.reasoning_effort:
            payload["reasoning_effort"] = request.reasoning_effort
        return payload

    def _build_turn_result(
        self,
        *,
        raw_text: str | None,
        reasoning_content: str | None,
        finish_reason: str | None,
        usage: dict[str, Any] | None,
        tool_calls: list[dict[str, Any]],
        raw_response: dict[str, Any],
        reasoning_present_override: bool | None = None,
        raw_has_output_items: bool = False,
        raw_has_output_text: bool = False,
        raw_output_item_types: tuple[str, ...] = (),
        raw_output_channels: tuple[str, ...] = (),
        function_call_items_count: int = 0,
        mcp_call_items_count: int = 0,
        incomplete_details: dict[str, Any] | None = None,
        truncation: str | None = None,
        final_text_source: str | None = None,
        adapter_type: str | None = None,
        harmony_enabled: bool = False,
        explicit_final_channel_present: bool = False,
        explicit_message_channel_present: bool = False,
        gpt_oss_replay_items: tuple[dict[str, Any], ...] = (),
        requested_max_tokens: int | None = None,
        effective_max_tokens: int | None = None,
        prompt_token_estimate: int | None = None,
        max_tokens_clipped: bool = False,
        max_tokens_clip_reason: str | None = None,
        transport_type: str | None = None,
        continuation_round_index: int | None = None,
        replayed_reasoning_chars_value: int = 0,
        replayed_reasoning_items_count: int = 0,
        replayed_tool_calls_count: int = 0,
        replayed_encrypted_reasoning_items_count: int = 0,
        finalization_status: str | None = None,
        finalization_failure_reason: str | None = None,
        encrypted_reasoning_include_requested: bool = False,
        encrypted_reasoning_include_accepted: bool | None = None,
        harmony_completion_class: str | None = None,
        harmony_completion_class_source: str | None = None,
        harmony_token_ids_present: bool = False,
        token_logprobs: tuple[float, ...] = (),
    ) -> ModelTurnResult:
        prompt_tokens, completion_tokens, total_tokens = _normalize_usage_dict(usage)
        final_text = self._truncate_at_canonical_final_line(raw_text or "")
        return ModelTurnResult(
            raw_text=raw_text,
            final_text=final_text,
            reasoning_present=bool(reasoning_content) if reasoning_present_override is None else reasoning_present_override,
            reasoning_content=reasoning_content,
            finish_reason=finish_reason,
            usage_prompt_tokens=prompt_tokens,
            usage_completion_tokens=completion_tokens,
            usage_total_tokens=total_tokens,
            tool_calls=tuple(
                ToolCall(
                    id=tool_call.get("id"),
                    type=tool_call.get("type"),
                    name=tool_call.get("name"),
                    arguments_raw=str(tool_call.get("arguments_raw", "")),
                    arguments=tool_call.get("arguments"),
                    call_id=tool_call.get("call_id"),
                    recipient=tool_call.get("recipient"),
                )
                for tool_call in tool_calls
            ),
            raw_response=raw_response,
            endpoint_used=self.endpoint_used,
            backend_type=self.backend_type,
            raw_has_output_items=raw_has_output_items,
            raw_has_output_text=raw_has_output_text,
            raw_output_item_types=raw_output_item_types,
            raw_output_channels=raw_output_channels,
            function_call_items_count=function_call_items_count,
            mcp_call_items_count=mcp_call_items_count,
            incomplete_details=incomplete_details,
            truncation=truncation,
            final_text_source=final_text_source,
            adapter_type=adapter_type or self.backend_type,
            harmony_enabled=harmony_enabled,
            explicit_final_channel_present=explicit_final_channel_present,
            explicit_message_channel_present=explicit_message_channel_present,
            gpt_oss_replay_items=gpt_oss_replay_items,
            requested_max_tokens=requested_max_tokens,
            effective_max_tokens=effective_max_tokens,
            prompt_token_estimate=prompt_token_estimate,
            max_tokens_clipped=max_tokens_clipped,
            max_tokens_clip_reason=max_tokens_clip_reason,
            transport_type=transport_type,
            continuation_round_index=continuation_round_index,
            replayed_reasoning_chars=replayed_reasoning_chars_value,
            replayed_reasoning_items_count=replayed_reasoning_items_count,
            replayed_tool_calls_count=replayed_tool_calls_count,
            replayed_encrypted_reasoning_items_count=replayed_encrypted_reasoning_items_count,
            finalization_status=finalization_status,
            finalization_failure_reason=finalization_failure_reason,
            encrypted_reasoning_include_requested=encrypted_reasoning_include_requested,
            encrypted_reasoning_include_accepted=encrypted_reasoning_include_accepted,
            harmony_completion_class=harmony_completion_class,
            harmony_completion_class_source=harmony_completion_class_source,
            harmony_token_ids_present=harmony_token_ids_present,
            token_logprobs=token_logprobs,
        )

    def generate_request(self, request: GenerationRequest) -> dict[str, Any]:
        data = self._post_chat_completion(self._payload_from_request(request))
        raw_output_text = self._extract_content(data)
        return self._build_turn_result(
            raw_text=raw_output_text,
            reasoning_content=self._extract_reasoning_content(data),
            finish_reason=self._extract_finish_reason(data),
            usage=self._extract_usage(data),
            tool_calls=self._extract_tool_calls(data),
            raw_response=data,
            final_text_source="message.content",
        ).as_dict()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.generate_with_metadata(system_prompt, user_prompt)["output_text"]

    def generate_with_metadata(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        request = _legacy_request(
            system_prompt,
            user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return self.generate_request(request)

    def generate_structured_from_request(self, request: GenerationRequest) -> dict[str, Any]:
        guided_regex_override = None
        if isinstance(request.metadata, dict):
            override = request.metadata.get("guided_regex_override")
            if isinstance(override, str) and override.strip():
                guided_regex_override = override.strip()
        if guided_regex_override:
            structured_attempts = [("guided_regex", {"guided_regex": guided_regex_override})]
        else:
            structured_attempts = [
                ("guided_regex", {"guided_regex": r"FINAL_ANSWER:\s*-?\d+"}),
                (
                    "guided_json",
                    {
                        "guided_json": {
                            "type": "object",
                            "properties": {"final_answer": {"type": "string", "pattern": r"^-?\d+$"}},
                            "required": ["final_answer"],
                            "additionalProperties": False,
                        }
                    },
                ),
            ]
        errors: list[str] = []
        for strategy, guided_payload in structured_attempts:
            try:
                payload = self._payload_from_request(request)
                if isinstance(payload, tuple):
                    payload = payload[0]
                payload.update({"max_tokens": min(request.max_tokens, 64), "temperature": 0.0, "top_p": 1.0})
                payload.update(guided_payload)
                data = self._post_chat_completion(payload)
                raw_content = self._extract_content(data)
                content = _normalize_structured_output(strategy, raw_content, guided_payload)
                if content is None:
                    errors.append(f"{strategy}: structured_output_validation_failed")
                    continue
                return {
                    "output_text": content,
                    "raw_output_text": content,
                    "strategy": strategy,
                    "finish_reason": self._extract_finish_reason(data),
                    "usage": self._extract_usage(data),
                    "reasoning_content": self._extract_reasoning_content(data),
                }
            except Exception as exc:
                errors.append(f"{strategy}: {exc}")
        raise RuntimeError("structured_finalization_unavailable: " + " | ".join(errors))

    def generate_structured_finalization(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        request = _legacy_request(
            system_prompt,
            user_prompt,
            max_tokens=min(self.max_tokens, 64),
            temperature=0.0,
            top_p=1.0,
        )
        return self.generate_structured_from_request(request)


class ResponsesModel(OpenAICompatibleModel):
    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1/responses"

    @property
    def endpoint_used(self) -> str:
        return "/v1/responses"

    @property
    def backend_type(self) -> str:
        return "responses"

    @staticmethod
    def _output_items(data: dict[str, Any]) -> list[dict[str, Any]]:
        return [item for item in (data.get("output") or []) if isinstance(item, dict)]

    def _raw_output_item_types(self, data: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            str(item.get("type"))
            for item in self._output_items(data)
            if isinstance(item.get("type"), str)
        )

    def _instructions_and_input(self, messages: tuple[dict[str, Any], ...]) -> tuple[str | None, list[dict[str, Any]]]:
        instructions: list[str] = []
        input_items: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if role in {"system", "developer"}:
                if isinstance(content, str) and content.strip():
                    instructions.append(content)
                continue
            input_items.append({"role": role, "content": content})
        return ("\n\n".join(instructions) if instructions else None), input_items

    def _payload_from_request(self, request: GenerationRequest) -> dict[str, Any]:
        instructions, input_items = self._instructions_and_input(request.messages)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": input_items,
            "max_output_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if instructions:
            payload["instructions"] = instructions
        if request.reasoning_effort:
            payload["reasoning"] = {"effort": request.reasoning_effort}
        if request.tools:
            payload["tools"] = list(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        return payload

    def _extract_message_text_parts(self, item: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
        return texts

    def _extract_visible_output_text(self, data: dict[str, Any]) -> tuple[str, str | None]:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text, "response.output_text"

        collected_parts: list[str] = []
        for item in data.get("output") or []:
            if isinstance(item, dict) and item.get("type") == "message":
                collected_parts.extend(self._extract_message_text_parts(item))
        return "\n".join(part for part in collected_parts if part), ("output.message.content" if collected_parts else None)

    def _extract_reasoning_content(self, data: dict[str, Any]) -> str | None:
        reasoning_parts: list[str] = []
        for item in self._output_items(data):
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                continue
            summary = item.get("summary")
            if isinstance(summary, list):
                for part in summary:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        reasoning_parts.append(part["text"])
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        reasoning_parts.append(part["text"])
            if isinstance(item.get("text"), str):
                reasoning_parts.append(item["text"])
        return "\n".join(part for part in reasoning_parts if part) or None

    def _extract_finish_reason(self, data: dict[str, Any]) -> str | None:
        incomplete_details = data.get("incomplete_details")
        if isinstance(incomplete_details, dict) and isinstance(incomplete_details.get("reason"), str):
            return incomplete_details["reason"]
        status = data.get("status")
        if status == "completed":
            return "stop"
        return status if isinstance(status, str) else None

    def _extract_usage(self, data: dict[str, Any]) -> dict[str, Any] | None:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt_tokens, completion_tokens, total_tokens = _normalize_usage_dict(usage)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _extract_tool_calls(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        parsed_calls: list[dict[str, Any]] = []
        for item in self._output_items(data):
            item_type = item.get("type")
            if item_type not in {"function_call", "mcp_call"}:
                continue
            arguments_raw = item.get("arguments", "")
            parsed_arguments = _parse_json_arguments(arguments_raw)
            is_python_mcp_exec = (
                item_type == "mcp_call"
                and item.get("server_label") == "python"
                and isinstance(arguments_raw, str)
                and bool(arguments_raw.strip())
            )
            if parsed_arguments is None and is_python_mcp_exec:
                parsed_arguments = {
                    "code": arguments_raw,
                    "intent": f"{item.get('server_label', 'python')}:{item.get('name', 'exec')}",
                }
            recipient = None
            if item_type == "function_call" and isinstance(item.get("name"), str):
                recipient = f"functions.{item['name']}"
            elif item_type == "mcp_call":
                server_label = item.get("server_label")
                name = item.get("name")
                if isinstance(server_label, str) and server_label and isinstance(name, str) and name:
                    recipient = f"{server_label}.{name}"
            parsed_calls.append(
                ToolCall(
                    id=item.get("id"),
                    type=item_type,
                    name="python_exec" if is_python_mcp_exec else item.get("name"),
                    arguments_raw=arguments_raw if isinstance(arguments_raw, str) else "",
                    arguments=parsed_arguments,
                    call_id=item.get("call_id"),
                    recipient=recipient,
                ).as_dict()
            )
        return parsed_calls

    def generate_request(self, request: GenerationRequest) -> dict[str, Any]:
        data = self._post_json(self.api_url, self._payload_from_request(request))
        raw_output_text, final_text_source = self._extract_visible_output_text(data)
        raw_output_item_types = self._raw_output_item_types(data)
        return self._build_turn_result(
            raw_text=raw_output_text,
            reasoning_content=self._extract_reasoning_content(data),
            finish_reason=self._extract_finish_reason(data),
            usage=self._extract_usage(data),
            tool_calls=self._extract_tool_calls(data),
            raw_response=data,
            raw_has_output_items=bool(self._output_items(data)),
            raw_has_output_text=bool(data.get("output_text")),
            raw_output_item_types=raw_output_item_types,
            function_call_items_count=sum(1 for item_type in raw_output_item_types if item_type == "function_call"),
            mcp_call_items_count=sum(1 for item_type in raw_output_item_types if item_type == "mcp_call"),
            incomplete_details=data.get("incomplete_details") if isinstance(data.get("incomplete_details"), dict) else None,
            truncation=data.get("truncation") if isinstance(data.get("truncation"), str) else None,
            final_text_source=final_text_source,
        ).as_dict()


class GptOssResponsesModel(ResponsesModel):
    @property
    def backend_type(self) -> str:
        return "gpt_oss_responses"

    @staticmethod
    def _assistant_message_input_item(message: dict[str, Any]) -> dict[str, Any]:
        input_item: dict[str, Any] = {
            "role": str(message.get("role", "assistant")),
            "content": message.get("content", ""),
        }
        for key in ("channel", "recipient", "content_type", "name", "tool_call_id", "call_id"):
            value = message.get(key)
            if isinstance(value, str) and value:
                input_item[key] = value
        return input_item

    @staticmethod
    def _reasoning_input_item(message: dict[str, Any]) -> dict[str, Any] | None:
        reasoning_id = message.get("id")
        if not isinstance(reasoning_id, str) or not reasoning_id:
            return None
        encrypted_content = message.get("encrypted_content")
        content = str(message.get("content", "") or "")
        item: dict[str, Any] = {
            "type": "reasoning",
            "id": reasoning_id,
            "summary": [],
        }
        if isinstance(message.get("status"), str) and message.get("status"):
            item["status"] = message["status"]
        if isinstance(encrypted_content, str) and encrypted_content:
            item["encrypted_content"] = encrypted_content
        if content.strip():
            item["content"] = [{"type": "reasoning_text", "text": content}]
        if not item.get("encrypted_content") and "content" not in item:
            return None
        return item

    @staticmethod
    def _function_call_input_item(message: dict[str, Any]) -> dict[str, Any] | None:
        item_type = message.get("type")
        recipient = message.get("recipient")
        supports_function_call_replay = item_type in {"function_call", "mcp_call"}
        if not supports_function_call_replay and (
            not isinstance(recipient, str) or not recipient.startswith("functions.")
        ):
            return None
        tool_name = message.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            if isinstance(recipient, str) and "." in recipient:
                tool_name = recipient.rsplit(".", 1)[-1]
            elif isinstance(recipient, str) and recipient:
                tool_name = recipient
            else:
                return None
        call_id = message.get("call_id") or message.get("tool_call_id") or message.get("id")
        if not isinstance(call_id, str) or not call_id:
            return None
        arguments_raw = message.get("arguments_raw")
        if not isinstance(arguments_raw, str):
            arguments_raw = str(message.get("content", "") or "")
        item: dict[str, Any] = {
            "type": "function_call",
            "call_id": call_id,
            "name": tool_name,
            "arguments": arguments_raw,
        }
        if isinstance(message.get("id"), str) and message.get("id"):
            item["id"] = message["id"]
        if isinstance(message.get("status"), str) and message.get("status"):
            item["status"] = message["status"]
        return item

    @staticmethod
    def _tool_output_input_item(message: dict[str, Any]) -> dict[str, Any] | None:
        call_id = message.get("tool_call_id") or message.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return None
        item: dict[str, Any] = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": str(message.get("content", "") or ""),
        }
        if isinstance(message.get("id"), str) and message.get("id"):
            item["id"] = message["id"]
        if isinstance(message.get("status"), str) and message.get("status"):
            item["status"] = message["status"]
        return item

    @staticmethod
    def _should_request_encrypted_reasoning(request: GenerationRequest) -> bool:
        return not bool(request.metadata.get("disable_encrypted_reasoning_include"))

    @staticmethod
    def _should_retry_without_encrypted_reasoning_include(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        if response is None or getattr(response, "status_code", None) != 400:
            return False
        response_text = getattr(response, "text", "") or ""
        lowered = response_text.lower()
        return "include" in lowered or "reasoning.encrypted_content" in lowered

    def _instructions_and_input(self, messages: tuple[dict[str, Any], ...]) -> tuple[str | None, list[dict[str, Any]]]:
        instructions: list[str] = []
        input_items: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if role in {"system", "developer"}:
                if isinstance(content, str) and content.strip():
                    instructions.append(content)
                continue
            input_item: dict[str, Any]
            if role == "assistant" and message.get("channel") == "analysis":
                input_item = self._reasoning_input_item(message) or self._assistant_message_input_item(message)
            elif role == "assistant":
                input_item = self._function_call_input_item(message) or self._assistant_message_input_item(message)
            elif role == "tool":
                input_item = self._tool_output_input_item(message) or self._assistant_message_input_item(
                    {"role": "assistant", "content": content}
                )
            else:
                input_item = {"role": role, "content": content}
            input_items.append(input_item)
        return ("\n\n".join(instructions) if instructions else None), input_items

    def _extract_replay_items(self, data: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        replay_items: list[dict[str, Any]] = []
        for item in self._output_items(data):
            item_type = item.get("type")
            if item_type == "reasoning":
                reasoning_parts: list[str] = []
                encrypted_content = item.get("encrypted_content") if isinstance(item.get("encrypted_content"), str) else None
                for key in ("summary", "content"):
                    value = item.get(key)
                    if isinstance(value, list):
                        for part in value:
                            if isinstance(part, dict) and isinstance(part.get("text"), str) and part.get("text"):
                                reasoning_parts.append(part["text"])
                if isinstance(item.get("text"), str) and item.get("text"):
                    reasoning_parts.append(item["text"])
                if reasoning_parts or encrypted_content:
                    replay_items.append(
                        make_replay_item(
                            role="assistant",
                            channel="analysis",
                            content="\n".join(reasoning_parts),
                            item_type="reasoning",
                            item_id=item.get("id") if isinstance(item.get("id"), str) else None,
                            status=item.get("status") if isinstance(item.get("status"), str) else None,
                            encrypted_content=encrypted_content,
                        )
                    )
                continue

            if item_type == "message":
                text_parts = self._extract_message_text_parts(item)
                if text_parts:
                    replay_items.append(
                        make_replay_item(
                            role="assistant",
                            channel="commentary",
                            content="\n".join(text_parts),
                            item_type="message",
                            item_id=item.get("id") if isinstance(item.get("id"), str) else None,
                            status=item.get("status") if isinstance(item.get("status"), str) else None,
                        )
                    )
                continue

            if item_type == "function_call":
                arguments_raw = item.get("arguments", "")
                replay_items.append(
                    make_replay_item(
                        role="assistant",
                        channel="commentary",
                        recipient=f"functions.{item.get('name')}" if isinstance(item.get("name"), str) else None,
                        content_type="<|constrain|> json",
                        content=arguments_raw if isinstance(arguments_raw, str) else "",
                        name=item.get("name") if isinstance(item.get("name"), str) else None,
                        item_type="function_call",
                        item_id=item.get("id") if isinstance(item.get("id"), str) else None,
                        status=item.get("status") if isinstance(item.get("status"), str) else None,
                        call_id=(
                            item.get("call_id")
                            if isinstance(item.get("call_id"), str)
                            else (item.get("id") if isinstance(item.get("id"), str) else None)
                        ),
                        tool_call_id=(
                            item.get("call_id")
                            if isinstance(item.get("call_id"), str)
                            else (item.get("id") if isinstance(item.get("id"), str) else None)
                        ),
                        arguments_raw=arguments_raw if isinstance(arguments_raw, str) else "",
                        arguments=_parse_json_arguments(arguments_raw),
                    )
                )
                continue

            if item_type == "mcp_call":
                arguments_raw = item.get("arguments", "")
                server_label = item.get("server_label")
                name = item.get("name")
                recipient = (
                    f"{server_label}.{name}"
                    if isinstance(server_label, str) and server_label and isinstance(name, str) and name
                    else None
                )
                parsed_arguments = _parse_json_arguments(arguments_raw)
                if parsed_arguments is None and recipient == "python.exec" and isinstance(arguments_raw, str) and arguments_raw.strip():
                    parsed_arguments = {"code": arguments_raw, "intent": recipient}
                replay_items.append(
                    make_replay_item(
                        role="assistant",
                        channel="commentary",
                        recipient=recipient,
                        content=arguments_raw if isinstance(arguments_raw, str) else "",
                        name="python_exec" if recipient == "python.exec" else (name if isinstance(name, str) else None),
                        item_type="mcp_call",
                        item_id=item.get("id") if isinstance(item.get("id"), str) else None,
                        status=item.get("status") if isinstance(item.get("status"), str) else None,
                        call_id=(
                            item.get("call_id")
                            if isinstance(item.get("call_id"), str)
                            else (item.get("id") if isinstance(item.get("id"), str) else None)
                        ),
                        tool_call_id=(
                            item.get("call_id")
                            if isinstance(item.get("call_id"), str)
                            else (item.get("id") if isinstance(item.get("id"), str) else None)
                        ),
                        arguments_raw=arguments_raw if isinstance(arguments_raw, str) else "",
                        arguments=parsed_arguments,
                    )
                )

        return normalize_replay_items(replay_items)

    def _payload_from_request(self, request: GenerationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
        instructions, input_items = self._instructions_and_input(request.messages)
        prompt_token_estimate = _estimate_gpt_oss_prompt_tokens(request)
        effective_max_tokens, clipped, clip_reason = _gpt_oss_budget(
            requested_max_tokens=request.max_tokens,
            prompt_token_estimate=prompt_token_estimate,
        )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": input_items,
            "max_output_tokens": effective_max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if instructions:
            payload["instructions"] = instructions
        if request.reasoning_effort:
            payload["reasoning"] = {"effort": request.reasoning_effort}
        if self._should_request_encrypted_reasoning(request):
            payload["include"] = ["reasoning.encrypted_content"]
        if request.tools:
            payload["tools"] = list(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        return payload, {
            "requested_max_tokens": request.max_tokens,
            "effective_max_tokens": effective_max_tokens,
            "prompt_token_estimate": prompt_token_estimate,
            "max_tokens_clipped": clipped,
            "max_tokens_clip_reason": clip_reason,
            "encrypted_reasoning_include_requested": self._should_request_encrypted_reasoning(request),
        }

    def generate_request(self, request: GenerationRequest) -> dict[str, Any]:
        requests = require_requests()
        try:
            payload, budget_meta = self._payload_from_request(request)
            (
                replay_reasoning_chars,
                replay_reasoning_items,
                replay_tool_calls,
                replay_encrypted_reasoning_items,
            ) = _request_replay_metrics(request)
            include_accepted = bool(payload.get("include"))
            try:
                data = self._post_json(self.api_url, payload)
            except requests.RequestException as exc:
                if payload.get("include") and self._should_retry_without_encrypted_reasoning_include(exc):
                    payload = dict(payload)
                    payload.pop("include", None)
                    include_accepted = False
                    data = self._post_json(self.api_url, payload)
                else:
                    raise
            raw_output_text, final_text_source = self._extract_visible_output_text(data)
            raw_output_item_types = self._raw_output_item_types(data)
            replay_items = self._extract_replay_items(data)
            return self._build_turn_result(
                raw_text=raw_output_text,
                reasoning_content=self._extract_reasoning_content(data),
                reasoning_present_override=any(item.get("type") == "reasoning" for item in self._output_items(data)),
                finish_reason=self._extract_finish_reason(data),
                usage=self._extract_usage(data),
                tool_calls=self._extract_tool_calls(data),
                raw_response=data,
                raw_has_output_items=bool(self._output_items(data)),
                raw_has_output_text=bool(data.get("output_text")),
                raw_output_item_types=raw_output_item_types,
                function_call_items_count=sum(1 for item_type in raw_output_item_types if item_type == "function_call"),
                mcp_call_items_count=sum(1 for item_type in raw_output_item_types if item_type == "mcp_call"),
                incomplete_details=data.get("incomplete_details") if isinstance(data.get("incomplete_details"), dict) else None,
                truncation=data.get("truncation") if isinstance(data.get("truncation"), str) else None,
                final_text_source=final_text_source,
                adapter_type=self.backend_type,
                harmony_enabled=False,
                explicit_final_channel_present=False,
                explicit_message_channel_present=any(item.get("type") == "message" for item in self._output_items(data)),
                gpt_oss_replay_items=replay_items,
                requested_max_tokens=budget_meta["requested_max_tokens"],
                effective_max_tokens=budget_meta["effective_max_tokens"],
                prompt_token_estimate=budget_meta["prompt_token_estimate"],
                max_tokens_clipped=budget_meta["max_tokens_clipped"],
                max_tokens_clip_reason=budget_meta["max_tokens_clip_reason"],
                transport_type="responses",
                continuation_round_index=request.metadata.get("continuation_round_index"),
                replayed_reasoning_chars_value=replay_reasoning_chars,
                replayed_reasoning_items_count=replay_reasoning_items,
                replayed_tool_calls_count=replay_tool_calls,
                replayed_encrypted_reasoning_items_count=replay_encrypted_reasoning_items,
                encrypted_reasoning_include_requested=budget_meta["encrypted_reasoning_include_requested"],
                encrypted_reasoning_include_accepted=include_accepted,
            ).as_dict()
        except requests.RequestException:
            if not USE_CHAT_COMPLETIONS_FALLBACK:
                raise
            data = self._post_chat_completion(OpenAICompatibleModel._payload_from_request(self, request))
            fallback_result = OpenAICompatibleModel._build_turn_result(
                self,
                raw_text=self._extract_content(data),
                reasoning_content=self._extract_reasoning_content(data),
                finish_reason=self._extract_finish_reason(data),
                usage=self._extract_usage(data),
                tool_calls=self._extract_tool_calls(data),
                raw_response=data,
                final_text_source="message.content",
            ).as_dict()
            fallback_result["endpoint_used"] = "/v1/chat/completions"
            fallback_result["backend_type"] = "chat_completions"
            return fallback_result


class GptOssHarmonyModel(OpenAICompatibleModel):
    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1/completions"

    @property
    def endpoint_used(self) -> str:
        return "/v1/completions+harmony"

    @property
    def backend_type(self) -> str:
        return "gpt_oss_harmony"

    @staticmethod
    def _extract_completion_choice(data: dict[str, Any]) -> dict[str, Any]:
        try:
            choice = data["choices"][0]
        except (TypeError, KeyError, IndexError) as exc:
            raise ValueError(f"Harmony completion response missing choices[0]: {data}") from exc
        if not isinstance(choice, dict):
            raise ValueError(f"Harmony completion choice must be a dict: {choice!r}")
        return choice

    def _payload_from_request(self, request: GenerationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
        rendered = render_harmony_prompt(
            messages=request.messages,
            tools=request.tools,
            reasoning_effort=request.reasoning_effort if GPT_OSS_EXPECT_HARMONY else None,
            enable_python_mcp=GPT_OSS_HARMONY_ENABLE_PYTHON_MCP,
        )
        effective_max_tokens, clipped, clip_reason = _gpt_oss_budget(
            requested_max_tokens=request.max_tokens,
            prompt_token_estimate=len(rendered.prompt_tokens),
        )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": rendered.prompt_text,
            "max_tokens": effective_max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "add_special_tokens": False,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "return_token_ids": True,
        }
        if ENABLE_DEEPCONF or ENABLE_DEEPCONF_LOGPROBS:
            payload["logprobs"] = 1
        if rendered.stop_token_ids:
            # The end-of-message boundary token is required for Harmony channel
            # transitions inside a single completion. Stopping on it truncates
            # analysis -> commentary/tool -> final handoffs before they can form.
            from harmony_bridge import HARMONY_END_MESSAGE_TOKEN_ID

            filtered_stop_token_ids = [
                token_id for token_id in rendered.stop_token_ids
                if token_id != HARMONY_END_MESSAGE_TOKEN_ID
            ]
            if filtered_stop_token_ids:
                payload["stop_token_ids"] = filtered_stop_token_ids
        if request.stop_sequences:
            payload["stop"] = list(request.stop_sequences)
        # Fix 1: Nudge model toward emitting HARMONY_CALL_TOKEN_ID (200012)
        # via logit_bias on /v1/completions — tool_choice="required" is silently
        # dropped by this endpoint, but logit_bias IS a first-class SamplingParams
        # field in vLLM and works on completions.
        from config import GPT_OSS_HARMONY_TOOL_CALL_LOGIT_BIAS
        from harmony_bridge import HARMONY_CALL_TOKEN_ID
        if (
            GPT_OSS_HARMONY_TOOL_CALL_LOGIT_BIAS > 0
            and request.tool_choice in ("required", "auto")
            and request.tools
        ):
            bias_strength = GPT_OSS_HARMONY_TOOL_CALL_LOGIT_BIAS
            if request.tool_choice == "required":
                bias_strength = max(bias_strength, 5.0)
            payload["logit_bias"] = {str(HARMONY_CALL_TOKEN_ID): bias_strength}
        render_meta = {
            "harmony_prompt_token_count": len(rendered.prompt_tokens),
            "harmony_prompt_preview": rendered.prompt_text[:1000],
            "requested_max_tokens": request.max_tokens,
            "effective_max_tokens": effective_max_tokens,
            "prompt_token_estimate": len(rendered.prompt_tokens),
            "max_tokens_clipped": clipped,
            "max_tokens_clip_reason": clip_reason,
            "logit_bias_applied": bool(payload.get("logit_bias")),
        }
        return payload, render_meta

    def _extract_completion_usage(self, data: dict[str, Any]) -> dict[str, Any] | None:
        usage = data.get("usage")
        return usage if isinstance(usage, dict) else None

    @staticmethod
    def _normalize_finish_reason(choice: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            return "max_output_tokens", {"reason": "max_output_tokens"}
        return finish_reason if isinstance(finish_reason, str) else None, None

    @staticmethod
    def _completion_text(choice: dict[str, Any]) -> str:
        text = choice.get("text")
        if text is None:
            return ""
        if not isinstance(text, str):
            raise ValueError(f"Harmony completion choice text must be a string, got {type(text)!r}")
        return text

    @staticmethod
    def _completion_token_ids(choice: dict[str, Any]) -> list[int]:
        token_ids = choice.get("token_ids")
        if isinstance(token_ids, list) and all(isinstance(token, int) for token in token_ids):
            return token_ids
        if GPT_OSS_HARMONY_ALLOW_TEXT_FALLBACK:
            return []
        raise HarmonyIntegrationError(
            "Harmony completion response did not include token_ids. "
            "Enable a backend that supports return_token_ids or set GPT_OSS_HARMONY_ALLOW_TEXT_FALLBACK=1."
        )

    @staticmethod
    def _completion_token_logprobs(choice: dict[str, Any]) -> tuple[float, ...]:
        raw_logprobs = choice.get("logprobs")
        if not isinstance(raw_logprobs, dict):
            return ()
        token_logprobs = raw_logprobs.get("token_logprobs")
        if not isinstance(token_logprobs, list):
            return ()

        normalized: list[float] = []
        for value in token_logprobs:
            if isinstance(value, (int, float)):
                normalized.append(float(value))
                continue
            if isinstance(value, dict):
                logprob = value.get("logprob")
                if isinstance(logprob, (int, float)):
                    normalized.append(float(logprob))
        return tuple(normalized)

    @staticmethod
    def _empty_parsed_completion() -> dict[str, Any]:
        return {
            "visible_text": "",
            "reasoning_content": None,
            "tool_calls": (),
            "replay_items": (),
            "raw_output_item_types": (),
            "raw_output_channels": (),
            "explicit_final_channel_present": False,
            "explicit_message_channel_present": False,
            "raw_messages": (),
            "completion_status": "INCOMPLETE",
            "completion_status_source": "none",
            "terminal_token_id": None,
        }

    @staticmethod
    def _coerce_parsed_completion(parsed_completion: Any) -> dict[str, Any]:
        if isinstance(parsed_completion, dict):
            return {
                "visible_text": parsed_completion["visible_text"],
                "reasoning_content": parsed_completion["reasoning_content"],
                "tool_calls": tuple(parsed_completion["tool_calls"]),
                "replay_items": tuple(parsed_completion["replay_items"]),
                "raw_output_item_types": tuple(parsed_completion["raw_output_item_types"]),
                "raw_output_channels": tuple(parsed_completion["raw_output_channels"]),
                "explicit_final_channel_present": bool(parsed_completion["explicit_final_channel_present"]),
                "explicit_message_channel_present": bool(parsed_completion["explicit_message_channel_present"]),
                "raw_messages": tuple(parsed_completion["raw_messages"]),
                "completion_status": parsed_completion.get("completion_status"),
                "completion_status_source": parsed_completion.get("completion_status_source"),
                "terminal_token_id": parsed_completion.get("terminal_token_id"),
            }
        return {
            "visible_text": parsed_completion.visible_text,
            "reasoning_content": parsed_completion.reasoning_content,
            "tool_calls": tuple(parsed_completion.tool_calls),
            "replay_items": tuple(parsed_completion.replay_items),
            "raw_output_item_types": tuple(parsed_completion.raw_output_item_types),
            "raw_output_channels": tuple(parsed_completion.raw_output_channels),
            "explicit_final_channel_present": bool(parsed_completion.explicit_final_channel_present),
            "explicit_message_channel_present": bool(parsed_completion.explicit_message_channel_present),
            "raw_messages": tuple(parsed_completion.raw_messages),
            "completion_status": getattr(parsed_completion, "completion_status", None),
            "completion_status_source": getattr(parsed_completion, "completion_status_source", None),
            "terminal_token_id": getattr(parsed_completion, "terminal_token_id", None),
        }

    def _parse_completion_payload(self, *, completion_text: str, token_ids: list[int]) -> dict[str, Any]:
        if token_ids:
            parsed_completion = parse_harmony_completion(token_ids)
            return self._coerce_parsed_completion(parsed_completion)
        if GPT_OSS_HARMONY_ALLOW_TEXT_FALLBACK and completion_text:
            return self._coerce_parsed_completion(parse_harmony_completion_text(completion_text))
        if GPT_OSS_HARMONY_ALLOW_TEXT_FALLBACK:
            return self._empty_parsed_completion()
        raise HarmonyIntegrationError("No Harmony completion tokens available to parse.")

    def _build_response_from_completion(
        self,
        *,
        request: GenerationRequest,
        data: dict[str, Any],
        completion_text: str,
        render_meta: dict[str, Any],
        parsed_completion: dict[str, Any],
        finish_reason: str | None,
        incomplete_details: dict[str, Any] | None,
        token_logprobs: tuple[float, ...],
    ) -> dict[str, Any]:
        (
            replay_reasoning_chars,
            replay_reasoning_items,
            replay_tool_calls,
            replay_encrypted_reasoning_items,
        ) = _request_replay_metrics(request)
        raw_output_item_types = parsed_completion["raw_output_item_types"]
        raw_output_channels = parsed_completion["raw_output_channels"]
        raw_response = {
            **data,
            "raw_completion_text": completion_text,
            "harmony": {
                **render_meta,
                "parsed_messages": list(parsed_completion["raw_messages"]),
            },
        }
        visible_text = parsed_completion["visible_text"]
        return self._build_turn_result(
            raw_text=visible_text,
            reasoning_content=parsed_completion["reasoning_content"],
            reasoning_present_override=(
                any(item_type == "reasoning" for item_type in raw_output_item_types)
                or any(channel == "analysis" for channel in raw_output_channels)
            ),
            finish_reason=finish_reason,
            usage=self._extract_completion_usage(data),
            tool_calls=list(parsed_completion["tool_calls"]),
            raw_response=raw_response,
            raw_has_output_items=bool(raw_output_item_types),
            raw_has_output_text=bool(completion_text),
            raw_output_item_types=raw_output_item_types,
            raw_output_channels=raw_output_channels,
            function_call_items_count=sum(1 for item_type in raw_output_item_types if item_type == "function_call"),
            mcp_call_items_count=sum(1 for item_type in raw_output_item_types if item_type == "mcp_call"),
            incomplete_details=incomplete_details,
            truncation="max_output_tokens" if finish_reason == "max_output_tokens" else None,
            final_text_source="harmony.final"
            if parsed_completion["explicit_final_channel_present"]
            else ("harmony.message" if visible_text else None),
            adapter_type=self.backend_type,
            harmony_enabled=True,
            explicit_final_channel_present=parsed_completion["explicit_final_channel_present"],
            explicit_message_channel_present=parsed_completion["explicit_message_channel_present"],
            gpt_oss_replay_items=normalize_replay_items(parsed_completion["replay_items"]),
            requested_max_tokens=render_meta.get("requested_max_tokens"),
            effective_max_tokens=render_meta.get("effective_max_tokens"),
            prompt_token_estimate=render_meta.get("prompt_token_estimate"),
            max_tokens_clipped=bool(render_meta.get("max_tokens_clipped")),
            max_tokens_clip_reason=render_meta.get("max_tokens_clip_reason"),
            transport_type="harmony",
            continuation_round_index=data.get("request_metadata", {}).get("continuation_round_index")
            if isinstance(data.get("request_metadata"), dict)
            else None,
            replayed_reasoning_chars_value=replay_reasoning_chars,
            replayed_reasoning_items_count=replay_reasoning_items,
            replayed_tool_calls_count=replay_tool_calls,
            replayed_encrypted_reasoning_items_count=replay_encrypted_reasoning_items,
            harmony_completion_class=parsed_completion.get("completion_status"),
            harmony_completion_class_source=parsed_completion.get("completion_status_source"),
            harmony_token_ids_present=parsed_completion.get("completion_status_source") == "token_ids",
            token_logprobs=token_logprobs,
        ).as_dict()

    def generate_request(self, request: GenerationRequest) -> dict[str, Any]:
        payload, render_meta = self._payload_from_request(request)
        data = self._post_json(self.api_url, payload)
        data["request_metadata"] = dict(request.metadata)
        choice = self._extract_completion_choice(data)
        completion_text = self._completion_text(choice)
        finish_reason, incomplete_details = self._normalize_finish_reason(choice)
        token_ids = self._completion_token_ids(choice)
        token_logprobs = self._completion_token_logprobs(choice)
        parsed_completion = self._parse_completion_payload(completion_text=completion_text, token_ids=token_ids)
        return self._build_response_from_completion(
            request=request,
            data=data,
            completion_text=completion_text,
            render_meta=render_meta,
            parsed_completion=parsed_completion,
            finish_reason=finish_reason,
            incomplete_details=incomplete_details,
            token_logprobs=token_logprobs,
        )

    def stream_diagnostic_request(self, request: GenerationRequest) -> dict[str, Any]:
        requests = require_requests()
        payload, render_meta = self._payload_from_request(request)
        payload = {**payload, "stream": True}
        stream_started = requests.post(
            self.api_url,
            json=payload,
            headers=self._headers(),
            timeout=self.timeout_seconds,
            stream=True,
        )
        stream_started.raise_for_status()
        write_event = _stream_event_writer(request.metadata.get("gpt_oss_stream_events_path"))

        completion_text = ""
        usage: dict[str, Any] | None = None
        token_ids_seen: list[int] = []
        final_chunk: dict[str, Any] | None = None
        finish_reason: str | None = None
        incomplete_details: dict[str, Any] | None = None
        first_reasoning_ms: float | None = None
        first_visible_ms: float | None = None
        first_final_channel_ms: float | None = None
        previous_reasoning = ""
        previous_visible = ""
        previous_item_types: tuple[str, ...] = ()
        previous_channels: tuple[str, ...] = ()
        previous_tool_calls = 0
        events: list[dict[str, Any]] = []
        started_at = time.perf_counter()
        checkpoint_chars = 0
        harmony = None
        encoding = None
        parser = None

        for raw_line in stream_started.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_payload = line[5:].strip()
            if data_payload == "[DONE]":
                break
            chunk = json.loads(data_payload)
            final_chunk = chunk
            choice = self._extract_completion_choice(chunk)
            chunk_finish_reason, chunk_incomplete = self._normalize_finish_reason(choice)
            if chunk_finish_reason is not None:
                finish_reason = chunk_finish_reason
            if chunk_incomplete is not None:
                incomplete_details = chunk_incomplete
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]

            delta_text = self._completion_text(choice)
            if delta_text:
                if completion_text and delta_text.startswith(completion_text):
                    completion_text = delta_text
                else:
                    completion_text += delta_text

            if not completion_text:
                continue

            parsed_completion = None
            chunk_token_ids = choice.get("token_ids")
            if isinstance(chunk_token_ids, list) and all(isinstance(token, int) for token in chunk_token_ids):
                if harmony is None:
                    harmony = require_harmony()
                    encoding = harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)
                    parser = harmony.StreamableParser(encoding, harmony.Role.ASSISTANT, strict=False)
                if token_ids_seen and chunk_token_ids[: len(token_ids_seen)] == token_ids_seen:
                    new_token_ids = chunk_token_ids[len(token_ids_seen):]
                    token_ids_seen = list(chunk_token_ids)
                else:
                    new_token_ids = chunk_token_ids
                    token_ids_seen.extend(chunk_token_ids)
                for token in new_token_ids:
                    parser.process(token)
                parsed_completion = self._coerce_parsed_completion(
                    {
                        **normalize_harmony_messages(
                            parser.messages,
                            partial_channel=parser.current_channel,
                            partial_text=parser.current_content,
                        ).__dict__,
                    }
                )
            elif len(completion_text) - checkpoint_chars >= 256 or "<|" in delta_text or chunk_finish_reason is not None:
                checkpoint_chars = len(completion_text)
                try:
                    parsed_completion = self._coerce_parsed_completion(parse_harmony_completion_text(completion_text))
                except Exception as exc:
                    event = {
                        "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 3),
                        "event_type": "parse_error",
                        "error": str(exc),
                    }
                    events.append(event)
                    write_event(event)
                    continue

            if parsed_completion is None:
                continue

            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
            reasoning_text = parsed_completion["reasoning_content"] or ""
            visible_text = parsed_completion["visible_text"] or ""

            if reasoning_text and first_reasoning_ms is None:
                first_reasoning_ms = elapsed_ms
            if visible_text and first_visible_ms is None:
                first_visible_ms = elapsed_ms
            if parsed_completion["explicit_final_channel_present"] and first_final_channel_ms is None:
                first_final_channel_ms = elapsed_ms

            if reasoning_text.startswith(previous_reasoning):
                appended_reasoning = reasoning_text[len(previous_reasoning):]
            else:
                appended_reasoning = reasoning_text
            if visible_text.startswith(previous_visible):
                appended_visible = visible_text[len(previous_visible):]
            else:
                appended_visible = visible_text

            for line_text in [segment.strip() for segment in appended_reasoning.splitlines() if segment.strip()]:
                event = {
                    "elapsed_ms": elapsed_ms,
                    "event_type": "reasoning_line",
                    "text": line_text,
                }
                events.append(event)
                write_event(event)
            for line_text in [segment.strip() for segment in appended_visible.splitlines() if segment.strip()]:
                event = {
                    "elapsed_ms": elapsed_ms,
                    "event_type": "visible_line",
                    "text": line_text,
                }
                events.append(event)
                write_event(event)

            if (
                parsed_completion["raw_output_item_types"] != previous_item_types
                or parsed_completion["raw_output_channels"] != previous_channels
            ):
                event = {
                    "elapsed_ms": elapsed_ms,
                    "event_type": "output_shape",
                    "raw_output_item_types": list(parsed_completion["raw_output_item_types"]),
                    "raw_output_channels": list(parsed_completion["raw_output_channels"]),
                }
                events.append(event)
                write_event(event)
                previous_item_types = parsed_completion["raw_output_item_types"]
                previous_channels = parsed_completion["raw_output_channels"]

            current_tool_calls = len(parsed_completion["tool_calls"])
            if current_tool_calls > previous_tool_calls:
                for tool_call in parsed_completion["tool_calls"][previous_tool_calls:]:
                    event = {
                        "elapsed_ms": elapsed_ms,
                        "event_type": "tool_call",
                        "tool_call": tool_call,
                    }
                    events.append(event)
                    write_event(event)
                previous_tool_calls = current_tool_calls

            previous_reasoning = reasoning_text
            previous_visible = visible_text

        total_elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
        if final_chunk is None:
            raise RuntimeError("stream_diagnostic_failed: no streamed chunks received")

        final_choice = self._extract_completion_choice(final_chunk)
        if finish_reason is None:
            finish_reason, incomplete_details = self._normalize_finish_reason(final_choice)
        if parser is not None:
            parser.process_eos()
            parsed_completion = self._coerce_parsed_completion(
                {
                    **normalize_harmony_messages(
                        parser.messages,
                        partial_channel=parser.current_channel,
                        partial_text=parser.current_content,
                    ).__dict__,
                }
            )
        else:
            parsed_completion = self._coerce_parsed_completion(parse_harmony_completion_text(completion_text))
        final_data = {**final_chunk}
        final_data["request_metadata"] = dict(request.metadata)
        if usage is not None and "usage" not in final_data:
            final_data["usage"] = usage
        model_response = self._build_response_from_completion(
            request=request,
            data=final_data,
            completion_text=completion_text,
            render_meta=render_meta,
            parsed_completion=parsed_completion,
            finish_reason=finish_reason,
            incomplete_details=incomplete_details,
            token_logprobs=self._completion_token_logprobs(final_choice),
        )
        event = {
            "elapsed_ms": total_elapsed_ms,
            "event_type": "completion",
            "finish_reason": finish_reason,
            "incomplete_details": incomplete_details,
        }
        events.append(event)
        write_event(event)
        return {
            "model_response": model_response,
            "events": events,
            "streaming_enabled": True,
            "first_reasoning_ms": first_reasoning_ms,
            "first_visible_ms": first_visible_ms,
            "first_final_channel_ms": first_final_channel_ms,
            "total_elapsed_ms": total_elapsed_ms,
            "completion_text": completion_text,
        }

    def generate_structured_from_request(self, request: GenerationRequest) -> dict[str, Any]:
        constrained_request = GenerationRequest(
            messages=request.messages,
            max_tokens=min(request.max_tokens, 64),
            temperature=0.0,
            top_p=1.0,
            stop_sequences=request.stop_sequences,
            reasoning_effort=request.reasoning_effort,
            tools=(),
            tool_choice=None,
            metadata={**request.metadata, "purpose": "structured_finalization"},
        )

        def _structured_payload(guided_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            payload, render_meta = self._payload_from_request(constrained_request)
            payload = dict(payload)
            # Continue directly inside the visible final message content so guided decoding
            # constrains only the answer text and not Harmony control tokens.
            payload["prompt"] = f"{payload['prompt']}<|channel|>final<|message|>"
            payload["max_tokens"] = min(constrained_request.max_tokens, 64)
            payload["temperature"] = 0.0
            payload["top_p"] = 1.0
            payload["stop_token_ids"] = [HARMONY_RETURN_TOKEN_ID]
            payload.update(guided_payload)
            return payload, {
                **render_meta,
                "guided_decoding_requested": True,
                "guided_decoding_payload": sorted(guided_payload.keys()),
                "guided_final_prefix": "<|channel|>final<|message|>",
            }

        guided_regex_override = None
        if isinstance(request.metadata, dict):
            override = request.metadata.get("guided_regex_override")
            if isinstance(override, str) and override.strip():
                guided_regex_override = override.strip()
        if guided_regex_override:
            structured_attempts = [("guided_regex", {"guided_regex": guided_regex_override})]
        else:
            structured_attempts = [
                ("guided_regex", {"guided_regex": r"FINAL_ANSWER:\s*-?\d+"}),
                (
                    "guided_json",
                    {
                        "guided_json": {
                            "type": "object",
                            "properties": {"final_answer": {"type": "string", "pattern": r"^-?\d+$"}},
                            "required": ["final_answer"],
                            "additionalProperties": False,
                        }
                    },
                ),
            ]
        errors: list[str] = []
        for strategy, guided_payload in structured_attempts:
            try:
                payload, render_meta = _structured_payload(guided_payload)
                data = self._post_json(self.api_url, payload)
                data["request_metadata"] = dict(constrained_request.metadata)
                choice = self._extract_completion_choice(data)
                finish_reason, incomplete_details = self._normalize_finish_reason(choice)
                raw_content = self._completion_text(choice).replace("<|return|>", "").strip()
                content = _normalize_structured_output(strategy, raw_content, guided_payload)
                if content is None:
                    errors.append(f"{strategy}: structured_output_validation_failed")
                    continue
                return {
                    "output_text": content,
                    "raw_output_text": content,
                    "finish_reason": finish_reason,
                    "usage": self._extract_completion_usage(data),
                    "usage_prompt_tokens": (self._extract_completion_usage(data) or {}).get("prompt_tokens"),
                    "usage_completion_tokens": (self._extract_completion_usage(data) or {}).get("completion_tokens"),
                    "usage_total_tokens": (self._extract_completion_usage(data) or {}).get("total_tokens"),
                    "reasoning_present": False,
                    "reasoning_content": None,
                    "tool_calls": [],
                    "raw_response": {
                        **data,
                        "harmony": render_meta,
                    },
                    "raw_has_output_items": bool(content),
                    "raw_has_output_text": bool(content),
                    "raw_output_item_types": ["assistant:final"],
                    "raw_output_channels": ["final"],
                    "function_call_items_count": 0,
                    "mcp_call_items_count": 0,
                    "incomplete_details": incomplete_details,
                    "truncation": "max_output_tokens" if finish_reason == "max_output_tokens" else None,
                    "final_text_source": "harmony.guided_extraction",
                    "adapter_type": self.backend_type,
                    "harmony_enabled": True,
                    "explicit_final_channel_present": True,
                    "explicit_message_channel_present": True,
                    "gpt_oss_replay_items": [],
                    "requested_max_tokens": render_meta.get("requested_max_tokens"),
                    "effective_max_tokens": min(constrained_request.max_tokens, 64),
                    "prompt_token_estimate": render_meta.get("prompt_token_estimate"),
                    "max_tokens_clipped": bool(render_meta.get("max_tokens_clipped")),
                    "max_tokens_clip_reason": render_meta.get("max_tokens_clip_reason"),
                    "transport_type": "harmony",
                    "continuation_round_index": constrained_request.metadata.get("continuation_round_index"),
                    "replayed_reasoning_chars": 0,
                    "replayed_reasoning_items_count": 0,
                    "replayed_tool_calls_count": 0,
                    "finalization_status": None,
                    "finalization_failure_reason": None,
                    "harmony_completion_class": "COMPLETE",
                    "harmony_completion_class_source": strategy,
                    "harmony_token_ids_present": False,
                    "final_text_present": bool(content),
                    "final_text_chars": len(content),
                    "tool_calls_count": 0,
                    "strategy": strategy,
                    "guided_decoding_enforced": True,
                    "guided_decoding_downgraded": False,
                }
            except Exception as exc:
                errors.append(f"{strategy}: {exc}")

        response = self.generate_request(constrained_request)
        response["strategy"] = "harmony_completion"
        response["guided_decoding_enforced"] = False
        response["guided_decoding_downgraded"] = True
        response["guided_decoding_errors"] = errors
        return response

    def generate_with_metadata(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        request = _legacy_request(
            system_prompt,
            user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return self.generate_request(request)


class VllmModel(OpenAICompatibleModel):
    """Alias for OpenAI-compatible chat completion backends such as vLLM."""


class ForcedToolFirstModel(OpenAICompatibleModel):
    def __init__(
        self,
        base_url: str = TOOL_CALL_OPENAI_BASE_URL,
        api_key: str = TOOL_CALL_OPENAI_API_KEY,
        model_name: str = TOOL_CALL_OPENAI_MODEL_NAME,
        timeout_seconds: int = TOOL_CALL_REQUEST_TIMEOUT_SECONDS,
        max_tokens: int = TOOL_CALL_MAX_TOKENS,
        temperature: float = TOOL_CALL_TEMPERATURE,
        top_p: float = TOOL_CALL_TOP_P,
        followup_temperature: float = TOOL_CALL_FOLLOWUP_TEMPERATURE,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.followup_temperature = followup_temperature

    @staticmethod
    def python_exec_tool_schema() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "python_exec",
                    "description": "Execute exact Python code for search, verification, enumeration, or symbolic computation on a math problem.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Pure Python code to execute. Print the key output needed to finish the problem.",
                            },
                            "intent": {
                                "type": "string",
                                "description": "One short sentence describing what the code is checking or computing.",
                            },
                        },
                        "required": ["code", "intent"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def generate_required_tool_call(self, user_prompt: str) -> dict[str, Any]:
        request = GenerationRequest(
            messages=({"role": "user", "content": user_prompt},),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            tools=tuple(self.python_exec_tool_schema()),
            tool_choice="required",
        )
        return self.generate_request(request)

    def generate_final_answer_after_tool(self, user_prompt: str) -> dict[str, Any]:
        request = GenerationRequest(
            messages=({"role": "user", "content": user_prompt},),
            max_tokens=min(self.max_tokens, 128),
            temperature=self.followup_temperature,
            top_p=0.9,
        )
        return self.generate_request(request)


def long_problem_force_tool_first_enabled() -> bool:
    return LONG_PROBLEM_FORCE_TOOL_FIRST

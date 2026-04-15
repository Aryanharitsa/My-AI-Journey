from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from gpt_oss_replay import make_replay_item, normalize_replay_items

try:
    import openai_harmony as _openai_harmony
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dependency
    _openai_harmony = None
    _HARMONY_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised indirectly in integration tests
    _HARMONY_IMPORT_ERROR = None


class HarmonyIntegrationError(RuntimeError):
    """Raised when the optional openai-harmony dependency is unavailable or misused."""


HARMONY_RETURN_TOKEN_ID = 200002
HARMONY_END_MESSAGE_TOKEN_ID = 200007
HARMONY_CALL_TOKEN_ID = 200012


@dataclass(frozen=True)
class HarmonyRenderedPrompt:
    prompt_text: str
    prompt_tokens: tuple[int, ...]
    stop_token_ids: tuple[int, ...]
    conversation: Any


@dataclass(frozen=True)
class HarmonyParsedCompletion:
    visible_text: str
    reasoning_content: str | None
    tool_calls: tuple[dict[str, Any], ...]
    replay_items: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    raw_output_item_types: tuple[str, ...] = field(default_factory=tuple)
    raw_output_channels: tuple[str, ...] = field(default_factory=tuple)
    explicit_final_channel_present: bool = False
    explicit_message_channel_present: bool = False
    raw_messages: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    partial_channel: str | None = None
    partial_text: str | None = None
    completion_status: str | None = None
    completion_status_source: str | None = None
    terminal_token_id: int | None = None


def harmony_available() -> bool:
    return _openai_harmony is not None


def require_harmony():
    if _openai_harmony is None:
        raise HarmonyIntegrationError(
            "openai-harmony is not installed. Install it with `pip install openai-harmony`."
        ) from _HARMONY_IMPORT_ERROR
    return _openai_harmony


def _map_reasoning_effort(harmony, reasoning_effort: str | None):
    effort = (reasoning_effort or "").strip().lower()
    if not effort:
        return None
    effort_enum = getattr(harmony, "ReasoningEffort", None)
    if effort_enum is None:
        return None
    mapping = {
        "low": getattr(effort_enum, "LOW", None),
        "medium": getattr(effort_enum, "MEDIUM", None),
        "high": getattr(effort_enum, "HIGH", None),
    }
    return mapping.get(effort)


def _tool_descriptions_from_request(harmony, tools: tuple[dict[str, Any], ...]) -> list[Any]:
    descriptions = []
    for tool in tools:
        name = tool.get("name") or (tool.get("function") or {}).get("name")
        description = tool.get("description") or (tool.get("function") or {}).get("description") or ""
        parameters = tool.get("parameters") or (tool.get("function") or {}).get("parameters")
        if isinstance(name, str) and name.strip():
            descriptions.append(
                harmony.ToolDescription.new(
                    name=name,
                    description=description,
                    parameters=parameters if isinstance(parameters, dict) else None,
                )
            )
    return descriptions


def _message_text(message: Any) -> str:
    parts: list[str] = []
    for content in getattr(message, "content", []) or []:
        text = getattr(content, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
            continue
        if hasattr(content, "to_dict"):
            data = content.to_dict()
            if isinstance(data, dict) and isinstance(data.get("text"), str):
                parts.append(data["text"])
    return "\n".join(part for part in parts if part)


def _tool_call_from_message(message: Any) -> dict[str, Any] | None:
    recipient = getattr(message, "recipient", None)
    if not isinstance(recipient, str) or not recipient.strip():
        return None

    arguments_raw = _message_text(message)
    if recipient.startswith("functions."):
        tool_name = recipient.split(".", 1)[1]
        try:
            arguments = json.loads(arguments_raw) if arguments_raw.strip() else None
        except json.JSONDecodeError:
            arguments = None
        if arguments is not None and not isinstance(arguments, dict):
            arguments = None
        return {
            "id": None,
            "type": "function_call",
            "name": tool_name,
            "arguments_raw": arguments_raw,
            "arguments": arguments,
            "call_id": None,
            "recipient": recipient,
        }

    if "." in recipient:
        server_label, tool_name = recipient.split(".", 1)
        normalized_name = "python_exec" if recipient == "python.exec" else tool_name
        arguments = None
        if arguments_raw.strip():
            if recipient == "python.exec":
                arguments = {
                    "code": arguments_raw,
                    "intent": recipient,
                }
        return {
            "id": None,
            "type": "mcp_call",
            "name": normalized_name,
            "arguments_raw": arguments_raw,
            "arguments": arguments,
            "call_id": None,
            "recipient": recipient,
            "server_label": server_label,
        }
    return None


def render_harmony_prompt(
    *,
    messages: tuple[dict[str, Any], ...],
    tools: tuple[dict[str, Any], ...] = (),
    reasoning_effort: str | None = None,
    enable_python_mcp: bool = False,
) -> HarmonyRenderedPrompt:
    harmony = require_harmony()
    encoding = harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)

    system_content = harmony.SystemContent.new().with_required_channels(["analysis", "commentary", "final"])
    mapped_reasoning = _map_reasoning_effort(harmony, reasoning_effort)
    if mapped_reasoning is not None:
        system_content = system_content.with_reasoning_effort(mapped_reasoning)
    if enable_python_mcp:
        system_content = system_content.with_python_tool()

    tool_descriptions = _tool_descriptions_from_request(harmony, tools)
    conversation_messages = [
        harmony.Message.from_role_and_content(harmony.Role.SYSTEM, system_content),
    ]
    attached_function_tools = False

    for raw_message in messages:
        role = str(raw_message.get("role", "user")).lower()
        content = str(raw_message.get("content", "") or "")

        if role in {"system", "developer"}:
            developer_content = harmony.DeveloperContent.new().with_instructions(content)
            if tool_descriptions and not attached_function_tools:
                developer_content = developer_content.with_function_tools(tool_descriptions)
                attached_function_tools = True
            conversation_messages.append(
                harmony.Message.from_role_and_content(harmony.Role.DEVELOPER, developer_content)
            )
            continue

        if role == "assistant":
            message = harmony.Message.from_role_and_content(harmony.Role.ASSISTANT, content)
            if isinstance(raw_message.get("channel"), str):
                message = message.with_channel(raw_message["channel"])
            if isinstance(raw_message.get("recipient"), str):
                message = message.with_recipient(raw_message["recipient"])
            if isinstance(raw_message.get("content_type"), str):
                message = message.with_content_type(raw_message["content_type"])
            conversation_messages.append(message)
            continue

        if role == "tool":
            author_name = str(raw_message.get("name") or raw_message.get("author_name") or "tool")
            message = harmony.Message.from_author_and_content(
                harmony.Author.new(harmony.Role.TOOL, author_name),
                content,
            )
            if isinstance(raw_message.get("channel"), str):
                message = message.with_channel(raw_message["channel"])
            conversation_messages.append(message)
            continue

        conversation_messages.append(harmony.Message.from_role_and_content(harmony.Role.USER, content))

    if tool_descriptions and not attached_function_tools:
        developer_content = harmony.DeveloperContent.new().with_instructions("Use the declared tools when needed.")
        developer_content = developer_content.with_function_tools(tool_descriptions)
        conversation_messages.insert(
            1,
            harmony.Message.from_role_and_content(harmony.Role.DEVELOPER, developer_content),
        )

    conversation = harmony.Conversation.from_messages(conversation_messages)
    prompt_tokens = tuple(encoding.render_conversation_for_completion(conversation, harmony.Role.ASSISTANT))
    prompt_text = encoding.decode(prompt_tokens)
    return HarmonyRenderedPrompt(
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
        stop_token_ids=tuple(encoding.stop_tokens()),
        conversation=conversation,
    )


def normalize_harmony_messages(
    messages: list[Any],
    *,
    partial_channel: str | None = None,
    partial_text: str | None = None,
) -> HarmonyParsedCompletion:
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    replay_items: list[dict[str, Any]] = []
    raw_output_item_types: list[str] = []
    raw_output_channels: list[str] = []
    explicit_final_channel_present = False
    explicit_message_channel_present = False
    raw_messages: list[dict[str, Any]] = []

    for message in messages:
        if hasattr(message, "to_dict"):
            raw_messages.append(message.to_dict())
        role = getattr(getattr(message, "author", None), "role", None)
        role_value = str(getattr(role, "value", role)) if role is not None else "assistant"
        channel = getattr(message, "channel", None)
        if isinstance(channel, str) and channel:
            raw_output_channels.append(channel)

        tool_call = _tool_call_from_message(message)
        if tool_call is not None:
            tool_calls.append(tool_call)
            raw_output_item_types.append(tool_call["type"])
            replay_items.append(
                make_replay_item(
                    role="assistant",
                    content=tool_call["arguments_raw"],
                    channel=channel if isinstance(channel, str) and channel else "commentary",
                    recipient=tool_call.get("recipient"),
                    content_type=getattr(message, "content_type", None),
                    name=tool_call.get("name"),
                    item_type=tool_call.get("type"),
                    call_id=tool_call.get("call_id"),
                    arguments_raw=tool_call.get("arguments_raw"),
                    arguments=tool_call.get("arguments"),
                )
            )
            continue

        text = _message_text(message)
        if role_value == "assistant":
            raw_output_item_types.append(f"assistant:{channel or 'default'}")
            if channel == "analysis":
                if text:
                    reasoning_parts.append(text)
            if channel == "final":
                explicit_final_channel_present = True
            if text:
                replay_items.append(
                    make_replay_item(
                        role="assistant",
                        content=text,
                        channel=channel if isinstance(channel, str) else None,
                    )
                )
            if channel == "analysis":
                continue
            if channel == "final":
                explicit_final_channel_present = True
            if text:
                visible_parts.append(text)
                explicit_message_channel_present = True
            continue

        if role_value == "tool":
            raw_output_item_types.append("tool")
            if text:
                replay_items.append(
                    make_replay_item(
                        role="tool",
                        name=getattr(getattr(message, "author", None), "name", None),
                        channel=channel if isinstance(channel, str) else None,
                        content=text,
                    )
                )
            continue

        raw_output_item_types.append(f"{role_value}:{channel or 'default'}")

    if isinstance(partial_channel, str) and partial_channel:
        raw_output_channels.append(partial_channel)
        raw_output_item_types.append(f"assistant_partial:{partial_channel}")
        if partial_channel == "analysis" and partial_text:
            reasoning_parts.append(partial_text)
        elif partial_text:
            visible_parts.append(partial_text)
            explicit_message_channel_present = True
            if partial_channel == "final":
                explicit_final_channel_present = True
        if partial_text:
            replay_items.append(
                make_replay_item(
                    role="assistant",
                    content=partial_text,
                    channel=partial_channel,
                    partial=True,
                )
            )

    return HarmonyParsedCompletion(
        visible_text="\n".join(part for part in visible_parts if part),
        reasoning_content="\n".join(part for part in reasoning_parts if part) or None,
        tool_calls=tuple(tool_calls),
        replay_items=normalize_replay_items(replay_items),
        raw_output_item_types=tuple(raw_output_item_types),
        raw_output_channels=tuple(raw_output_channels),
        explicit_final_channel_present=explicit_final_channel_present,
        explicit_message_channel_present=explicit_message_channel_present,
        raw_messages=tuple(raw_messages),
        partial_channel=partial_channel,
        partial_text=partial_text,
    )


def classify_harmony_completion_tokens(token_ids: list[int] | tuple[int, ...]) -> tuple[str, int | None]:
    if not token_ids:
        return "INCOMPLETE", None
    terminal_token_id = token_ids[-1]
    if terminal_token_id == HARMONY_RETURN_TOKEN_ID:
        return "COMPLETE", terminal_token_id
    if terminal_token_id == HARMONY_CALL_TOKEN_ID:
        return "TOOL_CALL", terminal_token_id
    return "INCOMPLETE", terminal_token_id


def parse_harmony_completion(token_ids: list[int]) -> HarmonyParsedCompletion:
    harmony = require_harmony()
    encoding = harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)
    parser = harmony.StreamableParser(encoding, harmony.Role.ASSISTANT, strict=False)
    for token in token_ids:
        parser.process(token)
    parser.process_eos()
    normalized = normalize_harmony_messages(
        parser.messages,
        partial_channel=parser.current_channel,
        partial_text=parser.current_content,
    )
    completion_status, terminal_token_id = classify_harmony_completion_tokens(token_ids)
    payload = dict(normalized.__dict__)
    payload.update(
        {
            "completion_status": completion_status,
            "completion_status_source": "token_ids",
            "terminal_token_id": terminal_token_id,
        }
    )
    return HarmonyParsedCompletion(**payload)


def parse_harmony_completion_text(completion_text: str) -> HarmonyParsedCompletion:
    harmony = require_harmony()
    encoding = harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)
    token_ids = encoding.encode(completion_text or "", allowed_special="all")
    normalized = parse_harmony_completion(token_ids)
    payload = dict(normalized.__dict__)
    payload["completion_status_source"] = "text_fallback"
    return HarmonyParsedCompletion(**payload)

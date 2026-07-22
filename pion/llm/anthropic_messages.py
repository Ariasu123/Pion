"""Anthropic Messages API streaming.

Python port of pi's `packages/ai/src/api/anthropic-messages.ts` (simplified
subset).

Contract (same as pi): `stream_simple` never raises for request/model/runtime
failures; failures are encoded as an `error` event carrying the final
AssistantMessage with stop_reason "error"/"aborted".
"""

from __future__ import annotations

import inspect
import json
from typing import Any, AsyncIterator, Optional

import httpx

from .event_stream import AssistantMessageEvent, AssistantMessageEventStream, StreamOptions
from .openai_completions import parse_partial_json
from .types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    StopReason,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

ANTHROPIC_VERSION = "2023-06-01"


class _Aborted(Exception):
    """Internal signal: the request was aborted via options.abort."""


class _HttpStatusError(Exception):
    """Non-2xx HTTP response, carrying a short body snippet."""

    def __init__(self, status_code: int, body: str):
        snippet = " ".join(body.split())[:300]
        super().__init__(f"HTTP {status_code}: {snippet}")
        self.status_code = status_code


def stream_simple(
    model: Model, context: Context, options: StreamOptions
) -> AssistantMessageEventStream:
    """Stream an assistant reply from the Anthropic Messages API."""
    return AssistantMessageEventStream(_run(model, context, options))


async def _run(
    model: Model, context: Context, options: StreamOptions
) -> AsyncIterator[AssistantMessageEvent]:
    output = AssistantMessage(api=model.api, provider=model.provider, model=model.id)
    try:
        payload = _build_payload(model, context, options)
        if options.on_payload is not None:
            replacement = options.on_payload(payload, model)
            if inspect.isawaitable(replacement):
                replacement = await replacement
            if replacement is not None:
                payload = replacement

        url = f"{model.base_url.rstrip('/')}/v1/messages"
        headers = {
            "x-api-key": options.api_key or "",
            "anthropic-version": ANTHROPIC_VERSION,
            **model.headers,
        }

        # Streaming scratch state. `positions` maps the Anthropic block index
        # to the position in output.content; `partial_json` accumulates the
        # tool_use input JSON per content position.
        positions: dict[int, int] = {}
        partial_json: dict[int, list[str]] = {}
        saw_message_start = False
        saw_message_stop = False

        timeout = httpx.Timeout(options.timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise _HttpStatusError(response.status_code, body)

                yield AssistantMessageEvent(type="start", partial=output)

                async for event_name, data in _iter_sse(response, options):
                    if event_name == "message_start":
                        saw_message_start = True
                        message = data.get("message") or {}
                        if message.get("id"):
                            output.response_id = message["id"]
                        _apply_usage(output, message.get("usage") or {}, model)

                    elif event_name == "content_block_start":
                        index = data.get("index", 0)
                        block = data.get("content_block") or {}
                        block_type = block.get("type")
                        position: Optional[int] = None
                        if block_type == "text":
                            output.content.append(TextContent(text=""))
                            position = len(output.content) - 1
                            yield AssistantMessageEvent(
                                type="text_start", content_index=position, partial=output
                            )
                        elif block_type in ("thinking", "redacted_thinking"):
                            output.content.append(
                                ThinkingContent(
                                    thinking="[Reasoning redacted]"
                                    if block_type == "redacted_thinking"
                                    else "",
                                    thinkingSignature=block.get("data")
                                    if block_type == "redacted_thinking"
                                    else None,
                                    redacted=True if block_type == "redacted_thinking" else None,
                                )
                            )
                            position = len(output.content) - 1
                            yield AssistantMessageEvent(
                                type="thinking_start", content_index=position, partial=output
                            )
                        elif block_type == "tool_use":
                            output.content.append(
                                ToolCall(
                                    id=block.get("id") or "",
                                    name=block.get("name") or "",
                                    arguments=block.get("input") or {},
                                )
                            )
                            position = len(output.content) - 1
                            partial_json[position] = []
                            yield AssistantMessageEvent(
                                type="toolcall_start", content_index=position, partial=output
                            )
                        if position is not None:
                            positions[index] = position

                    elif event_name == "content_block_delta":
                        position = positions.get(data.get("index", -1))
                        if position is None:
                            continue
                        delta = data.get("delta") or {}
                        delta_type = delta.get("type")
                        block = output.content[position]
                        if delta_type == "text_delta" and isinstance(block, TextContent):
                            block.text += delta.get("text", "")
                            yield AssistantMessageEvent(
                                type="text_delta",
                                content_index=position,
                                delta=delta.get("text", ""),
                                partial=output,
                            )
                        elif delta_type == "thinking_delta" and isinstance(block, ThinkingContent):
                            block.thinking += delta.get("thinking", "")
                            yield AssistantMessageEvent(
                                type="thinking_delta",
                                content_index=position,
                                delta=delta.get("thinking", ""),
                                partial=output,
                            )
                        elif delta_type == "signature_delta" and isinstance(block, ThinkingContent):
                            block.thinking_signature = (block.thinking_signature or "") + delta.get(
                                "signature", ""
                            )
                        elif delta_type == "input_json_delta" and isinstance(block, ToolCall):
                            text = delta.get("partial_json", "")
                            partial_json[position].append(text)
                            yield AssistantMessageEvent(
                                type="toolcall_delta",
                                content_index=position,
                                delta=text,
                                partial=output,
                            )

                    elif event_name == "content_block_stop":
                        index = data.get("index", -1)
                        position = positions.pop(index, None)
                        if position is None:
                            continue
                        block = output.content[position]
                        if isinstance(block, TextContent):
                            yield AssistantMessageEvent(
                                type="text_end",
                                content_index=position,
                                content=block.text,
                                partial=output,
                            )
                        elif isinstance(block, ThinkingContent):
                            yield AssistantMessageEvent(
                                type="thinking_end",
                                content_index=position,
                                content=block.thinking,
                                partial=output,
                            )
                        elif isinstance(block, ToolCall):
                            block.arguments = parse_partial_json(
                                "".join(partial_json.get(position, []))
                            )
                            yield AssistantMessageEvent(
                                type="toolcall_end",
                                content_index=position,
                                tool_call=block,
                                partial=output,
                            )

                    elif event_name == "message_delta":
                        delta = data.get("delta") or {}
                        stop_reason = delta.get("stop_reason")
                        if stop_reason:
                            reason, error_message = _map_stop_reason(stop_reason)
                            output.stop_reason = reason
                            if error_message:
                                output.error_message = error_message
                        _apply_usage(output, data.get("usage") or {}, model)

                    elif event_name == "message_stop":
                        saw_message_stop = True

                    elif event_name == "error":
                        raise RuntimeError(f"Anthropic SSE error: {json.dumps(data)[:300]}")

        _raise_if_aborted(options)
        if output.stop_reason in ("error", "aborted"):
            raise RuntimeError(output.error_message or "An unknown error occurred")
        if saw_message_start and not saw_message_stop:
            raise RuntimeError("Anthropic stream ended before message_stop")

        yield AssistantMessageEvent(type="done", reason=output.stop_reason, message=output)

    except _Aborted:
        output.stop_reason = "aborted"
        output.error_message = "Request was aborted"
        yield AssistantMessageEvent(type="error", reason="aborted", error=output)
    except Exception as exc:  # noqa: BLE001 — contract: never raise
        if options.abort is not None and options.abort.is_set():
            output.stop_reason = "aborted"
            output.error_message = "Request was aborted"
        else:
            output.stop_reason = "error"
            output.error_message = str(exc) or repr(exc)
        yield AssistantMessageEvent(type="error", reason=output.stop_reason, error=output)


async def _iter_sse(
    response: httpx.Response, options: StreamOptions
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Parse an Anthropic SSE byte stream into (event, data) pairs."""
    event_name: Optional[str] = None
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        _raise_if_aborted(options)
        if line == "":
            if event_name and data_lines:
                yield event_name, json.loads("\n".join(data_lines))
            event_name, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    # Flush a trailing event if the stream ended without a blank line.
    if event_name and data_lines:
        yield event_name, json.loads("\n".join(data_lines))


def _raise_if_aborted(options: StreamOptions) -> None:
    if options.abort is not None and options.abort.is_set():
        raise _Aborted()


def _build_payload(model: Model, context: Context, options: StreamOptions) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.id,
        "messages": _convert_messages(context),
        "max_tokens": options.max_tokens if options.max_tokens is not None else model.max_tokens,
        "stream": True,
    }
    if context.system_prompt:
        payload["system"] = context.system_prompt
    if options.temperature is not None:
        payload["temperature"] = options.temperature
    if context.tools:
        payload["tools"] = [_convert_tool(tool) for tool in context.tools]
    return payload


def _convert_messages(context: Context) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    source = context.messages
    i = 0
    while i < len(source):
        msg = source[i]

        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                if msg.content.strip():
                    messages.append({"role": "user", "content": msg.content})
            else:
                blocks: list[dict[str, Any]] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        if part.text.strip():
                            blocks.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": part.mime_type,
                                    "data": part.data,
                                },
                            }
                        )
                if blocks:
                    messages.append({"role": "user", "content": blocks})

        elif isinstance(msg, AssistantMessage):
            blocks = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    if block.text.strip():
                        blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ThinkingContent):
                    if block.redacted and block.thinking_signature:
                        blocks.append(
                            {"type": "redacted_thinking", "data": block.thinking_signature}
                        )
                    elif block.thinking_signature:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": block.thinking,
                                "signature": block.thinking_signature,
                            }
                        )
                    elif block.thinking.strip():
                        # Signatureless thinking (e.g. aborted stream) goes back as text.
                        blocks.append({"type": "text", "text": block.thinking})
                elif isinstance(block, ToolCall):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.arguments or {},
                        }
                    )
            if blocks:
                messages.append({"role": "assistant", "content": blocks})

        elif isinstance(msg, ToolResultMessage):
            # Group consecutive tool results into one user message.
            blocks = []
            while i < len(source) and isinstance(source[i], ToolResultMessage):
                result = source[i]
                assert isinstance(result, ToolResultMessage)
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.text() or "(no tool output)",
                        "is_error": result.is_error,
                    }
                )
                i += 1
            messages.append({"role": "user", "content": blocks})
            continue

        i += 1

    return messages


def _convert_tool(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def _apply_usage(output: AssistantMessage, raw: dict[str, Any], model: Model) -> None:
    """Update usage fields present in `raw` and recompute totals/cost.

    Anthropic reports input tokens in `message_start` and output tokens in the
    final `message_delta`; fields absent from `raw` keep their current values.
    """
    if raw.get("input_tokens") is not None:
        output.usage.input = raw["input_tokens"]
    if raw.get("output_tokens") is not None:
        output.usage.output = raw["output_tokens"]
    if raw.get("cache_read_input_tokens") is not None:
        output.usage.cache_read = raw["cache_read_input_tokens"]
    if raw.get("cache_creation_input_tokens") is not None:
        output.usage.cache_write = raw["cache_creation_input_tokens"]
    # Anthropic doesn't provide total_tokens; compute from components.
    output.usage.total_tokens = (
        output.usage.input
        + output.usage.output
        + output.usage.cache_read
        + output.usage.cache_write
    )
    output.usage.cost = model.compute_cost(output.usage)


def _map_stop_reason(reason: str) -> tuple[StopReason, Optional[str]]:
    if reason == "end_turn":
        return "stop", None
    if reason == "max_tokens":
        return "length", None
    if reason == "tool_use":
        return "toolUse", None
    if reason == "refusal":
        return "error", "The model refused to complete the request"
    if reason in ("pause_turn", "stop_sequence"):
        return "stop", None
    return "error", f"Unhandled stop reason: {reason}"

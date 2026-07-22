"""OpenAI-compatible chat completions streaming.

Python port of pi's `packages/ai/src/api/openai-completions.ts` (simplified
subset). Covers the OpenAI Chat Completions API shape, which also serves
DeepSeek, Kimi/Moonshot, Qwen, Zhipu and any OpenAI-compatible endpoint.

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
from .types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    StopReason,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


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
    """Stream an assistant reply from an OpenAI-compatible chat endpoint."""
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

        url = f"{model.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {options.api_key or ''}",
            **model.headers,
        }

        # Streaming scratch state, keyed by position in output.content.
        partial_args: dict[int, list[str]] = {}
        tool_positions: dict[int, int] = {}  # delta "index" -> content position
        text_position: Optional[int] = None
        has_finish_reason = False

        timeout = httpx.Timeout(options.timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise _HttpStatusError(response.status_code, body)

                yield AssistantMessageEvent(type="start", partial=output)

                async for line in response.aiter_lines():
                    _raise_if_aborted(options)
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)

                    if chunk.get("id") and not output.response_id:
                        output.response_id = chunk["id"]
                    chunk_model = chunk.get("model")
                    if (
                        isinstance(chunk_model, str)
                        and chunk_model
                        and chunk_model != model.id
                        and not output.response_model
                    ):
                        output.response_model = chunk_model
                    if chunk.get("usage"):
                        output.usage = _parse_usage(chunk["usage"], model)

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]

                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        reason, error_message = _map_finish_reason(finish_reason)
                        output.stop_reason = reason
                        if error_message:
                            output.error_message = error_message
                        has_finish_reason = True

                    delta = choice.get("delta") or {}

                    text_delta = delta.get("content")
                    if text_delta:
                        if text_position is None:
                            output.content.append(TextContent(text=""))
                            text_position = len(output.content) - 1
                            yield AssistantMessageEvent(
                                type="text_start",
                                content_index=text_position,
                                partial=output,
                            )
                        block = output.content[text_position]
                        assert isinstance(block, TextContent)
                        block.text += text_delta
                        yield AssistantMessageEvent(
                            type="text_delta",
                            content_index=text_position,
                            delta=text_delta,
                            partial=output,
                        )

                    for tool_delta in delta.get("tool_calls") or []:
                        index = tool_delta.get("index", 0)
                        position = tool_positions.get(index)
                        if position is None:
                            function = tool_delta.get("function") or {}
                            output.content.append(
                                ToolCall(
                                    id=tool_delta.get("id") or "",
                                    name=function.get("name") or "",
                                )
                            )
                            position = len(output.content) - 1
                            tool_positions[index] = position
                            partial_args[position] = []
                            yield AssistantMessageEvent(
                                type="toolcall_start",
                                content_index=position,
                                partial=output,
                            )
                        block = output.content[position]
                        assert isinstance(block, ToolCall)
                        if tool_delta.get("id") and not block.id:
                            block.id = tool_delta["id"]
                        function = tool_delta.get("function") or {}
                        if function.get("name") and not block.name:
                            block.name = function["name"]
                        args_delta = function.get("arguments") or ""
                        if args_delta:
                            partial_args[position].append(args_delta)
                        yield AssistantMessageEvent(
                            type="toolcall_delta",
                            content_index=position,
                            delta=args_delta,
                            partial=output,
                        )

        # Finalize all open blocks in order.
        for position, block in enumerate(output.content):
            if isinstance(block, TextContent):
                yield AssistantMessageEvent(
                    type="text_end",
                    content_index=position,
                    content=block.text,
                    partial=output,
                )
            elif isinstance(block, ToolCall):
                block.arguments = parse_partial_json("".join(partial_args.get(position, [])))
                yield AssistantMessageEvent(
                    type="toolcall_end",
                    content_index=position,
                    tool_call=block,
                    partial=output,
                )

        _raise_if_aborted(options)
        if output.stop_reason in ("error", "aborted"):
            raise RuntimeError(output.error_message or "Provider returned an error stop reason")
        if not has_finish_reason:
            raise RuntimeError("Stream ended without finish_reason")

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


def _raise_if_aborted(options: StreamOptions) -> None:
    if options.abort is not None and options.abort.is_set():
        raise _Aborted()


def _build_payload(model: Model, context: Context, options: StreamOptions) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.id,
        "messages": _convert_messages(context),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if options.temperature is not None:
        payload["temperature"] = options.temperature
    if options.max_tokens is not None:
        payload["max_tokens"] = options.max_tokens
    if context.tools:
        payload["tools"] = [_convert_tool(tool) for tool in context.tools]
    return payload


def _convert_messages(context: Context) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if context.system_prompt:
        messages.append({"role": "system", "content": context.system_prompt})

    for msg in context.messages:
        if isinstance(msg, UserMessage):
            if isinstance(msg.content, str):
                messages.append({"role": "user", "content": msg.content})
            else:
                parts: list[dict[str, Any]] = []
                for part in msg.content:
                    if isinstance(part, TextContent):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{part.mime_type};base64,{part.data}"
                                },
                            }
                        )
                if parts:
                    messages.append({"role": "user", "content": parts})

        elif isinstance(msg, AssistantMessage):
            text = "".join(
                block.text
                for block in msg.content
                if isinstance(block, TextContent) and block.text.strip()
            )
            tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
            # Skip empty assistant messages; some providers require
            # "either content or tool_calls, but not none".
            if not text and not tool_calls:
                continue
            converted: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                converted["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(converted)

        elif isinstance(msg, ToolResultMessage):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.tool_name,
                    "content": msg.text() or "(no tool output)",
                }
            )

    return messages


def _convert_tool(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _parse_usage(raw: dict[str, Any], model: Model) -> Usage:
    """Map OpenAI usage fields onto pion's Usage and compute cost.

    Follows pi's semantics: `cached_tokens`/`prompt_cache_hit_tokens` count as
    cache reads and are subtracted from the prompt tokens to get billable
    uncached input; OpenAI `completion_tokens` already include reasoning.
    """
    prompt_tokens = raw.get("prompt_tokens") or 0
    completion_tokens = raw.get("completion_tokens") or 0
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    cache_read = prompt_details.get("cached_tokens") or raw.get("prompt_cache_hit_tokens") or 0
    cache_write = prompt_details.get("cache_write_tokens") or 0
    input_tokens = max(0, prompt_tokens - cache_read - cache_write)
    usage = Usage(
        input=input_tokens,
        output=completion_tokens,
        cacheRead=cache_read,
        cacheWrite=cache_write,
        reasoning=completion_details.get("reasoning_tokens") or 0,
        totalTokens=input_tokens + completion_tokens + cache_read + cache_write,
    )
    usage.cost = model.compute_cost(usage)
    return usage


def _map_finish_reason(reason: str) -> tuple[StopReason, Optional[str]]:
    if reason in ("stop", "end"):
        return "stop", None
    if reason == "length":
        return "length", None
    if reason in ("function_call", "tool_calls"):
        return "toolUse", None
    if reason == "content_filter":
        return "error", "Provider finish_reason: content_filter"
    return "error", f"Provider finish_reason: {reason}"


def parse_partial_json(partial: str) -> dict[str, Any]:
    """Best-effort parse of possibly truncated streaming JSON arguments.

    Returns the parsed object; on truncation, salvages by closing open
    strings/braces (trimming incomplete tail tokens as needed). Falls back
    to `{}` when nothing parses.
    """
    if not partial or not partial.strip():
        return {}
    try:
        value = json.loads(partial)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    text = partial.rstrip()
    while text:
        try:
            value = json.loads(_close_open_constructs(text))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            text = text[:-1].rstrip()
    return {}


def _close_open_constructs(text: str) -> str:
    """Close any open string literal and open `{`/`[` brackets."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
    if escaped:
        # Dangling backslash at end of input; drop it.
        text = text[:-1]
    closers = '"' if in_string else ""
    closers += "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    return text + closers

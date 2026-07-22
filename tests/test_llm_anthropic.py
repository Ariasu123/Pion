"""Tests for the Anthropic Messages API provider."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from pion.llm.anthropic_messages import stream_simple
from pion.llm.event_stream import StreamOptions
from pion.llm.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

BASE_URL = "https://api.anthropic.test"
MESSAGES_URL = f"{BASE_URL}/v1/messages"


def make_model() -> Model:
    return Model(
        id="claude-test",
        provider="anthropic",
        api="anthropic-messages",
        baseUrl=BASE_URL,
        cost=ModelCost(input=3.0, output=15.0, cacheRead=0.3),
        contextWindow=200000,
        maxTokens=8192,
        reasoning=True,
    )


def make_context(with_tool: bool = False) -> Context:
    context = Context(
        systemPrompt="You are helpful.",
        messages=[UserMessage(content="Hi")],
    )
    if with_tool:
        context.tools = [
            Tool(
                name="get_weather",
                description="Get weather for a city",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ]
    return context


def sse_bytes(*events: tuple[str, dict]) -> bytes:
    parts = [f"event: {name}\ndata: {json.dumps(data)}" for name, data in events]
    return ("\n\n".join(parts) + "\n\n").encode()


def message_start(input_tokens: int = 12) -> tuple[str, dict]:
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "usage": {"input_tokens": input_tokens, "output_tokens": 1},
            },
        },
    )


def text_events(*texts: str, stop_reason: str = "end_turn", output_tokens: int = 6) -> bytes:
    events: list[tuple[str, dict]] = [
        message_start(),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ]
    for text in texts:
        events.append(
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            )
        )
    events += [
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": output_tokens},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return sse_bytes(*events)


async def collect(stream) -> tuple[list, AssistantMessage]:
    events = [event async for event in stream]
    return events, await stream.result()


@respx.mock
async def test_text_only_reply() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200,
            content=text_events("Hello", " world"),
            headers={"content-type": "text/event-stream"},
        )
    )
    stream = stream_simple(make_model(), make_context(), StreamOptions(api_key="sk-ant"))

    events, result = await collect(stream)

    assert [e.type for e in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert events[2].delta == "Hello"
    assert events[3].delta == " world"
    assert result.text() == "Hello world"
    assert result.stop_reason == "stop"
    assert result.response_id == "msg_1"

    request = route.calls[0].request
    assert request.headers["x-api-key"] == "sk-ant"
    assert request.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(request.content)
    assert body["stream"] is True
    assert body["system"] == "You are helpful."
    assert body["max_tokens"] == 8192  # default: model.max_tokens
    assert body["messages"] == [{"role": "user", "content": "Hi"}]


@respx.mock
async def test_tool_call_streamed_in_multiple_deltas() -> None:
    events = [
        message_start(),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
            },
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"city": "Par'}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": 'is", "units": "c"}'}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}},
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200, content=sse_bytes(*events), headers={"content-type": "text/event-stream"}
        )
    )
    stream = stream_simple(make_model(), make_context(with_tool=True), StreamOptions(api_key="k"))

    collected, result = await collect(stream)

    assert [e.type for e in collected] == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    deltas = [e.delta for e in collected if e.type == "toolcall_delta"]
    assert "".join(deltas) == '{"city": "Paris", "units": "c"}'
    tool_call = result.tool_calls()[0]
    assert tool_call.id == "toolu_1"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"city": "Paris", "units": "c"}
    assert result.stop_reason == "toolUse"

    body = json.loads(route.calls[0].request.content)
    assert body["tools"] == [
        {
            "name": "get_weather",
            "description": "Get weather for a city",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


@respx.mock
async def test_stop_reason_max_tokens() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200,
            content=text_events("truncated", stop_reason="max_tokens"),
            headers={"content-type": "text/event-stream"},
        )
    )
    stream = stream_simple(make_model(), make_context(), StreamOptions(api_key="k"))

    events, result = await collect(stream)

    assert events[-1].type == "done"
    assert events[-1].reason == "length"
    assert result.stop_reason == "length"


@respx.mock
async def test_usage_and_computed_cost() -> None:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_2",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 40,
                        "cache_creation_input_tokens": 10,
                    },
                },
            },
        ),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 25}},
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200, content=sse_bytes(*events), headers={"content-type": "text/event-stream"}
        )
    )
    model = make_model()  # cost: input 3.0, output 15.0, cache_read 0.3 per 1M tokens
    stream = stream_simple(model, make_context(), StreamOptions(api_key="k"))

    _, result = await collect(stream)

    assert result.usage.input == 100
    assert result.usage.output == 25
    assert result.usage.cache_read == 40
    assert result.usage.cache_write == 10
    assert result.usage.total_tokens == 175
    assert result.usage.cost.input == pytest.approx(100 * 3.0 / 1_000_000)
    assert result.usage.cost.output == pytest.approx(25 * 15.0 / 1_000_000)
    assert result.usage.cost.cache_read == pytest.approx(40 * 0.3 / 1_000_000)
    assert result.usage.cost.total == pytest.approx(
        result.usage.cost.input
        + result.usage.cost.output
        + result.usage.cost.cache_read
        + result.usage.cost.cache_write
    )


@respx.mock
async def test_http_401_yields_error_event() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            401,
            json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}},
        )
    )
    stream = stream_simple(make_model(), make_context(), StreamOptions(api_key="bad-key"))

    events, result = await collect(stream)

    assert [e.type for e in events] == ["error"]
    assert result.stop_reason == "error"
    assert result.error_message is not None
    assert "401" in result.error_message
    assert "invalid x-api-key" in result.error_message


@respx.mock
async def test_abort_yields_aborted_error() -> None:
    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200,
            content=text_events("never finished"),
            headers={"content-type": "text/event-stream"},
        )
    )
    abort = asyncio.Event()
    abort.set()
    stream = stream_simple(make_model(), make_context(), StreamOptions(api_key="k", abort=abort))

    events, result = await collect(stream)

    assert events[-1].type == "error"
    assert result.stop_reason == "aborted"
    assert result.error_message == "Request was aborted"


@respx.mock
async def test_conversation_with_tool_history_is_converted() -> None:
    route = respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            200,
            content=text_events("done"),
            headers={"content-type": "text/event-stream"},
        )
    )
    context = Context(
        messages=[
            UserMessage(content="weather in Paris?"),
            AssistantMessage(
                content=[
                    TextContent(text="Let me check."),
                    ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Paris"}),
                ]
            ),
            ToolResultMessage(
                toolCallId="toolu_1",
                toolName="get_weather",
                content=[TextContent(text="sunny, 22C")],
            ),
            UserMessage(content="thanks"),
        ]
    )
    stream = stream_simple(make_model(), context, StreamOptions(api_key="k"))

    _, result = await collect(stream)

    assert result.stop_reason == "stop"
    body = json.loads(route.calls[0].request.content)
    assert body["messages"] == [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "sunny, 22C",
                    "is_error": False,
                }
            ],
        },
        {"role": "user", "content": "thanks"},
    ]

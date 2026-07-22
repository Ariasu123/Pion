"""Tests for the OpenAI-compatible chat completions provider."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from pion.llm.event_stream import StreamOptions
from pion.llm.openai_completions import stream_simple
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

BASE_URL = "https://api.test/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"


def make_model() -> Model:
    return Model(
        id="test-model",
        provider="test",
        api="openai-completions",
        baseUrl=BASE_URL,
        cost=ModelCost(input=1.0, output=2.0, cacheRead=0.1),
        contextWindow=64000,
        maxTokens=8192,
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


def sse_bytes(*chunks: dict) -> bytes:
    parts = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    parts.append("data: [DONE]")
    return ("\n\n".join(parts) + "\n\n").encode()


def text_chunks(*texts: str, finish_reason: str = "stop", usage: dict | None = None) -> bytes:
    chunks: list[dict] = [
        {
            "id": "chatcmpl-1",
            "model": "test-model",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        }
    ]
    for text in texts:
        chunks.append(
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
        )
    chunks.append(
        {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
    )
    if usage is not None:
        chunks.append({"id": "chatcmpl-1", "choices": [], "usage": usage})
    return sse_bytes(*chunks)


async def collect(stream) -> tuple[list, AssistantMessage]:
    events = [event async for event in stream]
    return events, await stream.result()


@respx.mock
async def test_text_only_reply() -> None:
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=text_chunks(
                "Hello", " world",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    model = make_model()
    stream = stream_simple(model, make_context(), StreamOptions(api_key="sk-test"))

    events, result = await collect(stream)

    assert [e.type for e in events] == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert events[0].content_index is None
    assert events[2].delta == "Hello"
    assert events[3].delta == " world"
    assert all(e.content_index == 0 for e in events[1:5])
    assert result.text() == "Hello world"
    assert result.stop_reason == "stop"
    assert events[-1].message is result

    # Request shape.
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["model"] == "test-model"
    assert body["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]


@respx.mock
async def test_tool_call_streamed_in_multiple_deltas() -> None:
    chunks = [
        {
            "id": "chatcmpl-2",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-2",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": '{"city": "Par'}}]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-2",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'is", "units": "c"}'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-2",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
        {
            "id": "chatcmpl-2",
            "choices": [],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        },
    ]
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=sse_bytes(*chunks), headers={"content-type": "text/event-stream"}
        )
    )
    stream = stream_simple(make_model(), make_context(with_tool=True), StreamOptions(api_key="k"))

    events, result = await collect(stream)

    assert [e.type for e in events] == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    deltas = [e.delta for e in events if e.type == "toolcall_delta"]
    assert "".join(deltas) == '{"city": "Paris", "units": "c"}'
    tool_call = result.tool_calls()[0]
    assert tool_call.id == "call_1"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"city": "Paris", "units": "c"}
    assert result.stop_reason == "toolUse"

    body = json.loads(route.calls[0].request.content)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


@respx.mock
async def test_finish_reason_length() -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=text_chunks("truncated", finish_reason="length"),
            headers={"content-type": "text/event-stream"},
        )
    )
    stream = stream_simple(make_model(), make_context(), StreamOptions(api_key="k"))

    events, result = await collect(stream)

    assert events[-1].type == "done"
    assert events[-1].reason == "length"
    assert result.stop_reason == "length"
    assert result.text() == "truncated"


@respx.mock
async def test_usage_and_computed_cost() -> None:
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens_details": {"reasoning_tokens": 5},
    }
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=text_chunks("hi", usage=usage),
            headers={"content-type": "text/event-stream"},
        )
    )
    model = make_model()  # cost: input 1.0, output 2.0, cache_read 0.1 per 1M tokens
    stream = stream_simple(model, make_context(), StreamOptions(api_key="k"))

    _, result = await collect(stream)

    # 100 prompt - 40 cached = 60 uncached input tokens.
    assert result.usage.input == 60
    assert result.usage.output == 25
    assert result.usage.cache_read == 40
    assert result.usage.reasoning == 5
    assert result.usage.total_tokens == 125
    assert result.usage.cost.input == pytest.approx(60 * 1.0 / 1_000_000)
    assert result.usage.cost.output == pytest.approx(25 * 2.0 / 1_000_000)
    assert result.usage.cost.cache_read == pytest.approx(40 * 0.1 / 1_000_000)
    assert result.usage.cost.total == pytest.approx(
        result.usage.cost.input + result.usage.cost.output + result.usage.cost.cache_read
    )


@respx.mock
async def test_http_401_yields_error_event() -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(401, json={"error": {"message": "Incorrect API key"}})
    )
    stream = stream_simple(make_model(), make_context(), StreamOptions(api_key="bad-key"))

    events, result = await collect(stream)

    assert [e.type for e in events] == ["error"]
    assert result.stop_reason == "error"
    assert result.error_message is not None
    assert "401" in result.error_message
    assert "Incorrect API key" in result.error_message


@respx.mock
async def test_abort_yields_aborted_error() -> None:
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=text_chunks("never finished"),
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
async def test_truncated_tool_call_arguments_are_salvaged() -> None:
    chunks = [
        {
            "id": "chatcmpl-3",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_9",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Par'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {"id": "chatcmpl-3", "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
    ]
    respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=sse_bytes(*chunks), headers={"content-type": "text/event-stream"}
        )
    )
    stream = stream_simple(make_model(), make_context(with_tool=True), StreamOptions(api_key="k"))

    _, result = await collect(stream)

    assert result.stop_reason == "length"
    assert result.tool_calls()[0].arguments == {"city": "Par"}


@respx.mock
async def test_conversation_with_tool_history_is_converted() -> None:
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=text_chunks("done"),
            headers={"content-type": "text/event-stream"},
        )
    )
    context = Context(
        messages=[
            UserMessage(content="weather in Paris?"),
            AssistantMessage(
                content=[ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})]
            ),
            ToolResultMessage(
                toolCallId="call_1",
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
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "sunny, 22C"},
        {"role": "user", "content": "thanks"},
    ]

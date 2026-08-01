"""Tests for the agent loop (pion.agent) — loop semantics and Agent facade."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from pydantic import BaseModel

from pion.agent.agent import Agent
from pion.agent.loop import (
    AgentContext,
    AgentLoopConfig,
    run_agent_loop,
    run_agent_loop_continue,
)
from pion.llm.event_stream import AssistantMessageEvent, AssistantMessageEventStream
from pion.llm.types import (
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pion.tools.base import AgentToolResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def make_model() -> Model:
    return Model(
        id="fake-model",
        name="Fake",
        api="openai-completions",
        provider="fake",
        base_url="http://localhost:9",
    )


def tool_call(name: str, args: dict[str, Any], call_id: Optional[str] = None) -> ToolCall:
    return ToolCall(id=call_id or f"call-{name}-1", name=name, arguments=args)


class EchoArgs(BaseModel):
    text: str


class EchoTool:
    """Configurable test tool recording how/when it was executed."""

    def __init__(
        self,
        name: str = "echo",
        execution_mode: str = "parallel",
        delay: float = 0.0,
        log: Optional[list] = None,
        prefix: Optional[str] = None,
    ) -> None:
        self.name = name
        self.label = name
        self.description = f"Test tool {name}"
        self.Args = EchoArgs
        self.execution_mode = execution_mode
        self.delay = delay
        self.log = log  # shared list receiving ("start"|"end", name) markers
        self.prefix = prefix or name
        self.calls: list[str] = []

    @property
    def parameters(self) -> dict[str, Any]:
        return self.Args.model_json_schema()

    async def execute(self, tool_call_id, args, abort=None, on_update=None) -> AgentToolResult:
        if self.log is not None:
            self.log.append(("start", self.name))
        if self.delay:
            await asyncio.sleep(self.delay)
        self.calls.append(args.text)
        if self.log is not None:
            self.log.append(("end", self.name))
        return AgentToolResult.text(f"{self.prefix}:{args.text}")


def fake_stream_fn(scripts: list[dict[str, Any]]):
    """Build a stream_fn replaying one script per LLM call.

    Each script: {"text": str, "tool_calls": [ToolCall], "stop_reason": str,
    "error_message": str, "check_abort": bool}. `check_abort` makes the call
    yield an aborted message when options.abort is set. Scripts beyond the
    list length repeat the last one. Returns (stream_fn, calls).
    """
    calls: list[dict[str, Any]] = []

    def stream_fn(model, context, options=None) -> AssistantMessageEventStream:
        script = scripts[min(len(calls), len(scripts) - 1)]
        calls.append({"model": model, "context": context, "options": options})

        async def gen():
            if (
                script.get("check_abort")
                and options is not None
                and options.abort is not None
                and options.abort.is_set()
            ):
                aborted = AssistantMessage(
                    stop_reason="aborted", error_message="The operation was aborted"
                )
                yield AssistantMessageEvent(type="done", reason="aborted", message=aborted)
                return

            stop = script.get("stop_reason") or (
                "toolUse" if script.get("tool_calls") else "stop"
            )
            yield AssistantMessageEvent(type="start", partial=AssistantMessage())
            content: list = []
            text = script.get("text")
            if text:
                content.append(TextContent(text=text))
                yield AssistantMessageEvent(
                    type="text_delta",
                    content_index=0,
                    delta=text,
                    partial=AssistantMessage(content=list(content), stop_reason=stop),
                )
            for tc in script.get("tool_calls") or []:
                content.append(tc)
            final = AssistantMessage(
                content=content,
                stop_reason=stop,
                error_message=script.get("error_message"),
            )
            yield AssistantMessageEvent(type="done", reason=stop, message=final)

        return AssistantMessageEventStream(gen())

    return stream_fn, calls


def make_agent(scripts, tools=None, **kwargs) -> tuple[Agent, list, list]:
    """Agent with a fake stream; returns (agent, recorded calls, event types)."""
    stream_fn, calls = fake_stream_fn(scripts)
    agent = Agent(model=make_model(), tools=tools or [], stream_fn=stream_fn, **kwargs)
    events: list[str] = []
    agent.subscribe(lambda event: events.append(event.type))
    return agent, calls, events


def tool_results_of(agent: Agent) -> list[ToolResultMessage]:
    return [m for m in agent.messages if isinstance(m, ToolResultMessage)]


# ---------------------------------------------------------------------------
# Basic turns
# ---------------------------------------------------------------------------


async def test_single_text_turn():
    agent, calls, events = make_agent([{"text": "hello world"}])
    final = await agent.prompt("hi")

    assert final.text() == "hello world"
    assert [m.role for m in agent.messages] == ["user", "assistant"]
    assert agent.messages[1].text() == "hello world"
    assert not agent.is_streaming
    assert agent.error_message is None
    assert events == [
        "agent_start",
        "turn_start",
        "message_start",  # user prompt
        "message_end",
        "message_start",  # assistant (stream start)
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ]


async def test_two_round_tool_cycle():
    echo = EchoTool()
    agent, calls, events = make_agent(
        [
            {"tool_calls": [tool_call("echo", {"text": "ping"}, call_id="c1")]},
            {"text": "done"},
        ],
        tools=[echo],
    )
    final = await agent.prompt("go")

    assert final.text() == "done"
    assert echo.calls == ["ping"]
    assert [m.role for m in agent.messages] == ["user", "assistant", "toolResult", "assistant"]

    result = agent.messages[2]
    assert isinstance(result, ToolResultMessage)
    assert result.tool_call_id == "c1"
    assert result.tool_name == "echo"
    assert not result.is_error
    assert result.text() == "echo:ping"

    # The second LLM call saw the tool result, and tools were advertised.
    assert any(m.role == "toolResult" for m in calls[1]["context"].messages)
    assert [t.name for t in calls[0]["context"].tools] == ["echo"]


async def test_tool_can_return_an_explicit_error_result():
    class ErrorResultTool(EchoTool):
        async def execute(self, tool_call_id, args, abort=None, on_update=None):
            return AgentToolResult.text("remote error", is_error=True)

    agent, _, _ = make_agent(
        [
            {"tool_calls": [tool_call("echo", {"text": "x"})]},
            {"text": "handled"},
        ],
        tools=[ErrorResultTool()],
    )
    await agent.prompt("go")
    result = tool_results_of(agent)[0]
    assert result.is_error
    assert result.text() == "remote error"


async def test_events_sequence_sanity_tool_cycle():
    echo = EchoTool()
    agent, _, events = make_agent(
        [
            {"tool_calls": [tool_call("echo", {"text": "x"})]},
            {"text": "done"},
        ],
        tools=[echo],
    )
    await agent.prompt("go")

    assert events[0] == "agent_start"
    assert events[-1] == "agent_end"
    assert events.count("turn_start") == 2
    assert events.count("turn_end") == 2
    # Tool execution bracketed inside the first turn.
    assert events.index("tool_execution_start") < events.index("tool_execution_end")
    assert events.index("tool_execution_end") < events.index("turn_end")
    # toolResult message events come after execution end, before turn_end.
    end = events.index("tool_execution_end")
    assert events[end + 1 : end + 3] == ["message_start", "message_end"]


# ---------------------------------------------------------------------------
# Parallel / sequential execution
# ---------------------------------------------------------------------------


async def test_parallel_tool_execution_results_in_source_order():
    log: list = []
    slow = EchoTool(name="slow", delay=0.05, log=log)
    fast = EchoTool(name="fast", log=log)
    agent, _, _ = make_agent(
        [
            {
                "tool_calls": [
                    tool_call("slow", {"text": "a"}, call_id="c1"),
                    tool_call("fast", {"text": "b"}, call_id="c2"),
                ]
            },
            {"text": "ok"},
        ],
        tools=[slow, fast],
    )
    await agent.prompt("go")

    # Both ran concurrently: fast finished before slow even though slow started first.
    assert log == [("start", "slow"), ("start", "fast"), ("end", "fast"), ("end", "slow")]
    # ToolResultMessages are appended in assistant source order, not completion order.
    results = tool_results_of(agent)
    assert [r.tool_call_id for r in results] == ["c1", "c2"]
    assert [r.text() for r in results] == ["slow:a", "fast:b"]


async def test_sequential_when_tool_execution_mode_says_so():
    log: list = []
    seq = EchoTool(name="seq", execution_mode="sequential", delay=0.02, log=log)
    fast = EchoTool(name="fast", log=log)
    agent, _, _ = make_agent(
        [
            {
                "tool_calls": [
                    tool_call("seq", {"text": "a"}, call_id="c1"),
                    tool_call("fast", {"text": "b"}, call_id="c2"),
                ]
            },
            {"text": "ok"},
        ],
        tools=[seq, fast],
    )
    await agent.prompt("go")

    assert log == [("start", "seq"), ("end", "seq"), ("start", "fast"), ("end", "fast")]


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


async def test_length_stop_fails_all_tool_calls_without_executing():
    echo = EchoTool()
    agent, _, _ = make_agent(
        [
            {
                "tool_calls": [
                    tool_call("echo", {"text": "x"}, call_id="c1"),
                    tool_call("echo", {"text": "y"}, call_id="c2"),
                ],
                "stop_reason": "length",
            },
            {"text": "recovered"},
        ],
        tools=[echo],
    )
    final = await agent.prompt("go")

    assert echo.calls == []  # never executed
    results = tool_results_of(agent)
    assert len(results) == 2
    assert all(r.is_error for r in results)
    assert "token limit" in results[0].text()
    assert final.text() == "recovered"  # the model re-issued afterwards


async def test_unknown_tool_produces_error_result():
    agent, _, _ = make_agent(
        [
            {"tool_calls": [tool_call("nosuch", {"text": "x"})]},
            {"text": "ok"},
        ]
    )
    final = await agent.prompt("go")

    results = tool_results_of(agent)
    assert len(results) == 1
    assert results[0].is_error
    assert "Tool nosuch not found" in results[0].text()
    assert final.text() == "ok"  # loop recovered instead of raising


async def test_tool_exception_becomes_error_result():
    class BoomTool(EchoTool):
        async def execute(self, tool_call_id, args, abort=None, on_update=None):
            raise RuntimeError("boom")

    agent, _, _ = make_agent(
        [
            {"tool_calls": [tool_call("boom", {"text": "x"})]},
            {"text": "ok"},
        ],
        tools=[BoomTool(name="boom")],
    )
    final = await agent.prompt("go")

    results = tool_results_of(agent)
    assert len(results) == 1
    assert results[0].is_error
    assert "boom" in results[0].text()
    assert final.text() == "ok"


# ---------------------------------------------------------------------------
# Loop-level hooks (config)
# ---------------------------------------------------------------------------


def loop_setup(scripts, tools, **config_kwargs):
    stream_fn, calls = fake_stream_fn(scripts)
    config = AgentLoopConfig(model=make_model(), **config_kwargs)
    events: list[str] = []

    async def emit(event):
        events.append(event.type)

    context = AgentContext(system_prompt="", messages=[], tools=tools)
    return stream_fn, calls, config, emit, context, events


async def test_before_tool_call_block():
    echo = EchoTool()

    async def before(ctx):
        return {"block": True, "reason": "blocked by policy"}

    stream_fn, _, config, emit, context, _ = loop_setup(
        [
            {"tool_calls": [tool_call("echo", {"text": "x"})]},
            {"text": "ok"},
        ],
        [echo],
        before_tool_call=before,
    )
    new_messages = await run_agent_loop(
        [UserMessage(content="go")], context, config, emit, asyncio.Event(), stream_fn
    )

    results = [m for m in new_messages if isinstance(m, ToolResultMessage)]
    assert echo.calls == []
    assert len(results) == 1
    assert results[0].is_error
    assert "blocked by policy" in results[0].text()


async def test_after_tool_call_override():
    echo = EchoTool()

    async def after(ctx):
        return {"content": [TextContent(text="overridden")], "details": {"patched": True}}

    stream_fn, _, config, emit, context, _ = loop_setup(
        [
            {"tool_calls": [tool_call("echo", {"text": "x"})]},
            {"text": "ok"},
        ],
        [echo],
        after_tool_call=after,
    )
    new_messages = await run_agent_loop(
        [UserMessage(content="go")], context, config, emit, asyncio.Event(), stream_fn
    )

    results = [m for m in new_messages if isinstance(m, ToolResultMessage)]
    assert echo.calls == ["x"]  # the tool really ran
    assert results[0].text() == "overridden"
    assert results[0].details == {"patched": True}
    assert not results[0].is_error


async def test_steering_message_injected_mid_run():
    steering_calls = {"n": 0}

    async def get_steering():
        steering_calls["n"] += 1
        if steering_calls["n"] == 2:  # after the first turn
            return [UserMessage(content="steer")]
        return []

    stream_fn, calls, config, emit, context, _ = loop_setup(
        [{"text": "first"}, {"text": "second"}],
        [],
        get_steering_messages=get_steering,
    )
    new_messages = await run_agent_loop(
        [UserMessage(content="go")], context, config, emit, asyncio.Event(), stream_fn
    )

    assert len(calls) == 2
    assert [m.role for m in new_messages] == ["user", "assistant", "user", "assistant"]
    assert new_messages[2].content == "steer"
    # The steering message was part of the second LLM call's context.
    second_texts = [
        m.content for m in calls[1]["context"].messages if getattr(m, "role", None) == "user"
    ]
    assert "steer" in second_texts


async def test_follow_up_continues_after_stop():
    follow = {"done": False}

    async def get_follow_up():
        if not follow["done"]:
            follow["done"] = True
            return [UserMessage(content="follow up")]
        return []

    stream_fn, calls, config, emit, context, events = loop_setup(
        [{"text": "one"}, {"text": "two"}],
        [],
        get_follow_up_messages=get_follow_up,
    )
    new_messages = await run_agent_loop(
        [UserMessage(content="go")], context, config, emit, asyncio.Event(), stream_fn
    )

    assert len(calls) == 2
    assert [m.role for m in new_messages] == ["user", "assistant", "user", "assistant"]
    assert new_messages[2].content == "follow up"
    assert events[-1] == "agent_end"  # exactly one agent_end for the whole run
    assert events.count("agent_end") == 1


async def test_abort_mid_run_finishes_tool_then_ends():
    abort = asyncio.Event()

    class AbortTool(EchoTool):
        async def execute(self, tool_call_id, args, abort=None, on_update=None):
            abort.set()  # abort fires while the tool runs
            return AgentToolResult.text("tool finished")

    stream_fn, _, config, emit, context, events = loop_setup(
        [
            {"tool_calls": [tool_call("aborting", {"text": "x"})]},
            {"check_abort": True, "text": "unreachable"},
        ],
        [AbortTool(name="aborting")],
    )
    new_messages = await run_agent_loop(
        [UserMessage(content="go")], context, config, emit, abort, stream_fn
    )

    # The running tool finished normally...
    results = [m for m in new_messages if isinstance(m, ToolResultMessage)]
    assert len(results) == 1
    assert not results[0].is_error
    assert results[0].text() == "tool finished"
    # ...then the next stream came back aborted and the run ended with what we have.
    assistants = [m for m in new_messages if isinstance(m, AssistantMessage)]
    assert assistants[-1].stop_reason == "aborted"
    assert events[-1] == "agent_end"


async def test_run_agent_loop_continue():
    stream_fn, calls = fake_stream_fn([{"text": "continued"}])
    config = AgentLoopConfig(model=make_model())
    events: list[str] = []

    async def emit(event):
        events.append(event.type)

    context = AgentContext(messages=[UserMessage(content="still there")], tools=[])
    new_messages = await run_agent_loop_continue(
        context, config, emit, asyncio.Event(), stream_fn
    )

    assert [m.role for m in new_messages] == ["assistant"]
    assert calls[0]["context"].messages[0].content == "still there"

    # Continuing from an assistant message is rejected.
    try:
        await run_agent_loop_continue(context, config, emit, asyncio.Event(), stream_fn)
    except ValueError as exc:
        assert "assistant" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# Agent facade behavior
# ---------------------------------------------------------------------------


async def test_agent_error_state_on_stream_error():
    agent, _, _ = make_agent(
        [{"stop_reason": "error", "error_message": "provider exploded"}]
    )
    final = await agent.prompt("hi")

    assert final.stop_reason == "error"
    assert agent.error_message == "provider exploded"
    assert not agent.is_streaming


async def test_agent_dynamic_tool_management():
    agent, calls, _ = make_agent([{"text": "ok"}])
    echo = EchoTool()
    agent.add_tool(echo)
    assert [t.name for t in agent.tools] == ["echo"]
    await agent.prompt("hi")
    assert [t.name for t in calls[0]["context"].tools] == ["echo"]

    agent.remove_tool("echo")
    assert agent.tools == []


async def test_agent_pending_tool_calls_tracked():
    agent, _, _ = make_agent(
        [
            {"tool_calls": [tool_call("echo", {"text": "x"}, call_id="c1")]},
            {"text": "ok"},
        ],
        tools=[EchoTool()],
    )
    seen_pending: list[set] = []
    agent.subscribe(
        lambda event: seen_pending.append(set(agent.pending_tool_calls))
        if event.type == "tool_execution_end"
        else None
    )
    await agent.prompt("go")

    assert seen_pending == [set()]  # empty again by execution_end
    assert agent.pending_tool_calls == set()


# ---------------------------------------------------------------------------
# Lone-surrogate sanitization (reproduction tests for the serialization bug)
# ---------------------------------------------------------------------------


async def test_prompt_with_lone_surrogate_user_input_survives_persist(tmp_path):
    """User input decoded with surrogateescape must not crash session writes."""
    from pion.session import SessionManager

    agent, _, _ = make_agent([{"text": "ok"}])
    final = await agent.prompt("帮我 \udce6 写代码")
    assert final.text() == "ok"

    session = SessionManager(tmp_path / "s.jsonl")
    for message in agent.messages:  # pre-fix: model_dump_json raises here
        session.append_message(message)


async def test_prompt_sanitizes_surrogates_from_provider_stream():
    """Unpaired \\uXXXX escapes in provider JSON must not poison messages."""
    agent, _, _ = make_agent([{"text": "provider \udce6 output"}])
    final = await agent.prompt("go")
    assert "\udce6" not in final.text()
    assert "\udce6" not in agent.messages[-1].text()


async def test_error_message_with_surrogate_does_not_cascade():
    """A poisoned error_message must not break every subsequent prompt."""
    agent, _, _ = make_agent(
        [
            {"stop_reason": "error", "error_message": "boom \udce6"},
            {"text": "recovered"},
        ]
    )
    final = await agent.prompt("go")
    assert final.stop_reason == "error"
    assert "\udce6" not in (agent.error_message or "")
    # The next prompt must work (pre-fix the poisoned message kills it).
    recovered = await agent.prompt("again")
    assert recovered.text() == "recovered"

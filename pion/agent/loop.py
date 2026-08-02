"""The agent loop — faithful async port of pi's agent-loop.ts.

Works with pion `Message`s throughout and converts to the LLM `Context`
only at the provider call boundary. See packages/agent/src/agent-loop.ts
for the original; semantics preserved:

- Outer loop continues when queued follow-up messages arrive after the
  agent would stop; inner loop processes tool calls and steering messages.
- `turn_start` is emitted before each assistant response after the first
  (the initial `turn_start` is emitted by `run_agent_loop` itself).
- `message_start`/`message_end` wrap every message; streamed partials
  update the last context message and emit `message_update`.
- stop_reason "length" fails ALL tool calls in that message with an error
  tool result (truncated arguments are never executed).
- stop_reason "error"/"aborted" ends the run (turn_end + agent_end).
- Parallel tool execution via asyncio.gather, sequential when the config
  or any called tool's `execution_mode` says so.
- Unknown tool / validation failure / before_tool_call block / tool
  exceptions all become error ToolResultMessages — the loop never raises
  for tool failures.
- `terminate: True` on every result in a batch stops the inner loop.
- Abort (asyncio.Event) is honored between tool calls and is passed to
  the stream function, which encodes abortion as stop_reason "aborted".

Assistant response streaming lives in `streaming.py`; tool-call execution
lives in `tool_execution.py`. Both are pure moves out of this module.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional

from ..llm.event_stream import StreamOptions
from ..llm.types import (
    AssistantMessage,
    Context,
    Message,
    Model,
    ToolCall,
    ToolResultMessage,
)
from ..tools.base import AgentTool, AgentToolResult
from .events import AgentEvent

# Sink receiving every AgentEvent. Always awaited by the loop.
AgentEventSink = Callable[[AgentEvent], Awaitable[None]]

# Stream function shape: pion.llm.stream_simple satisfies this. May return
# the stream directly or an awaitable of it. Must never raise for request
# failures — failures are encoded as a final AssistantMessage with
# stop_reason "error"/"aborted".
StreamFn = Callable[[Model, Context, StreamOptions], Any]


async def _maybe_await(value: Any) -> Any:
    """Await `value` if it is awaitable, otherwise return it as-is."""
    if inspect.isawaitable(value):
        return await value
    return value


def _identity(messages: list[Message]) -> list[Message]:
    """Default convert_to_llm: messages are already LLM-compatible."""
    return list(messages)


# ---------------------------------------------------------------------------
# Context / config
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    """Context snapshot passed into the agent loop."""

    system_prompt: str = ""
    messages: list[Message] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)


@dataclass
class BeforeToolCallContext:
    """Context passed to `before_tool_call`."""

    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: Any  # validated arguments (pydantic model)
    context: AgentContext


@dataclass
class AfterToolCallContext:
    """Context passed to `after_tool_call`."""

    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: Any
    result: AgentToolResult
    is_error: bool
    context: AgentContext


@dataclass
class ShouldStopAfterTurnContext:
    """Context passed to `should_stop_after_turn`."""

    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[Message]


@dataclass
class AgentLoopConfig:
    """Configuration for one agent loop run.

    Hook contracts (same as pi): hooks must not raise for ordinary control
    flow — the loop converts their failures into error tool results where
    applicable. All hooks may be sync or async.
    """

    model: Model
    api_key: Optional[str] = None
    # AgentMessage[] -> Message[] before each LLM call. Default: identity.
    convert_to_llm: Callable[[list[Message]], Any] = _identity
    # Optional transform applied to the context messages before convert_to_llm.
    transform_context: Optional[Callable[[list[Message]], Any]] = None
    # May return {"block": True, "reason": str} to prevent execution.
    before_tool_call: Optional[Callable[[BeforeToolCallContext], Any]] = None
    # May return an override dict: content / details / is_error|isError / terminate.
    after_tool_call: Optional[Callable[[AfterToolCallContext], Any]] = None
    # Steering messages injected mid-run; follow-ups continue a finished run.
    get_steering_messages: Optional[Callable[[], Any]] = None
    get_follow_up_messages: Optional[Callable[[], Any]] = None
    # Return True to stop the loop gracefully after the current turn.
    should_stop_after_turn: Optional[Callable[[ShouldStopAfterTurnContext], Any]] = None
    tool_execution: Literal["parallel", "sequential"] = "parallel"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def run_agent_loop(
    prompts: list[Message],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    abort: Optional[asyncio.Event],
    stream_fn: StreamFn,
) -> list[Message]:
    """Start an agent loop with new prompt messages.

    The prompts are appended to the context and events are emitted for them.
    Returns the new messages produced by this run (prompts included).
    """
    new_messages: list[Message] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )

    await emit(AgentEvent(type="agent_start"))
    await emit(AgentEvent(type="turn_start"))
    for prompt in prompts:
        await emit(AgentEvent(type="message_start", message=prompt))
        await emit(AgentEvent(type="message_end", message=prompt))

    await _run_loop(current_context, new_messages, config, abort, emit, stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    abort: Optional[asyncio.Event],
    stream_fn: StreamFn,
) -> list[Message]:
    """Continue an agent loop from the current context without a new message.

    Used for retries — the context already ends with a user or toolResult
    message. Note: `context.messages` is mutated in place (same as pi).
    Returns only the messages produced by this run.
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if getattr(context.messages[-1], "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[Message] = []

    await emit(AgentEvent(type="agent_start"))
    await emit(AgentEvent(type="turn_start"))

    await _run_loop(context, new_messages, config, abort, emit, stream_fn)
    return new_messages


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def _run_loop(
    context: AgentContext,
    new_messages: list[Message],
    config: AgentLoopConfig,
    abort: Optional[asyncio.Event],
    emit: AgentEventSink,
    stream_fn: StreamFn,
) -> None:
    # Imported lazily: streaming.py / tool_execution.py import the config
    # dataclasses from this module, so top-level imports would be circular.
    from .streaming import _stream_assistant_response
    from .tool_execution import _execute_tool_calls, _fail_truncated_tool_calls

    first_turn = True
    # Check for steering messages at start (user may have typed while waiting).
    pending: list[Message] = []
    if config.get_steering_messages is not None:
        pending = (await _maybe_await(config.get_steering_messages())) or []

    # Outer loop: continues when queued follow-up messages arrive after the
    # agent would stop.
    while True:
        has_more_tool_calls = True

        # Inner loop: process tool calls and steering messages.
        while has_more_tool_calls or pending:
            if not first_turn:
                await emit(AgentEvent(type="turn_start"))
            else:
                first_turn = False

            # Inject pending messages before the next assistant response.
            if pending:
                for message in pending:
                    await emit(AgentEvent(type="message_start", message=message))
                    await emit(AgentEvent(type="message_end", message=message))
                    context.messages.append(message)
                    new_messages.append(message)
                pending = []

            message = await _stream_assistant_response(context, config, abort, emit, stream_fn)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await emit(AgentEvent(type="turn_end", message=message, tool_results=[]))
                await emit(AgentEvent(type="agent_end", messages=new_messages))
                return

            tool_calls = message.tool_calls()

            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                # A "length" stop means the output was cut off by the token
                # limit, so every tool call in the message may carry truncated
                # arguments. Fail them all instead of executing borked calls.
                if message.stop_reason == "length":
                    batch = await _fail_truncated_tool_calls(tool_calls, emit)
                else:
                    batch = await _execute_tool_calls(context, message, config, abort, emit)
                tool_results = batch.messages
                has_more_tool_calls = not batch.terminate
                for result in tool_results:
                    context.messages.append(result)
                    new_messages.append(result)

            await emit(AgentEvent(type="turn_end", message=message, tool_results=tool_results))

            if config.should_stop_after_turn is not None:
                stop_context = ShouldStopAfterTurnContext(
                    message=message,
                    tool_results=tool_results,
                    context=context,
                    new_messages=new_messages,
                )
                if await _maybe_await(config.should_stop_after_turn(stop_context)):
                    await emit(AgentEvent(type="agent_end", messages=new_messages))
                    return

            pending = []
            if config.get_steering_messages is not None:
                pending = (await _maybe_await(config.get_steering_messages())) or []

        # Agent would stop here. Check for follow-up messages.
        follow_ups: list[Message] = []
        if config.get_follow_up_messages is not None:
            follow_ups = (await _maybe_await(config.get_follow_up_messages())) or []
        if follow_ups:
            pending = follow_ups
            continue
        break

    await emit(AgentEvent(type="agent_end", messages=new_messages))

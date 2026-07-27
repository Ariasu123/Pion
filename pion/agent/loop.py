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
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional

from ..llm.event_stream import AssistantMessageEventStream, StreamOptions
from ..llm.types import (
    AssistantMessage,
    Context,
    Message,
    Model,
    TextContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    sanitize_message,
)
from ..tools.base import AgentTool, AgentToolResult, validate_arguments
from .events import AgentEvent

# Sink receiving every AgentEvent. Always awaited by the loop.
AgentEventSink = Callable[[AgentEvent], Awaitable[None]]

# Stream function shape: pion.llm.stream_simple satisfies this. May return
# the stream directly or an awaitable of it. Must never raise for request
# failures — failures are encoded as a final AssistantMessage with
# stop_reason "error"/"aborted".
StreamFn = Callable[[Model, Context, StreamOptions], Any]

TRUNCATED_TOOL_CALL_MESSAGE = (
    'Tool call "{name}" was not executed: the response hit the output token '
    "limit, so its arguments may be truncated. Re-issue the tool call with "
    "complete arguments."
)


async def _maybe_await(value: Any) -> Any:
    """Await `value` if it is awaitable, otherwise return it as-is."""
    if inspect.isawaitable(value):
        return await value
    return value


def _field(source: Any, *names: str) -> Any:
    """Read the first present, non-None field from a dict or object."""
    for name in names:
        if isinstance(source, dict):
            if name in source and source[name] is not None:
                return source[name]
        else:
            value = getattr(source, name, None)
            if value is not None:
                return value
    return None


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


# ---------------------------------------------------------------------------
# Assistant response streaming
# ---------------------------------------------------------------------------


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    abort: Optional[asyncio.Event],
    emit: AgentEventSink,
    stream_fn: StreamFn,
) -> AssistantMessage:
    """Stream one assistant response, updating the context as partials arrive."""
    messages = context.messages
    if config.transform_context is not None:
        messages = await _maybe_await(config.transform_context(messages))

    llm_messages = await _maybe_await(config.convert_to_llm(messages))

    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=[
            Tool(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool in context.tools
        ],
    )

    options = StreamOptions(api_key=config.api_key, abort=abort)
    response: AssistantMessageEventStream = await _maybe_await(
        stream_fn(config.model, llm_context, options)
    )

    partial: Optional[AssistantMessage] = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial = event.partial if event.partial is not None else AssistantMessage()
            context.messages.append(partial)
            added_partial = True
            await emit(AgentEvent(type="message_start", message=partial))
        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial is not None:
                if event.partial is not None:
                    partial = event.partial
                    context.messages[-1] = partial
                await emit(
                    AgentEvent(type="message_update", assistant_event=event, message=partial)
                )
        elif event.type in ("done", "error"):
            final_message = await response.result()
            # Scrub lone surrogates coming from the provider (unpaired
            # \uXXXX escapes) before the message enters the context.
            final_message = sanitize_message(final_message)
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await emit(AgentEvent(type="message_start", message=final_message))
            await emit(AgentEvent(type="message_end", message=final_message))
            return final_message

    # Stream ended without done/error: result() raises (contract violation).
    final_message = await response.result()
    final_message = sanitize_message(final_message)
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await emit(AgentEvent(type="message_start", message=final_message))
    await emit(AgentEvent(type="message_end", message=final_message))
    return final_message


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


@dataclass
class _FinalizedToolCall:
    """A tool call whose outcome is fully decided."""

    tool_call: ToolCall
    result: AgentToolResult
    is_error: bool


@dataclass
class _PreparedToolCall:
    """A tool call that passed validation and the before-hook."""

    tool_call: ToolCall
    tool: AgentTool
    args: Any  # validated arguments


@dataclass
class _ToolCallBatch:
    messages: list[ToolResultMessage]
    terminate: bool


def _error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})


def _should_terminate_batch(finalized: list[_FinalizedToolCall]) -> bool:
    return len(finalized) > 0 and all(f.result.terminate for f in finalized)


def _tool_result_message(finalized: _FinalizedToolCall) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        content=finalized.result.content or [],
        details=finalized.result.details,
        is_error=finalized.is_error,
    )


async def _emit_tool_result_message(message: ToolResultMessage, emit: AgentEventSink) -> None:
    await emit(AgentEvent(type="message_start", message=message))
    await emit(AgentEvent(type="message_end", message=message))


async def _fail_truncated_tool_calls(
    tool_calls: list[ToolCall],
    emit: AgentEventSink,
) -> _ToolCallBatch:
    """Fail every tool call of a token-limit-truncated assistant message.

    Streamed tool-call arguments are finalized with a best-effort JSON
    salvage parser, so a truncated message can yield tool calls whose
    arguments parse and validate but are silently incomplete. None of them
    are safe to execute; report each as an error so the model can re-issue.
    """
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await emit(
            AgentEvent(
                type="tool_execution_start",
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            )
        )
        finalized = _FinalizedToolCall(
            tool_call=tool_call,
            result=_error_tool_result(TRUNCATED_TOOL_CALL_MESSAGE.format(name=tool_call.name)),
            is_error=True,
        )
        await _emit_tool_execution_end(finalized, emit)
        result_message = _tool_result_message(finalized)
        await _emit_tool_result_message(result_message, emit)
        messages.append(result_message)
    return _ToolCallBatch(messages=messages, terminate=False)


async def _execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    abort: Optional[asyncio.Event],
    emit: AgentEventSink,
) -> _ToolCallBatch:
    tool_calls = assistant_message.tool_calls()
    has_sequential_tool_call = any(
        getattr(tool, "execution_mode", None) == "sequential"
        for tc in tool_calls
        for tool in [next((t for t in context.tools if t.name == tc.name), None)]
        if tool is not None
    )
    if config.tool_execution == "sequential" or has_sequential_tool_call:
        return await _execute_tool_calls_sequential(
            context, assistant_message, tool_calls, config, abort, emit
        )
    return await _execute_tool_calls_parallel(
        context, assistant_message, tool_calls, config, abort, emit
    )


async def _execute_tool_calls_sequential(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    abort: Optional[asyncio.Event],
    emit: AgentEventSink,
) -> _ToolCallBatch:
    finalized_calls: list[_FinalizedToolCall] = []
    messages: list[ToolResultMessage] = []

    for tool_call in tool_calls:
        await emit(
            AgentEvent(
                type="tool_execution_start",
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            )
        )
        preparation = await _prepare_tool_call(context, assistant_message, tool_call, config, abort)
        if isinstance(preparation, _FinalizedToolCall):
            finalized = preparation
        else:
            executed = await _execute_prepared_tool_call(preparation, abort, emit)
            finalized = await _finalize_executed_tool_call(
                context, assistant_message, preparation, executed, config
            )

        await _emit_tool_execution_end(finalized, emit)
        result_message = _tool_result_message(finalized)
        await _emit_tool_result_message(result_message, emit)
        finalized_calls.append(finalized)
        messages.append(result_message)

        if abort is not None and abort.is_set():
            break

    return _ToolCallBatch(
        messages=messages, terminate=_should_terminate_batch(finalized_calls)
    )


async def _execute_tool_calls_parallel(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    abort: Optional[asyncio.Event],
    emit: AgentEventSink,
) -> _ToolCallBatch:
    # Preflight sequentially, then execute concurrently. tool_execution_end
    # is emitted in completion order; tool-result message artifacts are
    # emitted afterwards in assistant source order.
    entries: list[Any] = []  # _FinalizedToolCall | async callable -> _FinalizedToolCall

    for tool_call in tool_calls:
        await emit(
            AgentEvent(
                type="tool_execution_start",
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                args=tool_call.arguments,
            )
        )
        preparation = await _prepare_tool_call(context, assistant_message, tool_call, config, abort)
        if isinstance(preparation, _FinalizedToolCall):
            await _emit_tool_execution_end(preparation, emit)
            entries.append(preparation)
        else:

            async def run(prepared: _PreparedToolCall = preparation) -> _FinalizedToolCall:
                executed = await _execute_prepared_tool_call(prepared, abort, emit)
                finalized = await _finalize_executed_tool_call(
                    context, assistant_message, prepared, executed, config
                )
                await _emit_tool_execution_end(finalized, emit)
                return finalized

            entries.append(run)
        if abort is not None and abort.is_set():
            break

    async def _ready(finalized: _FinalizedToolCall) -> _FinalizedToolCall:
        return finalized

    ordered: list[_FinalizedToolCall] = await asyncio.gather(
        *(entry() if callable(entry) else _ready(entry) for entry in entries)
    )

    messages: list[ToolResultMessage] = []
    for finalized in ordered:
        result_message = _tool_result_message(finalized)
        await _emit_tool_result_message(result_message, emit)
        messages.append(result_message)

    return _ToolCallBatch(messages=messages, terminate=_should_terminate_batch(ordered))


async def _prepare_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    abort: Optional[asyncio.Event],
) -> _PreparedToolCall | _FinalizedToolCall:
    """Resolve the tool, validate arguments, run the before-hook.

    Returns a finalized error outcome for anything that prevents execution.
    """
    tool = next((t for t in context.tools if t.name == tool_call.name), None)
    if tool is None:
        return _FinalizedToolCall(
            tool_call=tool_call,
            result=_error_tool_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        validated_args = validate_arguments(tool, tool_call.arguments)
        if config.before_tool_call is not None:
            before_result = await _maybe_await(
                config.before_tool_call(
                    BeforeToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=tool_call,
                        args=validated_args,
                        context=context,
                    )
                )
            )
            if abort is not None and abort.is_set():
                return _FinalizedToolCall(
                    tool_call=tool_call,
                    result=_error_tool_result("Operation aborted"),
                    is_error=True,
                )
            if before_result is not None and _field(before_result, "block"):
                reason = _field(before_result, "reason") or "Tool execution was blocked"
                return _FinalizedToolCall(
                    tool_call=tool_call,
                    result=_error_tool_result(reason),
                    is_error=True,
                )
        if abort is not None and abort.is_set():
            return _FinalizedToolCall(
                tool_call=tool_call,
                result=_error_tool_result("Operation aborted"),
                is_error=True,
            )
        return _PreparedToolCall(tool_call=tool_call, tool=tool, args=validated_args)
    except Exception as exc:
        return _FinalizedToolCall(
            tool_call=tool_call,
            result=_error_tool_result(str(exc)),
            is_error=True,
        )


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    abort: Optional[asyncio.Event],
    emit: AgentEventSink,
) -> tuple[AgentToolResult, bool]:
    """Run the tool. Exceptions become error results; never raises."""
    update_tasks: list[asyncio.Task] = []
    accepting_updates = True

    def on_update(partial_result: AgentToolResult) -> None:
        if not accepting_updates:
            return
        update_tasks.append(
            asyncio.get_running_loop().create_task(
                emit(
                    AgentEvent(
                        type="tool_execution_update",
                        tool_call_id=prepared.tool_call.id,
                        tool_name=prepared.tool_call.name,
                        args=prepared.tool_call.arguments,
                        partial_result=partial_result,
                    )
                )
            )
        )

    try:
        result = await prepared.tool.execute(
            prepared.tool_call.id,
            prepared.args,
            abort,
            on_update,
        )
        accepting_updates = False
        if update_tasks:
            await asyncio.gather(*update_tasks)
        return result, False
    except Exception as exc:
        accepting_updates = False
        if update_tasks:
            await asyncio.gather(*update_tasks)
        return _error_tool_result(str(exc)), True


async def _finalize_executed_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: tuple[AgentToolResult, bool],
    config: AgentLoopConfig,
) -> _FinalizedToolCall:
    """Apply the after-hook override to an executed tool result."""
    result, is_error = executed

    if config.after_tool_call is not None:
        try:
            override = await _maybe_await(
                config.after_tool_call(
                    AfterToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=prepared.tool_call,
                        args=prepared.args,
                        result=result,
                        is_error=is_error,
                        context=context,
                    )
                )
            )
            if override is not None:
                # Field-by-field merge; omitted fields keep executed values.
                result = AgentToolResult(
                    content=_field(override, "content") or result.content,
                    details=(
                        _field(override, "details")
                        if _field(override, "details") is not None
                        else result.details
                    ),
                    terminate=(
                        _field(override, "terminate")
                        if _field(override, "terminate") is not None
                        else result.terminate
                    ),
                )
                override_error = _field(override, "is_error", "isError")
                if override_error is not None:
                    is_error = override_error
        except Exception as exc:
            result = _error_tool_result(str(exc))
            is_error = True

    return _FinalizedToolCall(tool_call=prepared.tool_call, result=result, is_error=is_error)


async def _emit_tool_execution_end(
    finalized: _FinalizedToolCall, emit: AgentEventSink
) -> None:
    await emit(
        AgentEvent(
            type="tool_execution_end",
            tool_call_id=finalized.tool_call.id,
            tool_name=finalized.tool_call.name,
            result=finalized.result,
            is_error=finalized.is_error,
        )
    )

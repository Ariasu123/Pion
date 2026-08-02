"""Tool execution for the agent loop — split out of `loop.py` (pure move).

Covers tool-call preparation (resolve tool, validate arguments, before-hook),
sequential/parallel execution, result finalization (after-hook overrides) and
`tool_execution_*` event emission. Unknown tool / validation failure /
before_tool_call block / tool exceptions all become error results — these
helpers never raise for tool failures.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from ..llm.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage
from ..tools.base import AgentTool, AgentToolResult, validate_arguments
from .events import AgentEvent
from .loop import (
    AfterToolCallContext,
    AgentContext,
    AgentEventSink,
    AgentLoopConfig,
    BeforeToolCallContext,
    _maybe_await,
)

TRUNCATED_TOOL_CALL_MESSAGE = (
    'Tool call "{name}" was not executed: the response hit the output token '
    "limit, so its arguments may be truncated. Re-issue the tool call with "
    "complete arguments."
)


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
        return result, result.is_error
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
                    is_error=result.is_error,
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

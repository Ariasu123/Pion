"""Assistant response streaming — LLM event stream -> AgentEvent conversion.

Split out of `loop.py` (pure move). `_stream_assistant_response` streams one
assistant response, updating the context as partials arrive and emitting
`message_start`/`message_update`/`message_end` events.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

from ..llm.event_stream import AssistantMessageEventStream, StreamOptions
from ..llm.types import AssistantMessage, Context, Tool, sanitize_message
from .events import AgentEvent
from .loop import _maybe_await

if TYPE_CHECKING:
    from .loop import AgentContext, AgentEventSink, AgentLoopConfig, StreamFn


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

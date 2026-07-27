"""Stateful Agent — the public facade over the agent loop.

Holds the conversation (`messages`), the active tool list, and streaming
state; runs `run_agent_loop` for each prompt and re-broadcasts every
AgentEvent to subscribers. If an ExtensionManager is attached, the
extension hooks (before_agent_start / context / tool_call / tool_result /
agent_start / agent_end) are wired into the loop config here.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Optional

from ..hooks import ExtensionManager, ToolCallEvent, ToolResultEvent
from ..llm import stream_simple as _default_stream_fn
from ..llm.types import (
    AssistantMessage,
    Message,
    Model,
    UserMessage,
    sanitize_text,
)
from ..tools import DEFAULT_TOOLS
from ..tools.base import AgentTool
from .events import AgentEvent
from .loop import (
    AfterToolCallContext,
    AgentContext,
    AgentLoopConfig,
    BeforeToolCallContext,
    run_agent_loop,
)

# Subscriber callback: sync or async callable taking an AgentEvent.
EventHandler = Callable[[AgentEvent], Any]


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class Agent:
    """A coding agent: model + tools + messages + the loop."""

    def __init__(
        self,
        model: Model,
        tools: list[AgentTool] = DEFAULT_TOOLS,
        system_prompt: str = "",
        api_key: Optional[str] = None,
        stream_fn: Callable = _default_stream_fn,
        extension_manager: Optional[ExtensionManager] = None,
    ) -> None:
        self.model = model
        self.tools: list[AgentTool] = list(tools)
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.stream_fn = stream_fn
        self.extension_manager = extension_manager

        # Public state.
        self.messages: list[Message] = []
        self.is_streaming: bool = False
        self.pending_tool_calls: set[str] = set()
        self.error_message: Optional[str] = None
        # Exceptions raised by event subscribers (never break the loop).
        self.subscriber_errors: list[Exception] = []

        self._handlers: list[EventHandler] = []
        self._abort = asyncio.Event()

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def subscribe(self, handler: EventHandler) -> None:
        """Subscribe to AgentEvents; handler may be sync or async."""
        self._handlers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        """Remove a previously subscribed handler (no-op if absent)."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    # ------------------------------------------------------------------
    # Dynamic tool management
    # ------------------------------------------------------------------

    def add_tool(self, tool: AgentTool) -> None:
        """Add a tool; replaces an existing tool with the same name."""
        self.remove_tool(tool.name)
        self.tools.append(tool)

    def remove_tool(self, name: str) -> None:
        """Remove a tool by name (no-op if absent)."""
        self.tools = [t for t in self.tools if t.name != name]

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    async def prompt(self, text: str) -> AssistantMessage:
        """Send a user prompt and run the agent loop until it settles.

        Returns the final AssistantMessage of the run. New messages
        (including the prompt) are appended to `self.messages` so callers
        can persist them.
        """
        if self.is_streaming:
            raise RuntimeError("Agent is already streaming")
        self.is_streaming = True
        self.error_message = None
        self._abort.clear()
        try:
            system_prompt = self.system_prompt
            injected: list[Message] = []
            manager = self.extension_manager
            if manager is not None and manager.has_handlers("before_agent_start"):
                system_prompt, injected = await manager.run_before_agent_start(
                    prompt=text, system_prompt=system_prompt
                )

            prompts: list[Message] = [*injected, UserMessage(content=sanitize_text(text))]
            context = AgentContext(
                system_prompt=system_prompt,
                messages=list(self.messages),
                tools=self._all_tools(),
            )
            config = AgentLoopConfig(
                model=self.model,
                api_key=self.api_key,
                transform_context=self._hook_or_none("context"),
                before_tool_call=self._hook_or_none("tool_call"),
                after_tool_call=self._hook_or_none("tool_result"),
            )
            new_messages = await run_agent_loop(
                prompts, context, config, self._emit, self._abort, self.stream_fn
            )
            self.messages.extend(new_messages)

            final = next(
                (m for m in reversed(new_messages) if isinstance(m, AssistantMessage)), None
            )
            if final is None:  # pragma: no cover - the loop always streams at least once
                raise RuntimeError("Agent loop produced no assistant message")
            if final.stop_reason in ("error", "aborted"):
                self.error_message = final.error_message
            return final
        finally:
            self.is_streaming = False
            self.pending_tool_calls.clear()

    def abort(self) -> None:
        """Request abortion of the current run (honored between operations)."""
        self._abort.set()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _all_tools(self) -> list[AgentTool]:
        """Agent tools merged with extension-registered tools (by name)."""
        merged = {tool.name: tool for tool in self.tools}
        manager = self.extension_manager
        if manager is not None:
            for tool in manager.tools:
                merged[tool.name] = tool
        return list(merged.values())

    def _hook_or_none(self, event_name: str) -> Optional[Callable]:
        """Return the loop config hook for an extension event, if subscribed."""
        manager = self.extension_manager
        if manager is None or not manager.has_handlers(event_name):
            return None
        if event_name == "context":
            return manager.apply_context
        if event_name == "tool_call":
            return self._before_tool_call
        if event_name == "tool_result":
            return self._after_tool_call
        return None

    async def _before_tool_call(self, context: BeforeToolCallContext) -> Optional[dict]:
        """Adapt the loop's before_tool_call to extension "tool_call" handlers."""
        assert self.extension_manager is not None
        event = ToolCallEvent(
            tool_name=context.tool_call.name,
            tool_call_id=context.tool_call.id,
            args=context.tool_call.arguments,
        )
        return await self.extension_manager.run_tool_call(event)

    async def _after_tool_call(self, context: AfterToolCallContext) -> Optional[dict]:
        """Adapt the loop's after_tool_call to extension "tool_result" handlers."""
        assert self.extension_manager is not None
        event = ToolResultEvent(
            tool_name=context.tool_call.name,
            tool_call_id=context.tool_call.id,
            args=context.tool_call.arguments,
            content=context.result.content,
            details=context.result.details,
            is_error=context.is_error,
        )
        return await self.extension_manager.run_tool_result(event)

    async def _emit(self, event: AgentEvent) -> None:
        """Track state, notify subscribers, and forward lifecycle events to hooks."""
        if event.type == "tool_execution_start" and event.tool_call_id is not None:
            self.pending_tool_calls.add(event.tool_call_id)
        elif event.type == "tool_execution_end" and event.tool_call_id is not None:
            self.pending_tool_calls.discard(event.tool_call_id)

        for handler in list(self._handlers):
            try:
                await _maybe_await(handler(event))
            except Exception as exc:  # subscriber errors must not break the loop
                self.subscriber_errors.append(exc)

        manager = self.extension_manager
        if manager is not None:
            if event.type == "agent_start":
                await manager.notify("agent_start", {})
            elif event.type == "agent_end":
                await manager.notify("agent_end", {"messages": event.messages or []})

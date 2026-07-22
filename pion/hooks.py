"""Extension hook system — the soul of pi: primitives, not features.

Python port of pi's extension API surface (packages/coding-agent, see
docs/extensions.md), reduced to the event subset pion supports:

- before_agent_start     — may return {"system_prompt": str, "messages": [...]}
                           to replace the system prompt for this turn and/or
                           inject persistent messages before the user prompt.
                           Handlers are chained: each sees the system prompt
                           produced by the previous handlers.
- context                — fired before each LLM call. Handlers are chained:
                           each receives list[Message] and returns a replacement
                           list or None (keep current). Wired into the loop's
                           transform_context.
- tool_call              — fired after tool_execution_start, before the tool
                           executes. May return {"block": True, "reason": str}
                           to block execution. Wired into before_tool_call.
- tool_result            — fired after execution, before tool_execution_end.
                           Handlers chain like middleware: each sees the latest
                           result and may return a partial override dict
                           (content / details / is_error|isError / terminate).
                           Wired into after_tool_call.
- agent_start / agent_end / session_before_compact
                        — notification-only async handlers.

An extension is a plain `*.py` file exposing `setup(api)` (sync or async).
Handler errors are caught and collected into `ExtensionManager.errors`;
they never crash the agent.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .llm.types import Message
from .tools.base import AgentTool

KNOWN_EVENTS = (
    "before_agent_start",
    "context",
    "tool_call",
    "tool_result",
    "agent_start",
    "agent_end",
    "session_before_compact",
)


async def _maybe_await(value: Any) -> Any:
    """Await `value` if it is awaitable, otherwise return it as-is."""
    if inspect.isawaitable(value):
        return await value
    return value


# ---------------------------------------------------------------------------
# Event payloads passed to handlers
# ---------------------------------------------------------------------------


@dataclass
class BeforeAgentStartEvent:
    """Payload for "before_agent_start" handlers."""

    prompt: str
    system_prompt: str  # chained: includes earlier handlers' changes


@dataclass
class ToolCallEvent:
    """Payload for "tool_call" handlers."""

    tool_name: str
    tool_call_id: str
    args: dict[str, Any]  # raw tool-call arguments


@dataclass
class ToolResultEvent:
    """Payload for "tool_result" handlers.

    Chained: later handlers see the result after earlier handlers' overrides.
    """

    tool_name: str
    tool_call_id: str
    args: dict[str, Any]
    content: list = field(default_factory=list)
    details: Any = None
    is_error: bool = False


# ---------------------------------------------------------------------------
# Extension API
# ---------------------------------------------------------------------------


class ExtensionAPI:
    """The `api` object handed to each extension's `setup()`."""

    def __init__(self, manager: "ExtensionManager") -> None:
        self._manager = manager

    def on(self, event_name: str, handler: Callable) -> Callable:
        """Subscribe `handler` to an event. Returns the handler (decorator-friendly)."""
        self._manager.add_handler(event_name, handler)
        return handler

    def register_tool(self, tool: AgentTool) -> None:
        """Register (or replace, by name) a tool available to the agent."""
        self._manager.register_tool(tool)

    def register_command(self, name: str, command: Callable) -> None:
        """Register a slash command; `command` is a (sync or async) callable."""
        self._manager.commands[name] = command

    def remove_tool(self, name: str) -> None:
        """Remove a previously registered tool by name."""
        self._manager.remove_tool(name)


# ---------------------------------------------------------------------------
# Extension manager
# ---------------------------------------------------------------------------


class ExtensionManager:
    """Loads extensions, keeps handler/tool/command tables, runs hooks.

    Handler errors are caught and appended to `errors` — a broken extension
    must never crash the agent.
    """

    def __init__(self) -> None:
        self.handlers: dict[str, list[Callable]] = {name: [] for name in KNOWN_EVENTS}
        self.tools: list[AgentTool] = []
        self.commands: dict[str, Callable] = {}
        self.errors: list[Exception] = []
        self._api = ExtensionAPI(self)
        self._files: list[Path] = []
        self._load_counter = 0

    @property
    def api(self) -> ExtensionAPI:
        return self._api

    # ------------------------------------------------------------------
    # Registration (used by ExtensionAPI and tests)
    # ------------------------------------------------------------------

    def add_handler(self, event_name: str, handler: Callable) -> None:
        if event_name not in self.handlers:
            raise ValueError(
                f"Unknown extension event {event_name!r}; known events: {', '.join(KNOWN_EVENTS)}"
            )
        self.handlers[event_name].append(handler)

    def register_tool(self, tool: AgentTool) -> None:
        self.remove_tool(getattr(tool, "name", ""))
        self.tools.append(tool)

    def remove_tool(self, name: str) -> None:
        self.tools = [t for t in self.tools if getattr(t, "name", None) != name]

    def has_handlers(self, event_name: str) -> bool:
        return bool(self.handlers.get(event_name))

    # ------------------------------------------------------------------
    # Loading / reloading
    # ------------------------------------------------------------------

    async def load(self, extension_dirs: list[Path]) -> None:
        """Import every `*.py` file (sorted) under each dir and run its setup()."""
        self._files = [
            file for directory in extension_dirs for file in sorted(Path(directory).glob("*.py"))
        ]
        await self._load_all()

    async def reload(self) -> None:
        """Re-import all loaded extension files and re-run setup().

        Handler tables, commands and tools are rebuilt from scratch.
        """
        self.handlers = {name: [] for name in KNOWN_EVENTS}
        self.tools = []
        self.commands = {}
        await self._load_all()

    async def _load_all(self) -> None:
        for path in self._files:
            await self._load_file(path)

    async def _load_file(self, path: Path) -> None:
        """Import one extension file fresh and run its setup(api)."""
        self._load_counter += 1
        # A unique module name per import plus direct source compilation
        # bypasses both sys.modules and the .pyc cache, so reload() always
        # picks up file changes.
        module_name = f"_pion_extension_{path.stem}_{self._load_counter}"
        try:
            module = importlib.util.module_from_spec(
                importlib.machinery.ModuleSpec(module_name, loader=None)
            )
            module.__file__ = str(path)
            sys.modules[module_name] = module
            code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
            exec(code, module.__dict__)
            setup = getattr(module, "setup", None)
            if setup is None:
                raise AttributeError(f"Extension {path} does not define setup(api)")
            await _maybe_await(setup(self._api))
        except Exception as exc:
            self.errors.append(exc)

    # ------------------------------------------------------------------
    # Hook runners (wired into the Agent / loop)
    # ------------------------------------------------------------------

    async def run_before_agent_start(
        self, *, prompt: str, system_prompt: str
    ) -> tuple[str, list[Message]]:
        """Run before_agent_start handlers. Returns (system_prompt, injected messages)."""
        injected: list[Message] = []
        for handler in self.handlers["before_agent_start"]:
            try:
                result = await _maybe_await(
                    handler(BeforeAgentStartEvent(prompt=prompt, system_prompt=system_prompt))
                )
            except Exception as exc:
                self.errors.append(exc)
                continue
            if not result:
                continue
            replacement = result.get("system_prompt")
            if replacement:
                system_prompt = replacement
            messages = result.get("messages")
            if messages:
                injected.extend(messages)
        return system_prompt, injected

    async def apply_context(self, messages: list[Message]) -> list[Message]:
        """Run the chained "context" handlers over the outgoing message list."""
        current = messages
        for handler in self.handlers["context"]:
            try:
                result = await _maybe_await(handler(current))
            except Exception as exc:
                self.errors.append(exc)
                continue
            if result is not None:
                current = result
        return current

    async def run_tool_call(self, event: ToolCallEvent) -> Optional[dict]:
        """Run "tool_call" handlers; the first {"block": True} wins."""
        for handler in self.handlers["tool_call"]:
            try:
                result = await _maybe_await(handler(event))
            except Exception as exc:
                self.errors.append(exc)
                continue
            if result and result.get("block"):
                return result
        return None

    async def run_tool_result(self, event: ToolResultEvent) -> Optional[dict]:
        """Run chained "tool_result" handlers; returns the merged override dict."""
        merged: dict[str, Any] = {}
        for handler in self.handlers["tool_result"]:
            try:
                result = await _maybe_await(handler(event))
            except Exception as exc:
                self.errors.append(exc)
                continue
            if not result:
                continue
            merged.update(result)
            # Later handlers see the latest result after earlier overrides.
            if "content" in result:
                event.content = result["content"]
            if "details" in result:
                event.details = result["details"]
            if "is_error" in result:
                event.is_error = result["is_error"]
            if "isError" in result:
                event.is_error = result["isError"]
        return merged or None

    async def notify(self, event_name: str, payload: Optional[dict] = None) -> None:
        """Fire notification-only events (agent_start/agent_end/session_before_compact)."""
        for handler in self.handlers.get(event_name, []):
            try:
                await _maybe_await(handler(payload or {}))
            except Exception as exc:
                self.errors.append(exc)

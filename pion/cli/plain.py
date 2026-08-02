"""Plain line-oriented REPL and streaming output for the pion CLI."""

from __future__ import annotations

import asyncio
import inspect
import json
import signal
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape

from .. import __version__
from ..agent.events import AgentEvent
from ..controller import AgentSessionController
from ..llm.registry import ENV_KEY_NAMES
from ..llm.types import (
    AssistantMessage,
    Model,
    TextContent,
)
from ..session import should_compact
from ._shared import err_console


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without network or a REPL)
# ---------------------------------------------------------------------------


def parse_slash_command(text: str) -> tuple[str, str] | None:
    """Parse '/name args' into (name, args); None when not a slash command."""
    stripped = text.strip()
    if not stripped.startswith("/") or stripped == "/":
        return None
    parts = stripped[1:].split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    return name, args


def summarize_tool_args(name: str, args: dict[str, Any]) -> str:
    """One-line summary of a tool call: the most telling argument, single line."""
    for key in ("command", "path", "file_path"):
        if key in args and args[key] is not None:
            summary = str(args[key])
            break
    else:
        summary = json.dumps(args, ensure_ascii=False)
    summary = " ".join(summary.split())
    if len(summary) > 120:
        summary = summary[:117] + "..."
    return summary


def summarize_tool_result(text: str, limit: int = 200) -> str:
    """Short single-line summary of a tool result (first ~`limit` chars)."""
    summary = " ".join(text.split())
    if len(summary) > limit:
        summary = summary[: limit - 3] + "..."
    return summary


def api_key_env_name(model: Model) -> str:
    """Environment variable holding the API key for `model`'s provider."""
    return ENV_KEY_NAMES.get(model.provider, f"{model.provider.upper()}_API_KEY")


# ---------------------------------------------------------------------------
# Streaming renderer
# ---------------------------------------------------------------------------


class StreamRenderer:
    """Render AgentEvents to the console while a run streams.

    Assistant text prints as it arrives; thinking blocks render dim; tool
    calls get a one-line summary, tool results a short dim summary.
    """

    def __init__(self, out: Console) -> None:
        self.console = out
        self._open: str | None = None  # "text" | "thinking" line in progress

    def _close_line(self) -> None:
        if self._open is not None:
            self.console.print()
            self._open = None

    def handle(self, event: AgentEvent) -> None:
        """Subscriber callback for Agent.subscribe (sync variant)."""
        if event.type == "message_update" and event.assistant_event is not None:
            ev = event.assistant_event
            if ev.type == "thinking_delta" and ev.delta:
                if self._open != "thinking":
                    self._close_line()
                    self._open = "thinking"
                self.console.print(
                    ev.delta, end="", style="dim italic", markup=False, highlight=False
                )
            elif ev.type == "text_delta" and ev.delta:
                if self._open != "text":
                    self._close_line()
                    self._open = "text"
                self.console.print(ev.delta, end="", markup=False, highlight=False)
        elif event.type == "message_end":
            if isinstance(event.message, AssistantMessage):
                self._close_line()
        elif event.type == "tool_execution_start":
            self._close_line()
            name = event.tool_name or "tool"
            summary = escape(summarize_tool_args(name, event.args or {}))
            self.console.print(f"[cyan]● {escape(name)}[/cyan] [dim]{summary}[/dim]")
        elif event.type == "tool_execution_end":
            self._close_line()
            text = ""
            if event.result is not None:
                text = "".join(
                    block.text
                    for block in event.result.content
                    if isinstance(block, TextContent)
                )
            summary = escape(summarize_tool_result(text))
            style = "red" if event.is_error else "dim"
            self.console.print(f"  ⎿ {summary}", style=style, markup=False)


# ---------------------------------------------------------------------------
# The REPL / single-shot runner
# ---------------------------------------------------------------------------


class Repl:
    """Interactive session: agent + session store + extensions + console."""

    def __init__(
        self,
        controller: AgentSessionController,
        out: Console,
    ) -> None:
        self.controller = controller
        self.agent = controller.agent
        self.session = controller.session
        self.session_path = controller.session_path
        self.extensions = controller.extensions
        self.console = out

    # ------------------------------------------------------------------
    # Running prompts
    # ------------------------------------------------------------------

    async def _prompt_with_sigint(self, text: str) -> AssistantMessage:
        """Run one prompt; Ctrl-C during the run aborts the agent."""
        loop = asyncio.get_running_loop()
        installed = False
        try:
            loop.add_signal_handler(signal.SIGINT, self.controller.abort)
            installed = True
        except (NotImplementedError, RuntimeError, ValueError):
            # Platform without signal handlers (or not the main thread):
            # Ctrl-C falls back to interrupting the whole process.
            installed = False
        try:
            return await self.controller.prompt(text)
        finally:
            if installed:
                loop.remove_signal_handler(signal.SIGINT)

    async def handle_prompt(self, text: str, render: bool = True) -> AssistantMessage:
        """Run one prompt, persist new messages, auto-compact afterwards."""
        renderer = StreamRenderer(self.console) if render else None
        if renderer is not None:
            self.agent.subscribe(renderer.handle)
        try:
            final = await self._prompt_with_sigint(text)
        finally:
            if renderer is not None:
                self.agent.unsubscribe(renderer.handle)
        if render:
            if final.stop_reason == "error":
                self.console.print(
                    f"[red]Error: {escape(final.error_message or 'unknown error')}[/red]"
                )
            elif final.stop_reason == "aborted":
                self.console.print("[dim]Aborted.[/dim]")

        return final

    async def maybe_compact(self, force: bool = False) -> None:
        """Compact the session when the context crosses the threshold.

        Failures are reported but never crash the REPL.
        """
        if not self.agent.messages:
            if force:
                self.console.print("[dim]Nothing to compact yet.[/dim]")
            return
        if not force and not should_compact(self.agent.messages, self.agent.model):
            return
        try:
            summary = await self.controller.maybe_compact(force=force)
            if summary is None:
                return
            self.console.print(
                f"[dim]Context compacted — summary of {len(summary)} chars, "
                f"{len(self.agent.messages)} messages kept in context.[/dim]"
            )
        except Exception as exc:  # never crash the REPL over compaction
            self.console.print(
                f"[yellow]Compaction failed (continuing without it): "
                f"{escape(str(exc))}[/yellow]"
            )

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _print_help(self) -> None:
        self.console.print("[bold]Commands[/bold]")
        rows = [
            ("/help", "show this help"),
            (
                "/model <id>",
                "switch model (re-resolves the API key); no arg shows current",
            ),
            ("/compact", "force context compaction now"),
            ("/stats", "token usage and cost of the last response"),
            ("/exit", "quit (also Ctrl-D / EOF)"),
        ]
        for name, description in rows:
            self.console.print(f"  [cyan]{name:<14}[/cyan] {description}")
        if self.extensions is not None and self.extensions.commands:
            self.console.print("[bold]Extension commands[/bold]")
            for name in sorted(self.extensions.commands):
                self.console.print(f"  [cyan]/{name}[/cyan]")

    def _print_stats(self) -> None:
        usage = self.controller.last_usage
        if usage is None:
            self.console.print("[dim]No usage yet — run a prompt first.[/dim]")
            return
        cost = usage.cost.total or self.agent.model.compute_cost(usage).total
        self.console.print(
            f"[dim]last usage ({self.agent.model.id}): "
            f"input {usage.input}, output {usage.output}, "
            f"cache read {usage.cache_read}, cache write {usage.cache_write} — "
            f"cost ${cost:.6f}[/dim]"
        )

    def _switch_model(self, args: str) -> None:
        model_id = args.strip()
        if not model_id:
            self.console.print(f"[dim]Current model: {self.agent.model.id}[/dim]")
            return
        try:
            self.controller.switch_model(model_id)
        except KeyError as exc:
            self.console.print(f"[red]{escape(str(exc.args[0]))}[/red]")
            return
        model = self.agent.model
        self.console.print(f"[dim]Switched to {model.id} ({model.name}).[/dim]")
        if self.agent.api_key is None:
            self.console.print(
                f"[yellow]No API key found — set {api_key_env_name(model)} "
                f"or restart with --api-key.[/yellow]"
            )

    async def handle_slash(self, name: str, args: str) -> bool:
        """Handle one slash command. Returns True when the REPL should exit."""
        if name in ("exit", "quit"):
            return True
        if name == "help":
            self._print_help()
        elif name == "model":
            self._switch_model(args)
        elif name == "compact":
            await self.maybe_compact(force=True)
        elif name == "stats":
            self._print_stats()
        elif self.extensions is not None and name in self.extensions.commands:
            command = self.extensions.commands[name]
            result = command()
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                self.console.print(str(result))
        else:
            self.console.print(
                f"[red]Unknown command: /{escape(name)}[/red] — try /help"
            )
        return False

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        self.console.print(
            f"[bold]pion[/bold] {__version__} — model [cyan]{escape(self.agent.model.id)}[/cyan]"
        )
        self.console.print(
            f"[dim]cwd: {Path.cwd()}  session: {self.session_path}[/dim]"
        )
        self.console.print("[dim]/help for commands, /exit or Ctrl-D to quit[/dim]")

    async def run(self) -> None:
        """Interactive input()-based REPL loop."""
        self._print_banner()
        while True:
            try:
                line = input("pion> ")
            except EOFError:
                self.console.print("\n[dim]Bye.[/dim]")
                return
            line = line.strip()
            if not line:
                continue
            parsed = parse_slash_command(line)
            if parsed is not None:
                try:
                    should_exit = await self.handle_slash(*parsed)
                except Exception as exc:
                    self.console.print(f"[red]Command failed: {escape(str(exc))}[/red]")
                    should_exit = False
                if should_exit:
                    return
                continue
            try:
                await self.handle_prompt(line)
            except Exception as exc:  # a failed prompt must not kill the REPL
                self.console.print(f"[red]Error: {escape(str(exc))}[/red]")

    async def run_print(self, text: str) -> None:
        """Single-shot mode: run one prompt, print the final text, exit."""
        final = await self.handle_prompt(text, render=False)
        if final.stop_reason in ("error", "aborted"):
            err_console.print(
                f"[red]Error: {escape(final.error_message or final.stop_reason)}[/red]"
            )
            raise typer.Exit(1)
        sys.stdout.write(final.text() + "\n")

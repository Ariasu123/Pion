"""PionTUI: assembles the inline TUI around an AgentSessionController.

Layout (top→bottom, all inline in the terminal's main screen):
startup header → chat transcript → status slot (spinner) → editor → footer.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ... import __version__
from ...agent.events import AgentEvent
from ...controller import AgentSessionController, ControllerEvent
from ...llm.registry import list_models
from ...llm.types import (
    AssistantMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
)
from ..components import (
    CombinedAutocompleteProvider,
    Editor,
    Loader,
    SelectItem,
    SelectList,
    SlashCommand,
    Text,
)
from ..core.component import Component, Container
from ..core.keys import KeyDecoder
from ..core.renderer import InlineRenderer
from ..core.terminal import ProcessTerminal, Terminal
from ..theme import Theme, list_themes, load_theme, set_theme
from .chat import (
    AssistantMessageComponent,
    Notice,
    ToolExecutionComponent,
    UserMessageComponent,
    message_text,
)
from .footer import Footer, _context_tokens
from .queue import MessageQueue
from .selectors import OverlayPanel, TextInput, TreeSelector


@dataclass(frozen=True)
class TUIStatus:
    project: str
    sandbox: str
    mcp_servers: int = 0
    mcp_tools: int = 0


BUILTIN_COMMANDS = [
    SlashCommand("help", "Show available commands"),
    SlashCommand("model", "Switch model"),
    SlashCommand("compact", "Compact the session context"),
    SlashCommand("stats", "Show usage and cost"),
    SlashCommand("tree", "Navigate the session tree"),
    SlashCommand("theme", "Switch theme: /theme dark|light"),
    SlashCommand("exit", "Quit pion"),
]


class _Slot(Component):
    """A placeholder rendering its current child (or nothing)."""

    def __init__(self) -> None:
        self.child: Component | None = None

    def render(self, width: int) -> list[str]:
        return self.child.render(width) if self.child is not None else []

    def invalidate(self) -> None:
        if self.child is not None:
            self.child.invalidate()


class PionTUI:
    def __init__(
        self,
        controller: AgentSessionController,
        status: TUIStatus,
        theme: Theme | None = None,
        terminal: Terminal | None = None,
    ) -> None:
        self.controller = controller
        self.status = status
        self.theme = theme or load_theme("dark")
        set_theme(self.theme)
        self.terminal = terminal or ProcessTerminal()

        self.queue = MessageQueue()
        self.show_thinking = True
        self.tools_expanded = False

        # Layout.
        self.chat = Container()
        self._slot = _Slot()
        self._selector_slot = _Slot()  # inline selector above the editor
        self.editor = Editor(
            on_submit=self._submit,
            autocomplete=CombinedAutocompleteProvider(self._slash_commands(), Path.cwd()),
            theme=self.theme,
        )
        self.footer = Footer(
            controller,
            session_name=controller.session_path.name,
            queued=lambda: len(self.queue),
            theme=self.theme,
        )
        self.root = Container(
            [
                self._build_header(),
                self.chat,
                self._slot,
                self._selector_slot,
                self.editor,
                self.footer,
            ]
        )
        self.renderer = InlineRenderer(self.terminal, self.root)
        self._decoder = KeyDecoder()
        self._flush_handle: asyncio.TimerHandle | None = None

        self._active_assistant: AssistantMessageComponent | None = None
        self._tools: dict[str, ToolExecutionComponent] = {}
        self._exit = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        self._ticker: asyncio.Task | None = None
        self._selector: Component | None = None
        self._tree_selector: TreeSelector | None = None
        self._compaction_notice: Notice | None = None

    # -- construction helpers ------------------------------------------------

    def _slash_commands(self) -> list[SlashCommand]:
        commands = list(BUILTIN_COMMANDS)
        extensions = self.controller.extensions
        if extensions is not None:
            commands.extend(
                SlashCommand(name, "extension command", "extension")
                for name in sorted(extensions.commands)
            )
        return commands

    def _build_header(self) -> Container:
        theme = self.theme
        title = theme.styled("pion", "accent", bold=True) + theme.fg(
            "dim", f" v{__version__}"
        )
        details = f"{self.controller.agent.model.id} · {self.status.project} · sandbox {self.status.sandbox}"
        if self.status.mcp_servers:
            details += (
                f" · mcp {self.status.mcp_servers} server/"
                f"{self.status.mcp_tools} tools"
            )
        return Container(
            [
                Text("", pad_x=0),
                Text(title, pad_x=1, theme=theme),
                Text(details, pad_x=1, fg="dim", theme=theme),
                Text("/help for commands · ctrl+p palette · ctrl+b tree", pad_x=1, fg="dim", theme=theme),
            ]
        )

    # -- lifecycle --------------------------------------------------------------

    async def run_async(self) -> None:
        loop = asyncio.get_running_loop()
        self.renderer._loop = loop
        self.controller.agent.subscribe(self._on_agent_event)
        self.controller.subscribe(self._on_controller_event)
        self._render_history()
        self.terminal.start(self._on_bytes, self.renderer.on_resize)
        self.terminal.set_title(f"pion · {self.status.project}")
        self.renderer.render_now()
        try:
            await self._exit.wait()
        finally:
            self.controller.abort()
            for task in self._tasks:
                task.cancel()
            if self._ticker is not None:
                self._ticker.cancel()
            self.controller.agent.unsubscribe(self._on_agent_event)
            self.controller.unsubscribe(self._on_controller_event)
            self.renderer.close()
            self.terminal.stop()

    def quit(self) -> None:
        self._exit.set()

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # -- input ------------------------------------------------------------------

    def _on_bytes(self, data: bytes) -> None:
        for event in self._decoder.feed(data):
            self._dispatch(event)
        if self._flush_handle is not None:
            self._flush_handle.cancel()
        loop = asyncio.get_running_loop()
        self._flush_handle = loop.call_later(0.03, self._flush_keys)

    def _flush_keys(self) -> None:
        self._flush_handle = None
        for event in self._decoder.flush():
            self._dispatch(event)

    def _dispatch(self, key) -> None:
        overlay = self.renderer.overlays.top
        if self._selector is not None:
            self._selector.handle_input(key)
        elif overlay is not None:
            overlay.component.handle_input(key)
        elif key.key == "ctrl+q" or key.key == "ctrl+d" and not self.editor.text:
            self.quit()
        elif key.key == "ctrl+c":
            self.editor.clear()
        elif key.key == "escape":
            if self.editor.autocomplete_open:
                self.editor.dismiss_autocomplete()
            elif self.controller.is_busy:
                self.controller.abort()
        elif key.key == "ctrl+o":
            self.tools_expanded = not self.tools_expanded
        elif key.key == "ctrl+t":
            self.show_thinking = not self.show_thinking
        elif key.key == "ctrl+l":
            self.open_model_selector()
        elif key.key == "ctrl+b":
            self.open_tree()
        elif key.key == "ctrl+p":
            self.open_command_palette()
        elif key.key == "alt+enter":
            text = self.editor.text.strip()
            if text:
                self.queue.enqueue_followup(text)
                self.editor.clear()
        elif key.key == "alt+up":
            message = self.queue.pop_back()
            if message is not None:
                self.editor.set_text(message)
        else:
            self.editor.handle_input(key)
        self.renderer.request_render()

    # -- prompt / queue ---------------------------------------------------------

    def _submit(self, text: str) -> None:
        if text.startswith("/"):
            self._spawn(self._execute_command(text[1:]))
        elif self.controller.is_busy:
            self.queue.enqueue_steer(text)
            self.renderer.request_render()
        else:
            self._spawn(self._run_prompt(text))

    async def _run_prompt(self, text: str) -> None:
        self._set_busy("Working…")
        try:
            final = await self.controller.prompt(text)
            if final.stop_reason == "error":
                self.notice(
                    f"Error: {final.error_message or 'request failed'}", fg="error"
                )
        except Exception as exc:
            self.notice(f"Prompt failed: {exc}", fg="error")
        finally:
            self._set_busy(None)
            self.renderer.request_render()
            follow_up = self.queue.pop_next()
            if follow_up is not None:
                self._spawn(self._run_prompt(follow_up))

    def _set_busy(self, message: str | None) -> None:
        if message is None:
            self._slot.child = None
            if self._ticker is not None:
                self._ticker.cancel()
                self._ticker = None
        else:
            self._slot.child = Loader(message, theme=self.theme)
            if self._ticker is None or self._ticker.done():
                self._ticker = asyncio.ensure_future(self._tick())
        self.renderer.request_render()

    async def _tick(self) -> None:
        try:
            while True:
                await asyncio.sleep(0.1)
                self.renderer.request_render()
        except asyncio.CancelledError:
            pass

    # -- transcript ------------------------------------------------------------

    def notice(self, text: str, fg: str = "dim", glyph: str = "") -> Notice:
        note = Notice(text, fg=fg, glyph=glyph, theme=self.theme)
        self.chat.add(note)
        self.renderer.request_render()
        return note

    def _render_history(self) -> None:
        self.chat.clear()
        self._active_assistant = None
        self._tools = {}
        call_args: dict[str, dict[str, Any]] = {}
        for message in self.controller.agent.messages:
            if isinstance(message, UserMessage):
                self.chat.add(
                    UserMessageComponent(message_text(message), theme=self.theme)
                )
            elif isinstance(message, AssistantMessage):
                for tool_call in message.tool_calls():
                    call_args[tool_call.id] = tool_call.arguments
                if message.text() or any(
                    block.type == "thinking" for block in message.content
                ):
                    component = AssistantMessageComponent(
                        lambda: self.show_thinking, theme=self.theme
                    )
                    component.finalize(message)
                    self.chat.add(component)
            elif isinstance(message, ToolResultMessage):
                component = ToolExecutionComponent(
                    message.tool_call_id,
                    message.tool_name,
                    call_args.get(message.tool_call_id, {}),
                    lambda: self.tools_expanded,
                    theme=self.theme,
                )
                component.update_result(message.text(), message.is_error)
                self.chat.add(component)
        self.renderer.request_render()

    # -- agent / controller events --------------------------------------------

    def _on_agent_event(self, event: AgentEvent) -> None:
        if event.type == "message_start":
            if isinstance(event.message, AssistantMessage):
                self._active_assistant = AssistantMessageComponent(
                    lambda: self.show_thinking, streaming=True, theme=self.theme
                )
                self.chat.add(self._active_assistant)
            elif isinstance(event.message, UserMessage):
                self.chat.add(
                    UserMessageComponent(
                        message_text(event.message), theme=self.theme
                    )
                )
        elif event.type == "message_update" and event.assistant_event is not None:
            stream_event = event.assistant_event
            delta = getattr(stream_event, "delta", None)
            if self._active_assistant is not None and delta:
                if stream_event.type == "text_delta":
                    self._active_assistant.append_text(delta)
                elif stream_event.type == "thinking_delta":
                    self._active_assistant.append_thinking(delta)
        elif event.type == "message_end" and isinstance(
            event.message, AssistantMessage
        ):
            if self._active_assistant is not None:
                self._active_assistant.finalize(event.message)
                self._active_assistant = None
        elif event.type == "tool_execution_start" and event.tool_call_id:
            component = ToolExecutionComponent(
                event.tool_call_id,
                event.tool_name or "tool",
                event.args,
                lambda: self.tools_expanded,
                running=True,
                theme=self.theme,
            )
            self._tools[event.tool_call_id] = component
            self.chat.add(component)
        elif event.type == "tool_execution_update" and event.tool_call_id:
            component = self._tools.get(event.tool_call_id)
            if component is not None and event.partial_result is not None:
                component.update_progress(
                    self._result_text(event.partial_result.content)
                )
        elif event.type == "tool_execution_end" and event.tool_call_id:
            component = self._tools.get(event.tool_call_id)
            if component is not None:
                text = (
                    self._result_text(event.result.content) if event.result else ""
                )
                component.update_result(text, bool(event.is_error))
        self.renderer.request_render()

    def _on_controller_event(self, event: ControllerEvent) -> None:
        if event.type == "session_changed":
            if self._tree_selector is not None:
                self._tree_selector.refresh()
        elif event.type == "compaction_started":
            self._compaction_notice = self.notice("Compacting context…", glyph="◈")
        elif event.type == "compaction_finished":
            count = event.data.get("message_count", "?")
            if self._compaction_notice is not None:
                self._compaction_notice.set_text(
                    f"◈ Context compacted ({count} messages kept)"
                )
                self._compaction_notice = None
            else:
                self.notice(f"Context compacted ({count} messages kept)", glyph="◈")
        elif event.type == "error":
            self.notice(str(event.data.get("message", "Operation failed")), fg="error")
        self.renderer.request_render()

    @staticmethod
    def _result_text(content: list[Any]) -> str:
        texts = [block.text for block in content if isinstance(block, TextContent)]
        if texts:
            return "".join(texts)
        return "[non-text tool result]" if content else ""

    # -- selectors (inline above the editor) ---------------------------------

    def _open_selector(self, panel: Component) -> None:
        """Show a selector panel in the slot directly above the editor."""
        self._selector = panel
        self._selector_slot.child = panel
        self.renderer.request_render()

    def _close_selector(self) -> None:
        self._selector = None
        self._selector_slot.child = None
        self.renderer.request_render()

    def open_choice(
        self,
        title: str,
        items: list[tuple[str, str, str]],
        on_done,  # Callable[[str | None], None]
    ) -> None:
        def close(value: str | None) -> None:
            self._close_selector()
            on_done(value)

        select = SelectList(
            [SelectItem(value, label, description) for value, label, description in items],
            on_select=lambda value: close(value),
            on_cancel=lambda: close(None),
            theme=self.theme,
        )
        self._open_selector(OverlayPanel(title, select, theme=self.theme))

    def open_text_input(
        self,
        title: str,
        initial: str,
        placeholder: str,
        on_done,  # Callable[[str | None], None]
    ) -> None:
        def close(value: str | None) -> None:
            self._close_selector()
            on_done(value)

        input_ = TextInput(
            initial,
            placeholder,
            on_submit=lambda value: close(value),
            on_cancel=lambda: close(None),
            theme=self.theme,
        )
        self._open_selector(OverlayPanel(title, input_, theme=self.theme))

    def open_model_selector(self) -> None:
        models = {model.id: model for model in list_models()}
        current = self.controller.agent.model
        models[current.id] = current
        items = [
            (key, model.name or key, key) for key, model in sorted(models.items())
        ]

        def done(value: str | None) -> None:
            if value is None:
                return
            try:
                self.controller.switch_model(value)
            except KeyError as exc:
                self.notice(str(exc.args[0]), fg="error")
                return
            self._spawn(self.controller.notify_model_changed())
            self.notice(f"Model: {self.controller.agent.model.id}")

        self.open_choice("Select model", items, done)

    def open_command_palette(self) -> None:
        items = [
            (command.name, f"/{command.name}", command.description)
            for command in self._slash_commands()
        ]

        def done(value: str | None) -> None:
            if value is not None:
                self._spawn(self._execute_command(value))

        self.open_choice("Commands", items, done)

    # -- session tree -----------------------------------------------------------

    def open_tree(self) -> None:
        if self._tree_selector is not None:
            return
        selector = TreeSelector(
            self.controller.session,
            on_navigate=self._tree_navigate,
            on_label=self._tree_label,
            on_close=self._close_tree,
            theme=self.theme,
        )
        self._tree_selector = selector
        self._reopen_tree_panel()

    def _reopen_tree_panel(self) -> None:
        """Show the tree panel again after a sub-dialog is dismissed."""
        if self._tree_selector is not None:
            self._open_selector(
                OverlayPanel("Session tree", self._tree_selector, theme=self.theme)
            )

    def _close_tree(self) -> None:
        if self._tree_selector is not None:
            self._tree_selector = None
            self._close_selector()

    def _tree_navigate(self, entry_id: str) -> None:
        if self.controller.is_busy:
            self.notice("Wait for the current turn or press Esc to abort", fg="warning")
            return
        if not self.controller.would_abandon_suffix(entry_id):
            self._spawn(self._navigate(entry_id))
            return
        items = [
            ("plain", "Navigate without summary", "abandoned messages are dropped"),
            ("summary", "Summarize and navigate", "default branch summary"),
            ("custom", "Summarize with custom focus", "add instructions first"),
        ]

        def done(value: str | None) -> None:
            if value is None:
                self._reopen_tree_panel()
                return
            if value == "plain":
                self._spawn(self._navigate(entry_id))
            elif value == "summary":
                self._spawn(self._navigate(entry_id, summarize=True))
            else:

                def instructions_done(text: str | None) -> None:
                    if text is None:
                        self._reopen_tree_panel()
                    else:
                        self._spawn(
                            self._navigate(
                                entry_id, summarize=True, instructions=text
                            )
                        )

                self.open_text_input(
                    "Summary focus", "", "what should the summary keep?", instructions_done
                )

        self.open_choice("Branch abandons the current suffix", items, done)

    async def _navigate(
        self,
        entry_id: str,
        summarize: bool = False,
        instructions: str | None = None,
    ) -> None:
        self._close_tree()
        if summarize:
            self._set_busy("Summarizing branch…")
        try:
            result = await self.controller.navigate_tree(
                entry_id,
                summarize=summarize,
                custom_instructions=instructions,
            )
            self._render_history()
            if result.editor_text is not None:
                self.editor.set_text(result.editor_text)
        except Exception as exc:
            self.notice(f"Tree navigation failed: {exc}", fg="error")
        finally:
            if summarize:
                self._set_busy(None)
        self.renderer.request_render()

    def _tree_label(self, entry_id: str) -> None:
        current = self.controller.session.get_label(entry_id) or ""

        def done(value: str | None) -> None:
            if value is None:
                self._reopen_tree_panel()
                return

            async def apply() -> None:
                try:
                    await self.controller.set_label(entry_id, value or None)
                except Exception as exc:
                    self.notice(f"Could not update label: {exc}", fg="error")
                if self._tree_selector is not None:
                    self._tree_selector.refresh()
                self._reopen_tree_panel()
                self.renderer.request_render()

            self._spawn(apply())

        self.open_text_input("Entry label", current, "checkpoint name", done)

    # -- slash commands ---------------------------------------------------------

    async def _execute_command(self, raw: str) -> None:
        parts = raw.strip().split(maxsplit=1)
        name = parts[0].lower() if parts else ""
        args = parts[1].strip() if len(parts) > 1 else ""
        if name in ("exit", "quit"):
            self.quit()
        elif name == "help":
            lines = [
                f"/{command.name} — {command.description}"
                for command in self._slash_commands()
            ]
            lines.append("enter send · ctrl+j newline · esc abort")
            lines.append("ctrl+o expand tools · ctrl+t thinking · ctrl+l model")
            self.notice("\n".join(lines))
        elif name == "tree":
            self.open_tree()
        elif name == "model":
            if args:
                try:
                    self.controller.switch_model(args)
                    self._spawn(self.controller.notify_model_changed())
                    self.notice(f"Model: {self.controller.agent.model.id}")
                except KeyError as exc:
                    self.notice(str(exc.args[0]), fg="error")
            else:
                self.open_model_selector()
        elif name == "compact":
            try:
                summary = await self.controller.maybe_compact(force=True)
                if summary is None:
                    self.notice("Nothing to compact")
            except Exception as exc:
                self.notice(f"Compaction failed: {exc}", fg="error")
        elif name == "stats":
            self._show_stats()
        elif name == "theme":
            self._switch_theme(args)
        elif (
            self.controller.extensions is not None
            and name in self.controller.extensions.commands
        ):
            try:
                result = self.controller.extensions.commands[name]()
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    self.notice(str(result))
            except Exception as exc:
                self.notice(f"Command failed: {exc}", fg="error")
        else:
            self.notice(f"Unknown command: /{name}", fg="error")

    def _show_stats(self) -> None:
        usage = self.controller.last_usage
        tokens = _context_tokens(self.controller)
        context = min(
            100,
            round(tokens / max(1, self.controller.agent.model.context_window) * 100),
        )
        usage_text = "usage unavailable · cost unavailable"
        if usage is not None:
            cost = (
                usage.cost.total
                or self.controller.agent.model.compute_cost(usage).total
            )
            usage_text = (
                f"input {usage.input} · output {usage.output} · cost ${cost:.6f}"
            )
        self.notice(
            f"model {self.controller.agent.model.id} · "
            f"sandbox {self.status.sandbox} · "
            f"MCP {self.status.mcp_servers} server/{self.status.mcp_tools} tool · "
            f"context {context}% · {usage_text}"
        )

    def _switch_theme(self, name: str) -> None:
        if not name:
            self.notice(f"Theme: {self.theme.name} (available: {', '.join(list_themes())})")
            return
        try:
            self.theme = load_theme(name)
        except ValueError:
            self.notice(
                f"Unknown theme {name!r}; available: {', '.join(list_themes())}",
                fg="error",
            )
            return
        set_theme(self.theme)
        self.renderer.invalidate()


__all__ = ["PionTUI", "TUIStatus"]

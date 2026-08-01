"""Full-screen Textual application for Pion."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Click, Resize
from textual.message import Message as TextualMessage
from textual.widgets import Static

from ..agent.events import AgentEvent
from ..controller import AgentSessionController, ControllerEvent
from ..llm.registry import list_models
from ..llm.types import AssistantMessage, TextContent, ToolResultMessage
from ..session import estimate_tokens
from .screens import BranchDecision, BranchScreen, ChoiceScreen, TextInputScreen
from .theme import PION_DARK
from .widgets import (
    AssistantCard,
    ConversationView,
    HeaderBar,
    PromptComposer,
    PromptEditor,
    SessionTreePanel,
    ToolCard,
)


@dataclass(frozen=True)
class TUIStatus:
    project: str
    sandbox: str
    mcp_servers: int = 0
    mcp_tools: int = 0


class AgentEventMessage(TextualMessage):
    def __init__(self, event: AgentEvent) -> None:
        super().__init__()
        self.event = event


class ControllerEventMessage(TextualMessage):
    def __init__(self, event: ControllerEvent) -> None:
        super().__init__()
        self.event = event


class PionApp(App[None]):
    """Chat-centered Pion TUI with a first-class session tree."""

    CSS_PATH = "pion.tcss"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        ("ctrl+p", "command_palette", "Commands"),
        ("ctrl+b", "toggle_tree", "Session tree"),
        ("escape", "abort", "Abort"),
        ("ctrl+q", "quit_pion", "Quit"),
    ]

    def __init__(
        self,
        controller: AgentSessionController,
        status: TUIStatus,
    ) -> None:
        super().__init__()
        self.register_theme(PION_DARK)
        self.theme = PION_DARK.name
        self.controller = controller
        self.runtime_status = status
        self._active_assistant: AssistantCard | None = None
        self._tool_cards: dict[str, ToolCard] = {}
        self._tree_open = False

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="app-header")
        with Horizontal(id="workspace"):
            yield Static(id="tree-scrim")
            yield SessionTreePanel(
                self.controller.session,
                session_name=self.controller.session_path.name,
                id="session-tree",
            )
            with Vertical(id="main-pane"):
                yield ConversationView(id="conversation")
                yield PromptComposer(id="composer")

    async def on_mount(self) -> None:
        self.controller.agent.subscribe(self._receive_agent_event)
        self.controller.subscribe(self._receive_controller_event)
        await self._render_history()
        self._refresh_status()
        self.query_one(PromptEditor).focus()
        self._apply_responsive_layout(self.size.width)

    def on_unmount(self) -> None:
        self.controller.agent.unsubscribe(self._receive_agent_event)
        self.controller.unsubscribe(self._receive_controller_event)

    def _receive_agent_event(self, event: AgentEvent) -> None:
        self.post_message(AgentEventMessage(event))

    def _receive_controller_event(self, event: ControllerEvent) -> None:
        self.post_message(ControllerEventMessage(event))

    async def _render_history(self) -> None:
        view = self.query_one(ConversationView)
        await view.clear_content()
        self._active_assistant = None
        self._tool_cards.clear()
        rendered = False
        call_args: dict[str, dict[str, Any]] = {}
        for message in self.controller.agent.messages:
            if isinstance(message, AssistantMessage):
                for tool_call in message.tool_calls():
                    call_args[tool_call.id] = tool_call.arguments
                rendered = await view.add_message(message) is not None or rendered
            elif isinstance(message, ToolResultMessage):
                card = ToolCard(
                    message.tool_call_id,
                    message.tool_name,
                    call_args.get(message.tool_call_id),
                    result=message.text(),
                    is_error=message.is_error,
                )
                await view.mount_content(card)
                rendered = True
            else:
                rendered = await view.add_message(message) is not None or rendered
        if not rendered:
            await view.show_empty(self.runtime_status.project, Path.cwd())
        view.scroll_end(animate=False)

    @on(AgentEventMessage)
    async def handle_agent_event(self, message: AgentEventMessage) -> None:
        event = message.event
        view = self.query_one(ConversationView)
        follow_output = view.is_vertical_scroll_end
        if event.type == "message_start":
            if isinstance(event.message, AssistantMessage):
                card = AssistantCard(
                    streaming=True,
                    classes="chat-message assistant-message",
                )
                await view.mount_content(card)
                self._active_assistant = card
            elif event.message is not None and not isinstance(
                event.message, ToolResultMessage
            ):
                await view.add_message(event.message)
        elif event.type == "message_update" and event.assistant_event is not None:
            stream_event = event.assistant_event
            if self._active_assistant is not None and stream_event.delta:
                if stream_event.type == "text_delta":
                    self._active_assistant.append_delta("text", stream_event.delta)
                elif stream_event.type == "thinking_delta":
                    self._active_assistant.append_delta("thinking", stream_event.delta)
        elif event.type == "message_end" and isinstance(
            event.message, AssistantMessage
        ):
            if self._active_assistant is not None:
                await self._active_assistant.set_message(event.message)
            self._active_assistant = None
        elif event.type == "tool_execution_start" and event.tool_call_id:
            card = ToolCard(
                event.tool_call_id,
                event.tool_name or "tool",
                event.args,
                running=True,
            )
            self._tool_cards[event.tool_call_id] = card
            await view.mount_content(card)
        elif event.type == "tool_execution_update" and event.tool_call_id:
            card = self._tool_cards.get(event.tool_call_id)
            if card is not None and event.partial_result is not None:
                card.update_progress(self._result_text(event.partial_result.content))
        elif event.type == "tool_execution_end" and event.tool_call_id:
            card = self._tool_cards.get(event.tool_call_id)
            if card is not None:
                text = self._result_text(event.result.content) if event.result else ""
                card.update_result(text, bool(event.is_error))
        if follow_output:
            view.scroll_end(animate=False)
        self._refresh_status()

    @on(ControllerEventMessage)
    def handle_controller_event(self, message: ControllerEventMessage) -> None:
        event = message.event
        if event.type == "session_changed":
            self.query_one(SessionTreePanel).refresh_tree()
        elif event.type == "error":
            self.notify(
                str(event.data.get("message", "Operation failed")), severity="error"
            )
        elif event.type == "compaction_started":
            self.notify("Compacting session context…")
        elif event.type == "compaction_finished":
            self.notify("Session context compacted")
        self._refresh_status()

    @staticmethod
    def _result_text(content: list[Any]) -> str:
        texts = [block.text for block in content if isinstance(block, TextContent)]
        if texts:
            return "".join(texts)
        return "[non-text tool result]" if content else ""

    @on(PromptEditor.Submitted)
    def prompt_submitted(self, event: PromptEditor.Submitted) -> None:
        editor = self.query_one(PromptEditor)
        editor.clear()
        if event.text.startswith("/"):
            self._run_command(event.text[1:])
        else:
            self._run_prompt(event.text)

    @work(exclusive=True, group="prompt")
    async def _run_prompt(self, text: str) -> None:
        editor = self.query_one(PromptEditor)
        composer = self.query_one(PromptComposer)
        composer.set_running(True)
        self._refresh_status()
        try:
            final = await self.controller.prompt(text)
            if final.stop_reason in ("error", "aborted"):
                self.notify(
                    final.error_message or final.stop_reason,
                    severity="error" if final.stop_reason == "error" else "warning",
                )
        except Exception as exc:
            self.notify(f"Prompt failed: {exc}", severity="error")
        finally:
            composer.set_running(False)
            editor.focus()
            self._refresh_status()

    @work(exclusive=True, group="command")
    async def _run_command(self, raw: str) -> None:
        await self._execute_command(raw)

    async def _execute_command(self, raw: str) -> None:
        parts = raw.strip().split(maxsplit=1)
        name = parts[0].lower() if parts else ""
        args = parts[1].strip() if len(parts) > 1 else ""
        if name in ("exit", "quit"):
            self.exit()
        elif name == "help":
            self.notify(
                "/model /compact /stats /tree /exit · extension commands are supported"
            )
        elif name == "tree":
            self._show_tree_and_focus()
        elif name == "compact":
            try:
                summary = await self.controller.maybe_compact(force=True)
                self.notify(
                    "Nothing to compact" if summary is None else "Context compacted"
                )
            except Exception as exc:
                self.notify(f"Compaction failed: {exc}", severity="error")
        elif name == "stats":
            usage = self.controller.last_usage
            tokens = estimate_tokens(self.controller.agent.messages)
            context = min(
                100,
                round(
                    tokens / max(1, self.controller.agent.model.context_window) * 100
                ),
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
            self.notify(
                f"model {self.controller.agent.model.id} · "
                f"sandbox {self.runtime_status.sandbox} · "
                f"MCP {self.runtime_status.mcp_servers} server/"
                f"{self.runtime_status.mcp_tools} tool · context {context}% · "
                f"{usage_text}"
            )
        elif name == "model":
            await self._choose_model(args)
        elif (
            self.controller.extensions is not None
            and name in self.controller.extensions.commands
        ):
            try:
                result = self.controller.extensions.commands[name]()
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    self.notify(str(result))
            except Exception as exc:
                self.notify(f"Command failed: {exc}", severity="error")
        else:
            self.notify(f"Unknown command: /{name}", severity="error")

    async def _choose_model(self, model_id: str = "") -> None:
        if not model_id:
            models = {model.id: model for model in list_models()}
            models[self.controller.agent.model.id] = self.controller.agent.model
            choice = await self.push_screen_wait(
                ChoiceScreen(
                    "Select model",
                    [
                        (key, f"{model.name or key}  ({key})")
                        for key, model in models.items()
                    ],
                )
            )
            if choice is None:
                return
            model_id = choice
        try:
            self.controller.switch_model(model_id)
            await self.controller.notify_model_changed()
            self.notify(f"Model: {self.controller.agent.model.id}")
        except KeyError as exc:
            self.notify(str(exc.args[0]), severity="error")

    @on(SessionTreePanel.EntrySelected)
    def tree_entry_selected(self, event: SessionTreePanel.EntrySelected) -> None:
        self._navigate_tree(event.entry_id)

    @work(exclusive=True, group="tree")
    async def _navigate_tree(self, entry_id: str) -> None:
        if self.controller.is_busy:
            self.notify(
                "Wait for the current turn or press Esc to abort", severity="warning"
            )
            return
        decision = BranchDecision(False)
        if self.controller.would_abandon_suffix(entry_id):
            selected = await self.push_screen_wait(BranchScreen())
            if selected is None:
                return
            decision = selected
        try:
            result = await self.controller.navigate_tree(
                entry_id,
                summarize=decision.summarize,
                custom_instructions=decision.instructions,
            )
            await self._render_history()
            if result.editor_text is not None:
                editor = self.query_one(PromptEditor)
                editor.load_text(result.editor_text)
                lines = result.editor_text.split("\n")
                editor.move_cursor((len(lines) - 1, len(lines[-1])))
                editor.focus()
            self.query_one(SessionTreePanel).refresh_tree()
            self._set_tree_open(False)
        except Exception as exc:
            self.notify(f"Tree navigation failed: {exc}", severity="error")

    @on(SessionTreePanel.LabelRequested)
    def tree_label_requested(self, event: SessionTreePanel.LabelRequested) -> None:
        self._edit_label(event.entry_id)

    @work(exclusive=True, group="label")
    async def _edit_label(self, entry_id: str) -> None:
        value = await self.push_screen_wait(
            TextInputScreen(
                "Entry label",
                self.controller.session.get_label(entry_id) or "",
                "checkpoint name",
            )
        )
        if value is None:
            return
        try:
            await self.controller.set_label(entry_id, value)
            self.query_one(SessionTreePanel).refresh_tree()
        except Exception as exc:
            self.notify(f"Could not update label: {exc}", severity="error")

    def action_toggle_tree(self) -> None:
        self._set_tree_open(not self._tree_open)

    def _show_tree_and_focus(self) -> None:
        self._set_tree_open(True)

    def _set_tree_open(self, open_: bool) -> None:
        panel = self.query_one("#session-tree", SessionTreePanel)
        scrim = self.query_one("#tree-scrim", Static)
        self._tree_open = open_
        panel.set_class(open_, "-shown")
        scrim.set_class(open_, "-shown")
        if open_:
            panel.query_one("#session-tree-widget").focus()
        elif self.is_mounted:
            self.query_one(PromptEditor).focus()

    @on(Click, "#tree-scrim")
    def close_tree_from_scrim(self, event: Click) -> None:
        event.stop()
        self._set_tree_open(False)

    @work(exclusive=True, group="command")
    async def _open_command_palette(self) -> None:
        commands = [
            ("help", "Help"),
            ("model", "Switch model"),
            ("compact", "Compact context"),
            ("stats", "Show usage and cost"),
            ("tree", "Focus session tree"),
            ("exit", "Quit Pion"),
        ]
        if self.controller.extensions is not None:
            commands.extend(
                (name, f"Extension: /{name}")
                for name in sorted(self.controller.extensions.commands)
            )
        choice = await self.push_screen_wait(ChoiceScreen("Commands", commands))
        if choice:
            await self._execute_command(choice)

    def action_command_palette(self) -> None:
        self._open_command_palette()

    def action_abort(self) -> None:
        if self.controller.is_busy:
            self.controller.abort()
            self.notify("Abort requested", severity="warning")
        elif self._tree_open:
            self._set_tree_open(False)

    def action_quit_pion(self) -> None:
        self.controller.abort()
        self.exit()

    def on_resize(self, event: Resize) -> None:
        self._apply_responsive_layout(event.size.width)

    def _apply_responsive_layout(self, width: int) -> None:
        narrow = width < 80
        self.screen.set_class(narrow, "narrow")
        if self.is_mounted:
            self._refresh_status(width)

    def _refresh_status(self, width: int | None = None) -> None:
        agent = self.controller.agent
        tokens = estimate_tokens(agent.messages)
        percent = min(100, round(tokens / max(1, agent.model.context_window) * 100))
        self.query_one(HeaderBar).update_status(
            project=self.runtime_status.project,
            model=agent.model.id,
            context_percent=percent,
            running=self.controller.is_busy,
            width=self.size.width if width is None else width,
        )


__all__ = ["PionApp", "TUIStatus"]

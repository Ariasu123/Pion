"""Reusable Textual widgets for Pion's conversation and session tree."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as TextualMessage
from textual.widgets import Collapsible, Label, Markdown, Static, TextArea, Tree

from ..llm.types import (
    AssistantMessage,
    Message,
    TextContent,
    ThinkingContent,
    ToolResultMessage,
    UserMessage,
)
from ..session import SessionEntry, SessionManager, SessionTreeNode


def _truncate_middle(value: str, limit: int) -> str:
    """Keep both ends of a long status value within a fixed cell budget."""
    if len(value) <= limit:
        return value
    if limit < 5:
        return value[:limit]
    left = (limit - 1) // 2
    right = limit - left - 1
    return f"{value[:left]}…{value[-right:]}"


class HeaderBar(Horizontal):
    """Single-line application header with responsive status details."""

    def compose(self) -> ComposeResult:
        yield Static(id="header-summary", markup=False)
        yield Static("● READY · CTX 0%", id="header-status", markup=False)

    def update_status(
        self,
        *,
        project: str,
        model: str,
        context_percent: int,
        running: bool,
        width: int,
    ) -> None:
        if width < 70:
            summary = f"PION · {_truncate_middle(project, max(8, width - 32))}"
        elif width < 100:
            available = max(22, width - 32)
            project_limit = max(8, available // 2)
            model_limit = max(8, available - project_limit - 7)
            summary = (
                f"PION · {_truncate_middle(project, project_limit)}"
                f" · {_truncate_middle(model, model_limit)}"
            )
        else:
            summary = f"PION · {project} · {model}"

        status = "RUNNING" if running else "READY"
        self.query_one("#header-summary", Static).update(summary)
        status_widget = self.query_one("#header-status", Static)
        status_widget.update(f"● {status} · CTX {context_percent}%")
        status_widget.set_class(running, "running")


class EmptyState(Vertical):
    """Quiet first-run guidance shown only when the active branch is empty."""

    def __init__(self, project: str, cwd: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.project = project
        self.cwd = cwd

    def compose(self) -> ComposeResult:
        yield Static("PION", classes="empty-brand")
        yield Static(self.project, classes="empty-project", markup=False)
        yield Static(str(self.cwd), classes="empty-path", markup=False)
        yield Static(
            "Enter to send · Ctrl+J for a new line",
            classes="empty-help",
        )
        yield Static(
            "Ctrl+P for commands · Ctrl+B for session history",
            classes="empty-help",
        )


class PromptComposer(Vertical):
    """Minimal growing prompt surface with an integrated prompt glyph."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="composer-input"):
            yield Static("›", id="composer-prompt")
            yield PromptEditor(
                id="prompt-editor",
                language=None,
                compact=True,
                highlight_cursor_line=False,
                placeholder="Ask Pion anything…",
            )
            yield Static("Enter send", id="composer-hint", markup=False)

    def set_running(self, running: bool) -> None:
        self.set_class(running, "running")
        self.query_one(PromptEditor).disabled = running
        self.query_one("#composer-hint", Static).update(
            "Esc abort" if running else "Enter send"
        )


def message_text(message: Message) -> str:
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return message.content
        return "".join(
            block.text for block in message.content if isinstance(block, TextContent)
        )
    return message.text()


class PromptEditor(TextArea):
    class Submitted(TextualMessage):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    MIN_ROWS = 1
    MAX_ROWS = 6

    def on_mount(self) -> None:
        self._resize_for_content()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._resize_for_content()

    def _resize_for_content(self) -> None:
        rows = max(self.MIN_ROWS, min(self.MAX_ROWS, len(self.document.lines)))
        self.styles.height = rows
        composer = self.parent.parent if self.parent is not None else None
        if isinstance(composer, PromptComposer):
            # Two cells account for the composer's top and bottom border.
            composer.styles.height = rows + 2

    async def _on_key(self, event: Any) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
            return
        if event.key == "ctrl+j":
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        await super()._on_key(event)


class ChatMessage(Horizontal):
    def __init__(self, role: str, text: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.role = role
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static("›", classes="message-glyph")
        with Vertical(classes="message-body"):
            yield Markdown(self._text or " ", classes="message-markdown")

    async def set_text(self, text: str) -> None:
        self._text = text
        await self.query_one(Markdown).update(text or " ")


class ThinkingPanel(Collapsible):
    """Collapsed-by-default reasoning details for an assistant message."""

    def __init__(self, text: str = "", *, running: bool = False) -> None:
        self.thinking_text = text
        self.running = running
        self.started_at = time.monotonic() if running else None
        self.duration_seconds: float | None = None
        classes = "thinking-panel"
        if not text:
            classes += " empty"
        super().__init__(
            Static(text or " ", classes="thinking-content", markup=False),
            title=self._make_title(),
            collapsed=True,
            collapsed_symbol="›",
            expanded_symbol="⌄",
            classes=classes,
        )

    def _make_title(self) -> str:
        if self.running:
            return "Thinking…"
        if self.duration_seconds is not None:
            return f"Thought for {self.duration_seconds:.1f}s"
        return "Thinking"

    def update_thinking(self, text: str) -> None:
        self.thinking_text = text
        self.remove_class("empty")
        self.query_one(".thinking-content", Static).update(text or " ")

    def complete(self, text: str) -> None:
        self.update_thinking(text)
        if self.started_at is not None:
            self.duration_seconds = time.monotonic() - self.started_at
        self.running = False
        self.title = self._make_title()


class AssistantCard(ChatMessage):
    def __init__(
        self,
        text: str = "",
        thinking: str = "",
        *,
        streaming: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__("PION", text, **kwargs)
        self._thinking = thinking
        self._streaming = streaming
        self._pending_text = ""
        self._pending_thinking = ""
        self._flush_timer: Any = None

    def compose(self) -> ComposeResult:
        yield Static("◆", classes="message-glyph assistant-glyph")
        with Vertical(classes="message-body"):
            yield ThinkingPanel(self._thinking, running=self._streaming)
            yield Markdown(self._text or " ", classes="message-markdown")

    def append_delta(self, kind: Literal["text", "thinking"], delta: str) -> None:
        if kind == "thinking":
            self._pending_thinking += delta
        else:
            self._pending_text += delta
        if self._flush_timer is None:
            self._flush_timer = self.set_timer(1 / 30, self._flush_deltas)

    async def _flush_deltas(self) -> None:
        self._flush_timer = None
        if self._pending_thinking:
            self._thinking += self._pending_thinking
            self._pending_thinking = ""
            self.query_one(ThinkingPanel).update_thinking(self._thinking)
        if self._pending_text:
            self._text += self._pending_text
            self._pending_text = ""
            await self.query_one(Markdown).update(self._text or " ")

    async def set_message(self, message: AssistantMessage) -> None:
        self._pending_text = ""
        self._pending_thinking = ""
        self._text = "".join(
            block.text for block in message.content if isinstance(block, TextContent)
        )
        self._thinking = "".join(
            block.thinking
            for block in message.content
            if isinstance(block, ThinkingContent)
        )
        thinking_panel = self.query_one(ThinkingPanel)
        if self._thinking:
            thinking_panel.complete(self._thinking)
        else:
            thinking_panel.add_class("empty")
        await self.query_one(Markdown).update(self._text or " ")


class ToolCard(Collapsible):
    def __init__(
        self,
        call_id: str,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        result: str = "",
        is_error: bool = False,
        running: bool = False,
    ) -> None:
        self.call_id = call_id
        self.tool_name = name
        self.args = args or {}
        self.result_text = result
        self.is_error = is_error
        self.running = running
        self.started_at = time.monotonic() if running else None
        self.duration_seconds: float | None = None
        super().__init__(
            Static(self._details(), classes="tool-details", markup=False),
            title=self._make_title(),
            collapsed=True,
            collapsed_symbol="›",
            expanded_symbol="⌄",
            classes="tool-card error" if is_error else "tool-card",
        )

    def _summary(self) -> str:
        for key in ("command", "path", "file_path", "query"):
            if key in self.args:
                value = " ".join(str(self.args[key]).split())
                return value[:80] + ("…" if len(value) > 80 else "")
        raw = json.dumps(self.args, ensure_ascii=False)
        return raw[:80] + ("…" if len(raw) > 80 else "")

    def _make_title(self) -> str:
        state = (
            "● RUNNING" if self.running else ("✕ ERROR" if self.is_error else "✓ DONE")
        )
        elapsed = (
            f" {self.duration_seconds:.1f}s"
            if self.duration_seconds is not None
            else ""
        )
        return f"{state}  {self.tool_name}{elapsed}  {self._summary()}"

    def _details(self) -> str:
        args = json.dumps(self.args, ensure_ascii=False, indent=2)
        result = self.result_text or ("running…" if self.running else "")
        source = ""
        if "__" in self.tool_name:
            server, _ = self.tool_name.split("__", 1)
            source = f"SOURCE\nMCP server {server}\n\n"
        return f"{source}ARGUMENTS\n{args}\n\nOUTPUT\n{result}"

    def update_result(self, result: str, is_error: bool = False) -> None:
        if self.started_at is not None:
            self.duration_seconds = time.monotonic() - self.started_at
        self.running = False
        self.is_error = is_error
        self.result_text = result
        self.title = self._make_title()
        self.set_class(is_error, "error")
        self.query_one(".tool-details", Static).update(self._details())

    def update_progress(self, result: str) -> None:
        self.result_text = result
        self.running = True
        self.title = self._make_title()
        self.query_one(".tool-details", Static).update(self._details())


class ConversationView(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Vertical(id="conversation-stream")

    @property
    def stream(self) -> Vertical:
        return self.query_one("#conversation-stream", Vertical)

    async def clear_content(self) -> None:
        await self.stream.remove_children()

    async def show_empty(self, project: str, cwd: Path) -> None:
        await self.clear_content()
        await self.stream.mount(EmptyState(project, cwd, id="empty-state"))

    async def hide_empty(self) -> None:
        for empty in list(self.query(EmptyState)):
            await empty.remove()

    async def mount_content(self, widget: Any) -> None:
        await self.hide_empty()
        await self.stream.mount(widget)

    async def add_message(self, message: Message) -> ChatMessage | ToolCard | None:
        if isinstance(message, UserMessage):
            widget = ChatMessage(
                "YOU", message_text(message), classes="chat-message user-message"
            )
        elif isinstance(message, AssistantMessage):
            text = message_text(message)
            thinking = "".join(
                block.thinking
                for block in message.content
                if isinstance(block, ThinkingContent)
            )
            if not text and not thinking:
                return None
            widget = AssistantCard(
                text,
                thinking,
                classes="chat-message assistant-message",
            )
        else:
            widget = ToolCard(
                message.tool_call_id,
                message.tool_name,
                result=message.text(),
                is_error=message.is_error,
            )
        await self.mount_content(widget)
        return widget


TreeFilter = Literal["default", "no-tools", "user-only", "labeled-only", "all"]
TREE_FILTERS: tuple[TreeFilter, ...] = (
    "default",
    "no-tools",
    "user-only",
    "labeled-only",
    "all",
)


class SessionTreePanel(Vertical):
    class EntrySelected(TextualMessage):
        def __init__(self, entry_id: str) -> None:
            super().__init__()
            self.entry_id = entry_id

    class LabelRequested(TextualMessage):
        def __init__(self, entry_id: str) -> None:
            super().__init__()
            self.entry_id = entry_id

    BINDINGS = [
        ("ctrl+o", "cycle_filter", "Filter"),
        ("shift+l", "label_entry", "Label"),
    ]

    def __init__(
        self,
        session: SessionManager,
        *,
        session_name: str = "session.jsonl",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.session = session
        self.session_name = session_name
        self.filter_mode: TreeFilter = "default"

    def compose(self) -> ComposeResult:
        with Vertical(id="tree-header"):
            with Horizontal(id="tree-title-row"):
                yield Static("SESSION", id="tree-title")
                yield Static("0 entries", id="tree-count")
            yield Static(self.session_name, id="tree-session", markup=False)
        tree: Tree[str] = Tree("Session", id="session-tree-widget")
        tree.show_root = False
        yield tree
        with Vertical(id="tree-footer"):
            with Horizontal(id="tree-filter-row"):
                yield Label("default", id="tree-filter")
                yield Label("Ctrl+O filter", id="tree-filter-hint")
            yield Label(
                "Enter select · Shift+L label\nCtrl+B close · Esc close", id="tree-help"
            )

    def on_mount(self) -> None:
        self.refresh_tree()

    def refresh_tree(self) -> None:
        tree = self.query_one(Tree)
        tree.clear()
        self.query_one("#tree-count", Static).update(
            f"{len(self.session.get_entries())} entries"
        )
        active = {entry.id for entry in self.session.get_branch()}

        def add_nodes(nodes: tuple[SessionTreeNode, ...], parent: Any) -> bool:
            added_any = False
            for node in nodes:
                child_matches = self._has_visible_descendant(node)
                if not self._matches(node) and not child_matches:
                    continue
                if node.entry.type == "label":
                    if add_nodes(node.children, parent):
                        added_any = True
                    continue
                is_current = node.entry.id == self.session.leaf_id
                label = self._entry_label(node.entry, current=is_current)
                if node.label:
                    label.append(f"  {node.label}", style="#D7A85B italic")
                branch = parent.add(label, data=node.entry.id)
                if node.entry.id in active:
                    branch.expand()
                add_nodes(node.children, branch)
                added_any = True
            return added_any

        add_nodes(self.session.get_tree(), tree.root)
        tree.root.expand()

    def _has_visible_descendant(self, node: SessionTreeNode) -> bool:
        return any(
            self._matches(child) or self._has_visible_descendant(child)
            for child in node.children
        )

    def _matches(self, node: SessionTreeNode) -> bool:
        entry = node.entry
        if self.filter_mode == "all":
            return True
        if self.filter_mode == "labeled-only":
            return bool(node.label)
        if self.filter_mode == "user-only":
            return entry.type == "message" and isinstance(entry.message, UserMessage)
        if self.filter_mode == "no-tools":
            return (
                not (
                    entry.type == "message"
                    and isinstance(entry.message, ToolResultMessage)
                )
                and entry.type != "label"
            )
        return entry.type in (
            "message",
            "compaction",
            "branch_summary",
        ) and not isinstance(entry.message, ToolResultMessage)

    @staticmethod
    def _entry_label(entry: SessionEntry, *, current: bool = False) -> Text:
        symbol = "·"
        symbol_style = "#5E6675"
        body = "entry"
        if entry.type == "branch_summary":
            symbol, symbol_style = "◇", "#D97757"
            body = SessionTreePanel._short(entry.summary or "branch summary")
        elif entry.type == "compaction":
            symbol, symbol_style = "◈", "#8B93A3"
            body = SessionTreePanel._short(entry.summary or "compaction")
        elif entry.type == "custom":
            body = "custom"
        elif entry.type == "label":
            symbol, symbol_style = "⌑", "#D7A85B"
            body = f"label: {entry.label or 'cleared'}"
        elif isinstance(entry.message, UserMessage):
            symbol, symbol_style = "●", "#D97757"
            body = SessionTreePanel._short(message_text(entry.message))
        elif isinstance(entry.message, AssistantMessage):
            symbol, symbol_style = "◆", "#68B984"
            body = SessionTreePanel._short(message_text(entry.message) or "tool call")
        elif isinstance(entry.message, ToolResultMessage):
            symbol = "✕" if entry.message.is_error else "▪"
            symbol_style = "#E06C75" if entry.message.is_error else "#8B93A3"
            body = entry.message.tool_name

        label = Text()
        if current:
            label.append("▌", style="bold #D97757")
        label.append(f"{symbol} ", style=symbol_style)
        label.append(body, style="bold #E7E9EE" if current else "#B7BDC8")
        return label

    @staticmethod
    def _short(text: str, limit: int = 20) -> str:
        compact = " ".join(text.split())
        return compact[:limit] + ("…" if len(compact) > limit else "")

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        if event.node.data:
            self.post_message(self.EntrySelected(event.node.data))

    def action_cycle_filter(self) -> None:
        index = TREE_FILTERS.index(self.filter_mode)
        self.filter_mode = TREE_FILTERS[(index + 1) % len(TREE_FILTERS)]
        self.query_one("#tree-filter", Label).update(self.filter_mode)
        self.refresh_tree()

    def action_label_entry(self) -> None:
        node = self.query_one(Tree).cursor_node
        if node is not None and node.data:
            self.post_message(self.LabelRequested(node.data))


@dataclass(frozen=True)
class ToolSnapshot:
    call_id: str
    name: str
    args: dict[str, Any]

"""Overlay building blocks: framed panel, single-line input, session tree.

Selectors are modal overlays composited over the transcript — pi's
tree-selector / model-selector shape, replacing the old Textual drawer and
modal screens.
"""

from __future__ import annotations

from collections.abc import Callable

from ...llm.types import AssistantMessage, ToolResultMessage, UserMessage
from ...session.manager import SessionManager, SessionTreeNode
from ..core.component import Component
from ..core.keys import KeyEvent
from ..core.renderer import CURSOR_MARKER
from ..core.text_utils import (
    SGR_RESET,
    apply_bg,
    pad_line,
    truncate_to_width,
)
from ..theme import Theme, get_theme
from .chat import message_text

TREE_FILTERS = ("default", "no-tools", "user-only", "labeled-only", "all")


class OverlayPanel(Component):
    """Title + body framed by full-width `─` rules on an opaque background."""

    def __init__(self, title: str, body: Component, theme: Theme | None = None) -> None:
        self.title = title
        self.body = body
        self._theme = theme

    def handle_input(self, key: KeyEvent) -> None:
        self.body.handle_input(key)

    def invalidate(self) -> None:
        self.body.invalidate()

    def render(self, width: int) -> list[str]:
        theme = self._theme or get_theme()
        border = theme.fg("borderMuted", "─" * max(1, width))
        title = " " + theme.styled(self.title, "accent", bold=True)
        body_lines = self.body.render(max(1, width - 2))
        lines = [border, title] + [" " + line for line in body_lines] + [border]
        bg = theme.bg_open("overlayBg")
        return [apply_bg(pad_line(line, width), width, bg) for line in lines]


class TextInput(Component):
    """Single-line input with a fake cursor (for labels / custom focus)."""

    def __init__(
        self,
        initial: str = "",
        placeholder: str = "",
        on_submit: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        theme: Theme | None = None,
    ) -> None:
        self.text = initial
        self.cursor = len(initial)
        self.placeholder = placeholder
        self.on_submit = on_submit
        self.on_cancel = on_cancel
        self._theme = theme

    def handle_input(self, key: KeyEvent) -> None:
        if key.key == "text":
            self.text = self.text[: self.cursor] + key.text + self.text[self.cursor :]
            self.cursor += len(key.text)
        elif key.key == "paste":
            pasted = key.text.replace("\n", " ")
            self.text = self.text[: self.cursor] + pasted + self.text[self.cursor :]
            self.cursor += len(pasted)
        elif key.key == "backspace" and self.cursor > 0:
            self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
            self.cursor -= 1
        elif key.key == "delete" and self.cursor < len(self.text):
            self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
        elif key.key == "left":
            self.cursor = max(0, self.cursor - 1)
        elif key.key == "right":
            self.cursor = min(len(self.text), self.cursor + 1)
        elif key.key == "home":
            self.cursor = 0
        elif key.key == "end":
            self.cursor = len(self.text)
        elif key.key == "enter":
            if self.on_submit is not None:
                self.on_submit(self.text)
        elif key.key == "escape":
            if self.on_cancel is not None:
                self.on_cancel()

    def render(self, width: int) -> list[str]:
        theme = self._theme or get_theme()
        if self.text:
            cell = self.text[self.cursor : self.cursor + 1] or " "
            line = (
                self.text[: self.cursor]
                + CURSOR_MARKER
                + "\x1b[7m"
                + cell
                + SGR_RESET
                + self.text[self.cursor + 1 :]
            )
        else:
            line = (
                CURSOR_MARKER
                + "\x1b[7m "
                + SGR_RESET
                + theme.fg("dim", self.placeholder)
            )
        hint = theme.fg("keyHint", "enter submit · esc cancel")
        return [truncate_to_width(line, width), hint]


def _short(text: str, limit: int = 28) -> str:
    compact = " ".join(text.split())
    return compact[:limit] + ("…" if len(compact) > limit else "")


class TreeSelector(Component):
    """Flat navigable view over the branched session JSONL tree."""

    max_visible = 12

    def __init__(
        self,
        session: SessionManager,
        on_navigate: Callable[[str], None] | None = None,
        on_label: Callable[[str], None] | None = None,
        on_close: Callable[[], None] | None = None,
        theme: Theme | None = None,
    ) -> None:
        self.session = session
        self.on_navigate = on_navigate
        self.on_label = on_label
        self.on_close = on_close
        self._theme = theme
        self.filter_mode = "default"
        self.selected = 0
        self._rows: list[tuple[str, int, str]] = []  # (entry_id, depth, label)
        self.refresh()

    # -- tree → flat rows ---------------------------------------------------

    def refresh(self) -> None:
        rows: list[tuple[str, int, str]] = []

        def walk(nodes: tuple[SessionTreeNode, ...], depth: int) -> None:
            for node in nodes:
                if not self._matches(node) and not self._has_visible_descendant(node):
                    continue
                if node.entry.type == "label":
                    walk(node.children, depth)
                    continue
                rows.append((node.entry.id, depth, self._label(node)))
                walk(node.children, depth + 1)

        walk(self.session.get_tree(), 0)
        self._rows = rows
        if self.session.leaf_id is not None:
            for index, (entry_id, _, _) in enumerate(rows):
                if entry_id == self.session.leaf_id:
                    self.selected = index
                    break
        self.selected = max(0, min(self.selected, len(rows) - 1))

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
            return not (
                entry.type == "message"
                and isinstance(entry.message, ToolResultMessage)
            ) and entry.type != "label"
        return entry.type in ("message", "compaction", "branch_summary") and not (
            isinstance(entry.message, ToolResultMessage)
        )

    def _label(self, node: SessionTreeNode) -> str:
        entry = node.entry
        theme = self._theme or get_theme()
        if entry.type == "branch_summary":
            glyph, color = "◇", "warning"
            body = _short(entry.summary or "branch summary")
        elif entry.type == "compaction":
            glyph, color = "◈", "muted"
            body = _short(entry.summary or "compaction")
        elif entry.type == "custom":
            glyph, color = "·", "muted"
            body = "custom"
        elif isinstance(entry.message, UserMessage):
            glyph, color = "●", "accent"
            body = _short(message_text(entry.message))
        elif isinstance(entry.message, AssistantMessage):
            glyph, color = "◆", "success"
            body = _short(message_text(entry.message) or "tool call")
        elif isinstance(entry.message, ToolResultMessage):
            glyph = "✕" if entry.message.is_error else "▪"
            color = "error" if entry.message.is_error else "muted"
            body = entry.message.tool_name
        else:
            glyph, color = "·", "muted"
            body = "entry"
        current = entry.id == self.session.leaf_id
        marker = theme.fg("accent", "▌") if current else ""
        label = marker + theme.fg(color, f"{glyph} ") + body
        if node.label:
            label += "  " + theme.styled(node.label, "warning", italic=True)
        return label

    # -- input --------------------------------------------------------------

    def handle_input(self, key: KeyEvent) -> None:
        if key.key == "up":
            self.selected = (self.selected - 1) % max(1, len(self._rows))
        elif key.key == "down":
            self.selected = (self.selected + 1) % max(1, len(self._rows))
        elif key.key == "enter":
            if self._rows and self.on_navigate is not None:
                self.on_navigate(self._rows[self.selected][0])
        elif key.key == "ctrl+o":
            index = TREE_FILTERS.index(self.filter_mode)
            self.filter_mode = TREE_FILTERS[(index + 1) % len(TREE_FILTERS)]
            self.refresh()
        elif key.key == "shift+l" or (key.key == "text" and key.text == "L"):
            if self._rows and self.on_label is not None:
                self.on_label(self._rows[self.selected][0])
        elif key.key == "escape":
            if self.on_close is not None:
                self.on_close()

    # -- render ---------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        theme = self._theme or get_theme()
        header = theme.fg("dim", f"filter: {self.filter_mode}")
        hint = theme.fg(
            "keyHint", "enter go · ctrl+o filter · shift+l label · esc close"
        )
        lines = [header]
        if not self._rows:
            lines.append(theme.fg("dim", "  no entries"))
        else:
            half = self.max_visible // 2
            start = max(
                0, min(self.selected - half, len(self._rows) - self.max_visible)
            )
            window = self._rows[start : start + self.max_visible]
            for index, (_, depth, label) in enumerate(window):
                row = "  " * depth + label
                row = truncate_to_width(row, width - 2)
                if start + index == self.selected:
                    row = apply_bg(
                        pad_line(row, width - 2), width - 2, theme.bg_open("selectedBg")
                    )
                lines.append(row)
            hidden = len(self._rows) - len(window)
            if hidden > 0:
                lines.append(theme.fg("dim", f"… {hidden} more"))
        lines.append(hint)
        return lines

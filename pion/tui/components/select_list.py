"""Selection list used by all pickers (port of pi-tui's select-list.ts)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.component import Component
from ..core.keys import KeyEvent
from ..core.text_utils import apply_bg, pad_line, truncate_to_width
from ..theme import Theme, get_theme
from .fuzzy import fuzzy_filter


@dataclass(frozen=True)
class SelectItem:
    value: str
    label: str
    description: str = ""


class SelectList(Component):
    def __init__(
        self,
        items: list[SelectItem],
        max_visible: int = 8,
        on_select: Callable[[str], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        filterable: bool = True,
        theme: Theme | None = None,
    ) -> None:
        self.items = items
        self.max_visible = max_visible
        self.on_select = on_select
        self.on_cancel = on_cancel
        self.filterable = filterable
        self.filter_text = ""
        self.selected = 0
        self._theme = theme

    def set_items(self, items: list[SelectItem]) -> None:
        self.items = items
        self.selected = 0

    def filtered(self) -> list[SelectItem]:
        if not self.filter_text:
            return list(self.items)
        return fuzzy_filter(
            self.filter_text,
            self.items,
            lambda item: f"{item.label} {item.description}",
        )

    def current(self) -> SelectItem | None:
        items = self.filtered()
        if not items:
            return None
        self.selected = max(0, min(self.selected, len(items) - 1))
        return items[self.selected]

    def handle_input(self, key: KeyEvent) -> None:
        items = self.filtered()
        if key.key == "up":
            self.selected = (self.selected - 1) % max(1, len(items))
        elif key.key == "down":
            self.selected = (self.selected + 1) % max(1, len(items))
        elif key.key == "enter":
            item = self.current()
            if item is not None and self.on_select is not None:
                self.on_select(item.value)
        elif key.key == "escape":
            if self.filter_text:
                self.filter_text = ""
                self.selected = 0
            elif self.on_cancel is not None:
                self.on_cancel()
        elif key.key == "backspace":
            if self.filter_text:
                self.filter_text = self.filter_text[:-1]
                self.selected = 0
            elif self.on_cancel is not None:
                self.on_cancel()
        elif key.key == "text" and self.filterable:
            self.filter_text += key.text
            self.selected = 0

    def render(self, width: int) -> list[str]:
        theme = self._theme or get_theme()
        items = self.filtered()
        lines: list[str] = []
        if self.filterable:
            prompt = theme.fg("accent", "/ ") + (self.filter_text or "")
            if not self.filter_text:
                prompt += theme.fg("dim", "type to filter")
            lines.append(prompt)
        if not items:
            lines.append(theme.fg("dim", "  no matches"))
            return lines
        self.selected = max(0, min(self.selected, len(items) - 1))
        half = self.max_visible // 2
        start = max(0, min(self.selected - half, len(items) - self.max_visible))
        window = items[start : start + self.max_visible]
        for index, item in enumerate(window):
            marker = theme.fg("accent", "> ") if start + index == self.selected else "  "
            line = marker + truncate_to_width(item.label, max(1, width - 4))
            if item.description:
                room = width - 4 - len(item.label)
                if room > 6:
                    line += "  " + theme.fg(
                        "description", truncate_to_width(item.description, room - 2)
                    )
            line = pad_line(line, width)
            if start + index == self.selected:
                line = apply_bg(line, width, theme.bg_open("selectedBg"))
            lines.append(line)
        hidden = len(items) - len(window)
        if hidden > 0:
            lines.append(theme.fg("dim", f"  … {hidden} more"))
        return lines

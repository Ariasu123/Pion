"""Basic display components: Text, Box, DynamicBorder, Loader."""

from __future__ import annotations

import time

from ..core.component import Component, Container
from ..core.text_utils import apply_bg, pad_line, repaint, wrap_text_with_ansi
from ..theme import Theme, get_theme

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Text(Component):
    """Wrapped plain text with padding and optional full-width background."""

    def __init__(
        self,
        text: str = "",
        pad_x: int = 1,
        pad_y: int = 0,
        fg: str | None = None,
        bg: str | None = None,
        theme: Theme | None = None,
    ) -> None:
        self.text = text
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.fg = fg
        self.bg = bg
        self._theme = theme
        self._cache: tuple[int, list[str]] | None = None

    def set_text(self, text: str) -> None:
        if text != self.text:
            self.text = text
            self.invalidate()

    def invalidate(self) -> None:
        self._cache = None

    def render(self, width: int) -> list[str]:
        if self._cache and self._cache[0] == width:
            return list(self._cache[1])
        theme = self._theme or get_theme()
        inner = max(1, width - 2 * self.pad_x)
        lines: list[str] = []
        for raw in self.text.split("\n"):
            lines.extend(wrap_text_with_ansi(raw, inner))
        if self.fg:
            open_seq = theme.fg_open(self.fg)
            lines = [repaint(line, open_seq) for line in lines]
        pad = " " * self.pad_x
        lines = [pad + pad_line(line, inner) + pad for line in lines]
        if self.bg:
            bg_open = theme.bg_open(self.bg)
            lines = [apply_bg(line, width, bg_open) for line in lines]
        blank = apply_bg("", width, theme.bg_open(self.bg)) if self.bg else ""
        out = [blank] * self.pad_y + lines + [blank] * self.pad_y
        self._cache = (width, out)
        return list(out)


class Box(Container):
    """Children padded and painted with a full-width background band.

    This is the only "card" shape in the pion TUI — structure comes from
    background color, never from drawn borders.
    """

    def __init__(
        self,
        pad_x: int = 1,
        pad_y: int = 1,
        bg: str | None = None,
        children: list[Component] | None = None,
        theme: Theme | None = None,
    ) -> None:
        super().__init__(children)
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.bg = bg
        self._theme = theme

    def render(self, width: int) -> list[str]:
        theme = self._theme or get_theme()
        inner = max(1, width - 2 * self.pad_x)
        pad = " " * self.pad_x
        lines: list[str] = []
        for child in self.children:
            for line in child.render(inner):
                lines.append(pad + pad_line(line, inner) + pad)
        if self.bg:
            bg_open = theme.bg_open(self.bg)
            lines = [apply_bg(line, width, bg_open) for line in lines]
            blank = apply_bg("", width, bg_open)
        else:
            blank = ""
        return [blank] * self.pad_y + lines + [blank] * self.pad_y


class DynamicBorder(Component):
    """A full-width horizontal rule (the only drawn line in the UI)."""

    def __init__(self, color: str = "borderMuted", theme: Theme | None = None) -> None:
        self.color = color
        self._theme = theme

    def render(self, width: int) -> list[str]:
        theme = self._theme or get_theme()
        return [theme.fg(self.color, "─" * max(1, width))]


class Loader(Component):
    """Braille spinner with a message and an optional dim hint."""

    def __init__(
        self,
        message: str = "",
        hint: str | None = "esc cancel",
        theme: Theme | None = None,
    ) -> None:
        self.message = message
        self.hint = hint
        self._theme = theme
        self._start = time.monotonic()

    def render(self, width: int) -> list[str]:
        theme = self._theme or get_theme()
        elapsed = time.monotonic() - self._start
        frame = SPINNER_FRAMES[int(elapsed / 0.08) % len(SPINNER_FRAMES)]
        line = theme.fg("spinner", frame) + " " + self.message
        if self.hint:
            line += "  " + theme.fg("keyHint", self.hint)
        return ["", " " + line]

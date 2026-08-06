"""Markdown → ANSI component built on Rich.

Rich renders Markdown into a fixed-width ANSI string; we capture it and
post-process lines (trim, pad). While streaming, an unclosed code fence is
auto-closed and a trailing partial fence is trimmed, so intermediate frames
don't flicker.
"""

from __future__ import annotations

import io

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.theme import Theme as RichTheme

from ..core.component import Component
from ..core.text_utils import pad_line, repaint, truncate_to_width
from ..theme import Theme, get_theme


def _rich_theme(theme: Theme) -> RichTheme:
    def hexof(token: str) -> str:
        value = theme._resolve(token)
        return value if value.startswith("#") else "default"

    return RichTheme(
        {
            "markdown.h1": f"bold {hexof('mdHeading')}",
            "markdown.h2": f"bold {hexof('mdHeading')}",
            "markdown.h3": f"bold {hexof('text')}",
            "markdown.h4": f"bold {hexof('text')}",
            "markdown.code": hexof("mdCode"),
            "markdown.code_block": hexof("mdCodeBlock"),
            "markdown.link": f"underline {hexof('mdLink')}",
            "markdown.item.bullet": hexof("accent"),
            "markdown.item.number": hexof("accent"),
            "markdown.block_quote": hexof("mdQuote"),
            "markdown.hr": hexof("borderMuted"),
            "markdown.strong": "bold",
            "markdown.em": "italic",
        }
    )


def _sanitize_streaming(text: str) -> str:
    """Trim a trailing partial fence and close an unclosed fence."""
    # A trailing run of backticks that is not a complete fence marker.
    stripped = text.rstrip("\n")
    tail = stripped.rsplit("\n", 1)[-1]
    if tail and set(tail) == {"`"} and len(tail) < 3:
        stripped = stripped[: -len(tail)]
    if stripped.count("```") % 2 == 1:
        stripped += "\n```"
    return stripped


class Markdown(Component):
    def __init__(
        self,
        text: str,
        pad_x: int = 1,
        pad_y: int = 0,
        fg: str | None = None,
        italic: bool = False,
        streaming: bool = False,
        theme: Theme | None = None,
    ) -> None:
        self.text = text
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.fg = fg
        self.italic = italic
        self.streaming = streaming
        self._theme = theme
        self._cache: tuple[int, list[str]] | None = None

    def set_text(self, text: str, streaming: bool | None = None) -> None:
        if streaming is not None:
            self.streaming = streaming
        if text != self.text:
            self.text = text
            self.invalidate()

    def invalidate(self) -> None:
        self._cache = None

    def _render_ansi(self, inner: int) -> list[str]:
        theme = self._theme or get_theme()
        text = self.text.strip("\n")
        if self.streaming:
            text = _sanitize_streaming(text)
        if not text.strip():
            return []
        buffer = io.StringIO()
        console = Console(
            file=buffer,
            width=max(8, inner),
            force_terminal=True,
            color_system="truecolor" if theme._truecolor else "standard",
            theme=_rich_theme(theme),
            legacy_windows=False,
            soft_wrap=False,
        )
        console.print(RichMarkdown(text, code_theme="ansi_dark"))
        lines = buffer.getvalue().split("\n")
        # Rich pads lines to full width with spaces; trim trailing blanks.
        lines = [line.rstrip() for line in lines]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return lines

    def render(self, width: int) -> list[str]:
        if self._cache and self._cache[0] == width:
            return list(self._cache[1])
        theme = self._theme or get_theme()
        inner = max(8, width - 2 * self.pad_x)
        lines = self._render_ansi(inner)
        prefix = ""
        if self.fg:
            prefix += theme.fg_open(self.fg)
        if self.italic:
            prefix += "\x1b[3m"
        if prefix:
            lines = [repaint(line, prefix) for line in lines]
        pad = " " * self.pad_x
        out = [pad + pad_line(truncate_to_width(line, inner), inner) + pad for line in lines]
        out = [""] * self.pad_y + out + [""] * self.pad_y
        self._cache = (width, out)
        return list(out)

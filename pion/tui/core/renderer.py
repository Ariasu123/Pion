"""Differential inline renderer (port of pi-tui's tui-main-screen.ts).

The renderer draws into the terminal's main screen: content scrolls into
native scrollback as it grows, and exiting the UI leaves the transcript
visible. Frames are produced by rendering the whole component tree to a list
of styled lines, then diffed line-by-line against the previous frame; only
changed lines are rewritten. Every update is wrapped in synchronized output
(ESC[?2026h/l) and emitted as a single write.
"""

from __future__ import annotations

import asyncio

from .component import Component
from .overlays import OverlayStack
from .terminal import Terminal
from .text_utils import SGR_RESET, visible_width

CURSOR_MARKER = "\x1b_pi:c\x07"  # zero-width APC sequence marking the cursor

MIN_RENDER_INTERVAL = 0.016  # ~60fps coalescing

_SYNC_START = "\x1b[?2026h"
_SYNC_END = "\x1b[?2026l"
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_CLEAR_LINE = "\x1b[2K"


class RenderError(RuntimeError):
    """A component produced a line wider than the terminal."""


class InlineRenderer:
    def __init__(
        self,
        terminal: Terminal,
        root: Component,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.terminal = terminal
        self.root = root
        self.overlays = OverlayStack()
        self._loop = loop
        self._prev_lines: list[str] | None = None
        self._prev_width: int = 0
        self._render_scheduled = False
        self._closed = False
        self._force_clear = False
        self.cursor: tuple[int, int] | None = None  # (row, col) in the frame
        # Rows the hardware cursor is parked above the frame's last line
        # (after _position_cursor moved it to the editor's fake cursor).
        # Every paint must first restore the "cursor at frame end" invariant.
        self._parked_above_end = 0

    # -- scheduling -----------------------------------------------------

    def request_render(self) -> None:
        if self._closed or self._render_scheduled:
            return
        try:
            loop = self._loop or asyncio.get_running_loop()
        except RuntimeError:
            self.render_now()  # no event loop (e.g. tests): paint immediately
            return
        self._render_scheduled = True
        loop.call_later(MIN_RENDER_INTERVAL, self._render_if_pending)

    def _render_if_pending(self) -> None:
        self._render_scheduled = False
        if not self._closed:
            self.render_now()

    def on_resize(self) -> None:
        """Terminal size changed: force a full redraw on the next render."""
        if self.terminal.columns != self._prev_width:
            self._prev_lines = None  # triggers full redraw
            self._force_clear = True
        self.request_render()

    def invalidate(self) -> None:
        self.root.invalidate()
        self.request_render()

    def close(self) -> None:
        """Leave the transcript in scrollback and restore the cursor."""
        if self._closed:
            return
        if self._prev_lines is not None:
            self.terminal.write(self._restore_cursor() + "\r\n" + _SHOW_CURSOR)
        self._closed = True

    def _restore_cursor(self) -> str:
        """Move the hardware cursor back to the frame's last line.

        After `_position_cursor` parked the cursor at the editor's fake
        cursor, every paint (and close) must first undo that move — all
        diff arithmetic assumes the cursor is at the end of the last line.
        """
        if self._parked_above_end:
            out = f"\x1b[{self._parked_above_end}B\r"
            self._parked_above_end = 0
            return out
        return ""

    # -- frame production ------------------------------------------------

    def _render_frame(self) -> list[str]:
        width = self.terminal.columns
        lines = self.root.render(width)
        cursor: tuple[int, int] | None = None
        cleaned: list[str] = []
        for line in lines:
            idx = line.find(CURSOR_MARKER)
            if idx != -1:
                col = visible_width(line[:idx])
                line = line[:idx] + line[idx + len(CURSOR_MARKER) :]
                cursor = (len(cleaned), col)
            cleaned.append(line)
        cleaned = self.overlays.composite(
            cleaned, width, self.terminal.rows
        )
        for i, line in enumerate(cleaned):
            if visible_width(line) > width:
                raise RenderError(
                    f"Rendered line {i} exceeds terminal width {width}: "
                    f"{line!r}. Use visible_width/truncate_to_width."
                )
            # Normalize: every line ends with an SGR reset; styles never
            # bleed across lines.
            if not line.endswith(SGR_RESET):
                cleaned[i] = line + SGR_RESET
        self.cursor = cursor
        return cleaned

    # -- painting ---------------------------------------------------------

    def render_now(self) -> None:
        if self._closed:
            return
        width = self.terminal.columns
        rows = self.terminal.rows
        lines = self._render_frame()
        prev = self._prev_lines

        buf: list[str] = [_SYNC_START, _HIDE_CURSOR, self._restore_cursor()]
        if prev is None or self._prev_width != width:
            if self._force_clear:
                # Width changed: clear screen + scrollback and repaint.
                buf.append("\x1b[2J\x1b[H\x1b[3J")
                self._force_clear = False
            buf.append("".join(line + "\r\n" for line in lines[:-1]))
            if lines:
                buf.append(lines[-1])
        else:
            self._paint_diff(buf, prev, lines, rows)
        buf.append(self._position_cursor(lines))
        buf.append(_SYNC_END)
        self.terminal.write("".join(buf))
        self._prev_lines = lines
        self._prev_width = width

    @staticmethod
    def _paint_diff(
        buf: list[str], prev: list[str], lines: list[str], rows: int
    ) -> None:
        old_n, new_n = len(prev), len(lines)
        first = 0
        common = min(old_n, new_n)
        while first < common and prev[first] == lines[first]:
            first += 1
        if first == old_n == new_n:
            return  # frame unchanged; only the cursor may move
        tail = 0
        if old_n == new_n:
            # Tail matching is only safe for equal-length frames; when the
            # length changes, the whole suffix must be repainted anyway.
            while tail < common - first and prev[old_n - 1 - tail] == lines[new_n - 1 - tail]:
                tail += 1
        old_changed_end = old_n - tail  # exclusive
        new_changed_end = new_n if old_n != new_n else new_n - tail

        if old_n > rows and first < old_n - rows:
            # Change is above the visible viewport: repaint everything.
            buf.append("\x1b[2J\x1b[H\x1b[3J")
            buf.append("".join(line + "\r\n" for line in lines[:-1]))
            if lines:
                buf.append(lines[-1])
            return

        # Move from the end of the old last line to the first changed line.
        if first >= old_n:
            # Pure append: advance to the line below the old frame.
            buf.append("\r\n")
        else:
            up = (old_n - 1) - first
            buf.append("\r")
            if up:
                buf.append(f"\x1b[{up}A")

        # Rewrite changed lines, appending new ones as needed.
        for i in range(first, new_changed_end):
            buf.append(_CLEAR_LINE)
            buf.append(lines[i])
            if i < new_n - 1:
                buf.append("\r\n")

        # Clear leftover lines when the frame shrank.
        leftover = old_changed_end - new_changed_end
        if leftover > 0:
            if new_changed_end > first:
                # Cursor is at the end of the new last line; the stale lines
                # are directly below it.
                for _ in range(leftover):
                    buf.append(f"\x1b[1B{_CLEAR_LINE}")
                buf.append(f"\x1b[{leftover}A\r")
            else:
                # Pure shrink: cursor is at the start of old line `first`.
                buf.append(_CLEAR_LINE)
                for _ in range(leftover - 1):
                    buf.append(f"\x1b[1B{_CLEAR_LINE}")
                up = (old_changed_end - 1) - (new_n - 1)
                if up > 0:
                    buf.append(f"\x1b[{up}A")
        # Cursor is now at the end of the new last line.

    def _position_cursor(self, lines: list[str]) -> str:
        self._parked_above_end = 0
        if self.cursor is None or not lines:
            return ""
        row, col = self.cursor
        row = min(row, len(lines) - 1)
        up = (len(lines) - 1) - row
        out = "\r"
        if up:
            out += f"\x1b[{up}A"
        if col:
            out += f"\x1b[{col}C"
        self._parked_above_end = up
        return out + _SHOW_CURSOR

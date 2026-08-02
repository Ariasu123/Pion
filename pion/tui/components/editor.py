"""Multi-line prompt editor (port of pi-tui's editor.ts, reduced).

Framed by full-width `─` rules top and bottom; long lines hard-wrap; the
cursor is a reverse-video cell plus a CURSOR_MARKER so the renderer can park
the hardware cursor on it (IME-friendly). Supports history, word navigation,
kill operations, bracketed paste, and an autocomplete dropdown (slash
commands + @files).
"""

from __future__ import annotations

from collections.abc import Callable

from ..core.component import Component
from ..core.keys import KeyEvent
from ..core.renderer import CURSOR_MARKER
from ..core.text_utils import (
    SGR_RESET,
    apply_bg,
    char_width,
    pad_line,
    truncate_to_width,
    visible_width,
)
from ..theme import Theme, get_theme
from .autocomplete import CombinedAutocompleteProvider, Suggestion

_WORD_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _wrap_plain(line: str, width: int) -> list[str]:
    """Hard-wrap a plain (ANSI-free) line by display cells."""
    if not line:
        return [""]
    segments: list[str] = []
    current = ""
    used = 0
    for ch in line:
        cw = char_width(ch)
        if used + cw > width:
            segments.append(current)
            current, used = "", 0
        current += ch
        used += cw
    segments.append(current)
    return segments


class Editor(Component):
    def __init__(
        self,
        on_submit: Callable[[str], None] | None = None,
        autocomplete: CombinedAutocompleteProvider | None = None,
        theme: Theme | None = None,
    ) -> None:
        self.on_submit = on_submit
        self.autocomplete = autocomplete
        self._theme = theme
        self.lines: list[str] = [""]
        self.cursor_row = 0
        self.cursor_col = 0
        self.history: list[str] = []
        self._history_index: int | None = None
        self._suggestion: Suggestion | None = None
        self._ac_selected = 0
        self.border_color = "borderMuted"

    # -- state ------------------------------------------------------------

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def set_text(self, text: str) -> None:
        self.lines = text.split("\n") or [""]
        self.cursor_row = len(self.lines) - 1
        self.cursor_col = len(self.lines[-1])
        self._history_index = None
        self._refresh_autocomplete()

    def clear(self) -> None:
        self.set_text("")

    def _cursor_index(self) -> int:
        return sum(len(line) + 1 for line in self.lines[: self.cursor_row]) + self.cursor_col

    def _clamp_cursor(self) -> None:
        self.cursor_row = max(0, min(self.cursor_row, len(self.lines) - 1))
        self.cursor_col = max(0, min(self.cursor_col, len(self.lines[self.cursor_row])))

    # -- editing primitives -------------------------------------------------

    def _insert(self, text: str) -> None:
        parts = text.split("\n")
        line = self.lines[self.cursor_row]
        before, after = line[: self.cursor_col], line[self.cursor_col :]
        if len(parts) == 1:
            self.lines[self.cursor_row] = before + text + after
            self.cursor_col += len(text)
        else:
            new_lines = [before + parts[0], *parts[1:-1], parts[-1] + after]
            self.lines[self.cursor_row : self.cursor_row + 1] = new_lines
            self.cursor_row += len(parts) - 1
            self.cursor_col = len(parts[-1])
        self._history_index = None

    def _backspace(self) -> None:
        if self.cursor_col > 0:
            line = self.lines[self.cursor_row]
            self.lines[self.cursor_row] = (
                line[: self.cursor_col - 1] + line[self.cursor_col :]
            )
            self.cursor_col -= 1
        elif self.cursor_row > 0:
            previous = self.lines[self.cursor_row - 1]
            self.cursor_col = len(previous)
            self.lines[self.cursor_row - 1] = previous + self.lines[self.cursor_row]
            del self.lines[self.cursor_row]
            self.cursor_row -= 1

    def _delete(self) -> None:
        line = self.lines[self.cursor_row]
        if self.cursor_col < len(line):
            self.lines[self.cursor_row] = (
                line[: self.cursor_col] + line[self.cursor_col + 1 :]
            )
        elif self.cursor_row < len(self.lines) - 1:
            self.lines[self.cursor_row] = line + self.lines[self.cursor_row + 1]
            del self.lines[self.cursor_row + 1]

    def _newline(self) -> None:
        self._insert("\n")

    def _kill_to_eol(self) -> None:
        line = self.lines[self.cursor_row]
        if self.cursor_col < len(line):
            self.lines[self.cursor_row] = line[: self.cursor_col]
        elif self.cursor_row < len(self.lines) - 1:
            self._delete()

    def _kill_line(self) -> None:
        self.lines[self.cursor_row] = ""
        self.cursor_col = 0

    def _delete_word_back(self) -> None:
        while self.cursor_col == 0 and self.cursor_row > 0:
            self._backspace()
        line = self.lines[self.cursor_row]
        col = self.cursor_col
        while col > 0 and line[col - 1] == " ":
            col -= 1
        while col > 0 and line[col - 1] in _WORD_CHARS:
            col -= 1
        while col > 0 and line[col - 1] not in _WORD_CHARS and line[col - 1] != " ":
            col -= 1
        self.lines[self.cursor_row] = line[:col] + line[self.cursor_col :]
        self.cursor_col = col

    def _move_word(self, direction: int) -> None:
        text = self.text
        index = self._cursor_index()
        if direction < 0:
            index = max(0, index - 1)
            while index > 0 and text[index] not in _WORD_CHARS:
                index -= 1
            while index > 0 and text[index - 1] in _WORD_CHARS:
                index -= 1
        else:
            index = min(len(text), index + 1)
            while index < len(text) and text[index] in _WORD_CHARS:
                index += 1
            while index < len(text) and text[index] not in _WORD_CHARS:
                index += 1
        # Convert linear index back to row/col.
        remaining = index
        for row, line in enumerate(self.lines):
            if remaining <= len(line):
                self.cursor_row, self.cursor_col = row, remaining
                return
            remaining -= len(line) + 1
        self.cursor_row = len(self.lines) - 1
        self.cursor_col = len(self.lines[-1])

    # -- history ------------------------------------------------------------

    def _history_move(self, direction: int) -> None:
        if not self.history:
            return
        if self._history_index is None:
            if direction > 0:
                return
            self._history_index = len(self.history)
        self._history_index += direction
        self._history_index = max(0, min(self._history_index, len(self.history)))
        if self._history_index == len(self.history):
            text = ""
        else:
            text = self.history[self._history_index]
        # Load without resetting _history_index (unlike set_text).
        self.lines = text.split("\n") or [""]
        self.cursor_row = len(self.lines) - 1
        self.cursor_col = len(self.lines[-1])
        self._refresh_autocomplete()

    # -- autocomplete ---------------------------------------------------------

    @property
    def autocomplete_open(self) -> bool:
        return self._suggestion is not None and bool(self._suggestion.items)

    def _refresh_autocomplete(self) -> None:
        if self.autocomplete is None:
            self._suggestion = None
            return
        self._suggestion = self.autocomplete.suggest(self.text, self._cursor_index())
        self._ac_selected = 0

    def _accept_suggestion(self) -> None:
        suggestion = self._suggestion
        if suggestion is None or not suggestion.items:
            return
        item = suggestion.items[self._ac_selected % len(suggestion.items)]
        text = self.text
        cursor = self._cursor_index()
        new_text = text[: suggestion.replace_start] + item.value + " " + text[cursor:]
        self.set_text(new_text)
        self._suggestion = None

    def dismiss_autocomplete(self) -> None:
        self._suggestion = None

    # -- input ---------------------------------------------------------------

    def handle_input(self, key: KeyEvent) -> None:
        if self.autocomplete_open:
            if key.key == "down":
                self._ac_selected = (self._ac_selected + 1) % len(self._suggestion.items)
                return
            if key.key == "up":
                self._ac_selected = (self._ac_selected - 1) % len(self._suggestion.items)
                return
            if key.key in ("tab", "enter"):
                self._accept_suggestion()
                return
            if key.key == "escape":
                self.dismiss_autocomplete()
                return

        if key.key == "text":
            self._insert(key.text)
        elif key.key == "paste":
            self._insert(key.text.rstrip("\n"))
        elif key.key == "enter":
            text = self.text.strip()
            if text:
                if not self.history or self.history[-1] != text:
                    self.history.append(text)
                if self.on_submit is not None:
                    self.on_submit(text)
            self.clear()
            return
        elif key.key in ("ctrl+j", "shift+enter"):
            self._newline()
        elif key.key == "backspace":
            self._backspace()
        elif key.key in ("delete",):
            self._delete()
        elif key.key == "ctrl+k":
            self._kill_to_eol()
        elif key.key == "ctrl+u":
            self._kill_line()
        elif key.key in ("ctrl+w", "alt+backspace"):
            self._delete_word_back()
        elif key.key in ("home", "ctrl+a"):
            self.cursor_col = 0
        elif key.key in ("end", "ctrl+e"):
            self.cursor_col = len(self.lines[self.cursor_row])
        elif key.key == "left":
            if self.cursor_col > 0:
                self.cursor_col -= 1
            elif self.cursor_row > 0:
                self.cursor_row -= 1
                self.cursor_col = len(self.lines[self.cursor_row])
        elif key.key == "right":
            if self.cursor_col < len(self.lines[self.cursor_row]):
                self.cursor_col += 1
            elif self.cursor_row < len(self.lines) - 1:
                self.cursor_row += 1
                self.cursor_col = 0
        elif key.key in ("alt+left", "alt+b"):
            self._move_word(-1)
        elif key.key in ("alt+right", "alt+f"):
            self._move_word(1)
        elif key.key == "up":
            if self.cursor_row == 0:
                self._history_move(-1)
            else:
                self.cursor_row -= 1
                self._clamp_cursor()
        elif key.key == "down":
            if self.cursor_row == len(self.lines) - 1:
                self._history_move(1)
            else:
                self.cursor_row += 1
                self._clamp_cursor()
        elif key.key == "escape":
            return  # not consumed: the app handles abort/close
        else:
            return
        self._clamp_cursor()
        self._refresh_autocomplete()

    # -- rendering -------------------------------------------------------------

    def render(self, width: int) -> list[str]:
        theme = self._theme or get_theme()
        border = theme.fg(self.border_color, "─" * max(1, width))
        display: list[tuple[str, int, int]] = []  # (text, logical row, start col)
        for row, line in enumerate(self.lines):
            start = 0
            for segment in _wrap_plain(line, width):
                display.append((segment, row, start))
                start += len(segment)

        lines: list[str] = [border]
        cursor_done = False
        for text, row, start in display:
            if (
                not cursor_done
                and row == self.cursor_row
                and start <= self.cursor_col <= start + len(text)
            ):
                offset = self.cursor_col - start
                cell = text[offset : offset + 1] or " "
                cursor_cell = f"{CURSOR_MARKER}\x1b[7m{cell}{SGR_RESET}"
                text = text[:offset] + cursor_cell + text[offset + 1 :]
                cursor_done = True
            lines.append(text)
        if not cursor_done:  # defensive: pin cursor at the end
            lines.append(CURSOR_MARKER + "\x1b[7m " + SGR_RESET)

        if self.autocomplete_open:
            lines.extend(self._render_dropdown(width))
        lines.append(border)
        return lines

    def _render_dropdown(self, width: int) -> list[str]:
        theme = self._theme or get_theme()
        suggestion = self._suggestion
        assert suggestion is not None
        items = suggestion.items[:6]
        out: list[str] = []
        for index, item in enumerate(items):
            selected = index == (self._ac_selected % len(suggestion.items))
            marker = theme.fg("accent", "❯ ") if selected else "  "
            line = marker + truncate_to_width(item.label, max(1, width - 4))
            if item.description:
                room = width - 4 - visible_width(item.label)
                if room > 6:
                    line += "  " + theme.fg(
                        "description", truncate_to_width(item.description, room - 2)
                    )
            line = pad_line(line, width)
            if selected:
                line = apply_bg(line, width, theme.bg_open("selectedBg"))
            out.append(line)
        return out

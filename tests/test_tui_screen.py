"""Screen-level regression tests: emulate a real terminal with pyte.

FakeTerminal records the renderer's raw ANSI stream; feeding it into
pyte.Screen shows what a real terminal would display. These tests assert on
the *final screen contents*, which is where cursor-positioning bugs show up
(output-stream assertions in test_tui_core.py cannot catch them).
"""

from __future__ import annotations

import asyncio

import pyte
from test_agent import make_agent

from pion.controller import AgentSessionController
from pion.session import SessionManager
from pion.tui import PionTUI, TUIStatus
from pion.tui.core import CURSOR_MARKER, Component, FakeTerminal, InlineRenderer
from pion.tui.theme import load_theme

THEME = load_theme("dark", truecolor=True)
COLS, ROWS = 80, 24


class Screen:
    """FakeTerminal + pyte emulation of the visible screen."""

    def __init__(self, columns: int = COLS, rows: int = ROWS) -> None:
        self.terminal = FakeTerminal(columns, rows)
        self._screen = pyte.Screen(columns, rows)
        self._stream = pyte.ByteStream(self._screen)
        self._fed = 0

    def feed_new_output(self) -> None:
        data = "".join(self.terminal.written[self._fed :])
        self._fed = len(self.terminal.written)
        if data:
            self._stream.feed(data.encode("utf-8"))

    def lines(self) -> list[str]:
        self.feed_new_output()
        return ["".join(row).rstrip() for row in self._screen.display]

    def text(self) -> str:
        return "\n".join(self.lines())


class Lines(Component):
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, width: int) -> list[str]:
        return list(self.lines)


def test_unchanged_frame_repaint_keeps_screen():
    """A parked cursor must not shift the next repaint (screenshot 1 bug)."""
    screen = Screen()
    comp = Lines(["header", "─" * 5, "ab" + CURSOR_MARKER + "cd", "─" * 5, "footer"])
    renderer = InlineRenderer(screen.terminal, comp)
    renderer.render_now()
    before = screen.lines()
    # Second render with an unchanged frame (cursor parked mid-frame).
    renderer.render_now()
    after = screen.lines()
    assert before == after
    assert "header" in screen.text() and "footer" in screen.text()


def test_edit_after_parked_cursor_lands_on_editor_row():
    """Typing must echo on the editor row, not above the frame (screenshot 2)."""
    screen = Screen()
    comp = Lines(["header", "─" * 5, "" + CURSOR_MARKER, "─" * 5, "footer"])
    renderer = InlineRenderer(screen.terminal, comp)
    renderer.render_now()
    comp.lines = ["header", "─" * 5, "你" + CURSOR_MARKER, "─" * 5, "footer"]
    renderer.render_now()
    rows = screen.lines()
    editor_row = next(i for i, line in enumerate(rows) if "你" in line)
    header_row = next(i for i, line in enumerate(rows) if "header" in line)
    footer_row = next(i for i, line in enumerate(rows) if "footer" in line)
    assert header_row < editor_row < footer_row
    # Nothing above the header.
    assert all(not line.strip() for line in rows[:header_row])


def test_growth_after_parked_cursor_appends_below():
    screen = Screen()
    comp = Lines(["one", "x" + CURSOR_MARKER, "end"])
    renderer = InlineRenderer(screen.terminal, comp)
    renderer.render_now()
    comp.lines = ["one", "x", "inserted", "end"]
    renderer.render_now()
    rows = screen.lines()
    assert rows.index("inserted") == rows.index("x") + 1
    assert screen.text().count("end") == 1


async def test_full_session_screen(tmp_path):
    """End-to-end through PionTUI: one footer, no duplicates, no stray '['."""

    def make():
        agent, _, _ = make_agent([{"text": "screen reply OK"}])
        session_path = tmp_path / "s.jsonl"
        controller = AgentSessionController(
            agent, SessionManager(session_path), session_path
        )
        screen = Screen()
        tui = PionTUI(
            controller,
            TUIStatus(project="proj", sandbox="off"),
            theme=THEME,
            terminal=screen.terminal,
        )
        return tui, screen

    tui, screen = make()
    task = asyncio.ensure_future(tui.run_async())
    await asyncio.sleep(0.1)
    screen.feed_new_output()

    startup = screen.text()
    assert startup.count("pion v") == 1
    assert startup.count("session name") == 0  # sanity: no placeholder leaks

    tui.terminal.feed(b"hello\r")
    await asyncio.sleep(0.2)
    text = screen.text()
    assert "screen reply OK" in text
    assert "hello" in text
    # Exactly one footer (path line) and one header on screen.
    assert text.count("s.jsonl") == 1
    assert text.count("pion v") == 1
    # No mangled escape remnants.
    assert "[" not in text.replace("[non-text", "")

    tui.quit()
    await asyncio.wait_for(task, timeout=2)

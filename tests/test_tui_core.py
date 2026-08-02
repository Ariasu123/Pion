"""Unit tests for the pion-tui core kernel."""

from __future__ import annotations

import asyncio

from pion.tui.core import (
    Component,
    Container,
    FakeTerminal,
    InlineRenderer,
    KeyDecoder,
    KeyEvent,
    RenderError,
    Spacer,
    apply_bg,
    pad_line,
    slice_by_column,
    strip_ansi,
    truncate_to_width,
    visible_width,
    wrap_text_with_ansi,
)
from pion.tui.core.overlays import OverlayOptions

# -- text_utils -----------------------------------------------------------


def test_visible_width_plain_and_wide():
    assert visible_width("hello") == 5
    assert visible_width("你好") == 4
    assert visible_width("á") == 1  # combining mark


def test_visible_width_ignores_ansi():
    assert visible_width("\x1b[31mred\x1b[0m") == 3
    assert visible_width("\x1b]8;;http://x\x07link\x1b]8;;\x07") == 4


def test_strip_ansi():
    assert strip_ansi("\x1b[1;31mbold red\x1b[0m!") == "bold red!"


def test_truncate_to_width_plain():
    assert truncate_to_width("hello world", 8) == "hello wo"
    assert truncate_to_width("你好世界", 5) == "你好"  # wide char boundary
    assert truncate_to_width("hi", 5) == "hi"


def test_truncate_to_width_keeps_ansi_and_resets():
    out = truncate_to_width("\x1b[31mhello\x1b[0m", 3)
    assert strip_ansi(out).startswith("hel")
    assert "\x1b[31m" in out
    assert out.endswith("\x1b[0m") or "\x1b[0m" in out


def test_truncate_with_tail():
    out = truncate_to_width("hello world", 8, tail="…")
    assert visible_width(out) <= 8
    assert out.endswith("…")


def test_slice_by_column_plain():
    assert slice_by_column("hello world", 0, 5) == "hello\x1b[0m"
    assert slice_by_column("hello world", 6, 11) == "world\x1b[0m"
    assert slice_by_column("hello", 2, 2) == ""


def test_slice_by_column_preserves_style():
    line = "ab\x1b[31mcd\x1b[0mef"
    out = slice_by_column(line, 2, 4)
    assert strip_ansi(out) == "cd"
    assert "\x1b[31m" in out  # style re-emitted at slice start


def test_pad_line():
    assert pad_line("hi", 5) == "hi   "
    assert pad_line("hello", 5) == "hello"


def test_apply_bg_pads_and_repaints_resets():
    bg = "\x1b[48;2;1;2;3m"
    out = apply_bg("\x1b[31mhi\x1b[0m", 6, bg)
    assert visible_width(out) == 6
    assert out.startswith(bg)
    # Interior reset re-opens the bg.
    assert "\x1b[0m" + bg in out


def test_wrap_plain_words():
    lines = wrap_text_with_ansi("the quick brown fox", 9)
    assert lines == ["the quick", "brown fox"]


def test_wrap_hard_breaks_long_words():
    lines = wrap_text_with_ansi("abcdefghij", 4)
    assert lines == ["abcd", "efgh", "ij"]


def test_wrap_cjk():
    lines = wrap_text_with_ansi("你好世界啊", 5)
    assert [visible_width(line) for line in lines] == [4, 4, 2]


def test_wrap_preserves_ansi_and_recstyles_continuation():
    lines = wrap_text_with_ansi("\x1b[31mone two three\x1b[0m", 7)
    assert len(lines) == 2
    assert strip_ansi(lines[0]) == "one two"
    assert lines[1].startswith("\x1b[31m")
    assert strip_ansi(lines[1]) == "three"


def test_wrap_multiline():
    assert wrap_text_with_ansi("ab\ncd", 10) == ["ab", "cd"]


# -- keys -------------------------------------------------------------------


def decode(data: bytes, final: bool = True):
    decoder = KeyDecoder()
    events = decoder.feed(data)
    if final:
        events += decoder.flush()
    return events


def test_key_text_and_enter():
    assert decode(b"hi\r") == [KeyEvent("text", "hi"), KeyEvent("enter")]


def test_key_ctrl_and_specials():
    assert decode(b"\x0f") == [KeyEvent("ctrl+o")]
    assert decode(b"\n") == [KeyEvent("ctrl+j")]
    assert decode(b"\x7f") == [KeyEvent("backspace")]
    assert decode(b"\t") == [KeyEvent("tab")]


def test_key_arrows_and_modifiers():
    assert decode(b"\x1b[A") == [KeyEvent("up")]
    assert decode(b"\x1b[1;5A") == [KeyEvent("ctrl+up")]
    assert decode(b"\x1b[1;2B") == [KeyEvent("shift+down")]
    assert decode(b"\x1b[3~") == [KeyEvent("delete")]
    assert decode(b"\x1b[5~") == KeyEventList("pageup")


def test_key_escape_vs_alt():
    # Lone ESC held back until flush.
    decoder = KeyDecoder()
    assert decoder.feed(b"\x1b") == []
    assert decoder.flush() == [KeyEvent("escape")]
    # ESC + char is alt.
    assert decode(b"\x1ba") == [KeyEvent("alt+a", "a")]
    assert decode(b"\x1b\r") == [KeyEvent("alt+enter")]
    assert decode(b"\x1b\x1b[A") == [KeyEvent("alt+up")]


def test_key_batched_and_utf8():
    events = decode("héllo".encode())
    assert events == [KeyEvent("text", "héllo")]
    # Split UTF-8 across feeds.
    decoder = KeyDecoder()
    assert decoder.feed("你".encode()[:1]) == []
    assert decoder.feed("你".encode()[1:]) == [KeyEvent("text", "你")]


def test_key_bracketed_paste():
    events = decode(b"\x1b[200~line1\r\nline2\x1b[201~")
    assert events == [KeyEvent("paste", "line1\nline2")]


def test_key_kitty_shift_enter():
    assert decode(b"\x1b[13;2u") == [KeyEvent("shift+enter")]


def KeyEventList(*names):
    return [KeyEvent(n) for n in names]


# -- renderer -----------------------------------------------------------------


class Lines(Component):
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, width: int) -> list[str]:
        return list(self.lines)


def render_sync(renderer: InlineRenderer) -> None:
    renderer.render_now()


def test_renderer_first_paint():
    term = FakeTerminal(20, 10)
    renderer = InlineRenderer(term, Lines(["one", "two"]))
    render_sync(renderer)
    out = term.output()
    assert out.startswith("\x1b[?2026h")
    assert out.endswith("\x1b[?2026l")
    assert "one\x1b[0m\r\n" in out and "two" in out
    assert "\x1b[?25l" in out  # cursor hidden during paint


def test_renderer_diff_rewrites_only_changed_line():
    term = FakeTerminal(20, 10)
    comp = Lines(["one", "two", "three"])
    renderer = InlineRenderer(term, comp)
    render_sync(renderer)
    term.written.clear()
    comp.lines = ["one", "TWO", "three"]
    render_sync(renderer)
    out = term.output()
    assert "TWO" in out
    assert "one" not in out  # untouched lines are not rewritten
    assert "\x1b[1A" in out  # moved up from last line to line index 1


def test_renderer_append():
    term = FakeTerminal(20, 10)
    comp = Lines(["one"])
    renderer = InlineRenderer(term, comp)
    render_sync(renderer)
    term.written.clear()
    comp.lines = ["one", "two"]
    render_sync(renderer)
    out = term.output()
    assert "\r\n" in out and "two" in out


def test_renderer_shrink_clears_leftover():
    term = FakeTerminal(20, 10)
    comp = Lines(["a", "b", "c"])
    renderer = InlineRenderer(term, comp)
    render_sync(renderer)
    term.written.clear()
    comp.lines = ["a"]
    render_sync(renderer)
    out = term.output()
    assert out.count("\x1b[2K") == 2  # two stale lines erased


def test_renderer_full_redraw_on_width_change():
    term = FakeTerminal(20, 10)
    comp = Lines(["a", "b"])
    renderer = InlineRenderer(term, comp)
    render_sync(renderer)
    term.written.clear()
    term.resize(30, 10)
    renderer.on_resize()
    render_sync(renderer)
    out = term.output()
    assert "\x1b[2J\x1b[H\x1b[3J" in out
    assert "a\x1b[0m\r\n" in out


def test_renderer_overwide_line_raises():
    term = FakeTerminal(5, 10)
    renderer = InlineRenderer(term, Lines(["this is too long"]))
    try:
        render_sync(renderer)
    except RenderError:
        return
    raise AssertionError("expected RenderError")


def test_renderer_cursor_marker_positions_cursor():
    from pion.tui.core import CURSOR_MARKER

    term = FakeTerminal(20, 10)
    comp = Lines(["ab" + CURSOR_MARKER + "cd", "next"])
    renderer = InlineRenderer(term, comp)
    render_sync(renderer)
    out = term.output()
    assert CURSOR_MARKER not in out
    assert "\x1b[1A" in out  # back up to the marked line
    assert "\x1b[2C" in out  # and right to column 2
    assert out.endswith("\x1b[?25h\x1b[?2026l")  # cursor shown before sync end


def test_renderer_restores_parked_cursor_before_next_paint():
    from pion.tui.core import CURSOR_MARKER

    term = FakeTerminal(20, 10)
    comp = Lines(["ab" + CURSOR_MARKER + "cd", "next"])
    renderer = InlineRenderer(term, comp)
    render_sync(renderer)  # cursor ends parked on line 0, 1 row above the end
    term.written.clear()
    comp.lines = ["ab" + CURSOR_MARKER + "cd!", "next"]
    render_sync(renderer)
    out = term.output()
    # The paint must first move back down to the frame's last line.
    assert out.startswith("\x1b[?2026h\x1b[?25l\x1b[1B\r")


def test_renderer_close_restores_cursor_position():
    from pion.tui.core import CURSOR_MARKER

    term = FakeTerminal(20, 10)
    comp = Lines(["ab" + CURSOR_MARKER + "cd", "next"])
    renderer = InlineRenderer(term, comp)
    render_sync(renderer)
    term.written.clear()
    renderer.close()
    assert term.output().startswith("\x1b[1B\r\r\n")


def test_renderer_overlay_composites():
    term = FakeTerminal(20, 10)
    base = Lines(["base one", "base two", "base three"])
    renderer = InlineRenderer(term, base)
    renderer.overlays.push(Lines(["OVERLAY"]), OverlayOptions(width=10))
    render_sync(renderer)
    out = term.output()
    assert "OVERLAY" in out
    # Overlay padded to its width.
    assert "OVERLAY   " in out


def test_renderer_container_and_spacer():
    term = FakeTerminal(20, 10)
    root = Container([Lines(["a"]), Spacer(2), Lines(["b"])])
    renderer = InlineRenderer(term, root)
    render_sync(renderer)
    out = term.output()
    assert "a\x1b[0m\r\n" in out and "b" in out


def test_request_render_coalesces():
    async def run() -> str:
        term = FakeTerminal(20, 10)
        comp = Lines(["x"])
        renderer = InlineRenderer(term, comp)
        renderer.request_render()
        renderer.request_render()
        assert term.output() == ""  # not yet painted
        await asyncio.sleep(0.05)
        return term.output()

    out = asyncio.run(run())
    assert "x" in out

"""Tests for the prompt editor."""

from __future__ import annotations

from pion.tui.components import CombinedAutocompleteProvider, Editor, SlashCommand
from pion.tui.core import CURSOR_MARKER, KeyEvent, strip_ansi, visible_width
from pion.tui.theme import load_theme

THEME = load_theme("dark", truecolor=True)


def make_editor(**kwargs) -> Editor:
    kwargs.setdefault("theme", THEME)
    return Editor(**kwargs)


def type_text(editor: Editor, text: str) -> None:
    editor.handle_input(KeyEvent("text", text))


def render_plain(editor: Editor, width: int = 40) -> list[str]:
    return [strip_ansi(line).replace(CURSOR_MARKER, "") for line in editor.render(width)]


def test_type_and_submit():
    submitted = []
    editor = make_editor(on_submit=submitted.append)
    type_text(editor, "hello")
    editor.handle_input(KeyEvent("enter"))
    assert submitted == ["hello"]
    assert editor.text == ""
    assert editor.history == ["hello"]


def test_multiline_with_ctrl_j():
    editor = make_editor()
    type_text(editor, "one")
    editor.handle_input(KeyEvent("ctrl+j"))
    type_text(editor, "two")
    assert editor.text == "one\ntwo"
    assert editor.cursor_row == 1 and editor.cursor_col == 3


def test_backspace_merges_lines():
    editor = make_editor()
    editor.set_text("ab\ncd")
    editor.cursor_row, editor.cursor_col = 1, 0
    editor.handle_input(KeyEvent("backspace"))
    assert editor.text == "abcd"
    assert editor.cursor_row == 1 - 1 and editor.cursor_col == 2


def test_word_navigation_and_delete():
    editor = make_editor()
    editor.set_text("foo bar baz")
    editor.handle_input(KeyEvent("alt+left"))
    assert editor.cursor_col == 8
    editor.handle_input(KeyEvent("alt+backspace"))
    assert editor.text == "foo baz"
    editor.handle_input(KeyEvent("ctrl+w"))
    assert editor.text == "baz"


def test_kill_operations():
    editor = make_editor()
    editor.set_text("hello world")
    editor.cursor_col = 5
    editor.handle_input(KeyEvent("ctrl+k"))
    assert editor.text == "hello"
    editor.handle_input(KeyEvent("ctrl+u"))
    assert editor.text == ""


def test_history_navigation():
    editor = make_editor()
    editor.history = ["first", "second"]
    editor.handle_input(KeyEvent("up"))
    assert editor.text == "second"
    editor.handle_input(KeyEvent("up"))
    assert editor.text == "first"
    editor.handle_input(KeyEvent("down"))
    editor.handle_input(KeyEvent("down"))
    assert editor.text == ""


def test_paste_multiline():
    editor = make_editor()
    editor.handle_input(KeyEvent("paste", "a\nb\nc"))
    assert editor.text == "a\nb\nc"
    assert editor.cursor_row == 2


def test_render_borders_and_cursor_marker():
    editor = make_editor()
    type_text(editor, "hi")
    lines = editor.render(20)
    assert strip_ansi(lines[0]) == "─" * 20
    assert strip_ansi(lines[-1]) == "─" * 20
    body = lines[1]
    assert CURSOR_MARKER in body
    assert "\x1b[7m" in body  # reverse-video cursor cell
    assert all(visible_width(line.replace(CURSOR_MARKER, "")) <= 20 for line in lines)


def test_long_line_wraps_without_overflow():
    editor = make_editor()
    type_text(editor, "x" * 55)
    lines = editor.render(20)
    body = lines[1:-1]
    assert len(body) == 3  # 55 chars wrapped at 20
    assert all(
        visible_width(line.replace(CURSOR_MARKER, "")) <= 20 for line in body
    )


def test_autocomplete_flow(tmp_path):
    provider = CombinedAutocompleteProvider(
        [SlashCommand("tree", "session tree"), SlashCommand("stats", "usage")],
        tmp_path,
    )
    submitted = []
    editor = make_editor(on_submit=submitted.append, autocomplete=provider)
    type_text(editor, "/t")
    assert editor.autocomplete_open
    lines = render_plain(editor)
    assert any("/tree" in line for line in lines)
    editor.handle_input(KeyEvent("enter"))  # accepts suggestion, not submit
    assert submitted == []
    assert editor.text == "/tree "
    editor.handle_input(KeyEvent("enter"))
    assert submitted == ["/tree"]


def test_autocomplete_dismiss_on_escape(tmp_path):
    provider = CombinedAutocompleteProvider([SlashCommand("tree")], tmp_path)
    editor = make_editor(autocomplete=provider)
    type_text(editor, "/t")
    assert editor.autocomplete_open
    editor.handle_input(KeyEvent("escape"))
    assert not editor.autocomplete_open
    assert editor.text == "/t"


def test_cjk_cursor_positioning():
    editor = make_editor()
    type_text(editor, "你好")
    editor.handle_input(KeyEvent("left"))
    lines = editor.render(30)
    body = lines[1]
    marker_at = body.index(CURSOR_MARKER)
    # Marker placed before the second (wide) character.
    assert strip_ansi(body[:marker_at]) == "你"

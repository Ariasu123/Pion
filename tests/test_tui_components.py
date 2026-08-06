"""Tests for the general-purpose components."""

from __future__ import annotations

from pion.tui.components import (
    Box,
    CombinedAutocompleteProvider,
    DynamicBorder,
    Loader,
    Markdown,
    SelectItem,
    SelectList,
    SlashCommand,
    Text,
    fuzzy_filter,
    fuzzy_match,
)
from pion.tui.core import KeyEvent, strip_ansi, visible_width
from pion.tui.theme import load_theme

THEME = load_theme("dark", truecolor=True)


def test_text_wraps_and_pads():
    text = Text("hello world foo", pad_x=1, theme=THEME)
    lines = text.render(10)
    assert all(visible_width(line) <= 10 for line in lines)
    assert strip_ansi(lines[0]).startswith(" ")


def test_text_bg_band_full_width():
    text = Text("hi", pad_x=1, pad_y=1, bg="userMessageBg", theme=THEME)
    lines = text.render(20)
    assert len(lines) == 3
    assert all("\x1b[48;2;" in line for line in lines)
    assert all(visible_width(line) == 20 for line in lines)


def test_box_pads_children_with_bg():
    box = Box(pad_x=1, pad_y=1, bg="toolSuccessBg", children=[Text("ok", pad_x=0)], theme=THEME)
    lines = box.render(16)
    assert len(lines) == 3
    assert all(visible_width(line) == 16 for line in lines)
    assert all("\x1b[48;2;40;50;40m" in line for line in lines)


def test_dynamic_border():
    border = DynamicBorder(theme=THEME)
    (line,) = border.render(12)
    assert strip_ansi(line) == "─" * 12


def test_loader_renders_spinner_and_hint():
    loader = Loader("Working…", theme=THEME)
    lines = loader.render(40)
    plain = [strip_ansi(line) for line in lines]
    assert plain[0] == ""
    assert "Working…" in plain[1] and "esc cancel" in plain[1]


def test_markdown_basic_and_width():
    md = Markdown("# Title\n\nSome **bold** text and `code`.", theme=THEME)
    lines = md.render(50)
    plain = "\n".join(strip_ansi(line) for line in lines)
    assert "Title" in plain and "bold" in plain
    assert all(visible_width(line) <= 50 for line in lines)


def test_markdown_wraps_long_cjk_paragraph():
    # One long unbroken CJK paragraph must be folded to width, never exceed it.
    text = "已读完。这是一份关于大模型推理引擎方向的调研报告，" * 10
    md = Markdown(text, theme=THEME)
    lines = md.render(50)
    assert len(lines) > 1
    assert all(visible_width(line) <= 50 for line in lines)


def test_markdown_streaming_closes_fence():
    md = Markdown("```py\nprint(1)\n", streaming=True, theme=THEME)
    lines = md.render(60)
    plain = "\n".join(strip_ansi(line) for line in lines)
    assert "print(1)" in plain


def test_markdown_cache_invalidate():
    md = Markdown("one", theme=THEME)
    first = md.render(40)
    md.set_text("two")
    assert "two" in "\n".join(strip_ansi(l) for l in md.render(40))
    assert "one" not in "\n".join(strip_ansi(l) for l in md.render(40))
    assert "one" in "\n".join(strip_ansi(l) for l in first)


def test_fuzzy():
    assert fuzzy_match("mdl", "model") is not None
    assert fuzzy_match("xyz", "model") is None
    items = fuzzy_filter("tre", ["/tree", "/stats", "/help"], lambda s: s)
    assert items[0] == "/tree"


def test_select_list_navigation_and_select():
    chosen = []
    sl = SelectList(
        [SelectItem("a", "alpha"), SelectItem("b", "beta")],
        on_select=chosen.append,
        theme=THEME,
    )
    sl.handle_input(KeyEvent("down"))
    sl.handle_input(KeyEvent("enter"))
    assert chosen == ["b"]


def test_select_list_filter():
    sl = SelectList(
        [SelectItem("a", "alpha"), SelectItem("b", "beta")], theme=THEME
    )
    sl.handle_input(KeyEvent("text", "b"))
    sl.handle_input(KeyEvent("text", "e"))
    assert [i.label for i in sl.filtered()] == ["beta"]
    lines = sl.render(30)
    assert "beta" in strip_ansi("".join(lines))
    assert all(visible_width(line) <= 30 for line in lines)


def test_select_list_selected_row_has_bg():
    sl = SelectList([SelectItem("a", "alpha")], filterable=False, theme=THEME)
    (line,) = sl.render(20)
    assert "\x1b[48;2;58;58;74m" in line  # selectedBg #3a3a4a
    assert visible_width(line) == 20


def test_autocomplete_slash(tmp_path):
    provider = CombinedAutocompleteProvider(
        [SlashCommand("tree", "session tree"), SlashCommand("stats", "usage")],
        tmp_path,
    )
    suggestion = provider.suggest("/tr", 3)
    assert suggestion is not None
    assert suggestion.replace_start == 0
    assert suggestion.items[0].value == "/tree"
    assert provider.suggest("hello /tr", 9) is None  # not at line start


def test_autocomplete_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("")
    (tmp_path / "README.md").write_text("")
    provider = CombinedAutocompleteProvider([], tmp_path)
    suggestion = provider.suggest("check @src/ma", len("check @src/ma"))
    assert suggestion is not None
    assert suggestion.items[0].value == "@src/main.py"

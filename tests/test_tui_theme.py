"""Tests for the JSON theme system."""

from __future__ import annotations

from pion.tui.theme import Theme, list_themes, load_theme
from pion.tui.theme.theme import _rgb_to_256


def test_load_dark_theme():
    theme = load_theme("dark", truecolor=True)
    assert theme.name == "dark"
    out = theme.fg("error", "boom")
    assert out.startswith("\x1b[38;2;204;102;102m")  # #cc6666
    assert out.endswith("\x1b[0m")
    assert "boom" in out


def test_var_indirection():
    theme = load_theme("dark", truecolor=True)
    # mdLink → var mdLink → hex
    assert theme.fg_open("mdLink") == "\x1b[38;2;129;162;190m"


def test_bg_token():
    theme = load_theme("dark", truecolor=True)
    assert theme.fg_open("toolErrorBg") != theme.bg_open("toolErrorBg")
    assert theme.bg_open("toolErrorBg") == "\x1b[48;2;60;40;40m"


def test_256_fallback():
    theme = load_theme("dark", truecolor=False)
    open_seq = theme.fg_open("error")
    assert open_seq.startswith("\x1b[38;5;")


def test_unknown_token_no_style():
    theme = Theme("t", {"colors": {}}, truecolor=True)
    assert theme.fg("nope", "plain") == "plain"
    assert theme.bg_open("nope") == ""


def test_styles():
    theme = load_theme("dark")
    assert theme.bold("x") == "\x1b[1mx\x1b[0m"
    assert "\x1b[2m" in theme.dim("x")
    styled = theme.styled("x", "accent", bold=True)
    assert "\x1b[1m" in styled and theme.fg_open("accent") in styled


def test_list_themes_and_light():
    assert {"dark", "light"} <= set(list_themes())
    light = load_theme("light", truecolor=True)
    assert light.fg_open("text") != load_theme("dark", truecolor=True).fg_open("text")


def test_rgb_to_256():
    assert _rgb_to_256(255, 255, 255) == 231
    assert _rgb_to_256(0, 0, 0) == 16
    assert 16 <= _rgb_to_256(204, 102, 102) <= 231

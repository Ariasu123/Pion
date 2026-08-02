"""JSON theme system (port of pi's theme mechanism).

Themes are JSON files with CSS-like `vars` (palette) and semantic `colors`
tokens referencing vars, hex values, or ANSI color names. `Theme.fg/bg`
wrap text in SGR sequences; hex colors render as truecolor with a weighted
xterm-256 fallback when the terminal does not advertise 24-bit support.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..core.text_utils import SGR_RESET

_ANSI_NAMES = {
    "black": 0,
    "red": 1,
    "green": 2,
    "yellow": 3,
    "blue": 4,
    "magenta": 5,
    "cyan": 6,
    "white": 7,
    "brightBlack": 8,
    "brightRed": 9,
    "brightGreen": 10,
    "brightYellow": 11,
    "brightBlue": 12,
    "brightMagenta": 13,
    "brightCyan": 14,
    "brightWhite": 15,
}

_THEME_DIR = Path(__file__).parent


def supports_truecolor() -> bool:
    return os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgb_to_256(r: int, g: int, b: int) -> int:
    """Approximate an RGB color with the xterm-256 palette."""
    # Grayscale ramp when channels are close.
    if max(r, g, b) - min(r, g, b) < 12:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + round((r - 8) / 247 * 24)

    def cube(v: int) -> int:
        return 0 if v < 48 else min(5, round((v - 55) / 40))

    rc, gc, bc = cube(r), cube(g), cube(b)
    return 16 + 36 * rc + 6 * gc + bc


class Theme:
    def __init__(self, name: str, data: dict, truecolor: bool = True) -> None:
        self.name = name
        self._truecolor = truecolor
        self._vars: dict[str, str] = dict(data.get("vars", {}))
        self._colors: dict[str, str] = dict(data.get("colors", {}))
        self._fg_cache: dict[str, str] = {}
        self._bg_cache: dict[str, str] = {}

    # -- resolution ------------------------------------------------------

    def _resolve(self, token: str) -> str:
        value = self._colors.get(token, token)
        seen = set()
        while value in self._vars and value not in seen:
            seen.add(value)
            value = self._vars[value]
        return value

    def _sgr_open(self, token: str, background: bool) -> str:
        value = self._resolve(token)
        if value.startswith("#"):
            r, g, b = _hex_to_rgb(value)
            if self._truecolor:
                code = f"{'48' if background else '38'};2;{r};{g};{b}"
            else:
                code = f"{'48' if background else '38'};5;{_rgb_to_256(r, g, b)}"
            return f"\x1b[{code}m"
        if value in _ANSI_NAMES:
            idx = _ANSI_NAMES[value]
            if background:
                number = 40 + idx if idx < 8 else 100 + idx - 8
            else:
                number = 30 + idx if idx < 8 else 90 + idx - 8
            return f"\x1b[{number}m"
        return ""  # unknown token: no styling

    def fg_open(self, token: str) -> str:
        if token not in self._fg_cache:
            self._fg_cache[token] = self._sgr_open(token, background=False)
        return self._fg_cache[token]

    def bg_open(self, token: str) -> str:
        if token not in self._bg_cache:
            self._bg_cache[token] = self._sgr_open(token, background=True)
        return self._bg_cache[token]

    # -- styling API -------------------------------------------------------

    def fg(self, token: str, text: str) -> str:
        open_seq = self.fg_open(token)
        return f"{open_seq}{text}{SGR_RESET}" if open_seq else text

    def bg(self, token: str, text: str) -> str:
        open_seq = self.bg_open(token)
        return f"{open_seq}{text}{SGR_RESET}" if open_seq else text

    @staticmethod
    def bold(text: str) -> str:
        return f"\x1b[1m{text}{SGR_RESET}"

    @staticmethod
    def dim(text: str) -> str:
        return f"\x1b[2m{text}{SGR_RESET}"

    @staticmethod
    def italic(text: str) -> str:
        return f"\x1b[3m{text}{SGR_RESET}"

    @staticmethod
    def underline(text: str) -> str:
        return f"\x1b[4m{text}{SGR_RESET}"

    def styled(self, text: str, fg: str | None = None, *, bold=False, dim=False, italic=False) -> str:
        codes = ""
        if bold:
            codes += "\x1b[1m"
        if dim:
            codes += "\x1b[2m"
        if italic:
            codes += "\x1b[3m"
        if fg:
            codes += self.fg_open(fg)
        return f"{codes}{text}{SGR_RESET}" if codes else text


def load_theme(name: str = "dark", truecolor: bool | None = None) -> Theme:
    path = _THEME_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Unknown theme {name!r}; expected one of {list_themes()}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if truecolor is None:
        truecolor = supports_truecolor()
    return Theme(name, data, truecolor)


def list_themes() -> list[str]:
    return sorted(p.stem for p in _THEME_DIR.glob("*.json"))


_current: Theme | None = None


def get_theme() -> Theme:
    global _current
    if _current is None:
        _current = load_theme("dark")
    return _current


def set_theme(theme: Theme) -> None:
    global _current
    _current = theme

"""ANSI-aware display-width helpers (port of pi-tui's utils.ts core).

All rendered lines are measured in terminal display cells, not characters.
Wide (CJK) characters count as 2 cells, combining marks as 0. ANSI escape
sequences are preserved by every transformation and never counted.
"""

from __future__ import annotations

import re

from wcwidth import wcwidth

# CSI sequences, OSC sequences (terminated by BEL or ST), and 2-char escapes.
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC
    r"|\x1b[@-Z\\-_]"  # 2-char escape
)
_SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")

SGR_RESET = "\x1b[0m"
# Reset SGR and any OSC-8 hyperlink before switching line segments.
SEGMENT_RESET = "\x1b[0m\x1b]8;;\x07"


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def char_width(ch: str) -> int:
    w = wcwidth(ch)
    return max(0, w)


def visible_width(text: str) -> int:
    return sum(char_width(ch) for ch in strip_ansi(text))


def _tokens(text: str):
    """Yield (piece, width) pairs; ANSI sequences have width 0."""
    pos = 0
    for match in _ANSI_RE.finditer(text):
        if match.start() > pos:
            for ch in text[pos : match.start()]:
                yield ch, char_width(ch)
        yield match.group(0), 0
        pos = match.end()
    for ch in text[pos:]:
        yield ch, char_width(ch)


def _active_sgr(text: str) -> str:
    """Return the SGR state (concatenated codes) active at the end of text."""
    state = ""
    for match in _SGR_RE.finditer(text):
        params = match.group(1)
        if params in ("", "0"):
            state = ""
        else:
            state += match.group(0)
    return state


def truncate_to_width(text: str, width: int, tail: str = "") -> str:
    """Truncate to at most `width` cells, appending `tail` when truncated."""
    if visible_width(text) <= width:
        return text
    budget = width - visible_width(tail)
    budget = max(budget, 0)
    out: list[str] = []
    used = 0
    for piece, w in _tokens(text):
        if w == 0:
            out.append(piece)
            continue
        if used + w > budget:
            break
        out.append(piece)
        used += w
    result = "".join(out)
    if "\x1b" in result:
        result += SGR_RESET
    return result + tail


def slice_by_column(text: str, start: int, end: int) -> str:
    """Return the display-column slice [start, end), preserving styles.

    SGR state active at `start` is re-emitted so the slice renders with the
    same styling; the slice always ends with an SGR reset.
    """
    if end <= start:
        return ""
    out: list[str] = []
    col = 0
    started = False
    for piece, w in _tokens(text):
        if w == 0:
            if started:
                out.append(piece)
            elif _SGR_RE.fullmatch(piece):
                if piece in ("\x1b[m", SGR_RESET):
                    out.clear()
                else:
                    out.append(piece)
            continue
        if col + w > end:
            break
        if col + w > start:
            started = True
            out.append(piece)
        col += w
    if not out:
        return ""
    return "".join(out) + SGR_RESET


def pad_line(text: str, width: int, bg: str = "") -> str:
    """Pad a line with spaces to exactly `width` cells.

    When `bg` (an SGR open sequence) is given, the padding is painted with it.
    """
    missing = width - visible_width(text)
    if missing <= 0:
        return text
    if bg:
        return text + bg + " " * missing + SGR_RESET
    return text + " " * missing


def apply_bg(text: str, width: int, bg: str) -> str:
    """Paint the whole line with background `bg`, padding to `width` cells.

    Interior SGR resets are rewritten to re-open the background so it
    survives across embedded styled spans.
    """
    repainted = text.replace(SGR_RESET, SGR_RESET + bg).replace(
        "\x1b[m", "\x1b[m" + bg
    )
    missing = width - visible_width(text)
    padding = " " * missing if missing > 0 else ""
    return bg + repainted + padding + SGR_RESET


def repaint(text: str, sgr_open: str) -> str:
    """Re-open `sgr_open` after every interior reset; used to force a base
    style (e.g. dim/italic) over pre-styled content."""
    if not sgr_open:
        return text
    return sgr_open + text.replace(SGR_RESET, SGR_RESET + sgr_open) + SGR_RESET


def wrap_text_with_ansi(text: str, width: int) -> list[str]:
    """Word-wrap text to `width` cells, preserving ANSI styling.

    Breaks prefer the last space on the line; unbreakable long runs are
    hard-wrapped. Active SGR state is carried onto continuation lines.
    """
    width = max(width, 1)
    out: list[str] = []
    for raw_line in text.replace("\t", "   ").split("\n"):
        out.extend(_wrap_line(raw_line, width))
    return out


def _wrap_line(line: str, width: int) -> list[str]:
    if visible_width(line) <= width:
        return [line]
    tokens = list(_tokens(line))
    lines: list[str] = []
    cur: list[tuple[str, int]] = []  # tokens on the current output line
    cur_w = 0
    last_space: int | None = None  # index into `cur` of last breakable space
    idx = 0
    while idx < len(tokens):
        piece, w = tokens[idx]
        if w == 0:
            cur.append((piece, w))
            idx += 1
            continue
        if piece == " ":
            if cur_w == 0:
                idx += 1
                continue  # no leading spaces on continuation lines
            last_space = len(cur)
        if cur_w + w > width:
            if last_space is not None:
                emit, rest = cur[:last_space], cur[last_space + 1 :]
            else:
                emit, rest = cur, []
            emitted = "".join(p for p, _ in emit)
            lines.append(emitted)
            prefix = _active_sgr(emitted)
            cur = ([(prefix, 0)] if prefix else []) + rest
            cur_w = sum(wd for _, wd in rest)
            last_space = None
            if piece == " ":
                idx += 1  # the breaking space is consumed, not reprocessed
            continue  # reprocess the overflowing token on the new line
        cur.append((piece, w))
        cur_w += w
        idx += 1
    lines.append("".join(p for p, _ in cur))
    return lines

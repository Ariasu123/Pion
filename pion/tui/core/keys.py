"""Terminal input decoding (port of pi-tui's keys.ts, reduced).

Raw bytes from stdin are decoded into `KeyEvent`s. Canonical key names:
`enter`, `escape`, `tab`, `backspace`, `delete`, `up/down/left/right`,
`home`, `end`, `pageup`, `pagedown`, `space`, printable text as `text`,
bracketed paste as `paste`, and modifier combos like `ctrl+o`,
`alt+enter`, `shift+up`, `ctrl+shift+p`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class KeyEvent:
    key: str  # canonical name, "text", or "paste"
    text: str = ""  # payload for "text"/"paste"


def matches(event: KeyEvent, name: str) -> bool:
    return event.key == name


_PASTE_START = "\x1b[200~"
_PASTE_END = "\x1b[201~"

_CSI_RE = re.compile(rb"\x1b\[([0-9;?]*)([~uA-Za-z])")
_OSC_RE = re.compile(rb"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

_CSI_FINAL_KEYS = {
    "A": "up",
    "B": "down",
    "C": "right",
    "D": "left",
    "H": "home",
    "F": "end",
    "Z": "shift+tab",
}
_CSI_TILDE_KEYS = {
    1: "home",
    2: "insert",
    3: "delete",
    4: "end",
    5: "pageup",
    6: "pagedown",
    7: "home",
    8: "end",
}
# Kitty keyboard protocol code points that differ from plain text.
_KITTY_U_KEYS = {
    13: "enter",
    9: "tab",
    27: "escape",
    127: "backspace",
}
_MOD_BITS = ((1, "shift"), (2, "alt"), (4, "ctrl"))


def _apply_mods(base: str, modifier: int) -> str:
    modifier -= 1
    parts = [name for bit, name in _MOD_BITS if modifier & bit]
    if not parts:
        return base
    if len(base) == 1 and "shift" in parts:
        # shift+letter arrives as the uppercase letter already; keep others.
        parts = [p for p in parts if p != "shift"] if base.isalpha() else parts
        if not parts:
            return base
    return "+".join(parts) + "+" + base


def _decode_csi(params: str, final: str) -> KeyEvent | None:
    if params.startswith("?"):
        return None  # mode reports etc.
    fields = params.split(";") if params else []
    try:
        numbers = [int(f) for f in fields if f]
    except ValueError:
        return None
    modifier = numbers[1] if len(numbers) > 1 else 1
    if final == "~":
        number = numbers[0] if numbers else 0
        base = _CSI_TILDE_KEYS.get(number)
        if base is None:
            return None
        return KeyEvent(_apply_mods(base, modifier))
    if final == "u":  # kitty keyboard protocol
        code = numbers[0] if numbers else 0
        base = _KITTY_U_KEYS.get(code)
        if base is None:
            try:
                ch = chr(code)
            except ValueError:
                return None
            if modifier == 1 and ch.isprintable():
                return KeyEvent("text", ch)
            base = ch
        return KeyEvent(_apply_mods(base, modifier))
    base = _CSI_FINAL_KEYS.get(final)
    if base is None:
        return None
    return KeyEvent(_apply_mods(base, modifier))


class KeyDecoder:
    """Incremental byte-stream → KeyEvent decoder.

    `feed()` returns all complete events; an ambiguous trailing ESC is held
    back until `flush()` (called after a short idle timeout) resolves it as
    a bare `escape` key.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[KeyEvent]:
        self._buf.extend(data)
        return self._drain(final=False)

    def flush(self) -> list[KeyEvent]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[KeyEvent]:
        events: list[KeyEvent] = []
        buf = self._buf
        while buf:
            # Bracketed paste: swallow until the end marker.
            if buf.startswith(b"\x1b[200~"):
                end = buf.find(b"\x1b[201~", 6)
                if end == -1:
                    if not final:
                        break
                    text = bytes(buf[6:]).decode("utf-8", "replace")
                    buf.clear()
                    events.append(KeyEvent("paste", text))
                    continue
                text = bytes(buf[6:end]).decode("utf-8", "replace")
                del buf[: end + 6]
                events.append(KeyEvent("paste", text.replace("\r\n", "\n").replace("\r", "\n")))
                continue

            byte = buf[0]
            if byte != 0x1B:
                event = self._decode_non_escape(final)
                if event is None:
                    break  # incomplete multi-byte char; wait for more bytes
                events.append(event)
                if not buf:
                    break
                continue

            # Escape sequences.
            if len(buf) == 1:
                if not final:
                    break
                del buf[:1]
                events.append(KeyEvent("escape"))
                continue
            match = _CSI_RE.match(bytes(buf))
            if match:
                del buf[: match.end()]
                event = _decode_csi(match.group(1).decode(), match.group(2).decode())
                if event is not None:
                    events.append(event)
                continue
            if bytes(buf[:2]) == b"\x1b[":
                if not final:
                    break  # incomplete CSI; wait for more bytes
            osc = _OSC_RE.match(bytes(buf))
            if osc:
                del buf[: osc.end()]
                continue
            if bytes(buf[:2]) == b"\x1b]":
                if not final:
                    break
                del buf[:1]
                events.append(KeyEvent("escape"))
                continue
            # ESC + something else → alt+key (or alt+enter / alt+backspace).
            del buf[:1]
            inner = self._decode_single_char(final)
            if inner is None:
                events.append(KeyEvent("escape"))
                continue
            if inner.key == "text":
                events.append(KeyEvent("alt+" + inner.text.lower(), inner.text))
            elif inner.key.startswith(("alt+", "ctrl+alt+")):
                events.append(inner)
            elif inner.key == "enter":
                events.append(KeyEvent("alt+enter"))
            elif inner.key == "backspace":
                events.append(KeyEvent("alt+backspace"))
            elif inner.key == "tab":
                events.append(KeyEvent("alt+tab"))
            elif inner.key.startswith("ctrl+"):
                events.append(KeyEvent("ctrl+alt+" + inner.key[5:]))
            else:
                events.append(KeyEvent("alt+" + inner.key))
        return events

    def _decode_single_char(self, final: bool) -> KeyEvent | None:
        """Consume exactly one key (used after ESC for alt-combos)."""
        buf = self._buf
        if not buf:
            return None
        byte = buf[0]
        if byte == 0x1B:
            # ESC ESC ... → alt+sequence, e.g. alt+up = ESC ESC [ A
            match = _CSI_RE.match(bytes(buf))
            if match:
                del buf[: match.end()]
                event = _decode_csi(match.group(1).decode(), match.group(2).decode())
                if event is not None:
                    return KeyEvent("alt+" + event.key, event.text)
                return None
            if len(buf) == 1 and not final:
                return None
            del buf[:1]
            return KeyEvent("escape")
        if byte < 0x20 or byte == 0x7F:
            del buf[:1]
            return _decode_control(byte)
        if byte < 0x80:
            del buf[:1]
            return KeyEvent("text", chr(byte))
        # One multi-byte UTF-8 character.
        for size in (2, 3, 4):
            if len(buf) >= size:
                try:
                    ch = bytes(buf[:size]).decode("utf-8")
                except UnicodeDecodeError:
                    continue
                del buf[:size]
                return KeyEvent("text", ch)
        if final:
            del buf[:1]
            return KeyEvent("text", "�")
        return None

    def _decode_non_escape(self, final: bool) -> KeyEvent | None:
        """Consume a control key or a run of printable text; None if incomplete."""
        buf = self._buf
        byte = buf[0]
        if byte < 0x20 or byte == 0x7F:
            del buf[:1]
            return _decode_control(byte)
        # Printable run: up to (excluding) the next control/ESC byte.
        run = 0
        while run < len(buf) and buf[run] >= 0x20 and buf[run] != 0x7F:
            run += 1
        data = bytes(buf[:run])
        try:
            text = data.decode("utf-8")
            consumed = run
        except UnicodeDecodeError as exc:
            if not final and exc.end == len(data):
                # Truncated multi-byte character: decode the complete prefix
                # and hold the partial tail in the buffer.
                consumed = exc.start
                text = data[:consumed].decode("utf-8")
                if not text:
                    return None
            else:
                text = data.decode("utf-8", "replace")
                consumed = run
        del buf[:consumed]
        return KeyEvent("text", text)


def _decode_control(byte: int) -> KeyEvent:
    if byte == 0x0D:
        return KeyEvent("enter")
    if byte == 0x0A:
        return KeyEvent("ctrl+j")
    if byte == 0x09:
        return KeyEvent("tab")
    if byte == 0x7F:
        return KeyEvent("backspace")
    if byte == 0x08:
        return KeyEvent("ctrl+backspace")
    if byte == 0x20:
        return KeyEvent("text", " ")
    if 0x01 <= byte <= 0x1A:
        return KeyEvent("ctrl+" + chr(byte + 0x60))
    if byte == 0x1C:
        return KeyEvent("ctrl+\\")
    if byte == 0x1D:
        return KeyEvent("ctrl+]")
    if byte == 0x1E:
        return KeyEvent("ctrl+^")
    if byte == 0x1F:
        return KeyEvent("ctrl+_")
    return KeyEvent("text", chr(byte))

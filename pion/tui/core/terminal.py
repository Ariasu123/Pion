"""Terminal abstraction (port of pi-tui's terminal.ts, reduced to POSIX).

`ProcessTerminal` puts the tty in raw mode, enables bracketed paste, and
delivers input bytes + resize notifications through callbacks wired into the
asyncio event loop. `FakeTerminal` is an in-memory implementation for tests.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import Callable

InputHandler = Callable[[bytes], None]
ResizeHandler = Callable[[], None]

_BRACKETED_PASTE_ON = "\x1b[?2004h"
_BRACKETED_PASTE_OFF = "\x1b[?2004l"
_CURSOR_ON = "\x1b[?25h"


class Terminal:
    @property
    def columns(self) -> int:
        raise NotImplementedError

    @property
    def rows(self) -> int:
        raise NotImplementedError

    def start(self, on_input: InputHandler, on_resize: ResizeHandler) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def write(self, data: str) -> None:
        raise NotImplementedError

    def set_title(self, title: str) -> None:
        self.write(f"\x1b]0;{title}\x07")


class ProcessTerminal(Terminal):
    """Real POSIX terminal on stdin/stdout."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop
        self._stdin = sys.stdin
        self._stdout = sys.stdout
        self._old_attrs: list | None = None
        self._on_input: InputHandler | None = None
        self._on_resize: ResizeHandler | None = None
        self._old_sigwinch = None
        self._started = False

    @property
    def columns(self) -> int:
        return os.get_terminal_size(self._stdout.fileno()).columns or 80

    @property
    def rows(self) -> int:
        return os.get_terminal_size(self._stdout.fileno()).lines or 24

    def start(self, on_input: InputHandler, on_resize: ResizeHandler) -> None:
        import termios
        import tty

        self._loop = self._loop or asyncio.get_running_loop()
        self._on_input = on_input
        self._on_resize = on_resize
        fd = self._stdin.fileno()
        self._old_attrs = termios.tcgetattr(fd)
        tty.setraw(fd)
        self._loop.add_reader(fd, self._readable)
        self._old_sigwinch = signal.getsignal(signal.SIGWINCH)
        signal.signal(signal.SIGWINCH, self._sigwinch)
        self.write(_BRACKETED_PASTE_ON + _CURSOR_ON)
        self._started = True

    def _readable(self) -> None:
        try:
            data = os.read(self._stdin.fileno(), 65536)
        except OSError:
            return
        if data and self._on_input is not None:
            self._on_input(data)

    def _sigwinch(self, signum, frame) -> None:
        if self._loop is not None and self._on_resize is not None:
            self._loop.call_soon_threadsafe(self._on_resize)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        fd = self._stdin.fileno()
        if self._loop is not None:
            try:
                self._loop.remove_reader(fd)
            except Exception:
                pass
        if self._old_sigwinch is not None:
            signal.signal(signal.SIGWINCH, self._old_sigwinch)
        if self._old_attrs is not None:
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_attrs)
        self.write(_BRACKETED_PASTE_OFF + _CURSOR_ON)

    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._stdout.flush()


class FakeTerminal(Terminal):
    """In-memory terminal for tests: captures output, accepts scripted input."""

    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        self._columns = columns
        self._rows = rows
        self.written: list[str] = []
        self._on_input: InputHandler | None = None
        self._on_resize: ResizeHandler | None = None
        self.started = False

    @property
    def columns(self) -> int:
        return self._columns

    @property
    def rows(self) -> int:
        return self._rows

    def resize(self, columns: int, rows: int) -> None:
        self._columns, self._rows = columns, rows
        if self._on_resize is not None:
            self._on_resize()

    def start(self, on_input: InputHandler, on_resize: ResizeHandler) -> None:
        self._on_input = on_input
        self._on_resize = on_resize
        self.started = True

    def stop(self) -> None:
        self.started = False

    def write(self, data: str) -> None:
        self.written.append(data)

    def feed(self, data: bytes) -> None:
        if self._on_input is not None:
            self._on_input(data)

    def output(self) -> str:
        """Everything ever written, concatenated."""
        return "".join(self.written)

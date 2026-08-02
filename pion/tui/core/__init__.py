"""pion-tui core: a minimal inline differential-rendering TUI kernel."""

from .component import Component, Container, Spacer
from .keys import KeyDecoder, KeyEvent, matches
from .overlays import Overlay, OverlayOptions, OverlayStack
from .renderer import CURSOR_MARKER, InlineRenderer, RenderError
from .terminal import FakeTerminal, ProcessTerminal, Terminal
from .text_utils import (
    SGR_RESET,
    apply_bg,
    pad_line,
    repaint,
    slice_by_column,
    strip_ansi,
    truncate_to_width,
    visible_width,
    wrap_text_with_ansi,
)

__all__ = [
    "CURSOR_MARKER",
    "SGR_RESET",
    "Component",
    "Container",
    "FakeTerminal",
    "InlineRenderer",
    "KeyDecoder",
    "KeyEvent",
    "Overlay",
    "OverlayOptions",
    "OverlayStack",
    "ProcessTerminal",
    "RenderError",
    "Spacer",
    "Terminal",
    "apply_bg",
    "matches",
    "pad_line",
    "repaint",
    "slice_by_column",
    "strip_ansi",
    "truncate_to_width",
    "visible_width",
    "wrap_text_with_ansi",
]

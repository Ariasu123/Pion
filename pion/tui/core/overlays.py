"""Modal overlay stack + compositing (port of pi-tui's overlay logic).

Overlays are components drawn on top of the base frame. They are positioned
relative to the *visible* region (the last `rows` lines of the frame) and
spliced into base lines by display column.
"""

from __future__ import annotations

from dataclasses import dataclass

from .component import Component
from .text_utils import SEGMENT_RESET, pad_line, slice_by_column


@dataclass
class OverlayOptions:
    width: float | int = 0.6  # float in (0,1] = fraction of terminal width
    max_height: int | None = None
    anchor: str = "center"  # "center" | "top" | "bottom"


class Overlay:
    def __init__(
        self,
        component: Component,
        options: OverlayOptions | None = None,
        on_close=None,
    ) -> None:
        self.component = component
        self.options = options or OverlayOptions()
        self._on_close = on_close
        self._closed = False
        self._stack: OverlayStack | None = None

    def close(self) -> None:
        if not self._closed and self._stack is not None:
            self._stack._remove(self)

    def _notify_closed(self) -> None:
        if self._on_close is not None:
            self._on_close()

    def render(self, term_width: int) -> list[str]:
        width = self.options.width
        ow = max(8, int(term_width * width)) if isinstance(width, float) else width
        ow = min(ow, term_width)
        lines = self.component.render(ow)
        if self.options.max_height is not None:
            lines = lines[: self.options.max_height]
        return [pad_line(line, ow) for line in lines]

    def width_cells(self, term_width: int) -> int:
        width = self.options.width
        ow = max(8, int(term_width * width)) if isinstance(width, float) else width
        return min(ow, term_width)


class OverlayStack:
    def __init__(self) -> None:
        self._overlays: list[Overlay] = []

    def push(
        self,
        component: Component,
        options: OverlayOptions | None = None,
        on_close=None,
    ) -> Overlay:
        overlay = Overlay(component, options, on_close)
        overlay._stack = self
        self._overlays.append(overlay)
        return overlay

    def _remove(self, overlay: Overlay) -> None:
        if overlay in self._overlays:
            self._overlays.remove(overlay)
            overlay._closed = True
            overlay._notify_closed()

    @property
    def top(self) -> Overlay | None:
        return self._overlays[-1] if self._overlays else None

    def __bool__(self) -> bool:
        return bool(self._overlays)

    def composite(
        self, base: list[str], term_width: int, term_height: int
    ) -> list[str]:
        """Splice all overlays into a copy of the base frame lines."""
        if not self._overlays:
            return base
        frame = list(base)
        for overlay in self._overlays:
            olines = overlay.render(term_width)
            if not olines:
                continue
            ow = overlay.width_cells(term_width)
            ox = max(0, (term_width - ow) // 2)
            # The visible viewport is the last `term_height` base lines.
            view_top = max(0, len(frame) - term_height)
            if overlay.options.anchor == "top":
                oy = view_top + 1
            elif overlay.options.anchor == "bottom":
                oy = len(frame) - len(olines) - 1
            else:
                oy = view_top + max(0, (min(term_height, len(frame)) - len(olines)) // 2)
            oy = max(0, min(oy, max(0, len(frame) - 1)))
            for row, oline in enumerate(olines):
                target = oy + row
                if target >= len(frame):
                    break
                base_line = frame[target]
                left = slice_by_column(base_line, 0, ox)
                right = slice_by_column(base_line, ox + ow, term_width)
                frame[target] = (
                    left + SEGMENT_RESET + oline + SEGMENT_RESET + right
                )
        return frame

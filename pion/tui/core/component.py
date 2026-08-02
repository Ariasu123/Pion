"""Component protocol and the Container compositor (port of pi-tui/tui.ts)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .keys import KeyEvent


class Component:
    """Anything drawable: renders to a list of ANSI-styled lines.

    Each returned line MUST NOT exceed `width` display cells; the renderer
    raises if it does. Components may cache rendered output and must clear
    that cache in `invalidate()` (e.g. on theme change).
    """

    def render(self, width: int) -> list[str]:
        raise NotImplementedError

    def handle_input(self, key: KeyEvent) -> None:
        """Consume a key event; only called on the focused component."""

    def invalidate(self) -> None:
        """Drop cached render state."""


class Container(Component):
    """The only built-in compositor: a vertical stack of children."""

    def __init__(self, children: list[Component] | None = None) -> None:
        self.children: list[Component] = list(children or [])

    def add(self, child: Component) -> Component:
        self.children.append(child)
        return child

    def insert(self, index: int, child: Component) -> Component:
        self.children.insert(index, child)
        return child

    def remove(self, child: Component) -> None:
        if child in self.children:
            self.children.remove(child)

    def clear(self) -> None:
        self.children.clear()

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        for child in self.children:
            lines.extend(child.render(width))
        return lines

    def invalidate(self) -> None:
        for child in self.children:
            child.invalidate()


class Spacer(Component):
    """`n` blank lines."""

    def __init__(self, n: int = 1) -> None:
        self.n = n

    def render(self, width: int) -> list[str]:
        return [""] * self.n

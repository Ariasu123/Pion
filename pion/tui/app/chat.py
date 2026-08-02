"""Chat transcript components (port of pi-coding-agent's message components).

Structure follows pi: user messages are full-width background bands, tool
executions are state-tinted bands, assistant text has no chrome at all, and
blocks are separated by a one-line rhythm.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from ...llm.types import (
    AssistantMessage,
    TextContent,
    ThinkingContent,
)
from ..components.basic import Box, Text
from ..components.markdown import Markdown
from ..core.component import Component, Container, Spacer
from ..core.text_utils import truncate_to_width
from ..theme import Theme, get_theme

PREVIEW_LINES = 5  # collapsed tool output keeps this many trailing lines


def message_text(message) -> str:
    """Plain text of a user/assistant message."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.text for block in content if isinstance(block, TextContent))


class UserMessageComponent(Container):
    def __init__(self, text: str, theme: Theme | None = None) -> None:
        super().__init__(
            [
                Spacer(1),
                Box(
                    1,
                    1,
                    bg="userMessageBg",
                    children=[
                        Markdown(text, pad_x=0, fg="userMessageText", theme=theme)
                    ],
                    theme=theme,
                ),
            ]
        )


class AssistantMessageComponent(Container):
    """Assistant text + thinking. No border, no background."""

    def __init__(
        self,
        show_thinking: Callable[[], bool],
        streaming: bool = False,
        theme: Theme | None = None,
    ) -> None:
        super().__init__()
        self._theme = theme
        self._show_thinking = show_thinking
        self.streaming = streaming
        self.text = ""
        self.thinking = ""
        self.error: str | None = None
        self.add(Spacer(1))
        self._thinking_md = self.add(
            Markdown("", pad_x=1, fg="thinkingText", italic=True, theme=theme)
        )
        self._thinking_label = self.add(
            Text("", pad_x=1, fg="dim", theme=theme)
        )
        self._text_md = self.add(
            Markdown("", pad_x=1, streaming=streaming, theme=theme)
        )
        self._error_text = self.add(Text("", pad_x=1, fg="error", theme=theme))

    def append_text(self, delta: str) -> None:
        self.text += delta
        self._text_md.set_text(self.text, streaming=self.streaming)

    def append_thinking(self, delta: str) -> None:
        self.thinking += delta
        self._thinking_md.set_text(self.thinking, streaming=self.streaming)

    def finalize(self, message: AssistantMessage) -> None:
        self.streaming = False
        self.text = message.text()
        self.thinking = "".join(
            block.thinking
            for block in message.content
            if isinstance(block, ThinkingContent)
        )
        self._text_md.set_text(self.text, streaming=False)
        self._thinking_md.set_text(self.thinking, streaming=False)
        if message.stop_reason == "aborted":
            self.error = "Operation aborted"
        elif message.stop_reason == "error":
            self.error = f"Error: {message.error_message or 'unknown'}"
        elif message.stop_reason == "length":
            self.error = "Response truncated (max tokens reached)"
        if self.error:
            self._error_text.set_text(self.error)

    def render(self, width: int) -> list[str]:
        # Thinking: full italic-gray markdown, or a static label when hidden.
        if self.thinking and self._show_thinking():
            self._thinking_md.set_text(self.thinking, streaming=self.streaming)
            self._thinking_label.set_text("")
        elif self.thinking:
            self._thinking_md.text = ""
            self._thinking_md.invalidate()
            self._thinking_label.set_text("Thinking…")
        else:
            self._thinking_label.set_text("")
            self._thinking_md.set_text("")
        return super().render(width)


class ToolExecutionComponent(Container):
    """State-tinted background band: pending / success / error."""

    def __init__(
        self,
        call_id: str,
        name: str,
        args: dict | None,
        expanded: Callable[[], bool],
        running: bool = False,
        theme: Theme | None = None,
    ) -> None:
        super().__init__()
        self.call_id = call_id
        self.tool_name = name
        self.args = args or {}
        self._expanded = expanded
        self._theme = theme
        self.running = running
        self.is_error = False
        self.result_text = ""
        self.started_at = time.monotonic()
        self.duration: float | None = None

    def update_progress(self, text: str) -> None:
        self.result_text = text

    def update_result(self, text: str, is_error: bool) -> None:
        self.result_text = text
        self.is_error = is_error
        self.running = False
        self.duration = time.monotonic() - self.started_at

    def _summary(self) -> str:
        for key in ("command", "path", "file_path", "query"):
            if key in self.args:
                value = " ".join(str(self.args[key]).split())
                return value[:100] + ("…" if len(value) > 100 else "")
        raw = json.dumps(self.args, ensure_ascii=False)
        return raw[:100] + ("…" if len(raw) > 100 else "")

    def _title(self, theme: Theme) -> str:
        if self.tool_name == "bash" and "command" in self.args:
            command = " ".join(str(self.args["command"]).split())
            title = theme.styled(f"$ {command}", "toolTitle", bold=True)
        else:
            title = theme.styled(self.tool_name, "toolTitle", bold=True)
            summary = self._summary()
            if summary:
                title += " " + theme.fg("muted", summary)
        if "__" in self.tool_name:
            server = self.tool_name.split("__", 1)[0]
            title += " " + theme.fg("dim", f"(mcp {server})")
        return title

    def render(self, width: int) -> list[str]:
        theme = self._theme or get_theme()
        if self.running:
            bg = "toolPendingBg"
        elif self.is_error:
            bg = "toolErrorBg"
        else:
            bg = "toolSuccessBg"

        inner = max(8, width - 2)
        lines = [truncate_to_width(self._title(theme), inner)]

        output = self.result_text.rstrip("\n")
        if output:
            output_lines = output.split("\n")
            if not self._expanded() and len(output_lines) > PREVIEW_LINES:
                skipped = len(output_lines) - PREVIEW_LINES
                hint = theme.fg(
                    "dim", f"… ({skipped} earlier lines, ctrl+o to expand)"
                )
                lines.append(hint)
                output_lines = output_lines[-PREVIEW_LINES:]
            for line in output_lines:
                lines.append(theme.fg("toolOutput", truncate_to_width(line, inner)))

        if self.running:
            elapsed = time.monotonic() - self.started_at
            lines.append(theme.fg("dim", f"Elapsed {elapsed:.0f}s"))
        elif self.duration is not None:
            lines.append(theme.fg("dim", f"Took {self.duration:.1f}s"))

        box = Box(1, 1, bg=bg, children=[_RawLines(lines)], theme=theme)
        return [""] + box.render(width)


class _RawLines(Component):
    """Pre-styled lines passed through a Box unchanged."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def render(self, width: int) -> list[str]:
        return [truncate_to_width(line, width) for line in self.lines]


class Notice(Component):
    """A dim/colored one-off line in the transcript (compaction, errors…)."""

    def __init__(self, text: str, fg: str = "dim", glyph: str = "", theme=None):
        self._text = Text(
            f"{glyph} {text}".strip(), pad_x=1, fg=fg, theme=theme
        )

    def set_text(self, text: str) -> None:
        self._text.set_text(text)

    def render(self, width: int) -> list[str]:
        return self._text.render(width)

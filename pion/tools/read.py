"""read tool — read a UTF-8 text file with line numbers.

Python port of pi's read tool (packages/coding-agent/src/core/tools/read.ts),
text-only subset. Output is capped at MAX_LINES lines or MAX_BYTES bytes
(whichever is hit first); individual lines are truncated at MAX_LINE_CHARS.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..sandbox.base import SandboxRuntime
from ..sandbox.workspace import WorkspaceAccessError, WorkspaceGuard
from .base import AgentToolResult, OnUpdate

MAX_LINES = 1000
MAX_BYTES = 100 * 1024  # 100KB
MAX_LINE_CHARS = 2000


class ReadArgs(BaseModel):
    path: str = Field(description="Path to the file to read (relative or absolute)")
    offset: int = Field(
        default=1,
        description="Line number to start reading from (1-indexed). Negative values read from the end of the file.",
    )
    limit: Optional[int] = Field(default=None, description="Maximum number of lines to read")


class ReadTool:
    name = "read"
    label = "read"
    description = (
        f"Read the contents of a file as UTF-8 text with line numbers. "
        f"Output is truncated to {MAX_LINES} lines or {MAX_BYTES // 1024}KB (whichever is hit first); "
        f"lines longer than {MAX_LINE_CHARS} characters are truncated. "
        f"Use offset/limit for large files. A negative offset reads from the end of the file."
    )
    Args = ReadArgs
    execution_mode = "parallel"

    def __init__(
        self,
        guard: WorkspaceGuard | None = None,
        runtime: SandboxRuntime | None = None,
    ) -> None:
        self.guard = guard
        self.runtime = runtime

    @property
    def parameters(self) -> dict[str, Any]:
        return self.Args.model_json_schema()

    async def execute(
        self,
        tool_call_id: str,
        args: ReadArgs,
        abort: Optional[asyncio.Event] = None,
        on_update: Optional[OnUpdate] = None,
    ) -> AgentToolResult:
        if abort is not None and abort.is_set():
            return AgentToolResult.text("Error: operation aborted")

        try:
            path = (
                self.guard.resolve(args.path, "read")
                if self.guard is not None
                else Path(args.path).expanduser()
            )
        except WorkspaceAccessError as exc:
            return AgentToolResult.text(
                f"Error: {exc}",
                details=self._details({"denied": True}),
            )
        try:
            if self.guard is not None:
                fd = self.guard.open_file(args.path, "read", os.O_RDONLY)
                with os.fdopen(fd, "rb") as handle:
                    raw = handle.read()
            else:
                raw = path.read_bytes()
        except FileNotFoundError:
            return AgentToolResult.text(
                f"Error: file not found: {args.path}",
                details=self._details({}),
            )
        except IsADirectoryError:
            return AgentToolResult.text(
                f"Error: path is a directory, not a file: {args.path}",
                details=self._details({}),
            )
        except OSError as exc:
            return AgentToolResult.text(
                f"Error: could not read {args.path}: {exc}",
                details=self._details({}),
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return AgentToolResult.text(
                f"Error: {args.path} is not valid UTF-8 text",
                details=self._details({}),
            )

        lines = text.split("\n")
        total_lines = len(lines)

        # Convert 1-indexed offset to a 0-indexed start. Negative offset: from the end.
        if args.offset < 0:
            start = max(0, total_lines + args.offset)
        else:
            start = args.offset - 1
        if start >= total_lines:
            return AgentToolResult.text(
                f"Error: offset {args.offset} is beyond end of file ({total_lines} lines total)",
                details=self._details(
                    {
                        "truncated": False,
                        "linesReturned": 0,
                        "totalLines": total_lines,
                    }
                ),
            )

        selected = lines[start:]
        if args.limit is not None:
            selected = selected[: max(0, args.limit)]

        truncated = False
        if len(selected) > MAX_LINES:
            selected = selected[:MAX_LINES]
            truncated = True

        # Build numbered output, enforcing per-line and total byte caps.
        out_lines: list[str] = []
        out_bytes = 0
        for i, line in enumerate(selected):
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + " [... line truncated]"
                truncated = True
            numbered = f"{start + i + 1}\t{line}"
            size = len(numbered.encode("utf-8")) + (1 if out_lines else 0)  # +1 for the join "\n"
            if out_bytes + size > MAX_BYTES:
                truncated = True
                break
            out_lines.append(numbered)
            out_bytes += size

        output = "\n".join(out_lines)
        first_display = start + 1
        last_display = start + len(out_lines)
        if last_display < total_lines:
            note = f"[Showing lines {first_display}-{last_display} of {total_lines}. Use offset={last_display + 1} to continue.]"
            output = f"{output}\n\n{note}" if output else note

        return AgentToolResult.text(
            output,
            details=self._details(
                {
                    "truncated": truncated,
                    "linesReturned": len(out_lines),
                    "totalLines": total_lines,
                }
            ),
        )

    def _details(self, values: dict[str, object]) -> dict[str, object]:
        if self.runtime is None:
            return values
        return {**self.runtime.describe(), **values}

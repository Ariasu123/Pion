"""write tool — create or overwrite a file.

Python port of pi's write tool (packages/coding-agent/src/core/tools/write.ts).
Creates parent directories automatically; overwrites the whole file.
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


class WriteArgs(BaseModel):
    path: str = Field(description="Path to the file to write (relative or absolute)")
    content: str = Field(description="Content to write to the file")


class WriteTool:
    name = "write"
    label = "write"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
        "Automatically creates parent directories."
    )
    Args = WriteArgs
    execution_mode = "sequential"

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
        args: WriteArgs,
        abort: Optional[asyncio.Event] = None,
        on_update: Optional[OnUpdate] = None,
    ) -> AgentToolResult:
        if abort is not None and abort.is_set():
            return AgentToolResult.text("Error: operation aborted")

        try:
            path = (
                self.guard.resolve(args.path, "write")
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
                fd = self.guard.open_file(
                    args.path,
                    "write",
                    os.O_WRONLY | os.O_CREAT,
                    create_parents=True,
                )
                with os.fdopen(fd, "wb") as handle:
                    os.ftruncate(handle.fileno(), 0)
                    handle.write(args.content.encode("utf-8"))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(args.content, encoding="utf-8")
        except OSError as exc:
            return AgentToolResult.text(
                f"Error: could not write {args.path}: {exc}",
                details=self._details({}),
            )

        num_bytes = len(args.content.encode("utf-8"))
        return AgentToolResult.text(
            f"Successfully wrote {num_bytes} bytes to {args.path}",
            details=self._details({"bytes": num_bytes}),
        )

    def _details(self, values: dict[str, object]) -> dict[str, object]:
        if self.runtime is None:
            return values
        return {**self.runtime.describe(), **values}

"""write tool — create or overwrite a file.

Python port of pi's write tool (packages/coding-agent/src/core/tools/write.ts).
Creates parent directories automatically; overwrites the whole file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

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

        path = Path(args.path).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args.content, encoding="utf-8")
        except OSError as exc:
            return AgentToolResult.text(f"Error: could not write {args.path}: {exc}")

        num_bytes = len(args.content.encode("utf-8"))
        return AgentToolResult.text(
            f"Successfully wrote {num_bytes} bytes to {args.path}",
            details={"bytes": num_bytes},
        )

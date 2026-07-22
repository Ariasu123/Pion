"""edit tool — exact text replacement in a single file.

Python port of pi's edit tool (packages/coding-agent/src/core/tools/edit.ts),
single-replacement subset. old_string must occur exactly once in the file
unless replace_all is set.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .base import AgentToolResult, OnUpdate


class EditArgs(BaseModel):
    path: str = Field(description="Path to the file to edit (relative or absolute)")
    old_string: str = Field(description="Exact text to replace. Must be unique in the file unless replace_all is set.")
    new_string: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False, description="Replace every occurrence of old_string")


class EditTool:
    name = "edit"
    label = "edit"
    description = (
        "Edit a file using exact text replacement. old_string must match the file content exactly "
        "and must be unique in the file unless replace_all is set."
    )
    Args = EditArgs
    execution_mode = "sequential"

    @property
    def parameters(self) -> dict[str, Any]:
        return self.Args.model_json_schema()

    async def execute(
        self,
        tool_call_id: str,
        args: EditArgs,
        abort: Optional[asyncio.Event] = None,
        on_update: Optional[OnUpdate] = None,
    ) -> AgentToolResult:
        if abort is not None and abort.is_set():
            return AgentToolResult.text("Error: operation aborted")

        if not args.old_string:
            return AgentToolResult.text("Error: old_string must not be empty")

        path = Path(args.path).expanduser()
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return AgentToolResult.text(f"Error: file not found: {args.path}")
        except IsADirectoryError:
            return AgentToolResult.text(f"Error: path is a directory, not a file: {args.path}")
        except (OSError, UnicodeDecodeError) as exc:
            return AgentToolResult.text(f"Error: could not read {args.path}: {exc}")

        occurrences = content.count(args.old_string)
        if occurrences == 0:
            return AgentToolResult.text(
                f"Error: old_string not found in {args.path}. It must match the file content exactly.",
                details={"replacements": 0},
            )
        if occurrences > 1 and not args.replace_all:
            return AgentToolResult.text(
                f"Error: old_string occurs {occurrences} times in {args.path}. "
                "Provide more context to make it unique, or set replace_all to true.",
                details={"replacements": 0},
            )

        replacements = occurrences if args.replace_all else 1
        new_content = content.replace(args.old_string, args.new_string, -1 if args.replace_all else 1)
        try:
            path.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            return AgentToolResult.text(f"Error: could not write {args.path}: {exc}")

        return AgentToolResult.text(
            f"Successfully replaced {replacements} occurrence(s) in {args.path}.",
            details={"replacements": replacements},
        )

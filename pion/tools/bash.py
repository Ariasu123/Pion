"""bash tool — execute a shell command and stream its output.

Python port of pi's bash tool (packages/coding-agent/src/core/tools/bash.ts),
local-shell subset. Combined stdout/stderr, streamed to on_update. The
captured output is capped at MAX_OUTPUT_BYTES, keeping the tail. Non-zero
exit codes are reported in the result text, not raised.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from pydantic import BaseModel, Field

from .base import AgentToolResult, OnUpdate

MAX_OUTPUT_BYTES = 100 * 1024  # keep the last ~100KB of combined output
_READ_CHUNK = 4096


class BashArgs(BaseModel):
    command: str = Field(description="Bash command to execute")
    timeout_s: int = Field(default=120, description="Timeout in seconds; the process is killed when exceeded")


class BashTool:
    name = "bash"
    label = "bash"
    description = (
        f"Execute a bash command. Returns combined stdout and stderr, truncated to the last "
        f"{MAX_OUTPUT_BYTES // 1024}KB. Non-zero exit codes are reported in the output, not raised."
    )
    Args = BashArgs
    execution_mode = "sequential"

    @property
    def parameters(self) -> dict[str, Any]:
        return self.Args.model_json_schema()

    async def execute(
        self,
        tool_call_id: str,
        args: BashArgs,
        abort: Optional[asyncio.Event] = None,
        on_update: Optional[OnUpdate] = None,
    ) -> AgentToolResult:
        if abort is not None and abort.is_set():
            return AgentToolResult.text("Error: operation aborted", details={"exitCode": None, "truncated": False})

        try:
            proc = await asyncio.create_subprocess_shell(
                args.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            return AgentToolResult.text(
                f"Error: could not start command: {exc}",
                details={"exitCode": None, "truncated": False},
            )

        buffer = bytearray()
        truncated = False

        async def pump() -> None:
            nonlocal truncated
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(_READ_CHUNK)
                if not chunk:
                    break
                buffer.extend(chunk)
                if len(buffer) > MAX_OUTPUT_BYTES:
                    # Keep the tail: the most recent output is what matters.
                    del buffer[: len(buffer) - MAX_OUTPUT_BYTES]
                    truncated = True
                if on_update is not None:
                    on_update(AgentToolResult.text(buffer.decode("utf-8", errors="replace")))

        pump_task = asyncio.create_task(pump())
        wait_task = asyncio.create_task(proc.wait())
        abort_task = asyncio.create_task(abort.wait()) if abort is not None else None

        wait_set: set[asyncio.Task] = {wait_task}
        if abort_task is not None:
            wait_set.add(abort_task)

        done, pending = await asyncio.wait(
            wait_set, timeout=args.timeout_s, return_when=asyncio.FIRST_COMPLETED
        )

        timed_out = not done
        aborted = abort_task is not None and abort_task in done and not timed_out

        if timed_out or aborted:
            proc.kill()
        for task in pending:
            task.cancel()
        # Drain remaining output after a kill so partial output is preserved.
        await asyncio.gather(wait_task, pump_task, return_exceptions=True)
        if abort_task is not None:
            abort_task.cancel()

        output = buffer.decode("utf-8", errors="replace")
        if truncated:
            output = f"[output truncated: showing last ~{MAX_OUTPUT_BYTES // 1024}KB]\n\n{output}"

        if aborted:
            text = f"Command aborted.\n\n{output}" if output else "Command aborted."
            return AgentToolResult.text(text, details={"exitCode": None, "truncated": truncated})
        if timed_out:
            note = f"Error: command timed out after {args.timeout_s}s and was killed."
            text = f"{note}\n\n{output}" if output else note
            return AgentToolResult.text(text, details={"exitCode": None, "truncated": truncated})

        exit_code = proc.returncode
        text = output
        if exit_code != 0:
            note = f"[exit code {exit_code}]"
            text = f"{text}\n\n{note}" if text else note
        if not text:
            text = "(no output)"
        return AgentToolResult.text(text, details={"exitCode": exit_code, "truncated": truncated})

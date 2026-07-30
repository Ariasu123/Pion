"""bash tool — execute a shell command and stream its output.

Python port of pi's bash tool (packages/coding-agent/src/core/tools/bash.ts),
local-shell subset. Combined stdout/stderr, streamed to on_update. The
captured output is capped at MAX_OUTPUT_BYTES, keeping the tail. Non-zero
exit codes are reported in the result text, not raised.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..sandbox.base import (
    HostSandboxRuntime,
    SandboxError,
    SandboxRuntime,
    SandboxSettings,
)
from .base import AgentToolResult, OnUpdate

MAX_OUTPUT_BYTES = 100 * 1024  # keep the last ~100KB of combined output


class BashArgs(BaseModel):
    command: str = Field(description="Bash command to execute")
    timeout_s: int = Field(
        default=120,
        ge=1,
        description="Timeout in seconds; the execution environment is restarted when exceeded",
    )


class BashTool:
    name = "bash"
    label = "bash"
    description = (
        f"Execute a bash command. Returns combined stdout and stderr, truncated to the last "
        f"{MAX_OUTPUT_BYTES // 1024}KB. Non-zero exit codes are reported in the output, not raised."
    )
    Args = BashArgs
    execution_mode = "sequential"

    def __init__(self, runtime: SandboxRuntime | None = None) -> None:
        self.runtime = runtime

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
        runtime = self.runtime
        compatibility_mode = runtime is None
        if runtime is None:
            # ``BashTool()`` remains the historical, explicitly unsandboxed
            # compatibility surface. The CLI always injects a runtime.
            runtime = HostSandboxRuntime(
                Path.cwd(), SandboxSettings(backend="off")
            )

        def details(
            *,
            exit_code: int | None,
            truncated: bool,
            timed_out: bool = False,
            aborted: bool = False,
        ) -> dict[str, object]:
            result: dict[str, object] = {
                "exitCode": exit_code,
                "truncated": truncated,
            }
            if not compatibility_mode:
                result = {
                    **runtime.describe(),
                    **result,
                    "timedOut": timed_out,
                    "aborted": aborted,
                }
            return result

        if abort is not None and abort.is_set():
            return AgentToolResult.text(
                "Error: operation aborted",
                details=details(
                    exit_code=None,
                    truncated=False,
                    aborted=True,
                ),
            )

        try:
            result = await runtime.execute(
                args.command,
                timeout_s=args.timeout_s,
                abort=abort,
                on_update=(
                    (
                        lambda output: on_update(AgentToolResult.text(output))
                    )
                    if on_update is not None
                    else None
                ),
                max_output_bytes=MAX_OUTPUT_BYTES,
            )
        except SandboxError as exc:
            return AgentToolResult.text(
                f"Error: sandbox command failed: {exc}",
                details=details(exit_code=None, truncated=False),
            )

        output = result.output
        if result.truncated:
            output = f"[output truncated: showing last ~{MAX_OUTPUT_BYTES // 1024}KB]\n\n{output}"

        result_details = details(
            exit_code=result.exit_code,
            truncated=result.truncated,
            timed_out=result.timed_out,
            aborted=result.aborted,
        )
        if result.aborted:
            text = f"Command aborted.\n\n{output}" if output else "Command aborted."
            return AgentToolResult.text(text, details=result_details)
        if result.timed_out:
            note = f"Error: command timed out after {args.timeout_s}s and was killed."
            text = f"{note}\n\n{output}" if output else note
            return AgentToolResult.text(text, details=result_details)

        exit_code = result.exit_code
        text = output
        if exit_code != 0:
            note = f"[exit code {exit_code}]"
            text = f"{text}\n\n{note}" if text else note
        if not text:
            text = "(no output)"
        return AgentToolResult.text(text, details=result_details)

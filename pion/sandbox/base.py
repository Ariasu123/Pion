"""Sandbox runtime contracts plus the explicit unsandboxed host backend."""

from __future__ import annotations

import asyncio
import os
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .workspace import WorkspaceGuard

SandboxBackend = Literal["docker", "off"]
SandboxNetwork = Literal["bridge", "none"]
OutputCallback = Callable[[str], None]


class SandboxError(RuntimeError):
    """Base class for sandbox startup and execution failures."""


class SandboxUnavailableError(SandboxError):
    """Raised when a fail-closed sandbox backend cannot be started."""


class SandboxSettings(BaseModel):
    """Persistent and CLI-overridable sandbox policy."""

    model_config = ConfigDict(extra="forbid")

    backend: SandboxBackend = "docker"
    image: str | None = None
    network: SandboxNetwork = "bridge"
    memory_mb: int = Field(default=4096, ge=128)
    cpus: float = Field(default=2.0, gt=0)
    pids_limit: int = Field(default=256, ge=16)
    git_write: bool = False
    protect_paths: list[str] = Field(default_factory=lambda: [".env", ".env.*"])

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("sandbox image must not be empty")
        return value

    @field_validator("protect_paths")
    @classmethod
    def validate_protect_paths(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            pattern = value.strip().replace("\\", "/")
            parts = Path(pattern).parts
            if (
                not pattern
                or pattern.startswith("/")
                or Path(pattern).is_absolute()
                or ".." in parts
            ):
                raise ValueError(
                    "protected paths must be non-empty workspace-relative patterns"
                )
            normalized.append(pattern)
        return normalized


@dataclass
class SandboxCommandResult:
    """Normalized command result returned by every runtime backend."""

    output: str
    exit_code: int | None
    truncated: bool = False
    timed_out: bool = False
    aborted: bool = False


class SandboxRuntime(ABC):
    """Execution boundary shared by the default Pion tools."""

    backend: str

    def __init__(
        self,
        workspace: Path,
        settings: SandboxSettings,
        guard: WorkspaceGuard | None,
    ) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.settings = settings
        self.guard = guard

    @abstractmethod
    async def start(self) -> None:
        """Create or validate the execution environment."""

    @abstractmethod
    async def execute(
        self,
        command: str,
        *,
        timeout_s: int,
        abort: Optional[asyncio.Event],
        on_update: Optional[OutputCallback],
        max_output_bytes: int,
    ) -> SandboxCommandResult:
        """Execute one shell command inside this runtime."""

    @abstractmethod
    async def close(self) -> None:
        """Destroy transient runtime state."""

    def describe(self) -> dict[str, object]:
        return {"backend": self.backend, "containerId": None}


class HostSandboxRuntime(SandboxRuntime):
    """Explicit compatibility backend that executes directly on the host."""

    backend = "host"

    def __init__(self, workspace: Path, settings: SandboxSettings) -> None:
        # --sandbox off intentionally preserves the historical unrestricted
        # file-tool behavior. The CLI prints a high-visibility warning.
        super().__init__(workspace, settings, guard=None)

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def execute(
        self,
        command: str,
        *,
        timeout_s: int,
        abort: Optional[asyncio.Event],
        on_update: Optional[OutputCallback],
        max_output_bytes: int,
    ) -> SandboxCommandResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            raise SandboxError(f"could not start host command: {exc}") from exc

        buffer = bytearray()
        truncated = False

        async def pump() -> None:
            nonlocal truncated
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                buffer.extend(chunk)
                if len(buffer) > max_output_bytes:
                    del buffer[: len(buffer) - max_output_bytes]
                    truncated = True
                if on_update is not None:
                    on_update(buffer.decode("utf-8", errors="replace"))

        pump_task = asyncio.create_task(pump())
        wait_task = asyncio.create_task(proc.wait())
        abort_task = asyncio.create_task(abort.wait()) if abort is not None else None
        wait_set: set[asyncio.Task] = {wait_task}
        if abort_task is not None:
            wait_set.add(abort_task)

        try:
            done, pending = await asyncio.wait(
                wait_set, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
            )
            timed_out = not done
            aborted = (
                abort_task is not None
                and abort_task in done
                and wait_task not in done
                and not timed_out
            )
            if timed_out or aborted:
                self._kill_process_tree(proc)

            for task in pending:
                if task is not wait_task:
                    task.cancel()
            await asyncio.gather(wait_task, pump_task, return_exceptions=True)
        except asyncio.CancelledError:
            self._kill_process_tree(proc)
            await asyncio.gather(wait_task, pump_task, return_exceptions=True)
            raise
        finally:
            if abort_task is not None:
                abort_task.cancel()

        return SandboxCommandResult(
            output=buffer.decode("utf-8", errors="replace"),
            exit_code=None if timed_out or aborted else proc.returncode,
            truncated=truncated,
            timed_out=timed_out,
            aborted=aborted,
        )

    @staticmethod
    def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover - Windows CI is not currently used
                proc.kill()
        except (ProcessLookupError, PermissionError):
            proc.kill()

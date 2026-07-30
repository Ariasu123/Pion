"""Sandbox policy and runtime backends."""

from __future__ import annotations

from pathlib import Path

from .base import (
    HostSandboxRuntime,
    SandboxBackend,
    SandboxCommandResult,
    SandboxError,
    SandboxNetwork,
    SandboxRuntime,
    SandboxSettings,
    SandboxUnavailableError,
)
from .docker import DockerSandboxRuntime
from .workspace import WorkspaceAccessError, WorkspaceGuard


def build_runtime(settings: SandboxSettings, workspace: Path) -> SandboxRuntime:
    """Construct the configured runtime without starting side effects."""
    if settings.backend == "docker":
        return DockerSandboxRuntime(workspace, settings)
    return HostSandboxRuntime(workspace, settings)


__all__ = [
    "DockerSandboxRuntime",
    "HostSandboxRuntime",
    "SandboxBackend",
    "SandboxCommandResult",
    "SandboxError",
    "SandboxNetwork",
    "SandboxRuntime",
    "SandboxSettings",
    "SandboxUnavailableError",
    "WorkspaceAccessError",
    "WorkspaceGuard",
    "build_runtime",
]

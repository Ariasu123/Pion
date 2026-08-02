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
from .docker import DockerSandboxRuntime, check_docker_available
from .workspace import WorkspaceAccessError, WorkspaceGuard


def build_runtime(settings: SandboxSettings, workspace: Path) -> SandboxRuntime:
    """Construct the runtime for the default (unsandboxed) host mode.

    Sandboxed execution is mounted via the `pion mcp` server instead; see
    `sandbox.backend == "mcp"` in the CLI.
    """
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
    "check_docker_available",
]

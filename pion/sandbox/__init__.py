"""Pion sandbox policy, host runtime and standalone-package adapters."""

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
    `sandbox.backend == "mcp"` in the CLI. Refuse any other backend here rather
    than silently returning the host runtime — a caller expecting isolation must
    not get unsandboxed host execution by mistake.
    """
    if settings.backend != "off":
        raise SandboxError(
            f"build_runtime only serves the host backend; got {settings.backend!r}. "
            "Sandboxed execution runs via the `pion mcp` server."
        )
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

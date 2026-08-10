"""Compatibility entry point for the extracted Docker sandbox MCP server.

The MCP protocol, tools and Docker runtime live in ``sandbox-docker-mcp``.
This module only translates Pion's existing config/environment contract and
keeps ``pion mcp`` working for existing users.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from sandbox_docker_mcp.server import serve as external_serve

from ..config import load_config
from ..sandbox import (
    DockerSandboxRuntime,
    HostSandboxRuntime,
    SandboxRuntime,
    SandboxSettings,
)
from ..sandbox.docker import to_external_settings


def resolve_server_settings() -> SandboxSettings:
    """Apply legacy ``PION_SANDBOX_*`` overrides to Pion's saved policy."""

    try:
        settings = load_config().sandbox
    except Exception:
        settings = SandboxSettings()
    updates: dict[str, object] = {}
    env = os.environ
    if env.get("PION_SANDBOX_IMAGE"):
        updates["image"] = env["PION_SANDBOX_IMAGE"]
    if env.get("PION_SANDBOX_NETWORK") in ("bridge", "none"):
        updates["network"] = env["PION_SANDBOX_NETWORK"]
    if env.get("PION_SANDBOX_GIT_WRITE") == "1":
        updates["git_write"] = True
    if env.get("PION_SANDBOX_MEMORY_MB", "").isdigit():
        updates["memory_mb"] = int(env["PION_SANDBOX_MEMORY_MB"])
    if env.get("PION_SANDBOX_CPUS"):
        try:
            updates["cpus"] = float(env["PION_SANDBOX_CPUS"])
        except ValueError:
            pass
    if updates:
        settings = SandboxSettings.model_validate(
            {**settings.model_dump(mode="python"), **updates}
        )
    return settings


def build_server_runtime(settings: SandboxSettings, workspace: Path) -> SandboxRuntime:
    """Use host execution only for Pion's existing test compatibility switch."""

    if os.environ.get("PION_SANDBOX_BACKEND") == "off":
        return HostSandboxRuntime(workspace, settings)
    return DockerSandboxRuntime(workspace, settings)  # type: ignore[return-value]


async def serve(workspace: Path | None = None) -> None:
    active_workspace = workspace or Path.cwd()
    settings = resolve_server_settings()
    runtime = build_server_runtime(settings, active_workspace)
    await external_serve(
        workspace=active_workspace,
        settings=to_external_settings(settings),
        runtime=runtime,  # type: ignore[arg-type]
    )


def main() -> None:
    """Console compatibility entry for ``pion mcp``."""

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


__all__ = ["build_server_runtime", "main", "resolve_server_settings", "serve"]

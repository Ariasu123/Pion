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
    """Apply legacy ``PION_SANDBOX_*`` overrides to Pion's saved policy.

    This process executes sandboxed commands, so it fails closed: a corrupt
    saved config (``load_config`` raises) or an invalid ``PION_SANDBOX_*`` value
    aborts instead of silently reverting to the — possibly more permissive —
    built-in defaults. A missing config file is fine: ``load_config`` returns
    the defaults without raising.
    """

    settings = load_config().sandbox
    updates: dict[str, object] = {}
    env = os.environ
    if env.get("PION_SANDBOX_IMAGE"):
        updates["image"] = env["PION_SANDBOX_IMAGE"]
    network = env.get("PION_SANDBOX_NETWORK")
    if network:
        if network not in ("bridge", "none"):
            raise ValueError(
                f"PION_SANDBOX_NETWORK must be 'bridge' or 'none', got {network!r}"
            )
        updates["network"] = network
    if env.get("PION_SANDBOX_GIT_WRITE") == "1":
        updates["git_write"] = True
    memory = env.get("PION_SANDBOX_MEMORY_MB")
    if memory:
        if not memory.isdigit():
            raise ValueError(
                f"PION_SANDBOX_MEMORY_MB must be a positive integer, got {memory!r}"
            )
        updates["memory_mb"] = int(memory)
    cpus = env.get("PION_SANDBOX_CPUS")
    if cpus:
        try:
            updates["cpus"] = float(cpus)
        except ValueError:
            raise ValueError(
                f"PION_SANDBOX_CPUS must be a number, got {cpus!r}"
            ) from None
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

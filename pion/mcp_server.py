"""`pion mcp` — serve the default toolset (bash/read/write/edit) over MCP.

This is how sandboxing is mounted: the server runs the tools against a
DockerSandboxRuntime (the same hardened engine pion has always used —
non-root, cap-drop ALL, no docker.sock, read-only .git, secret masking),
so any MCP client (pion itself with sandbox.backend="mcp", Claude Code,
Cursor, …) can execute inside the disposable container.

Settings come from the `sandbox` section of ~/.pion/config.json with
environment overrides (used by the parent pion process to forward CLI
flags, and by tests to avoid Docker):

- PION_SANDBOX_BACKEND=off   run on the host instead of Docker
- PION_SANDBOX_IMAGE / PION_SANDBOX_NETWORK(=bridge|none)
- PION_SANDBOX_GIT_WRITE=1 / PION_SANDBOX_MEMORY_MB / PION_SANDBOX_CPUS
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from . import __version__
from .config import load_config
from .sandbox import (
    DockerSandboxRuntime,
    HostSandboxRuntime,
    SandboxRuntime,
    SandboxSettings,
)
from .tools import build_default_tools
from .tools.base import AgentToolResult, ToolCallError, validate_arguments


def resolve_server_settings() -> SandboxSettings:
    """Config file sandbox section + PION_SANDBOX_* environment overrides."""
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
    """Docker by default; host when PION_SANDBOX_BACKEND=off (tests, no Docker)."""
    if os.environ.get("PION_SANDBOX_BACKEND") == "off":
        return HostSandboxRuntime(workspace, settings)
    return DockerSandboxRuntime(workspace, settings)


def _result_text(result: AgentToolResult) -> str:
    parts = [block.text for block in result.content if block.type == "text"]
    return "".join(parts)


async def serve(workspace: Path | None = None) -> None:
    workspace = workspace or Path.cwd()
    settings = resolve_server_settings()
    runtime = build_server_runtime(settings, workspace)
    tools = {tool.name: tool for tool in build_default_tools(runtime)}

    server: Server = Server(f"pion-sandbox@{__version__}")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.parameters,
            )
            for tool in tools.values()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> types.CallToolResult:
        tool = tools.get(name)
        if tool is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )
        try:
            args = validate_arguments(tool, arguments)
            result = await tool.execute(
                tool_call_id=f"mcp-{uuid.uuid4().hex[:12]}",
                args=args,
            )
        except ToolCallError as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))],
                isError=True,
            )
        except Exception as exc:
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text", text=f"{type(exc).__name__}: {exc}"
                    )
                ],
                isError=True,
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=_result_text(result))],
            isError=result.is_error,
        )

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        try:
            await asyncio.wait_for(runtime.close(), timeout=10)
        except Exception:
            print("pion mcp: failed to clean up sandbox runtime", file=sys.stderr)


def main() -> None:
    """Console entry for `pion mcp`."""
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


__all__ = ["build_server_runtime", "main", "resolve_server_settings", "serve"]

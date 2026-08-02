"""End-to-end tests for `pion mcp` — the sandbox toolset served over stdio.

Uses PION_SANDBOX_BACKEND=off so no Docker daemon is required; the tool
surface and wire behavior are identical to the Docker-backed mode.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def sandbox_session(workspace):
    env = {**os.environ, "PION_SANDBOX_BACKEND": "off"}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "pion.cli", "mcp"],
        env=env,
        cwd=str(workspace),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def text_of(result) -> str:
    return "".join(
        block.text for block in result.content if getattr(block, "type", "") == "text"
    )


async def test_list_tools_exposes_default_toolset(tmp_path):
    async with sandbox_session(tmp_path) as session:
        result = await session.list_tools()
    tools = {tool.name: tool for tool in result.tools}
    assert {"bash", "read", "write", "edit"} <= set(tools)
    assert "command" in tools["bash"].inputSchema.get("properties", {})
    assert "path" in tools["read"].inputSchema.get("properties", {})


async def test_call_bash_returns_output(tmp_path):
    async with sandbox_session(tmp_path) as session:
        result = await session.call_tool("bash", {"command": "echo hello-mcp"})
    assert not result.isError
    assert "hello-mcp" in text_of(result)


async def test_call_bash_failure_is_error(tmp_path):
    # pion semantics: a non-zero exit code is information for the model,
    # surfaced as a note in the output rather than an MCP error.
    async with sandbox_session(tmp_path) as session:
        result = await session.call_tool("bash", {"command": "exit 3"})
    assert not result.isError
    assert "[exit code 3]" in text_of(result)


async def test_call_invalid_arguments_is_error(tmp_path):
    async with sandbox_session(tmp_path) as session:
        result = await session.call_tool("bash", {})
    assert result.isError


async def test_call_unknown_tool_is_error(tmp_path):
    async with sandbox_session(tmp_path) as session:
        result = await session.call_tool("nope", {})
    assert result.isError
    assert "Unknown tool" in text_of(result)


async def test_write_then_read_roundtrip(tmp_path):
    async with sandbox_session(tmp_path) as session:
        write = await session.call_tool(
            "write", {"path": "note.txt", "content": "from-mcp\n"}
        )
        assert not write.isError
        read = await session.call_tool("read", {"path": "note.txt"})
    assert "from-mcp" in text_of(read)
    assert (tmp_path / "note.txt").read_text() == "from-mcp\n"

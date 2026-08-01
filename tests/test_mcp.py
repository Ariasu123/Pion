"""Tests for the built-in stdio MCP client."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from mcp import types as mcp_types
from pydantic import ValidationError

from pion.config import MCPServerConfig
from pion.llm.types import ImageContent, TextContent
from pion.mcp import MCPClientManager, MCPTool


class FakeSession:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = []

    async def call_tool(self, name, arguments, **kwargs):
        self.calls.append((name, arguments, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


def remote_tool(schema=None) -> mcp_types.Tool:
    return mcp_types.Tool(
        name="lookup",
        description="Look something up",
        inputSchema=schema
        or {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def test_tool_keeps_original_schema_and_validates_arguments() -> None:
    tool = MCPTool("search", remote_tool(), FakeSession(), 12)
    assert tool.name == "search__lookup"
    assert tool.parameters["required"] == ["query"]
    assert tool.Args.model_validate({"query": "pion"}).model_dump() == {"query": "pion"}
    with pytest.raises(ValidationError, match="query.*required"):
        tool.Args.model_validate({})
    with pytest.raises(ValidationError, match="Additional properties"):
        tool.Args.model_validate({"query": "pion", "extra": True})


async def test_tool_converts_text_image_and_remote_error() -> None:
    session = FakeSession(
        mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(type="text", text="hello"),
                mcp_types.ImageContent(
                    type="image", data="aGVsbG8=", mimeType="image/png"
                ),
            ],
            structuredContent={"count": 2},
            isError=True,
        )
    )
    tool = MCPTool("demo", remote_tool(), session, 9)
    result = await tool.execute("call-1", tool.Args.model_validate({"query": "x"}))
    assert isinstance(result.content[0], TextContent)
    assert isinstance(result.content[1], ImageContent)
    assert result.is_error
    assert result.details["structuredContent"] == {"count": 2}
    assert session.calls[0][0:2] == ("lookup", {"query": "x"})


async def test_tool_failure_becomes_error_result() -> None:
    tool = MCPTool(
        "demo", remote_tool(), FakeSession(error=TimeoutError("too slow")), 1
    )
    result = await tool.execute("call-1", tool.Args.model_validate({"query": "x"}))
    assert result.is_error
    assert "too slow" in result.content[0].text


async def test_tool_failure_redacts_configured_environment_values() -> None:
    tool = MCPTool(
        "demo",
        remote_tool(),
        FakeSession(error=RuntimeError("token do-not-print rejected")),
        1,
        ["do-not-print"],
    )
    result = await tool.execute("call-1", tool.Args.model_validate({"query": "x"}))
    assert "do-not-print" not in result.content[0].text
    assert "***" in result.content[0].text


def _server_config(**updates) -> MCPServerConfig:
    script = Path(__file__).parent / "fixtures" / "mcp_stdio_server.py"
    values = {
        "command": sys.executable,
        "args": [str(script)],
        "timeout_seconds": 10,
    }
    values.update(updates)
    return MCPServerConfig(**values)


async def test_manager_discovers_calls_and_closes_real_stdio_server() -> None:
    manager = MCPClientManager(
        {
            "demo": _server_config(
                env={"PION_MCP_TEST_VALUE": "inherited-and-overridden"}
            )
        }
    )
    await manager.start()
    try:
        assert manager.errors == []
        assert manager.connected_server_count == 1
        assert {tool.name for tool in manager.tools} == {
            "demo__echo",
            "demo__environment",
        }
        echo = next(tool for tool in manager.tools if tool.name == "demo__echo")
        result = await echo.execute(
            "call-1", echo.Args.model_validate({"text": "hello"})
        )
        assert result.content[0].text == "mcp:hello"
        environment = next(
            tool for tool in manager.tools if tool.name == "demo__environment"
        )
        env_result = await environment.execute(
            "call-2",
            environment.Args.model_validate({"name": "PION_MCP_TEST_VALUE"}),
        )
        assert env_result.content[0].text == "inherited-and-overridden"
    finally:
        await manager.close()
    assert manager.connected_server_count == 0
    assert manager.tools == []


async def test_manager_isolates_failures_disabled_servers_and_conflicts() -> None:
    manager = MCPClientManager(
        {
            "disabled": _server_config(enabled=False),
            "missing": MCPServerConfig(command="definitely-not-a-real-pion-command"),
            "demo": _server_config(),
            "working": _server_config(),
        }
    )
    await manager.start({"demo__echo"})
    try:
        assert manager.connected_server_count == 1
        assert {tool.name for tool in manager.tools} == {
            "working__echo",
            "working__environment",
        }
        assert len(manager.errors) == 2
        assert manager.errors[0].startswith("missing:")
        assert "tool name conflict: demo__echo" in manager.errors[1]
    finally:
        await manager.close()


async def test_manager_redacts_configured_environment_values_from_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = MCPClientManager(
        {"secret": MCPServerConfig(command="missing", env={"TOKEN": "do-not-print"})}
    )

    async def fail(*args, **kwargs):
        raise RuntimeError("credential do-not-print rejected")

    monkeypatch.setattr(manager, "_connect", fail)
    await manager.start()
    assert "do-not-print" not in manager.errors[0]
    assert "***" in manager.errors[0]

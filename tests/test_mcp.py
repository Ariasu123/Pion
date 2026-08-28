"""Tests for the built-in stdio MCP client."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from mcp import types as mcp_types
from pydantic import ValidationError

import asyncio

from pion.config import MCPServerConfig
from pion.llm.types import ImageContent, TextContent
from pion.mcp import MCPClientManager, MCPServerConnection, MCPTool
from pion.mcp.client import _child_environment


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


async def test_manager_rejects_bad_remote_tool_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A remote tool whose *bare* name has an illegal character must be reported
    # by that bare name, and the whole server dropped as an isolated error.
    manager = MCPClientManager({"demo": MCPServerConfig(command="whatever")})

    async def fake_connect(name, config, stack):
        bad = mcp_types.Tool(
            name="bad name!",
            inputSchema={"type": "object", "additionalProperties": True},
        )
        tool = MCPTool(name, bad, FakeSession(), 5)
        return MCPServerConnection(name=name, session=FakeSession(), stack=stack, tools=[tool])

    monkeypatch.setattr(manager, "_connect", fake_connect)
    await manager.start()
    assert manager.connected_server_count == 0
    assert "remote tool names" in manager.errors[0]
    assert "bad name!" in manager.errors[0]


async def test_execute_aborts_in_flight_remote_call() -> None:
    # Aborting mid-call must not block until read_timeout: the racing abort
    # cancels the in-flight request and returns an error result promptly.
    started = asyncio.Event()

    class SlowSession:
        async def call_tool(self, name, arguments, **kwargs):
            started.set()
            await asyncio.sleep(60)
            raise AssertionError("should have been cancelled")

    tool = MCPTool("demo", remote_tool(), SlowSession(), 30)
    abort = asyncio.Event()
    task = asyncio.ensure_future(
        tool.execute("c1", tool.Args.model_validate({"query": "x"}), abort=abort)
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    abort.set()
    result = await asyncio.wait_for(task, timeout=1)
    assert result.is_error
    assert "aborted" in result.content[0].text


async def test_child_environment_strips_host_secrets_but_keeps_config_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pion's own provider credentials must not leak into an MCP child, but a
    # non-secret host var and anything the user set via config.env must survive.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("MY_GITHUB_TOKEN", "ghp-should-not-leak")
    monkeypatch.setenv("PION_MCP_TEST_VALUE", "inherited")

    env = _child_environment({"OPENAI_API_KEY": "explicitly-provided"})

    assert "ANTHROPIC_API_KEY" not in env
    assert "MY_GITHUB_TOKEN" not in env
    assert env["PION_MCP_TEST_VALUE"] == "inherited"
    # An explicit config.env secret is layered back on for the server that needs it.
    assert env["OPENAI_API_KEY"] == "explicitly-provided"

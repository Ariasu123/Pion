"""Built-in stdio MCP client and AgentTool adapter.

MCP servers run as trusted child processes of Pion on the host. They are not
inside the Docker sandbox used by the default file and shell tools.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Iterable
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, ClassVar, Optional

from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for
from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, model_validator

from ..config import MCPServerConfig
from ..llm.types import ImageContent, TextContent, sanitize_text
from ..tools.base import AgentToolResult, OnUpdate


class _Aborted(Exception):
    """Internal signal that an in-flight remote call was cancelled by abort."""


class MCPToolArguments(BaseModel):
    """Pydantic entry point whose concrete subclasses validate MCP JSON Schema."""

    model_config = ConfigDict(extra="allow")
    _schema_validator: ClassVar[Any]

    @model_validator(mode="after")
    def validate_mcp_schema(self) -> "MCPToolArguments":
        error = best_match(self._schema_validator.iter_errors(self.model_dump()))
        if error is not None:
            location = ".".join(str(part) for part in error.absolute_path)
            prefix = f"{location}: " if location else ""
            raise ValueError(prefix + error.message)
        return self


def _arguments_model(
    server_name: str, tool_name: str, schema: dict[str, Any]
) -> type[BaseModel]:
    """Build a Pydantic model backed by the MCP tool's full JSON Schema."""
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", f"{server_name}_{tool_name}")
    return type(
        f"MCPArgs_{safe_name}",
        (MCPToolArguments,),
        {"_schema_validator": validator},
    )


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _redact_values(message: str, values: Iterable[str]) -> str:
    redacted = sanitize_text(message) or "unknown error"
    for value in values:
        if value:
            redacted = redacted.replace(value, "***")
    return redacted


# Substrings (case-insensitive) that mark a host env var as a likely secret.
# Such vars are stripped before spawning an stdio MCP child so Pion's own
# provider credentials (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, ...) don't leak
# into every server. A server that genuinely needs one opts in explicitly via
# `config.env`, which is layered on top afterwards.
_SECRET_ENV_MARKERS = (
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "TOKEN",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "CREDENTIAL",
)


def _is_secret_env_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_ENV_MARKERS)


def _child_environment(config_env: dict[str, str]) -> dict[str, str]:
    """Host environment for an MCP child, minus likely-secret host vars.

    Everything else is inherited so servers still find their runtime
    (PATH/HOME/...); `config_env` is layered on top so an explicitly configured
    secret still reaches the child that needs it.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if not _is_secret_env_name(key)
    }
    environment.update(config_env)
    return environment


def _host_secret_values() -> tuple[str, ...]:
    """Values of likely-secret host env vars, for redacting from error text.

    Short values are skipped to avoid garbling unrelated text.
    """
    return tuple(
        value
        for key, value in os.environ.items()
        if _is_secret_env_name(key) and len(value) >= 6
    )


def _convert_content(block: Any) -> TextContent | ImageContent:
    if isinstance(block, mcp_types.TextContent):
        return TextContent(text=sanitize_text(block.text))
    if isinstance(block, mcp_types.ImageContent):
        return ImageContent(data=block.data, mimeType=block.mimeType)
    if isinstance(block, mcp_types.EmbeddedResource):
        resource = block.resource
        if isinstance(resource, mcp_types.TextResourceContents):
            return TextContent(text=sanitize_text(resource.text))
        mime_type = resource.mimeType or "application/octet-stream"
        if mime_type.startswith("image/"):
            return ImageContent(data=resource.blob, mimeType=mime_type)
    # Pion's message contract is text/image only. Preserve unsupported MCP
    # blocks as JSON text instead of silently dropping server output.
    dumped = (
        block.model_dump(mode="json", by_alias=True)
        if hasattr(block, "model_dump")
        else block
    )
    return TextContent(text=_json_text(dumped))


class MCPTool:
    """Expose one remote MCP tool through Pion's AgentTool protocol."""

    execution_mode = "parallel"

    def __init__(
        self,
        server_name: str,
        remote_tool: mcp_types.Tool,
        session: ClientSession,
        timeout_seconds: float,
        redact_values: Iterable[str] = (),
    ) -> None:
        self.server_name = server_name
        self.remote_name = remote_tool.name
        self.name = f"{server_name}__{remote_tool.name}"
        self.label = remote_tool.title or remote_tool.name
        self.description = (
            remote_tool.description or f"MCP tool {remote_tool.name} from {server_name}"
        )
        self._parameters = dict(remote_tool.inputSchema)
        self.Args = _arguments_model(server_name, remote_tool.name, self._parameters)
        self._session = session
        self._timeout = timedelta(seconds=timeout_seconds)
        self._redact_values = tuple(redact_values)

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self._parameters)

    async def execute(
        self,
        tool_call_id: str,
        args: BaseModel,
        abort: Optional[asyncio.Event] = None,
        on_update: Optional[OnUpdate] = None,
    ) -> AgentToolResult:
        if abort is not None and abort.is_set():
            return AgentToolResult.text("Error: operation aborted", is_error=True)
        try:
            result = await self._call_with_abort(args, abort)
        except _Aborted:
            return AgentToolResult.text("Error: operation aborted", is_error=True)
        except Exception as exc:
            return AgentToolResult.text(
                f"MCP tool {self.name} failed: "
                f"{_redact_values(str(exc), self._redact_values)}",
                details={"server": self.server_name, "tool": self.remote_name},
                is_error=True,
            )

        content = [_convert_content(block) for block in result.content]
        if not content and result.structuredContent is not None:
            content.append(TextContent(text=_json_text(result.structuredContent)))
        if not content and result.isError:
            content.append(TextContent(text=f"MCP tool {self.name} reported an error"))
        details = {
            "server": self.server_name,
            "tool": self.remote_name,
            "structuredContent": result.structuredContent,
            "meta": result.meta,
        }
        return AgentToolResult(
            content=content,
            details=details,
            is_error=bool(result.isError),
        )

    async def _call_with_abort(
        self, args: BaseModel, abort: Optional[asyncio.Event]
    ) -> mcp_types.CallToolResult:
        """Invoke the remote tool, racing the request against `abort`.

        Without this, an aborted turn would still block on a slow remote tool
        until `read_timeout` elapsed. On abort we cancel the in-flight request
        and raise `_Aborted`; the late JSON-RPC response is dropped by the
        session. If the call finishes first it is returned (or re-raises its own
        failure to the caller).
        """
        call = asyncio.ensure_future(
            self._session.call_tool(
                self.remote_name,
                args.model_dump(),
                read_timeout_seconds=self._timeout,
            )
        )
        if abort is None:
            return await call
        waiter = asyncio.ensure_future(abort.wait())
        try:
            await asyncio.wait({call, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        if not call.done():
            call.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await call
            raise _Aborted
        return call.result()


@dataclass
class MCPServerConnection:
    name: str
    session: ClientSession
    stack: AsyncExitStack
    tools: list[MCPTool] = field(default_factory=list)


class MCPClientManager:
    """Own all configured MCP sessions and their stdio child processes."""

    def __init__(self, servers: dict[str, MCPServerConfig]) -> None:
        self.servers = servers
        self.connections: list[MCPServerConnection] = []
        self.tools: list[MCPTool] = []
        self.errors: list[str] = []

    @property
    def connected_server_count(self) -> int:
        return len(self.connections)

    @property
    def connected_server_names(self) -> set[str]:
        """Names of servers that finished startup (failed ones are excluded)."""
        return {connection.name for connection in self.connections}

    async def start(self, reserved_tool_names: set[str] | None = None) -> None:
        used_names = set(reserved_tool_names or ())
        for name, config in self.servers.items():
            if not config.enabled:
                continue
            stack = AsyncExitStack()
            try:
                connection = await self._connect(name, config, stack)
                # Charset is validated on the *bare* remote name: the server
                # prefix is already constrained by PionConfig, so a bad character
                # can only originate from the remote tool, and reporting the bare
                # name points the error at the real culprit instead of the
                # prefixed "server__tool" string.
                bad_remote = sorted(
                    tool.remote_name
                    for tool in connection.tools
                    if not tool.remote_name
                    or re.fullmatch(r"[A-Za-z0-9_-]+", tool.remote_name) is None
                )
                if bad_remote:
                    raise ValueError(
                        "remote tool names must be non-empty and use only ASCII "
                        "letters, digits, '_' or '-': " + ", ".join(bad_remote)
                    )
                discovered_names = [tool.name for tool in connection.tools]
                # The exposed "server__tool" name is what the provider sees and
                # must fit its 64-character tool-name limit.
                too_long = sorted(name for name in discovered_names if len(name) > 64)
                if too_long:
                    raise ValueError(
                        "exposed tool names exceed the 64-character limit; shorten "
                        "the server name or the remote tool: " + ", ".join(too_long)
                    )
                if len(discovered_names) != len(set(discovered_names)):
                    raise ValueError("server returned duplicate tool names")
                conflicts = sorted(
                    tool_name
                    for tool_name in discovered_names
                    if tool_name in used_names
                )
                if conflicts:
                    raise ValueError("tool name conflict: " + ", ".join(conflicts))
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                # Genuine shutdown of our own coroutine — clean up and propagate.
                await self._safe_aclose(stack)
                raise
            except BaseException as exc:
                # asyncio.timeout cancelling the MCP anyio task group surfaces
                # as a (Base)ExceptionGroup, which `except Exception` would miss
                # — letting one slow or broken server kill the whole startup.
                # Degrade to a recorded per-server error instead.
                message = str(exc)
                close_error = await self._safe_aclose(stack)
                if close_error is not None:
                    message += f" (cleanup also failed: {close_error})"
                self.errors.append(f"{name}: {self._redact(message, config)}")
                continue
            self.connections.append(connection)
            self.tools.extend(connection.tools)
            used_names.update(tool.name for tool in connection.tools)

    async def _connect(
        self,
        name: str,
        config: MCPServerConfig,
        stack: AsyncExitStack,
    ) -> MCPServerConnection:
        environment = _child_environment(config.env)
        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=environment,
        )
        async with asyncio.timeout(config.timeout_seconds):
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(params)
            )
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=config.timeout_seconds),
                )
            )
            await session.initialize()
            remote_tools: list[mcp_types.Tool] = []
            cursor: str | None = None
            while True:
                page = await session.list_tools(cursor=cursor)
                remote_tools.extend(page.tools)
                cursor = page.nextCursor
                if cursor is None:
                    break
        redact = (*config.env.values(), *_host_secret_values())
        tools = [
            MCPTool(
                name,
                tool,
                session,
                config.timeout_seconds,
                redact,
            )
            for tool in remote_tools
        ]
        return MCPServerConnection(name=name, session=session, stack=stack, tools=tools)

    async def close(self) -> None:
        for connection in reversed(self.connections):
            close_error = await self._safe_aclose(connection.stack)
            if close_error is not None:
                config = self.servers[connection.name]
                self.errors.append(
                    f"{connection.name} shutdown: {self._redact(close_error, config)}"
                )
        self.connections.clear()
        self.tools.clear()

    @staticmethod
    async def _safe_aclose(stack: AsyncExitStack) -> str | None:
        """Close `stack`, returning the failure text instead of raising.

        The MCP session lives inside anyio task groups whose teardown can raise
        a `BaseExceptionGroup` that `except Exception` would miss; swallow every
        non-cancellation failure so cleanup never masks the real error or kills
        an unrelated server.
        """
        try:
            await stack.aclose()
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 - teardown failures are reported, not raised
            return str(exc)
        return None

    @staticmethod
    def _redact(message: str, config: MCPServerConfig) -> str:
        return _redact_values(
            message, (*config.env.values(), *_host_secret_values())
        )


__all__ = ["MCPClientManager", "MCPServerConnection", "MCPTool", "MCPToolArguments"]

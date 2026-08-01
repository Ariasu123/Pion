"""Small stdio MCP server used by Pion's integration tests."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP


server = FastMCP("pion-test-server")


@server.tool()
def echo(text: str) -> str:
    """Echo text from the integration-test server."""
    return f"mcp:{text}"


@server.tool()
def environment(name: str) -> str:
    """Read one child-process environment variable."""
    return os.environ.get(name, "")


if __name__ == "__main__":
    server.run(transport="stdio")

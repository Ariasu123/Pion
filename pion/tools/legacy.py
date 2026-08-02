"""Deprecated unguarded tool instances, kept only for library compatibility.

These module-level instances execute directly on the host with no sandbox
runtime and no workspace guard. The CLI and the `pion mcp` server never use
them — they call `build_default_tools(runtime)` with an explicit runtime.
New code should do the same; this module will be removed in a future release.
"""

from .base import AgentTool
from .bash import BashTool
from .edit import EditTool
from .read import ReadTool
from .write import WriteTool

READ_TOOL = ReadTool()
WRITE_TOOL = WriteTool()
EDIT_TOOL = EditTool()
BASH_TOOL = BashTool()

DEFAULT_TOOLS: list[AgentTool] = [READ_TOOL, WRITE_TOOL, EDIT_TOOL, BASH_TOOL]

__all__ = ["READ_TOOL", "WRITE_TOOL", "EDIT_TOOL", "BASH_TOOL", "DEFAULT_TOOLS"]

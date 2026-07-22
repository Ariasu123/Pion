"""Default tool set of pion.

Python port of pi's default tools (packages/coding-agent/src/core/tools/).
"""

from .bash import BashTool
from .base import AgentTool, AgentToolResult, OnUpdate, ToolCallError, validate_arguments
from .edit import EditTool
from .read import ReadTool
from .write import WriteTool

READ_TOOL = ReadTool()
WRITE_TOOL = WriteTool()
EDIT_TOOL = EditTool()
BASH_TOOL = BashTool()

DEFAULT_TOOLS: list[AgentTool] = [READ_TOOL, WRITE_TOOL, EDIT_TOOL, BASH_TOOL]

__all__ = [
    "AgentTool",
    "AgentToolResult",
    "OnUpdate",
    "ToolCallError",
    "validate_arguments",
    "READ_TOOL",
    "WRITE_TOOL",
    "EDIT_TOOL",
    "BASH_TOOL",
    "DEFAULT_TOOLS",
]

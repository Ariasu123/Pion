"""Default tool set of pion.

Python port of pi's default tools (packages/coding-agent/src/core/tools/).
"""

from ..sandbox.base import SandboxRuntime
from .base import (
    AgentTool,
    AgentToolResult,
    OnUpdate,
    ToolCallError,
    validate_arguments,
)
from .bash import BashTool
from .edit import EditTool
from .read import ReadTool
from .write import WriteTool

# These module-level instances intentionally preserve Pion's historical
# unrestricted host behavior for library callers. The CLI never uses them;
# it calls ``build_default_tools`` with an explicit sandbox runtime.
READ_TOOL = ReadTool()
WRITE_TOOL = WriteTool()
EDIT_TOOL = EditTool()
BASH_TOOL = BashTool()

DEFAULT_TOOLS: list[AgentTool] = [READ_TOOL, WRITE_TOOL, EDIT_TOOL, BASH_TOOL]


def build_default_tools(runtime: SandboxRuntime) -> list[AgentTool]:
    """Build one runtime-bound tool set for a CLI agent instance."""
    return [
        ReadTool(runtime.guard, runtime),
        WriteTool(runtime.guard, runtime),
        EditTool(runtime.guard, runtime),
        BashTool(runtime),
    ]

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
    "build_default_tools",
]

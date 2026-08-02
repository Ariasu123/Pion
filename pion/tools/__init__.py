"""Default tool set of pion.

Python port of pi's default tools (packages/coding-agent/src/core/tools/).
The supported entry point is `build_default_tools(runtime)`; the unguarded
module-level instances live in `pion.tools.legacy` (re-exported here for
compatibility).
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
from .legacy import BASH_TOOL, DEFAULT_TOOLS, EDIT_TOOL, READ_TOOL, WRITE_TOOL
from .read import ReadTool
from .write import WriteTool


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

"""Tool base contract — the AgentTool protocol and AgentToolResult.

Python port of pi's `AgentTool` / `AgentToolResult` (packages/agent/src/types.ts).

Design notes:
- Tool args are declared as a pydantic model per tool (`Args`); the JSON schema
  exposed to the LLM is derived from it, so validation and schema never drift.
- `execute` returns content for the LLM (`content`) plus structured `details`
  for UI/logs — pi's "two-part tool result" idea.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm.types import ImageContent, TextContent


@dataclass
class AgentToolResult:
    """Final or partial result produced by a tool."""

    content: list[TextContent | ImageContent] = field(default_factory=list)
    details: Any = None
    terminate: bool = False  # hint: stop the agent loop after this batch
    is_error: bool = False

    @classmethod
    def text(
        cls,
        text: str,
        details: Any = None,
        terminate: bool = False,
        is_error: bool = False,
    ) -> "AgentToolResult":
        return cls(
            content=[TextContent(text=text)],
            details=details,
            terminate=terminate,
            is_error=is_error,
        )


@dataclass
class ToolCallError(Exception):
    """Raised by arg validation; converted to an error tool result by the loop."""

    message: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


# Callback a tool uses to stream partial execution updates.
OnUpdate = Callable[[AgentToolResult], None]


class AgentTool(Protocol):
    """A tool the agent can call. Concrete tools live in this package."""

    name: str
    label: str
    description: str
    Args: type[BaseModel]
    execution_mode: str  # "sequential" | "parallel"

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for the LLM, derived from `Args`."""
        ...

    async def execute(
        self,
        tool_call_id: str,
        args: BaseModel,
        abort: Optional[asyncio.Event] = None,
        on_update: Optional[OnUpdate] = None,
    ) -> AgentToolResult:
        """Run the tool. Raise on failure instead of encoding errors in content."""
        ...


T = TypeVar("T", bound=BaseModel)


def validate_arguments(tool: AgentTool, raw: dict[str, Any]) -> BaseModel:
    """Validate raw tool-call arguments against the tool's Args model."""
    try:
        return tool.Args.model_validate(raw)
    except ValidationError as exc:
        raise ToolCallError(f"Invalid arguments for tool {tool.name}: {exc}") from exc

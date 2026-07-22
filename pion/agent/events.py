"""Agent event stream payloads.

Python port of pi's `AgentEvent` union (packages/agent/src/types.ts).
The TS union is represented here as a single dataclass discriminated by
`type`; every other field is optional and only populated for the event
types that carry it:

- agent_start / agent_end          — `messages` (agent_end only)
- turn_start / turn_end            — `message`, `tool_results` (turn_end only)
- message_start / message_end      — `message`
- message_update                   — `message`, `assistant_event`
- tool_execution_start             — `tool_call_id`, `tool_name`, `args`
- tool_execution_update            — + `partial_result`
- tool_execution_end               — `tool_call_id`, `tool_name`, `result`, `is_error`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from ..llm.event_stream import AssistantMessageEvent
from ..llm.types import Message, ToolResultMessage
from ..tools.base import AgentToolResult

AgentEventType = Literal[
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
]


@dataclass
class AgentEvent:
    """One event emitted by the agent loop (see `pion.agent.loop`)."""

    type: AgentEventType
    # message lifecycle / turn_end
    message: Optional[Message] = None
    # agent_end
    messages: Optional[list[Message]] = None
    # turn_end
    tool_results: Optional[list[ToolResultMessage]] = None
    # tool_execution_*
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    args: Optional[dict[str, Any]] = None
    # tool_execution_update
    partial_result: Optional[AgentToolResult] = None
    # tool_execution_end
    result: Optional[AgentToolResult] = None
    is_error: Optional[bool] = None
    # message_update
    assistant_event: Optional[AssistantMessageEvent] = None

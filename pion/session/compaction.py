"""Auto-compaction: token estimation, trigger check, and summarization.

Python port of the essentials of pi's
`packages/coding-agent/src/core/compaction/compaction.ts`. The LLM call is
injectable (`stream_fn`) so tests never touch the network.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..llm.event_stream import StreamOptions
from ..llm.types import (
    AssistantMessage,
    Context,
    Message,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    UserMessage,
)
from .manager import SessionManager

#: Tokens reserved for the LLM response (pi's default `reserveTokens`).
DEFAULT_RESERVE_TOKENS = 16384

#: Tool results are truncated to this length during serialization (as in pi).
TOOL_RESULT_CHAR_LIMIT = 2000

#: Structured checkpoint prompt, following pi's SUMMARIZATION_PROMPT.
SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages so work on the touched files can resume without re-reading the conversation."""

#: Signature-compatible with `pion.llm.stream_simple`.
StreamFn = Callable[[Model, Context, StreamOptions], Any]


def _message_chars(message: Message) -> int:
    """Character count of a message's meaningful content."""
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return len(message.content)
        return sum(len(c.text) for c in message.content if isinstance(c, TextContent))
    if isinstance(message, AssistantMessage):
        chars = 0
        for block in message.content:
            if isinstance(block, TextContent):
                chars += len(block.text)
            elif isinstance(block, ThinkingContent):
                chars += len(block.thinking)
            elif isinstance(block, ToolCall):
                chars += len(block.name) + len(json.dumps(block.arguments))
        return chars
    # ToolResultMessage
    return sum(len(c.text) for c in message.content if isinstance(c, TextContent))


def estimate_tokens(messages: list[Message]) -> int:
    """Rough token estimate for a message list (chars // 4 heuristic)."""
    return sum(_message_chars(message) for message in messages) // 4


def should_compact(
    messages: list[Message], model: Model, reserve: int = DEFAULT_RESERVE_TOKENS
) -> bool:
    """True when the estimated context size crosses the compaction threshold.

    Mirrors pi: compact when tokens exceed `context_window - reserve`, leaving
    room for the LLM's response.
    """
    return estimate_tokens(messages) > model.context_window - reserve


def serialize_conversation(messages: list[Message]) -> str:
    """Render messages as tagged plain text for the summarization prompt.

    Prevents the summarizing model from treating the input as a conversation
    to continue (same idea as pi's `serializeConversation`).
    """
    lines: list[str] = []
    for message in messages:
        if isinstance(message, UserMessage):
            if isinstance(message.content, str):
                text = message.content
            else:
                text = "".join(
                    c.text for c in message.content if isinstance(c, TextContent)
                )
            lines.append(f"[User]: {text}")
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ThinkingContent):
                    lines.append(f"[Assistant thinking]: {block.thinking}")
                elif isinstance(block, TextContent):
                    lines.append(f"[Assistant]: {block.text}")
            tool_calls = message.tool_calls()
            if tool_calls:
                calls = "; ".join(
                    f"{tc.name}({json.dumps(tc.arguments)})" for tc in tool_calls
                )
                lines.append(f"[Assistant tool calls]: {calls}")
        else:  # ToolResultMessage
            text = message.text()
            if len(text) > TOOL_RESULT_CHAR_LIMIT:
                text = (
                    text[:TOOL_RESULT_CHAR_LIMIT]
                    + f"... ({len(text) - TOOL_RESULT_CHAR_LIMIT} characters truncated)"
                )
            lines.append(f"[Tool result]: {text}")
    return "\n".join(lines)


async def _collect(stream: Any) -> AssistantMessage:
    """Drain a stream into its final AssistantMessage.

    Accepts either an `AssistantMessageEventStream` (via its `result()`) or a
    raw async iterator of `AssistantMessageEvent` (e.g. a test fake).
    """
    if hasattr(stream, "result"):
        return await stream.result()
    final: AssistantMessage | None = None
    async for event in stream:
        if event.type == "done" and event.message is not None:
            final = event.message
        elif event.type == "error" and event.error is not None:
            final = event.error
    if final is None:
        raise RuntimeError("stream ended without done/error event")
    return final


async def compact(
    session: SessionManager,
    messages: list[Message],
    model: Model,
    stream_fn: StreamFn,
    api_key: str | None = None,
) -> str:
    """Summarize `messages` with the LLM and record the compaction in `session`.

    The serialized conversation is sent to `stream_fn` (signature-compatible
    with `pion.llm.stream_simple`) together with `SUMMARIZATION_PROMPT`. The
    resulting summary is appended to the session as a compaction entry with
    `first_kept_entry_id=None` (the summary replaces the entire prior branch)
    and also returned. Callers that want to keep a recent tail should append
    the compaction entry themselves with the appropriate kept-entry id.
    """
    conversation_text = serialize_conversation(messages)
    prompt = (
        f"<conversation>\n{conversation_text}\n</conversation>\n\n"
        f"{SUMMARIZATION_PROMPT}"
    )
    context = Context(messages=[UserMessage(content=prompt)])
    stream = stream_fn(model, context, StreamOptions(api_key=api_key))
    response = await _collect(stream)
    if response.stop_reason == "error":
        raise RuntimeError(
            f"Summarization failed: {response.error_message or 'unknown error'}"
        )
    summary = response.text()
    session.append_compaction(summary, first_kept_entry_id=None)
    return summary

"""Assistant message streaming protocol.

Python port of pi's `AssistantMessageEventStream` contract:
- providers yield a sequence of `AssistantMessageEvent`s, starting with
  `start` and terminating with either `done` (success) or `error`.
- consumers iterate the stream via `async for`, and can `await stream.result()`
  at any point to get the final `AssistantMessage`.

Stream options mirror pi's `SimpleStreamOptions` (subset).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from .types import AssistantMessage, StopReason, ToolCall


@dataclass
class AssistantMessageEvent:
    type: str  # start|text_start|text_delta|text_end|thinking_start|thinking_delta|thinking_end
    # |toolcall_start|toolcall_delta|toolcall_end|done|error
    content_index: Optional[int] = None
    delta: Optional[str] = None
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    partial: Optional[AssistantMessage] = None
    reason: Optional[StopReason] = None
    message: Optional[AssistantMessage] = None  # done
    error: Optional[AssistantMessage] = None  # error


@dataclass
class StreamOptions:
    """Subset of pi's SimpleStreamOptions."""

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    reasoning: Optional[str] = None  # minimal|low|medium|high
    abort: Optional[asyncio.Event] = None  # set to abort the request
    on_payload: Optional[Any] = None  # callable(payload, model) -> payload | None
    session_id: Optional[str] = None
    timeout_s: float = 600.0
    extra: dict[str, Any] = field(default_factory=dict)


class AssistantMessageEventStream:
    """Async iterator over provider events plus a `result()` future.

    Wraps a provider async generator. Terminates on `done`/`error`; the final
    `AssistantMessage` is then available via `result()`.
    """

    def __init__(self, gen: AsyncIterator[AssistantMessageEvent]):
        self._gen = gen
        self._result: asyncio.Future[AssistantMessage] = asyncio.get_running_loop().create_future()

    def __aiter__(self) -> "AssistantMessageEventStream":
        return self

    async def __anext__(self) -> AssistantMessageEvent:
        if self._result.done():
            raise StopAsyncIteration
        try:
            event = await self._gen.__anext__()
        except StopAsyncIteration:
            if not self._result.done():
                self._result.set_exception(
                    RuntimeError("provider stream ended without done/error event")
                )
            raise
        except BaseException as exc:
            if not self._result.done():
                self._result.set_exception(exc)
            raise
        if event.type == "done" and event.message is not None:
            if not self._result.done():
                self._result.set_result(event.message)
        elif event.type == "error" and event.error is not None:
            if not self._result.done():
                self._result.set_result(event.error)
        return event

    async def result(self) -> AssistantMessage:
        """The final assistant message (awaits stream completion if needed)."""
        async for _ in self:
            pass
        return await self._result

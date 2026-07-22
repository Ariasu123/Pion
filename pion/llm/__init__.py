"""Unified multi-provider LLM streaming API.

Only two API shapes exist (following pi's core insight):
- `openai-completions`  — OpenAI chat completions; also DeepSeek, Kimi/Moonshot,
  Qwen, Zhipu and any OpenAI-compatible endpoint.
- `anthropic-messages`  — Anthropic Messages API (Claude).
"""

from .anthropic_messages import stream_simple as stream_anthropic
from .event_stream import AssistantMessageEvent, AssistantMessageEventStream, StreamOptions
from .openai_completions import stream_simple as stream_openai
from .types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    Model,
    ModelCost,
    StopReason,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)

_STREAMERS = {
    "openai-completions": stream_openai,
    "anthropic-messages": stream_anthropic,
}


def stream_simple(
    model: Model, context: Context, options: StreamOptions | None = None
) -> AssistantMessageEventStream:
    """Dispatch to the provider implementation for `model.api`.

    Contract (same as pi): never raises for request/model/runtime failures;
    failures are encoded as an `error` event carrying the final
    AssistantMessage with stop_reason "error"/"aborted".
    """
    return _STREAMERS[model.api](model, context, options or StreamOptions())


__all__ = [
    "AssistantMessage",
    "AssistantMessageEvent",
    "AssistantMessageEventStream",
    "Context",
    "ImageContent",
    "Message",
    "Model",
    "ModelCost",
    "StopReason",
    "StreamOptions",
    "TextContent",
    "ThinkingContent",
    "Tool",
    "ToolCall",
    "ToolResultMessage",
    "Usage",
    "UsageCost",
    "UserMessage",
    "stream_simple",
]

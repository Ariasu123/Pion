"""Core message / tool / model types — the shared contract of pion.

Python port of pi's `packages/ai/src/types.ts` (simplified subset).
Field aliases follow pi's camelCase JSON conventions so serialized
sessions stay close to pi's session format.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


def _now_ms() -> int:
    return int(time.time() * 1000)


class PiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class TextContent(PiModel):
    type: Literal["text"] = "text"
    text: str
    text_signature: Optional[str] = Field(default=None, alias="textSignature")


class ThinkingContent(PiModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: Optional[str] = Field(default=None, alias="thinkingSignature")
    redacted: Optional[bool] = None


class ImageContent(PiModel):
    type: Literal["image"] = "image"
    data: str  # base64-encoded image data
    mime_type: str = Field(alias="mimeType")


class ToolCall(PiModel):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    thought_signature: Optional[str] = Field(default=None, alias="thoughtSignature")


AssistantContent = Annotated[
    Union[TextContent, ThinkingContent, ToolCall], Field(discriminator="type")
]
UserContent = Annotated[Union[TextContent, ImageContent], Field(discriminator="type")]
ToolResultContent = Annotated[Union[TextContent, ImageContent], Field(discriminator="type")]

# ---------------------------------------------------------------------------
# Usage / cost
# ---------------------------------------------------------------------------


class UsageCost(PiModel):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")
    total: float = 0.0


class Usage(PiModel):
    input: int = 0
    output: int = 0
    cache_read: int = Field(default=0, alias="cacheRead")
    cache_write: int = Field(default=0, alias="cacheWrite")
    reasoning: Optional[int] = None  # subset of `output`, when the provider reports it
    total_tokens: int = Field(default=0, alias="totalTokens")
    cost: UsageCost = Field(default_factory=UsageCost)


StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]

# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class UserMessage(PiModel):
    role: Literal["user"] = "user"
    content: Union[str, list[UserContent]]
    timestamp: int = Field(default_factory=_now_ms)


class AssistantMessage(PiModel):
    role: Literal["assistant"] = "assistant"
    content: list[AssistantContent] = Field(default_factory=list)
    api: str = ""
    provider: str = ""
    model: str = ""
    response_model: Optional[str] = Field(default=None, alias="responseModel")
    response_id: Optional[str] = Field(default=None, alias="responseId")
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = Field(default="stop", alias="stopReason")
    error_message: Optional[str] = Field(default=None, alias="errorMessage")
    timestamp: int = Field(default_factory=_now_ms)

    def text(self) -> str:
        """Concatenated text blocks (convenience)."""
        return "".join(c.text for c in self.content if isinstance(c, TextContent))

    def tool_calls(self) -> list[ToolCall]:
        return [c for c in self.content if isinstance(c, ToolCall)]


class ToolResultMessage(PiModel):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str = Field(alias="toolCallId")
    tool_name: str = Field(alias="toolName")
    content: list[ToolResultContent] = Field(default_factory=list)
    details: Any = None
    usage: Optional[Usage] = None
    is_error: bool = Field(default=False, alias="isError")
    timestamp: int = Field(default_factory=_now_ms)

    def text(self) -> str:
        return "".join(c.text for c in self.content if isinstance(c, TextContent))


Message = Annotated[
    Union[UserMessage, AssistantMessage, ToolResultMessage], Field(discriminator="role")
]

# ---------------------------------------------------------------------------
# Tools / context
# ---------------------------------------------------------------------------


class Tool(PiModel):
    """A tool the LLM can call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class Context(PiModel):
    system_prompt: Optional[str] = Field(default=None, alias="systemPrompt")
    messages: list[Message] = Field(default_factory=list)
    tools: list[Tool] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Model descriptor
# ---------------------------------------------------------------------------


class ModelCost(PiModel):
    """USD per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")


class Model(PiModel):
    id: str
    name: str = ""
    api: Literal["openai-completions", "anthropic-messages"] = "openai-completions"
    provider: str = ""
    base_url: str = Field(alias="baseUrl")
    reasoning: bool = False
    input: list[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    cost: ModelCost = Field(default_factory=ModelCost)
    context_window: int = Field(default=128_000, alias="contextWindow")
    max_tokens: int = Field(default=8192, alias="maxTokens")
    headers: dict[str, str] = Field(default_factory=dict)

    def compute_cost(self, usage: Usage) -> UsageCost:
        cost = UsageCost(
            input=usage.input * self.cost.input / 1_000_000,
            output=usage.output * self.cost.output / 1_000_000,
            cacheRead=usage.cache_read * self.cost.cache_read / 1_000_000,
            cacheWrite=usage.cache_write * self.cost.cache_write / 1_000_000,
        )
        cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write
        return cost

"""Session layer: tree-structured JSONL session store plus auto-compaction."""

from .compaction import (
    DEFAULT_RESERVE_TOKENS,
    SUMMARIZATION_PROMPT,
    compact,
    estimate_tokens,
    serialize_conversation,
    should_compact,
)
from .manager import SessionEntry, SessionManager

__all__ = [
    "DEFAULT_RESERVE_TOKENS",
    "SUMMARIZATION_PROMPT",
    "SessionEntry",
    "SessionManager",
    "compact",
    "estimate_tokens",
    "serialize_conversation",
    "should_compact",
]

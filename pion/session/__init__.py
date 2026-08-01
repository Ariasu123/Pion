"""Session layer: tree-structured JSONL session store plus auto-compaction."""

from .compaction import (
    BRANCH_SUMMARIZATION_PROMPT,
    DEFAULT_RESERVE_TOKENS,
    SUMMARIZATION_PROMPT,
    compact,
    estimate_tokens,
    generate_summary,
    serialize_conversation,
    should_compact,
)
from .manager import SessionEntry, SessionManager, SessionTreeNode

__all__ = [
    "BRANCH_SUMMARIZATION_PROMPT",
    "DEFAULT_RESERVE_TOKENS",
    "SUMMARIZATION_PROMPT",
    "SessionEntry",
    "SessionManager",
    "SessionTreeNode",
    "compact",
    "estimate_tokens",
    "generate_summary",
    "serialize_conversation",
    "should_compact",
]

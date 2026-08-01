"""Tree-structured JSONL session store.

Python port of the essentials of pi's
`packages/coding-agent/src/core/session-manager.ts`:

- Entries form a tree via `id`/`parent_id`, enabling in-place branching.
- Sessions persist as append-only JSONL (one JSON object per line, camelCase
  keys via ``by_alias=True``).
- `build_context()` replays the active branch and honors compaction
  boundaries: the newest compaction on the path replaces everything before
  its ``first_kept_entry_id`` with a summary user message.

Differences from pi:
- Only three entry types exist here: "message", "compaction", "custom".
- Entry timestamps are epoch milliseconds (int), matching pion's message
  timestamps; pi uses ISO strings for entry timestamps.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field

from ..llm.types import Message, PiModel, UserMessage, sanitize_message


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id() -> str:
    return uuid.uuid4().hex


class SessionEntry(PiModel):
    """One node in the session tree.

    The payload fields are mutually exclusive and selected by `type`:
    - "message": `message` holds the pion message (participates in context).
    - "compaction": `summary` replaces everything before `first_kept_entry_id`.
    - "custom": `data` is extension state; it never enters the LLM context.
    """

    id: str = Field(default_factory=_new_id)
    parent_id: Optional[str] = Field(default=None, alias="parentId")
    type: Literal["message", "compaction", "custom"]
    timestamp: int = Field(default_factory=_now_ms)
    message: Optional[Message] = None
    summary: Optional[str] = None
    first_kept_entry_id: Optional[str] = Field(default=None, alias="firstKeptEntryId")
    data: Optional[dict[str, Any]] = None


class SessionManager:
    """Append-only session store with tree navigation.

    `path=None` creates an in-memory session (no persistence). New entries
    are appended as children of the current leaf, which then becomes the new
    leaf. Use `switch_leaf()` to move to an earlier entry and branch off.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._entries: dict[str, SessionEntry] = {}
        self._order: list[str] = []  # entry ids in insertion order
        self._leaf_id: str | None = None

    # ------------------------------------------------------------------
    # Loading / persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "SessionManager":
        """Replay a JSONL session file; future appends keep writing to it.

        Messages are sanitized on the way in: older files may contain
        unpaired \\uXXXX escapes (lone surrogates) that would otherwise make
        every later serialization fail.
        """
        manager = cls(path)
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = SessionEntry.model_validate(json.loads(line))
                if entry.message is not None:
                    entry.message = sanitize_message(entry.message)
                manager._entries[entry.id] = entry
                manager._order.append(entry.id)
                manager._leaf_id = entry.id
        return manager

    def _persist(self, entry: SessionEntry) -> None:
        if self._path is None:
            return
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json(by_alias=True, exclude_none=True) + "\n")

    def _append(self, entry: SessionEntry) -> str:
        self._entries[entry.id] = entry
        self._order.append(entry.id)
        self._leaf_id = entry.id
        self._persist(entry)
        return entry.id

    # ------------------------------------------------------------------
    # Appending (all return the new entry id)
    # ------------------------------------------------------------------

    def append_message(self, message: Message) -> str:
        """Append a conversation message as a child of the current leaf."""
        return self._append(
            SessionEntry(type="message", parentId=self._leaf_id, message=message)
        )

    def append_compaction(
        self, summary: str, first_kept_entry_id: str | None = None
    ) -> str:
        """Append a compaction entry.

        `first_kept_entry_id` marks the first entry that survives compaction;
        `None` means the summary replaces the entire prior branch.
        """
        return self._append(
            SessionEntry(
                type="compaction",
                parentId=self._leaf_id,
                summary=summary,
                firstKeptEntryId=first_kept_entry_id,
            )
        )

    def append_custom(self, data: dict[str, Any]) -> str:
        """Append extension state. Custom entries never enter the LLM context."""
        return self._append(
            SessionEntry(type="custom", parentId=self._leaf_id, data=data)
        )

    # ------------------------------------------------------------------
    # Tree navigation
    # ------------------------------------------------------------------

    @property
    def leaf_id(self) -> str | None:
        """Id of the current leaf (None before the first entry)."""
        return self._leaf_id

    def get_entry(self, entry_id: str) -> SessionEntry:
        """Look up an entry by id."""
        return self._entries[entry_id]

    def children(self, parent_id: str | None) -> list[SessionEntry]:
        """Direct children of `parent_id`, in insertion order."""
        return [
            self._entries[entry_id]
            for entry_id in self._order
            if self._entries[entry_id].parent_id == parent_id
        ]

    def branch_ids(self) -> list[str]:
        """Ids of all tree leaves (entries that are no one's parent)."""
        parents = {
            entry.parent_id
            for entry in self._entries.values()
            if entry.parent_id is not None
        }
        return [entry_id for entry_id in self._order if entry_id not in parents]

    def switch_leaf(self, entry_id: str) -> None:
        """Move the current position to an existing entry (branch off from it)."""
        if entry_id not in self._entries:
            raise KeyError(f"unknown entry id: {entry_id}")
        self._leaf_id = entry_id

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _path_entries(self) -> list[SessionEntry]:
        """Entries on the active branch, ordered root -> leaf."""
        chain: list[SessionEntry] = []
        current = self._leaf_id
        while current is not None:
            entry = self._entries[current]
            chain.append(entry)
            current = entry.parent_id
        chain.reverse()
        return chain

    def build_context(self) -> list[Message]:
        """Build the message list for the LLM along the active branch.

        If a compaction entry is on the path, the newest one wins: its summary
        is prepended as a user message, followed by the kept messages from
        `first_kept_entry_id` onward (up to the compaction entry) and then all
        messages after the compaction. Older summarized messages are omitted.
        If `first_kept_entry_id` is None or missing from the path, nothing
        before the compaction is kept. Custom entries never enter the context.
        """
        path = self._path_entries()

        compaction_idx = -1
        for i, entry in enumerate(path):
            if entry.type == "compaction":
                compaction_idx = i

        if compaction_idx < 0:
            return [
                entry.message
                for entry in path
                if entry.type == "message" and entry.message is not None
            ]

        compaction = path[compaction_idx]
        selected: list[SessionEntry] = []
        found_first_kept = False
        for entry in path[:compaction_idx]:
            if entry.id == compaction.first_kept_entry_id:
                found_first_kept = True
            if found_first_kept:
                selected.append(entry)
        selected.extend(path[compaction_idx + 1 :])

        # The summary is a deterministic projection of the persisted
        # compaction entry. Reusing its timestamp prevents repeated context
        # builds (and reloads) from producing different message snapshots.
        messages: list[Message] = [
            UserMessage(
                content=compaction.summary or "",
                timestamp=compaction.timestamp,
            )
        ]
        messages.extend(
            entry.message
            for entry in selected
            if entry.type == "message" and entry.message is not None
        )
        return messages

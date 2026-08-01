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
- Pion supports message, compaction, custom, branch-summary, and label entries.
- Entry timestamps are epoch milliseconds (int), matching pion's message
  timestamps; pi uses ISO strings for entry timestamps.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field

from ..llm.types import Message, PiModel, Usage, UserMessage, sanitize_message


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
    type: Literal["message", "compaction", "custom", "branch_summary", "label"]
    timestamp: int = Field(default_factory=_now_ms)
    message: Optional[Message] = None
    summary: Optional[str] = None
    first_kept_entry_id: Optional[str] = Field(default=None, alias="firstKeptEntryId")
    data: Optional[dict[str, Any]] = None
    from_id: Optional[str] = Field(default=None, alias="fromId")
    target_id: Optional[str] = Field(default=None, alias="targetId")
    label: Optional[str] = None
    usage: Optional[Usage] = None


@dataclass(frozen=True)
class SessionTreeNode:
    """Read-only session tree node with its latest resolved label."""

    entry: SessionEntry
    children: tuple["SessionTreeNode", ...] = ()
    label: str | None = None
    label_timestamp: int | None = None


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
        self._labels_by_id: dict[str, str] = {}
        self._label_timestamps_by_id: dict[str, int] = {}

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
                manager._apply_label_entry(entry)
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
        self._apply_label_entry(entry)
        self._persist(entry)
        return entry.id

    def _apply_label_entry(self, entry: SessionEntry) -> None:
        if entry.type != "label" or entry.target_id is None:
            return
        if entry.label:
            self._labels_by_id[entry.target_id] = entry.label
            self._label_timestamps_by_id[entry.target_id] = entry.timestamp
        else:
            self._labels_by_id.pop(entry.target_id, None)
            self._label_timestamps_by_id.pop(entry.target_id, None)

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

    def append_label_change(self, target_id: str, label: str | None) -> str:
        """Append a label update without rewriting the target entry."""
        if target_id not in self._entries:
            raise KeyError(f"unknown entry id: {target_id}")
        normalized = label.strip() if label and label.strip() else None
        return self._append(
            SessionEntry(
                type="label",
                parentId=self._leaf_id,
                targetId=target_id,
                label=normalized,
            )
        )

    def branch_with_summary(
        self,
        branch_from_id: str | None,
        summary: str,
        *,
        from_id: str | None,
        usage: Usage | None = None,
    ) -> str:
        """Atomically move to an earlier point and append a branch summary."""
        if branch_from_id is not None and branch_from_id not in self._entries:
            raise KeyError(f"unknown entry id: {branch_from_id}")
        self._leaf_id = branch_from_id
        return self._append(
            SessionEntry(
                type="branch_summary",
                parentId=branch_from_id,
                fromId=from_id,
                summary=summary,
                usage=usage,
            )
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

    def get_entries(self) -> list[SessionEntry]:
        """All entries in append order as defensive copies."""
        return [
            self._entries[entry_id].model_copy(deep=True) for entry_id in self._order
        ]

    def get_label(self, entry_id: str) -> str | None:
        return self._labels_by_id.get(entry_id)

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

    def branch(self, entry_id: str | None) -> None:
        """Move the active leaf to an entry, or before the root with None."""
        if entry_id is not None and entry_id not in self._entries:
            raise KeyError(f"unknown entry id: {entry_id}")
        self._leaf_id = entry_id

    def reset_leaf(self) -> None:
        self._leaf_id = None

    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]:
        """Entries from root to ``from_id`` (the active leaf by default)."""
        current = self._leaf_id if from_id is None else from_id
        if current is not None and current not in self._entries:
            raise KeyError(f"unknown entry id: {current}")
        chain: list[SessionEntry] = []
        while current is not None:
            entry = self._entries[current]
            chain.append(entry)
            current = entry.parent_id
        chain.reverse()
        return chain

    def get_tree(self) -> tuple[SessionTreeNode, ...]:
        """Return the full append-only session structure as immutable nodes."""
        children: dict[str | None, list[str]] = {}
        for entry_id in self._order:
            entry = self._entries[entry_id]
            parent = entry.parent_id if entry.parent_id in self._entries else None
            children.setdefault(parent, []).append(entry_id)

        def build(entry_id: str, ancestors: frozenset[str]) -> SessionTreeNode:
            entry = self._entries[entry_id]
            if entry_id in ancestors:  # defensive handling for corrupted files
                child_nodes: tuple[SessionTreeNode, ...] = ()
            else:
                child_nodes = tuple(
                    build(child_id, ancestors | {entry_id})
                    for child_id in children.get(entry_id, [])
                    if child_id != entry_id
                )
            return SessionTreeNode(
                entry=entry.model_copy(deep=True),
                children=child_nodes,
                label=self._labels_by_id.get(entry_id),
                label_timestamp=self._label_timestamps_by_id.get(entry_id),
            )

        return tuple(
            build(entry_id, frozenset()) for entry_id in children.get(None, [])
        )

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _path_entries(self) -> list[SessionEntry]:
        """Entries on the active branch, ordered root -> leaf."""
        return self.get_branch()

    @staticmethod
    def _context_message(entry: SessionEntry) -> Message | None:
        if entry.type == "message":
            return entry.message
        if entry.type == "branch_summary":
            return UserMessage(
                content=f"[Branch summary]\n{entry.summary or ''}",
                timestamp=entry.timestamp,
            )
        return None

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
                message for entry in path if (message := self._context_message(entry))
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
            message for entry in selected if (message := self._context_message(entry))
        )
        return messages

"""UI-neutral controller for an interactive Pion agent session."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .agent.agent import Agent
from .hooks import ExtensionManager
from .llm.registry import get_model, resolve_api_key
from .llm.types import Message, TextContent, Usage, UserMessage
from .session import (
    BRANCH_SUMMARIZATION_PROMPT,
    SessionEntry,
    SessionManager,
    estimate_tokens,
    generate_summary,
    should_compact,
)

ControllerEventType = Literal[
    "session_changed",
    "compaction_started",
    "compaction_finished",
    "branch_summary_started",
    "branch_summary_finished",
    "model_changed",
    "error",
]


@dataclass(frozen=True)
class ControllerEvent:
    type: ControllerEventType
    data: dict[str, Any]


@dataclass(frozen=True)
class TreeNavigationResult:
    selected_id: str
    leaf_id: str | None
    editor_text: str | None = None
    summary_entry_id: str | None = None


ControllerHandler = Callable[[ControllerEvent], Any]


class AgentSessionController:
    """Coordinates Agent, SessionManager and compaction for any UI."""

    def __init__(
        self,
        agent: Agent,
        session: SessionManager,
        session_path: Path,
        extensions: ExtensionManager | None = None,
    ) -> None:
        self.agent = agent
        self.session = session
        self.session_path = session_path
        self.extensions = extensions
        self.last_usage: Usage | None = None
        self.last_error: str | None = None
        self.subscriber_errors: list[Exception] = []
        self._handlers: list[ControllerHandler] = []
        self._tree_abort = asyncio.Event()
        self._tree_summarizing = False
        self._compacting = False

    @property
    def is_busy(self) -> bool:
        """Whether a turn, branch summary or compaction is currently running."""
        return self.agent.is_streaming or self._tree_summarizing or self._compacting

    def subscribe(self, handler: ControllerHandler) -> None:
        self._handlers.append(handler)

    def unsubscribe(self, handler: ControllerHandler) -> None:
        if handler in self._handlers:
            self._handlers.remove(handler)

    async def _emit(self, event_type: ControllerEventType, **data: Any) -> None:
        event = ControllerEvent(event_type, data)
        for handler in list(self._handlers):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                # Observers must not break persistence or model operations.
                self.subscriber_errors.append(exc)

    async def prompt(self, text: str) -> Any:
        self.last_error = None
        before = len(self.agent.messages)
        final = await self.agent.prompt(text)
        self.last_usage = final.usage
        if final.stop_reason == "error" and final.error_message:
            self.last_error = final.error_message
        for message in self.agent.messages[before:]:
            self.session.append_message(message)
        await self._emit("session_changed", leaf_id=self.session.leaf_id)
        try:
            await self.maybe_compact()
        except Exception as exc:
            self.last_error = f"Compaction failed: {exc}"
            await self._emit("error", source="compaction", message=self.last_error)
        return final

    def abort(self) -> None:
        self.agent.abort()
        self._tree_abort.set()

    def would_abandon_suffix(self, selected_id: str) -> bool:
        """Return whether navigating to an entry leaves active history behind."""
        selected = self.session.get_entry(selected_id)
        target_id = (
            selected.parent_id if self._user_text(selected) is not None else selected.id
        )
        return target_id != self.session.leaf_id

    async def maybe_compact(self, force: bool = False) -> str | None:
        if not self.agent.messages:
            return None
        if not force and not should_compact(self.agent.messages, self.agent.model):
            return None
        kept_id = self._find_first_kept_entry_id()
        if not force and not self._compaction_would_progress(kept_id):
            # The kept tail alone already exceeds the threshold, so summarizing
            # folds nothing away. Re-running it every turn would just thrash the
            # LLM without shrinking the context — skip until real progress is
            # possible.
            return None
        summary: str | None = None
        self._tree_abort.clear()
        self._compacting = True
        await self._emit("compaction_started", force=force)
        try:
            if self.extensions is not None:
                await self.extensions.notify(
                    "session_before_compact",
                    {
                        "session_path": str(self.session_path),
                        "message_count": len(self.agent.messages),
                    },
                )
            response = await generate_summary(
                self.agent.messages,
                self.agent.model,
                self.agent.stream_fn,
                api_key=self.agent.api_key,
                abort=self._tree_abort,
            )
            summary = response.text()
            self.session.append_compaction(summary, first_kept_entry_id=kept_id)
            self.agent.messages = self.session.build_context()
        finally:
            # Always pair compaction_started so the UI spinner can't get stuck,
            # even when generate_summary raises. On failure summary is None and
            # message_count reflects the unchanged context.
            self._compacting = False
            await self._emit(
                "compaction_finished",
                summary=summary,
                message_count=len(self.agent.messages),
            )
        await self._emit("session_changed", leaf_id=self.session.leaf_id)
        return summary

    def switch_model(self, model_id: str) -> None:
        model = get_model(model_id)
        self.agent.model = model
        self.agent.api_key = resolve_api_key(model)

    async def notify_model_changed(self) -> None:
        await self._emit("model_changed", model_id=self.agent.model.id)

    async def set_label(self, entry_id: str, label: str | None) -> str:
        label_entry_id = self.session.append_label_change(entry_id, label)
        self.agent.messages = self.session.build_context()
        await self._emit("session_changed", leaf_id=self.session.leaf_id)
        return label_entry_id

    async def navigate_tree(
        self,
        selected_id: str,
        *,
        summarize: bool = False,
        custom_instructions: str | None = None,
    ) -> TreeNavigationResult:
        if self.is_busy:
            raise RuntimeError(
                "Cannot navigate the session tree while the agent is running"
            )
        selected = self.session.get_entry(selected_id)
        editor_text = self._user_text(selected)
        target_id = selected.parent_id if editor_text is not None else selected.id
        old_leaf = self.session.leaf_id
        if target_id == old_leaf:
            return TreeNavigationResult(selected_id, old_leaf, editor_text)

        summary_entry_id: str | None = None
        if summarize:
            abandoned = self._abandoned_messages(old_leaf, target_id)
            if abandoned:
                instructions = BRANCH_SUMMARIZATION_PROMPT
                if custom_instructions:
                    instructions += (
                        f"\n\nAdditional focus:\n{custom_instructions.strip()}"
                    )
                self._tree_abort.clear()
                self._tree_summarizing = True
                await self._emit("branch_summary_started", selected_id=selected_id)
                try:
                    response = await generate_summary(
                        abandoned,
                        self.agent.model,
                        self.agent.stream_fn,
                        api_key=self.agent.api_key,
                        instructions=instructions,
                        abort=self._tree_abort,
                    )
                finally:
                    self._tree_summarizing = False
                summary_entry_id = self.session.branch_with_summary(
                    target_id,
                    response.text(),
                    from_id=old_leaf,
                    usage=response.usage,
                )
                await self._emit(
                    "branch_summary_finished",
                    selected_id=selected_id,
                    summary_entry_id=summary_entry_id,
                )
            else:
                self.session.branch(target_id)
        else:
            self.session.branch(target_id)

        self.agent.messages = self.session.build_context()
        await self._emit("session_changed", leaf_id=self.session.leaf_id)
        return TreeNavigationResult(
            selected_id,
            self.session.leaf_id,
            editor_text,
            summary_entry_id,
        )

    def _abandoned_messages(
        self, old_leaf: str | None, target_id: str | None
    ) -> list[Message]:
        old_branch = self.session.get_branch(old_leaf) if old_leaf is not None else []
        target_branch = (
            self.session.get_branch(target_id) if target_id is not None else []
        )
        common = 0
        while (
            common < len(old_branch)
            and common < len(target_branch)
            and old_branch[common].id == target_branch[common].id
        ):
            common += 1
        messages: list[Message] = []
        for entry in old_branch[common:]:
            if entry.type == "message" and entry.message is not None:
                messages.append(entry.message)
            elif entry.type == "branch_summary":
                messages.append(
                    UserMessage(
                        content=f"[Branch summary]\n{entry.summary or ''}",
                        timestamp=entry.timestamp,
                    )
                )
        return messages

    @staticmethod
    def _user_text(entry: SessionEntry) -> str | None:
        if entry.type != "message" or not isinstance(entry.message, UserMessage):
            return None
        if isinstance(entry.message.content, str):
            return entry.message.content
        return "".join(
            block.text
            for block in entry.message.content
            if isinstance(block, TextContent)
        )

    def _compaction_would_progress(self, kept_id: str | None) -> bool:
        """Whether compacting now would actually drop older messages.

        `kept_id` is where the post-compaction tail starts. If it is the very
        first message after the last compaction (or `None` means keep nothing),
        summarizing folds nothing away — so re-running it every turn would just
        thrash. Progress requires at least one message older than `kept_id`.
        """
        if kept_id is None:
            return True  # keep nothing -> the whole prior branch is folded away
        entries = self.session.get_branch()
        last_compaction = -1
        for index, entry in enumerate(entries):
            if entry.type == "compaction":
                last_compaction = index
        for entry in entries[last_compaction + 1 :]:
            if entry.type == "message" and entry.message is not None:
                return entry.id != kept_id
        return False

    def _find_first_kept_entry_id(self) -> str | None:
        entries = self.session.get_branch()
        last_compaction = -1
        for index, entry in enumerate(entries):
            if entry.type == "compaction":
                last_compaction = index
        candidates = [
            entry
            for entry in entries[last_compaction + 1 :]
            if entry.type == "message" and entry.message is not None
        ]
        budget = self.agent.model.context_window // 2
        kept: list[SessionEntry] = []
        total = 0
        for entry in reversed(candidates):
            tokens = estimate_tokens([entry.message])
            if kept and total + tokens > budget:
                break
            kept.append(entry)
            total += tokens
        kept.reverse()
        while kept and not isinstance(kept[0].message, UserMessage):
            kept.pop(0)
        return kept[0].id if kept else None


__all__ = [
    "AgentSessionController",
    "ControllerEvent",
    "ControllerEventType",
    "TreeNavigationResult",
]

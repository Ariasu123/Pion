"""Tests for the session layer: SessionManager tree store + compaction."""

from __future__ import annotations

import json
from pathlib import Path

from pion.llm.event_stream import AssistantMessageEvent
from pion.llm.types import (
    AssistantMessage,
    Model,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pion.session import (
    SessionEntry,
    SessionManager,
    compact,
    estimate_tokens,
    should_compact,
)


def _user(text: str) -> UserMessage:
    return UserMessage(content=text)


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)])


def _model(context_window: int = 1000) -> Model:
    return Model(id="test-model", base_url="http://localhost", context_window=context_window)


def _texts(messages: list) -> list[str]:
    """Flatten built context to comparable text payloads."""
    out = []
    for message in messages:
        if isinstance(message, UserMessage):
            out.append(message.content if isinstance(message.content, str) else "")
        else:
            out.append(message.text())
    return out


# ---------------------------------------------------------------------------
# Append + persistence
# ---------------------------------------------------------------------------


def test_append_and_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    manager = SessionManager(path)

    first = manager.append_message(_user("hello"))
    manager.append_message(_assistant("hi there"))
    manager.append_custom({"extension": "state"})
    manager.append_compaction("summary so far", first_kept_entry_id=first)

    # JSONL: one camelCase JSON object per line.
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 4
    assert '"parentId"' in lines[1]
    assert '"firstKeptEntryId"' in lines[3]
    for line in lines:
        SessionEntry.model_validate(json.loads(line))  # must parse back

    loaded = SessionManager.load(path)
    assert loaded.leaf_id == manager.leaf_id
    assert loaded.branch_ids() == manager.branch_ids()
    assert [m.model_dump() for m in loaded.build_context()] == [
        m.model_dump() for m in manager.build_context()
    ]

    # The loaded session stays writable (appends keep going to the same file).
    loaded.append_message(_user("continued"))
    assert len(path.read_text(encoding="utf-8").strip().split("\n")) == 5


def test_in_memory_session_writes_nothing(tmp_path: Path) -> None:
    manager = SessionManager()
    manager.append_message(_user("hello"))
    assert manager.leaf_id is not None
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Tree branching
# ---------------------------------------------------------------------------


def test_branching_and_switch_leaf() -> None:
    manager = SessionManager()
    root = manager.append_message(_user("root"))
    branch_b = manager.append_message(_assistant("branch B"))

    manager.switch_leaf(root)
    branch_c = manager.append_message(_user("branch C"))

    assert [entry.id for entry in manager.children(root)] == [branch_b, branch_c]
    assert sorted(manager.branch_ids()) == sorted([branch_b, branch_c])

    # Current leaf is on branch C: context reflects that branch only.
    assert _texts(manager.build_context()) == ["root", "branch C"]

    manager.switch_leaf(branch_b)
    assert _texts(manager.build_context()) == ["root", "branch B"]


def test_switch_leaf_rejects_unknown_id() -> None:
    manager = SessionManager()
    manager.append_message(_user("root"))
    try:
        manager.switch_leaf("does-not-exist")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


# ---------------------------------------------------------------------------
# Compaction boundary in build_context
# ---------------------------------------------------------------------------


def test_build_context_compaction_keeps_from_first_kept_entry() -> None:
    manager = SessionManager()
    manager.append_message(_user("old message"))
    kept = manager.append_message(_user("kept one"))
    manager.append_message(_assistant("kept two"))
    manager.append_compaction("SUMMARY", first_kept_entry_id=kept)
    manager.append_message(_user("after compaction"))

    context = manager.build_context()
    assert isinstance(context[0], UserMessage)
    assert context[0].content == "SUMMARY"
    assert _texts(context[1:]) == ["kept one", "kept two", "after compaction"]
    assert "old message" not in _texts(context)


def test_build_context_compaction_without_kept_entry_drops_history() -> None:
    manager = SessionManager()
    manager.append_message(_user("old one"))
    manager.append_message(_assistant("old two"))
    manager.append_compaction("SUMMARY", first_kept_entry_id=None)
    manager.append_message(_user("after"))

    assert _texts(manager.build_context()) == ["SUMMARY", "after"]


def test_build_context_uses_newest_compaction_on_path() -> None:
    manager = SessionManager()
    first = manager.append_message(_user("first"))
    manager.append_compaction("OLD SUMMARY", first_kept_entry_id=first)
    kept = manager.append_message(_user("kept"))
    manager.append_compaction("NEW SUMMARY", first_kept_entry_id=kept)
    manager.append_message(_user("latest"))

    # Only the newest compaction boundary applies.
    assert _texts(manager.build_context()) == ["NEW SUMMARY", "kept", "latest"]


def test_custom_entries_never_enter_context() -> None:
    manager = SessionManager()
    manager.append_message(_user("visible"))
    manager.append_custom({"secret": "state"})
    manager.append_message(_assistant("also visible"))

    assert _texts(manager.build_context()) == ["visible", "also visible"]


# ---------------------------------------------------------------------------
# Token estimation / trigger
# ---------------------------------------------------------------------------


def test_estimate_tokens_counts_content_chars() -> None:
    messages = [
        _user("a" * 40),  # 40 chars
        AssistantMessage(
            content=[
                TextContent(text="b" * 20),  # 20
                ThinkingContent(thinking="c" * 12),  # 12
                # 4 (name) + 15 (json of arguments)
                ToolCall(id="t1", name="read", arguments={"path": "ab"}),
            ]
        ),
        ToolResultMessage(
            tool_call_id="t1",
            tool_name="read",
            content=[TextContent(text="d" * 16)],  # 16
        ),
    ]
    # (40 + 20 + 12 + 4 + 15 + 16) // 4
    assert estimate_tokens(messages) == 26
    assert estimate_tokens([]) == 0


def test_should_compact_threshold() -> None:
    model = _model(context_window=1000)
    below = [_user("x" * 3200)]  # 800 estimated tokens
    above = [_user("x" * 3800)]  # 950 estimated tokens

    assert not should_compact(below, model, reserve=100)  # 800 <= 1000 - 100
    assert should_compact(above, model, reserve=100)  # 950 > 1000 - 100
    # Default reserve (16384) dwarfs a small window: always compact.
    assert should_compact(below, model)


# ---------------------------------------------------------------------------
# compact() with an injected stream
# ---------------------------------------------------------------------------


def _fake_stream_fn(summary_text: str, captured: dict):
    """Mimic `pion.llm.stream_simple`: return a raw async event generator."""

    def stream_fn(model, context, options):
        captured["model"] = model
        captured["context"] = context
        captured["options"] = options

        async def gen():
            yield AssistantMessageEvent(type="start", partial=AssistantMessage())
            yield AssistantMessageEvent(type="text_start", content_index=0)
            yield AssistantMessageEvent(
                type="text_delta", content_index=0, delta=summary_text
            )
            yield AssistantMessageEvent(type="text_end", content_index=0)
            yield AssistantMessageEvent(
                type="done",
                reason="stop",
                message=AssistantMessage(content=[TextContent(text=summary_text)]),
            )

        return gen()

    return stream_fn


async def test_compact_generates_and_records_summary() -> None:
    captured: dict = {}
    manager = SessionManager()
    manager.append_message(_user("please refactor the parser"))
    manager.append_message(_assistant("done, split it into lexer and parser"))
    old_messages = manager.build_context()

    summary = await compact(
        manager,
        old_messages,
        _model(),
        _fake_stream_fn("## Goal\nRefactor the parser", captured),
        api_key="sk-test",
    )

    assert summary == "## Goal\nRefactor the parser"

    # The summarization request carries the serialized conversation + prompt.
    assert captured["options"].api_key == "sk-test"
    prompt_message = captured["context"].messages[0]
    assert "<conversation>" in prompt_message.content
    assert "[User]: please refactor the parser" in prompt_message.content
    assert "## Goal" in prompt_message.content

    # The compaction was appended to the session and now bounds the context.
    leaf = manager.get_entry(manager.leaf_id)
    assert leaf.type == "compaction"
    assert leaf.summary == summary
    assert leaf.first_kept_entry_id is None
    assert _texts(manager.build_context()) == [summary]


async def test_compact_raises_on_error_event() -> None:
    def stream_fn(model, context, options):
        async def gen():
            yield AssistantMessageEvent(type="start", partial=AssistantMessage())
            yield AssistantMessageEvent(
                type="error",
                error=AssistantMessage(
                    stop_reason="error", error_message="provider exploded"
                ),
            )

        return gen()

    manager = SessionManager()
    try:
        await compact(manager, [_user("hi")], _model(), stream_fn)
    except RuntimeError as exc:
        assert "provider exploded" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

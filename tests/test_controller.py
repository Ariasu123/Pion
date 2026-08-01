"""Tests for the UI-neutral interactive session controller."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_agent import make_agent

from pion.controller import AgentSessionController
from pion.llm.types import AssistantMessage, TextContent, UserMessage
from pion.session import SessionManager


def _user(text: str) -> UserMessage:
    return UserMessage(content=text)


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)])


def _controller(tmp_path: Path, scripts):
    agent, _, _ = make_agent(scripts)
    session = SessionManager(tmp_path / "session.jsonl")
    return AgentSessionController(agent, session, tmp_path / "session.jsonl")


async def test_prompt_persists_messages_and_usage(tmp_path: Path) -> None:
    controller = _controller(tmp_path, [{"text": "hello"}])
    final = await controller.prompt("hi")
    assert final.text() == "hello"
    assert [entry.type for entry in controller.session.get_entries()] == [
        "message",
        "message",
    ]
    assert controller.last_usage == final.usage


async def test_navigate_to_user_prefills_editor_and_branches_from_parent(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, [{"text": "unused"}])
    root = controller.session.append_message(_user("first prompt"))
    controller.session.append_message(_assistant("first answer"))
    controller.agent.messages = controller.session.build_context()

    result = await controller.navigate_tree(root)
    assert result.editor_text == "first prompt"
    assert result.leaf_id is None
    assert controller.agent.messages == []


async def test_branch_summary_is_transactional_and_records_old_leaf(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, [{"text": "useful abandoned facts"}])
    root = controller.session.append_message(_user("first prompt"))
    old_leaf = controller.session.append_message(_assistant("old branch"))
    controller.agent.messages = controller.session.build_context()

    result = await controller.navigate_tree(root, summarize=True)
    assert result.editor_text == "first prompt"
    assert result.summary_entry_id is not None
    summary = controller.session.get_entry(result.summary_entry_id)
    assert summary.type == "branch_summary"
    assert summary.parent_id is None
    assert summary.from_id == old_leaf
    assert "useful abandoned facts" in (summary.summary or "")


async def test_branch_summary_failure_keeps_session_unchanged(tmp_path: Path) -> None:
    controller = _controller(
        tmp_path,
        [{"stop_reason": "error", "error_message": "summary failed"}],
    )
    root = controller.session.append_message(_user("first prompt"))
    old_leaf = controller.session.append_message(_assistant("old branch"))
    before = [entry.model_dump() for entry in controller.session.get_entries()]
    controller.agent.messages = controller.session.build_context()

    with pytest.raises(RuntimeError, match="summary failed"):
        await controller.navigate_tree(root, summarize=True)

    assert controller.session.leaf_id == old_leaf
    assert [entry.model_dump() for entry in controller.session.get_entries()] == before


async def test_controller_label_updates_context_without_exposing_label(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, [{"text": "unused"}])
    target = controller.session.append_message(_user("visible"))
    controller.agent.messages = controller.session.build_context()
    await controller.set_label(target, "checkpoint")
    assert controller.session.get_label(target) == "checkpoint"
    assert len(controller.agent.messages) == 1
    assert controller.agent.messages[0].content == "visible"


async def test_force_compaction_appends_one_transactional_entry(tmp_path: Path) -> None:
    controller = _controller(tmp_path, [{"text": "summary"}])
    controller.session.append_message(_user("prompt"))
    controller.session.append_message(_assistant("answer"))
    controller.agent.messages = controller.session.build_context()

    assert await controller.maybe_compact(force=True) == "summary"
    compactions = [
        entry
        for entry in controller.session.get_entries()
        if entry.type == "compaction"
    ]
    assert len(compactions) == 1
    assert compactions[0].summary == "summary"


async def test_controller_observer_error_does_not_break_prompt(tmp_path: Path) -> None:
    controller = _controller(tmp_path, [{"text": "answer"}])

    def broken_observer(event) -> None:
        raise RuntimeError("renderer failed")

    controller.subscribe(broken_observer)
    final = await controller.prompt("prompt")
    assert final.text() == "answer"
    assert len(controller.subscriber_errors) >= 1

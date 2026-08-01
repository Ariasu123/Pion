"""Headless tests for the Textual Pion UI."""

from __future__ import annotations

from pathlib import Path

from test_agent import EchoTool, make_agent, tool_call
from textual.widgets import Input, Markdown, Static

from pion.controller import AgentSessionController
from pion.llm.types import AssistantMessage, TextContent, ThinkingContent, UserMessage
from pion.session import SessionManager
from pion.tui import PionApp, TUIStatus
from pion.tui.screens import BranchScreen, ChoiceScreen, TextInputScreen
from pion.tui.theme import PION_DARK
from pion.tui.widgets import (
    ChatMessage,
    EmptyState,
    HeaderBar,
    PromptComposer,
    PromptEditor,
    SessionTreePanel,
    ThinkingPanel,
    ToolCard,
)


def make_tui(tmp_path: Path, scripts, tools=None):
    agent, _, _ = make_agent(scripts, tools=tools)
    session = SessionManager(tmp_path / "session.jsonl")
    controller = AgentSessionController(
        agent,
        session,
        tmp_path / "session.jsonl",
    )
    return PionApp(controller, TUIStatus("Pion", "docker", 2, 8)), controller


async def test_tree_is_a_hidden_drawer_at_every_width(tmp_path: Path) -> None:
    app, _ = make_tui(tmp_path, [{"text": "unused"}])
    async with app.run_test(size=(120, 40)) as pilot:
        panel = app.query_one(SessionTreePanel)
        scrim = app.query_one("#tree-scrim", Static)
        assert not panel.display
        assert not scrim.display

        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert not panel.display

        await pilot.press("ctrl+b")
        await pilot.pause()
        assert panel.has_class("-shown")
        assert panel.display
        assert scrim.display
        assert 26 <= panel.region.width <= 32

        await pilot.press("ctrl+b")
        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        assert app.screen.has_class("narrow")
        assert not panel.display
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert panel.display
        assert panel.region.width <= 32


async def test_tree_drawer_closes_with_escape_and_scrim(tmp_path: Path) -> None:
    app, _ = make_tui(tmp_path, [{"text": "unused"}])
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app._tree_open

        await pilot.press("escape")
        await pilot.pause()
        assert not app._tree_open


async def test_escape_aborts_before_closing_an_open_drawer(tmp_path: Path) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+b")
        controller.agent.is_streaming = True
        await pilot.press("escape")
        await pilot.pause()
        assert controller.agent._abort.is_set()
        assert app._tree_open

        controller.agent.is_streaming = False
        await pilot.press("escape")
        await pilot.pause()
        assert not app._tree_open

        await pilot.press("ctrl+b")
        await pilot.pause()
        await pilot.click("#tree-scrim")
        await pilot.pause()
        assert not app._tree_open


async def test_visual_chrome_and_empty_state(tmp_path: Path) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        header = app.query_one(HeaderBar)
        composer = app.query_one(PromptComposer)

        assert app.theme == "pion-dark"
        assert PION_DARK.primary == "#D97757"
        assert header.region == (0, 0, 120, 1)
        assert composer.region.height == 3
        assert len(app.query("#footer-hints")) == 0
        assert len(app.query(EmptyState)) == 1
        summary = str(app.query_one("#header-summary").render())
        assert "PION · Pion" in summary
        assert controller.agent.model.id in summary
        assert str(app.query_one("#header-status").render()) == "● READY · CTX 0%"

        controller.agent.is_streaming = True
        app._refresh_status()
        assert "RUNNING" in str(app.query_one("#header-status").render())
        controller.agent.is_streaming = False


async def test_header_responsive_tiers(tmp_path: Path) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    controller.agent.model.id = "provider/very-long-model-name-for-testing"
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        medium = str(app.query_one("#header-summary").render())
        assert "PION" in medium
        assert "provider" in medium

        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        compact = str(app.query_one("#header-summary").render())
        assert compact.startswith("PION ·")
        assert "provider" not in compact
        assert "CTX" in str(app.query_one("#header-status").render())


async def test_prompt_submission_streams_and_persists(tmp_path: Path) -> None:
    app, controller = make_tui(tmp_path, [{"text": "hello from tui"}])
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.query_one(PromptEditor)
        editor.load_text("hi")
        editor.focus()
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert editor.text == ""
        assert len(controller.session.get_entries()) == 2
        assert [card.role for card in app.query(ChatMessage)] == ["YOU", "PION"]
        assert len(app.query(EmptyState)) == 0


async def test_ctrl_j_inserts_newline_without_submitting(tmp_path: Path) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.query_one(PromptEditor)
        editor.load_text("line one")
        editor.move_cursor((0, len("line one")))
        editor.focus()
        await pilot.press("ctrl+j")
        await pilot.pause()
        assert editor.text == "line one\n"
        assert controller.session.get_entries() == []


async def test_composer_grows_between_one_and_six_rows(tmp_path: Path) -> None:
    app, _ = make_tui(tmp_path, [{"text": "unused"}])
    async with app.run_test(size=(120, 30)) as pilot:
        editor = app.query_one(PromptEditor)
        composer = app.query_one(PromptComposer)

        editor.load_text("\n".join(str(index) for index in range(12)))
        await pilot.pause()
        assert editor.region.height == 6
        assert composer.region.height == 8

        editor.clear()
        await pilot.pause()
        assert editor.region.height == 1
        assert composer.region.height == 3
        assert str(app.query_one("#composer-hint").render()) == "Enter send"

        composer.set_running(True)
        await pilot.pause()
        assert editor.disabled
        assert str(app.query_one("#composer-hint").render()) == "Esc abort"
        composer.set_running(False)

        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        assert not app.query_one("#composer-hint").display


async def test_thinking_is_collapsed_and_retained(tmp_path: Path) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    controller.session.append_message(UserMessage(content="think"))
    controller.session.append_message(
        AssistantMessage(
            content=[
                ThinkingContent(thinking="private reasoning details"),
                TextContent(text="final answer"),
            ]
        )
    )
    controller.agent.messages = controller.session.build_context()

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        thinking = app.query_one(ThinkingPanel)
        assert thinking.collapsed
        assert thinking.thinking_text == "private reasoning details"
        assert "Thinking" in thinking.title


async def test_narrow_chinese_message_wraps_without_horizontal_scroll(
    tmp_path: Path,
) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    controller.session.append_message(
        UserMessage(
            content="请帮我重新设计这个终端界面，并且保持所有键盘操作都可以使用。"
        )
    )
    controller.session.append_message(
        AssistantMessage(content=[TextContent(text="我会保持功能不变并改善视觉层级。")])
    )
    controller.agent.messages = controller.session.build_context()

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.query_one("#conversation").max_scroll_x == 0
        assert app.query_one(PromptEditor).region.width > 55


async def test_transcript_has_shared_edges_and_minimal_role_glyphs(
    tmp_path: Path,
) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    controller.session.append_message(UserMessage(content="A compact prompt"))
    controller.session.append_message(
        AssistantMessage(
            content=[
                ThinkingContent(thinking="check details"),
                TextContent(text="A concise answer"),
            ]
        )
    )
    controller.agent.messages = controller.session.build_context()

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        messages = list(app.query(ChatMessage))
        assert len(messages) == 2
        assert messages[0].region.x == messages[1].region.x
        assert messages[0].region.width == messages[1].region.width
        assert messages[0].region.height <= 2
        assert str(messages[0].query_one(".message-glyph").render()) == "›"
        assert str(messages[1].query_one(".message-glyph").render()) == "◆"
        rendered = " ".join(str(widget.render()) for widget in app.query(Static))
        assert "YOU" not in rendered
        assert "PION" not in " ".join(
            str(widget.render()) for widget in messages[1].query(Static)
        )


async def test_tool_call_becomes_collapsed_result_card(tmp_path: Path) -> None:
    app, _ = make_tui(
        tmp_path,
        [
            {"tool_calls": [tool_call("echo", {"text": "ping"}, call_id="c1")]},
            {"text": "done"},
        ],
        tools=[EchoTool()],
    )
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.query_one(PromptEditor)
        editor.load_text("go")
        editor.focus()
        await pilot.press("enter")
        await pilot.pause(0.2)

        cards = list(app.query(ToolCard))
        assert len(cards) == 1
        assert cards[0].collapsed
        assert cards[0].result_text == "echo:ping"
        assert not cards[0].is_error
        assert "DONE" in cards[0].title


async def test_tree_filter_cycles_and_session_changes_refresh(tmp_path: Path) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    controller.session.append_message(UserMessage(content="tree prompt"))
    controller.agent.messages = controller.session.build_context()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = app.query_one(SessionTreePanel)
        assert panel.filter_mode == "default"
        panel.action_cycle_filter()
        await pilot.pause()
        assert panel.filter_mode == "no-tools"
        assert "tree prompt" in str(
            panel.query_one("#session-tree-widget").root.children[0].label
        )

        panel.query_one("#session-tree-widget").focus()
        await pilot.pause()
        assert "Enter select" in str(panel.query_one("#tree-help").render())


async def test_branch_modal_switches_user_node_and_prefills_editor(
    tmp_path: Path,
) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    root = controller.session.append_message(UserMessage(content="edit this prompt"))
    controller.session.append_message(
        AssistantMessage(content=[TextContent(text="old answer")])
    )
    controller.agent.messages = controller.session.build_context()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("ctrl+b")
        app._navigate_tree(root)
        await pilot.pause()
        assert isinstance(app.screen, BranchScreen)
        await pilot.click("#branch-no")
        await pilot.pause()
        assert app.screen is app.screen_stack[0]
        assert app.query_one(PromptEditor).text == "edit this prompt"
        assert controller.session.leaf_id is None
        assert not app._tree_open


async def test_modal_escape_and_label_persistence(tmp_path: Path) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    target = controller.session.append_message(UserMessage(content="checkpoint"))
    controller.agent.messages = controller.session.build_context()

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert isinstance(app.screen, ChoiceScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is app.screen_stack[0]

        app._edit_label(target)
        await pilot.pause()
        assert isinstance(app.screen, TextInputScreen)
        app.screen.query_one(Input).value = "milestone"
        await pilot.press("enter")
        await pilot.pause()
        assert controller.session.get_label(target) == "milestone"


async def test_selecting_current_assistant_leaf_skips_branch_modal(
    tmp_path: Path,
) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    controller.session.append_message(UserMessage(content="prompt"))
    leaf = controller.session.append_message(
        AssistantMessage(content=[TextContent(text="answer")])
    )
    controller.agent.messages = controller.session.build_context()

    async with app.run_test(size=(120, 40)) as pilot:
        app._navigate_tree(leaf)
        await pilot.pause()
        assert app.screen is app.screen_stack[0]
        assert controller.session.leaf_id == leaf


async def test_mcp_tool_card_shows_source_and_elapsed_time(tmp_path: Path) -> None:
    app, _ = make_tui(tmp_path, [{"text": "unused"}])
    async with app.run_test(size=(120, 40)) as pilot:
        card = ToolCard(
            "mcp-call",
            "filesystem__read_file",
            {"path": "README.md"},
            running=True,
        )
        await app.query_one("#conversation").mount(card)
        card.update_result("contents")
        await pilot.pause()
        details = str(card.query_one(".tool-details").render())
        assert "MCP server filesystem" in details
        assert card.duration_seconds is not None


async def test_markdown_and_long_paths_do_not_scroll_horizontally(
    tmp_path: Path,
) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    markdown = """# Heading

- one
- two

> quiet quote

[`link`](https://example.com) and `inline code`

```python
/a/very/long/path/that/keeps/going/without/a/natural/break/for/a/long/time/example.py
```

中文内容应该在窄终端中保持正确换行，不应产生横向滚动。
"""
    controller.session.append_message(UserMessage(content="format this"))
    controller.session.append_message(
        AssistantMessage(content=[TextContent(text=markdown)])
    )
    controller.agent.messages = controller.session.build_context()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        assert len(app.query(Markdown)) == 2
        assert app.query_one("#conversation").max_scroll_x == 0


async def test_stats_reports_runtime_and_usage_fields(tmp_path: Path) -> None:
    app, controller = make_tui(tmp_path, [{"text": "unused"}])
    notifications: list[str] = []
    app.notify = lambda message, **kwargs: notifications.append(str(message))  # type: ignore[method-assign]
    async with app.run_test(size=(120, 30)):
        await app._execute_command("stats")
        text = notifications[-1]
        assert f"model {controller.agent.model.id}" in text
        assert "sandbox docker" in text
        assert "MCP 2 server/8 tool" in text
        assert "context" in text
        assert "usage unavailable" in text

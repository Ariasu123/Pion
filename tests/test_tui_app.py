"""Integration tests for PionTUI driven through a FakeTerminal."""

from __future__ import annotations

import asyncio

from test_agent import EchoTool, make_agent, tool_call

from pion.controller import AgentSessionController
from pion.llm.types import AssistantMessage, TextContent, ThinkingContent
from pion.session import SessionManager
from pion.tui import PionTUI, TUIStatus
from pion.tui.app.chat import AssistantMessageComponent, ToolExecutionComponent
from pion.tui.core import FakeTerminal, strip_ansi
from pion.tui.core.renderer import CURSOR_MARKER
from pion.tui.theme import load_theme

THEME = load_theme("dark", truecolor=True)
USER_BG = "\x1b[48;2;52;53;65m"  # userMsgBg #343541
SUCCESS_BG = "\x1b[48;2;40;50;40m"  # toolSuccessBg #283228
ERROR_BG = "\x1b[48;2;60;40;40m"  # toolErrorBg #3c2828


def make_tui(tmp_path, scripts, tools=None, columns=80, rows=24):
    agent, calls, _ = make_agent(scripts, tools=tools)
    session_path = tmp_path / "session.jsonl"
    session = SessionManager(session_path)
    controller = AgentSessionController(agent, session, session_path)
    terminal = FakeTerminal(columns, rows)
    tui = PionTUI(
        controller,
        TUIStatus(project="proj", sandbox="off"),
        theme=THEME,
        terminal=terminal,
    )
    return tui, terminal, calls


def plain(frame_lines):
    return "\n".join(strip_ansi(line).replace(CURSOR_MARKER, "") for line in frame_lines)


async def settle(duration: float = 0.1):
    await asyncio.sleep(duration)


async def run_tui(tui):
    task = asyncio.ensure_future(tui.run_async())
    await settle()
    return task


async def stop_tui(tui, task):
    tui.quit()
    await asyncio.wait_for(task, timeout=2)


async def test_startup_header_and_footer(tmp_path):
    tui, terminal, _ = make_tui(tmp_path, [{"text": "hi"}])
    task = await run_tui(tui)
    out = terminal.output()
    assert "pion" in out and "fake-model" in out
    assert "proj" in out and "sandbox off" in out
    assert "session.jsonl" in plain(tui.root.render(80))
    await stop_tui(tui, task)


async def test_prompt_streams_and_renders(tmp_path):
    tui, terminal, calls = make_tui(tmp_path, [{"text": "hello world"}])
    task = await run_tui(tui)
    terminal.feed(b"hi there\r")
    await settle()
    out = terminal.output()
    assert USER_BG in out  # user message band
    assert "hi there" in out
    assert "hello world" in out  # assistant reply
    assert len(calls) == 1
    # Session persisted through the controller.
    assert tui.controller.session.get_entries()
    await stop_tui(tui, task)


async def test_tool_card_lifecycle(tmp_path):
    echo = EchoTool()
    tui, terminal, _ = make_tui(
        tmp_path,
        [
            {"tool_calls": [tool_call("echo", {"text": "ping"}, call_id="c1")]},
            {"text": "done"},
        ],
        tools=[echo],
    )
    task = await run_tui(tui)
    terminal.feed(b"go\r")
    await settle()
    out = terminal.output()
    assert "echo" in out and "ping" in out
    assert "echo:ping" in out  # tool output
    assert SUCCESS_BG in out
    assert "Took" in out
    await stop_tui(tui, task)


async def test_tool_card_error_tint(tmp_path):
    class FailingTool(EchoTool):
        async def execute(self, tool_call_id, args, abort=None, on_update=None):
            raise RuntimeError("boom")

    tui, terminal, _ = make_tui(
        tmp_path,
        [
            {"tool_calls": [tool_call("fail", {"text": "x"}, call_id="c9")]},
            {"text": "recovered"},
        ],
        tools=[FailingTool(name="fail")],
    )
    task = await run_tui(tui)
    terminal.feed(b"go\r")
    await settle()
    assert ERROR_BG in terminal.output()
    await stop_tui(tui, task)


async def test_abort_appends_notice(tmp_path):
    echo = EchoTool(delay=0.3)
    tui, terminal, _ = make_tui(
        tmp_path,
        [
            {"tool_calls": [tool_call("echo", {"text": "slow"}, call_id="c1")]},
            {"check_abort": True},
        ],
        tools=[echo],
    )
    task = await run_tui(tui)
    terminal.feed(b"go\r")
    await settle()
    assert tui.controller.is_busy
    terminal.feed(b"\x1b")  # escape → abort
    await settle(0.8)
    out = terminal.output()
    assert "Operation aborted" in out
    assert not tui.controller.is_busy
    await stop_tui(tui, task)


async def test_message_queue_while_busy(tmp_path):
    echo = EchoTool(delay=0.2)
    tui, terminal, _ = make_tui(
        tmp_path,
        [
            {"tool_calls": [tool_call("echo", {"text": "one"}, call_id="c1")]},
            {"text": "first done"},
            {"text": "second done"},
        ],
        tools=[echo],
    )
    task = await run_tui(tui)
    terminal.feed(b"first\r")
    await settle(0.05)  # busy now
    assert tui.controller.is_busy
    terminal.feed(b"second\r")  # queued, not sent
    await settle(0.05)
    assert len(tui.queue) == 1
    assert "queued" in plain(tui.root.render(80))
    await settle(0.5)  # both runs complete
    assert "first done" in terminal.output()
    assert "second done" in terminal.output()
    assert len(tui.queue) == 0
    await stop_tui(tui, task)


async def test_alt_up_returns_queued_message(tmp_path):
    echo = EchoTool(delay=0.3)
    tui, terminal, _ = make_tui(
        tmp_path,
        [{"tool_calls": [tool_call("echo", {"text": "x"}, call_id="c1")]}, {"text": "ok"}],
        tools=[echo],
    )
    task = await run_tui(tui)
    terminal.feed(b"first\r")
    await settle(0.05)
    terminal.feed(b"second\r")
    await settle(0.05)
    assert len(tui.queue) == 1
    terminal.feed(b"\x1b\x1b[A")  # alt+up
    await settle()
    assert len(tui.queue) == 0
    assert tui.editor.text == "second"
    tui.controller.abort()
    await stop_tui(tui, task)


async def test_stats_command(tmp_path):
    tui, terminal, _ = make_tui(tmp_path, [{"text": "hi"}])
    task = await run_tui(tui)
    terminal.feed(b"/stats\r")  # first Enter accepts the autocomplete suggestion
    await settle()
    terminal.feed(b"\r")  # second Enter submits
    await settle()
    out = terminal.output()
    assert "model fake-model" in out
    assert "context" in out
    await stop_tui(tui, task)


async def test_unknown_command(tmp_path):
    tui, terminal, _ = make_tui(tmp_path, [{"text": "hi"}])
    task = await run_tui(tui)
    terminal.feed(b"/nope\r")
    await settle()
    assert "Unknown command: /nope" in terminal.output()
    await stop_tui(tui, task)


async def test_theme_command_switches(tmp_path):
    tui, terminal, _ = make_tui(tmp_path, [{"text": "hi"}])
    task = await run_tui(tui)
    terminal.feed(b"/theme light\r")
    await settle()
    assert tui.theme.name == "light"
    terminal.feed(b"/theme nope\r")
    await settle()
    assert "Unknown theme" in terminal.output()
    assert tui.theme.name == "light"
    await stop_tui(tui, task)


async def test_tree_overlay_open_close(tmp_path):
    tui, terminal, _ = make_tui(tmp_path, [{"text": "hello"}])
    task = await run_tui(tui)
    terminal.feed(b"hi\r")
    await settle()
    terminal.feed(b"\x02")  # ctrl+b
    await settle()
    assert "Session tree" in terminal.output()
    assert tui._tree_overlay is not None
    terminal.feed(b"\x1b")  # escape closes
    await settle(0.2)
    assert tui._tree_overlay is None
    await stop_tui(tui, task)


async def test_tree_navigation_prefills_editor(tmp_path):
    tui, terminal, _ = make_tui(
        tmp_path, [{"text": "reply one"}, {"text": "reply two"}]
    )
    task = await run_tui(tui)
    terminal.feed(b"first\r")
    await settle()
    terminal.feed(b"second\r")
    await settle()
    terminal.feed(b"\x02")  # open tree
    await settle()
    # Rows: user first, assistant, user second, assistant(leaf). Select "second".
    terminal.feed(b"\x1b[A")  # up
    await settle()
    terminal.feed(b"\r")  # enter → branch decision overlay
    await settle()
    assert "abandons" in terminal.output()
    terminal.feed(b"\r")  # "Navigate without summary"
    await settle(0.2)
    assert tui.editor.text == "second"
    await stop_tui(tui, task)


async def test_model_selector_opens(tmp_path):
    tui, terminal, _ = make_tui(tmp_path, [{"text": "hi"}])
    task = await run_tui(tui)
    terminal.feed(b"\x0c")  # ctrl+l
    await settle()
    assert "Select model" in terminal.output()
    terminal.feed(b"\x1b")
    await settle(0.2)
    await stop_tui(tui, task)


async def test_ctrl_o_expands_tool_output(tmp_path):
    class LongTool(EchoTool):
        async def execute(self, tool_call_id, args, abort=None, on_update=None):
            from pion.tools.base import AgentToolResult

            return AgentToolResult.text("\n".join(f"line {i}" for i in range(10)))

    tui, terminal, _ = make_tui(
        tmp_path,
        [
            {"tool_calls": [tool_call("long", {"text": "x"}, call_id="c1")]},
            {"text": "done"},
        ],
        tools=[LongTool(name="long")],
    )
    task = await run_tui(tui)
    terminal.feed(b"go\r")
    await settle()
    out = terminal.output()
    assert "earlier lines, ctrl+o to expand" in out
    assert "line 0" not in out  # collapsed: only last 5 lines
    terminal.feed(b"\x0f")  # ctrl+o → expand all
    await settle()
    assert "line 0" in terminal.output()
    await stop_tui(tui, task)


async def test_ctrl_t_toggles_thinking(tmp_path):
    show = True
    component = AssistantMessageComponent(lambda: show, theme=THEME)
    message = AssistantMessage(
        content=[ThinkingContent(thinking="deep thought"), TextContent(text="answer")]
    )
    component.finalize(message)
    shown = plain(component.render(60))
    assert "deep thought" in shown and "answer" in shown
    show = False
    component.invalidate()
    hidden = plain(component.render(60))
    assert "deep thought" not in hidden
    assert "Thinking…" in hidden


def test_tool_component_mcp_source():
    component = ToolExecutionComponent(
        "c1", "weather__forecast", {"query": "berlin"}, lambda: False, theme=THEME
    )
    component.update_result("sunny", False)
    rendered = plain(component.render(70))
    assert "mcp weather" in rendered
    assert "sunny" in rendered

"""Tests for the extension hook system (pion.hooks)."""

from __future__ import annotations

from pion.agent.agent import Agent
from pion.hooks import ExtensionManager
from pion.llm.types import ToolResultMessage, UserMessage

from test_agent import fake_stream_fn, make_model, tool_call

# An extension registering a tool, a tool_call blocker, a context
# transformer, and a command. v2 changes the tool output and command result
# so reload() is observable.
EXTENSION_V1 = '''
from pydantic import BaseModel
from pion.tools.base import AgentToolResult


class HiArgs(BaseModel):
    text: str


class HiTool:
    name = "hi"
    label = "hi"
    description = "Say hi"
    Args = HiArgs
    execution_mode = "parallel"

    @property
    def parameters(self):
        return self.Args.model_json_schema()

    async def execute(self, tool_call_id, args, abort=None, on_update=None):
        return AgentToolResult.text("hi-v1:" + args.text)


def setup(api):
    api.register_tool(HiTool())

    async def blocker(event):
        if event.tool_name == "hi" and event.args.get("text") == "bad":
            return {"block": True, "reason": "no bad words"}
        return None

    api.on("tool_call", blocker)

    def transformer(messages):
        from pion.llm.types import UserMessage

        return [*messages, UserMessage(content="injected-by-extension")]

    api.on("context", transformer)

    async def hello():
        return "hello-v1"

    api.register_command("hello", hello)
'''

EXTENSION_V2 = EXTENSION_V1.replace("hi-v1", "hi-v2").replace("hello-v1", "hello-v2")

BROKEN_EXTENSION = '''
def setup(api):
    def boom(messages):
        raise RuntimeError("extension exploded")

    api.on("context", boom)
'''


def write_extension(tmp_path, source: str, name: str = "myext.py"):
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir(exist_ok=True)
    path = ext_dir / name
    path.write_text(source)
    return ext_dir, path


def make_hooked_agent(scripts, manager):
    stream_fn, calls = fake_stream_fn(scripts)
    agent = Agent(
        model=make_model(), tools=[], stream_fn=stream_fn, extension_manager=manager
    )
    return agent, calls


# ---------------------------------------------------------------------------
# Loading and wiring
# ---------------------------------------------------------------------------


async def test_load_extension_and_wire_hooks(tmp_path):
    ext_dir, _ = write_extension(tmp_path, EXTENSION_V1)
    manager = ExtensionManager()
    await manager.load([ext_dir])

    assert manager.errors == []
    assert [t.name for t in manager.tools] == ["hi"]
    assert "hello" in manager.commands
    assert await manager.commands["hello"]() == "hello-v1"

    scripts = [
        {"tool_calls": [tool_call("hi", {"text": "yo"}, call_id="c1")]},
        {"text": "done"},
    ]
    agent, calls = make_hooked_agent(scripts, manager)
    final = await agent.prompt("go")

    # Extension-registered tool executed through the loop.
    results = [m for m in agent.messages if isinstance(m, ToolResultMessage)]
    assert len(results) == 1
    assert not results[0].is_error
    assert results[0].text() == "hi-v1:yo"
    assert final.text() == "done"

    # Context transformer ran before every LLM call.
    first_call = calls[0]["context"]
    assert any(
        getattr(m, "role", None) == "user" and m.content == "injected-by-extension"
        for m in first_call.messages
    )
    # The extension tool was advertised to the LLM.
    assert "hi" in [t.name for t in first_call.tools]


async def test_tool_call_blocked_by_extension(tmp_path):
    ext_dir, _ = write_extension(tmp_path, EXTENSION_V1)
    manager = ExtensionManager()
    await manager.load([ext_dir])

    scripts = [
        {"tool_calls": [tool_call("hi", {"text": "bad"}, call_id="c1")]},
        {"text": "ok"},
    ]
    agent, _ = make_hooked_agent(scripts, manager)
    await agent.prompt("go")

    results = [m for m in agent.messages if isinstance(m, ToolResultMessage)]
    assert len(results) == 1
    assert results[0].is_error
    assert "no bad words" in results[0].text()


async def test_before_agent_start_injection(tmp_path):
    manager = ExtensionManager()

    def before(event):
        return {
            "system_prompt": event.system_prompt + "\nEXTRA",
            "messages": [UserMessage(content="injected-by-hook")],
        }

    manager.api.on("before_agent_start", before)

    scripts = [{"text": "ok"}]
    agent, calls = make_hooked_agent(scripts, manager)
    agent.system_prompt = "BASE"
    await agent.prompt("go")

    assert calls[0]["context"].system_prompt == "BASE\nEXTRA"
    # Injected message persisted ahead of the user prompt.
    contents = [m.content for m in agent.messages if m.role == "user"]
    assert contents == ["injected-by-hook", "go"]


async def test_notification_handlers_and_errors_never_crash(tmp_path):
    manager = ExtensionManager()
    seen: list = []
    manager.api.on("agent_start", lambda payload: seen.append(("start", payload)))
    manager.api.on("agent_end", lambda payload: seen.append(("end", len(payload["messages"]))))
    manager.api.on(
        "session_before_compact", lambda payload: seen.append(("compact", payload))
    )

    scripts = [{"text": "ok"}]
    agent, _ = make_hooked_agent(scripts, manager)
    await agent.prompt("go")
    await manager.notify("session_before_compact", {"reason": "manual"})

    assert seen[0] == ("start", {})
    assert seen[1][0] == "end"
    assert seen[2] == ("compact", {"reason": "manual"})
    assert manager.errors == []


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


async def test_handler_exception_isolated(tmp_path):
    ext_dir, _ = write_extension(tmp_path, BROKEN_EXTENSION, name="broken.py")
    manager = ExtensionManager()
    await manager.load([ext_dir])

    scripts = [{"text": "fine"}]
    agent, _ = make_hooked_agent(scripts, manager)
    final = await agent.prompt("go")

    assert final.text() == "fine"  # agent unaffected
    assert len(manager.errors) == 1
    assert "extension exploded" in str(manager.errors[0])


async def test_unknown_event_rejected():
    manager = ExtensionManager()
    try:
        manager.api.on("not_an_event", lambda event: None)
    except ValueError as exc:
        assert "not_an_event" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------


async def test_reload_picks_up_file_changes(tmp_path):
    ext_dir, path = write_extension(tmp_path, EXTENSION_V1)
    manager = ExtensionManager()
    await manager.load([ext_dir])
    assert await manager.commands["hello"]() == "hello-v1"

    path.write_text(EXTENSION_V2)
    await manager.reload()

    # Tables were rebuilt (not duplicated) from the new file contents.
    assert await manager.commands["hello"]() == "hello-v2"
    assert len(manager.tools) == 1
    assert len(manager.handlers["tool_call"]) == 1
    assert len(manager.handlers["context"]) == 1

    scripts = [
        {"tool_calls": [tool_call("hi", {"text": "yo"}, call_id="c1")]},
        {"text": "done"},
    ]
    agent, _ = make_hooked_agent(scripts, manager)
    await agent.prompt("go")

    results = [m for m in agent.messages if isinstance(m, ToolResultMessage)]
    assert results[0].text() == "hi-v2:yo"

"""Offline end-to-end demo of pion with a scripted fake LLM provider.

Runs a real Agent with the real DEFAULT_TOOLS (read/write/edit/bash) in a
temporary directory — no network involved. The fake provider first emits a
`write` tool call, then a `bash` (cat) tool call, then a final text summary.

Run: uv run python demos/mock_e2e.py   (prints MOCK E2E OK, exit 0)
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from pion.agent.agent import Agent
from pion.agent.events import AgentEvent
from pion.llm.event_stream import AssistantMessageEvent, AssistantMessageEventStream
from pion.llm.types import (
    AssistantMessage,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

USER_PROMPT = "create hello.txt containing 'hello pion' and then show it with bash"
FILE_CONTENT = "hello pion\n"

# ---------------------------------------------------------------------------
# Fake provider (shape-copied from tests/test_agent.py)
# ---------------------------------------------------------------------------


def make_model() -> Model:
    """Offline stand-in model descriptor."""
    return Model(
        id="mock-model",
        name="Mock",
        api="openai-completions",
        provider="mock",
        base_url="http://localhost:9",
    )


def tool_call(name: str, args: dict[str, Any], call_id: Optional[str] = None) -> ToolCall:
    return ToolCall(id=call_id or f"call-{name}-1", name=name, arguments=args)


def fake_stream_fn(scripts: list[dict[str, Any]]):
    """Build a stream_fn replaying one script per LLM call.

    Each script: {"text": str, "tool_calls": [ToolCall]}. Returns
    (stream_fn, calls) where calls records every invocation.
    """
    calls: list[dict[str, Any]] = []

    def stream_fn(model, context, options=None) -> AssistantMessageEventStream:
        script = scripts[min(len(calls), len(scripts) - 1)]
        calls.append({"model": model, "context": context, "options": options})

        async def gen():
            stop = "toolUse" if script.get("tool_calls") else "stop"
            yield AssistantMessageEvent(type="start", partial=AssistantMessage())
            content: list = []
            text = script.get("text")
            if text:
                content.append(TextContent(text=text))
                yield AssistantMessageEvent(
                    type="text_delta",
                    content_index=0,
                    delta=text,
                    partial=AssistantMessage(content=list(content), stop_reason=stop),
                )
            for tc in script.get("tool_calls") or []:
                content.append(tc)
            final = AssistantMessage(content=content, stop_reason=stop)
            yield AssistantMessageEvent(type="done", reason=stop, message=final)

        return AssistantMessageEventStream(gen())

    return stream_fn, calls


# ---------------------------------------------------------------------------
# Transcript printer: events -> readable lines
# ---------------------------------------------------------------------------


def make_transcript_printer():
    """Event handler printing the run as a readable transcript."""
    state = {"open": False}  # an assistant text line is mid-print

    def handle(event: AgentEvent) -> None:
        if event.type == "message_start" and isinstance(event.message, UserMessage):
            print(f"[user] {event.message.content}")
        elif (
            event.type == "message_update"
            and event.assistant_event is not None
            and event.assistant_event.type == "text_delta"
            and event.assistant_event.delta
        ):
            if not state["open"]:
                print("[assistant] ", end="")
                state["open"] = True
            print(event.assistant_event.delta, end="", flush=True)
        elif event.type == "message_end" and isinstance(event.message, AssistantMessage):
            if state["open"]:
                print()
                state["open"] = False
        elif event.type == "tool_execution_start":
            if state["open"]:
                print()
                state["open"] = False
            print(f"[tool call] {event.tool_name} {json.dumps(event.args)}")
        elif event.type == "tool_execution_end":
            text = ""
            if event.result is not None:
                text = "".join(
                    block.text
                    for block in event.result.content
                    if isinstance(block, TextContent)
                )
            print(f"[tool result] {' '.join(text.split())[:200]}")

    return handle


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="pion-demo-"))
    previous_cwd = Path.cwd()
    os.chdir(workdir)  # the real tools resolve relative paths against cwd

    scripts = [
        {
            "text": "I'll create hello.txt with the requested content.",
            "tool_calls": [
                tool_call("write", {"path": "hello.txt", "content": FILE_CONTENT})
            ],
        },
        {
            "text": "Now let me show it with bash.",
            "tool_calls": [tool_call("bash", {"command": "cat hello.txt"})],
        },
        {
            "text": "Done — created hello.txt containing 'hello pion' and verified it with cat."
        },
    ]
    stream_fn, calls = fake_stream_fn(scripts)

    try:
        agent = Agent(model=make_model(), stream_fn=stream_fn)  # real DEFAULT_TOOLS
        agent.subscribe(make_transcript_printer())

        print(f"# pion mock e2e demo (workdir: {workdir})")
        final = await agent.prompt(USER_PROMPT)

        # --- assertions ---
        hello = workdir / "hello.txt"
        assert hello.exists(), "hello.txt was not created"
        assert hello.read_text(encoding="utf-8") == FILE_CONTENT, (
            f"unexpected content: {hello.read_text(encoding='utf-8')!r}"
        )

        bash_results = [
            m
            for m in agent.messages
            if isinstance(m, ToolResultMessage) and m.tool_name == "bash"
        ]
        assert bash_results, "no bash tool result recorded"
        assert "hello pion" in bash_results[0].text(), (
            f"bash output missing content: {bash_results[0].text()!r}"
        )

        assert len(calls) == 3, f"expected 3 LLM calls, got {len(calls)}"
        assert "hello pion" in final.text()
    finally:
        os.chdir(previous_cwd)

    print("MOCK E2E OK")


if __name__ == "__main__":
    asyncio.run(main())

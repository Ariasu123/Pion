"""Tests for the pion CLI (pion/cli.py).

Covers the typer entry point (--help/--version/startup validation) and the
pure helpers factored out of the REPL. No network, no interactive REPL.
"""

from __future__ import annotations

from datetime import datetime
import re

import pytest
from typer.testing import CliRunner

from pion import __version__
from pion.config import PionConfig, ProfileConfig, load_config, save_config
import pion.cli as cli
from pion.cli import (
    app,
    api_key_env_name,
    build_system_prompt,
    default_session_path,
    find_first_kept_entry_id,
    parse_slash_command,
    summarize_tool_args,
    summarize_tool_result,
)
from pion.llm.registry import get_model
from pion.llm.types import AssistantMessage, Model, TextContent, UserMessage
from pion.session import SessionManager

runner = CliRunner()

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    """Return CLI output without terminal styling escape sequences."""
    return ANSI_ESCAPE_RE.sub("", text)


@pytest.fixture(autouse=True)
def isolated_user_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / ".pion" / "config.json"
    monkeypatch.setattr(cli, "default_config_path", lambda: path)


async def _noop_async_main(*args, **kwargs) -> None:
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_mentions_model() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--model" in strip_ansi(result.output)


def test_version_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_unknown_model_exits_nonzero_and_lists_available_ids() -> None:
    result = runner.invoke(app, ["--model", "no-such-model"])
    assert result.exit_code != 0
    assert "no-such-model" in result.output
    for available in ("deepseek-v4-flash", "claude-sonnet-4-5", "kimi-k2-0905-preview"):
        assert available in result.output


def test_missing_api_key_names_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = runner.invoke(app, [])
    assert result.exit_code == 1
    assert "no usable profile or API key" in result.output


@pytest.mark.parametrize(
    ("protocol_input", "expected_api"),
    [
        ("openai", "openai-completions"),
        ("anthropic", "anthropic-messages"),
    ],
)
def test_configure_wizard_saves_profile_and_hides_key(
    monkeypatch: pytest.MonkeyPatch,
    protocol_input: str,
    expected_api: str,
) -> None:
    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    monkeypatch.setattr(cli, "_async_main", _noop_async_main)

    result = runner.invoke(
        app,
        ["--configure"],
        input=(
            "work\n"
            f"{protocol_input}\n"
            "https://gateway.example/v1\n"
            "custom-model\n"
            "super-secret\n"
        ),
    )

    assert result.exit_code == 0, result.output
    assert "super-secret" not in result.output
    config = load_config(cli.default_config_path())
    assert config.active_profile == "work"
    assert config.profiles["work"].api == expected_api
    assert config.profiles["work"].api_key == "super-secret"


def test_multiple_profiles_prompt_for_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PionConfig(
        active_profile="first",
        profiles={
            "first": ProfileConfig(
                api="openai-completions",
                base_url="https://first.example/v1",
                api_key="first-key",
                model="first-model",
            ),
            "second": ProfileConfig(
                api="anthropic-messages",
                base_url="https://second.example",
                api_key="second-key",
                model="second-model",
            ),
        },
    )
    save_config(config, cli.default_config_path())
    captured = {}

    async def capture(model, api_key, *args, **kwargs) -> None:
        captured["model"] = model
        captured["key"] = api_key

    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    monkeypatch.setattr(cli, "_async_main", capture)
    result = runner.invoke(app, [], input="2\n")

    assert result.exit_code == 0, result.output
    assert "Available profiles" in result.output
    assert captured["model"].id == "second-model"
    assert captured["key"] == "second-key"
    assert load_config(cli.default_config_path()).active_profile == "second"


def test_profile_flag_skips_selection_and_cli_overrides_are_saved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PionConfig(
        active_profile="work",
        profiles={
            "work": ProfileConfig(
                api="openai-completions",
                base_url="https://old.example/v1",
                api_key="old-key",
                model="old-model",
            )
        },
    )
    save_config(config, cli.default_config_path())
    captured = {}

    async def capture(model, api_key, *args, **kwargs) -> None:
        captured["model"] = model
        captured["key"] = api_key

    monkeypatch.setattr(cli, "_async_main", capture)
    result = runner.invoke(
        app,
        [
            "--profile",
            "work",
            "--model",
            "new-model",
            "--base-url",
            "https://new.example/v1",
            "--api-key",
            "new-key",
        ],
    )

    assert result.exit_code == 0, result.output
    saved = load_config(cli.default_config_path()).profiles["work"]
    assert saved.model == "new-model"
    assert saved.base_url == "https://new.example/v1"
    assert saved.api_key == "new-key"
    assert captured["model"].id == "new-model"
    assert captured["key"] == "new-key"


def test_configure_existing_profile_keeps_key_when_left_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PionConfig(
        active_profile="work",
        profiles={
            "work": ProfileConfig(
                api="openai-completions",
                base_url="https://old.example/v1",
                api_key="keep-this-key",
                model="old-model",
            )
        },
    )
    save_config(config, cli.default_config_path())
    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    monkeypatch.setattr(cli, "_async_main", _noop_async_main)

    result = runner.invoke(
        app,
        ["--configure"],
        input=(
            "\n"  # keep profile name
            "\n"  # keep protocol
            "https://new.example/v1\n"
            "new-model\n"
            "\n"  # keep API key
        ),
    )

    assert result.exit_code == 0, result.output
    saved = load_config(cli.default_config_path()).profiles["work"]
    assert saved.api_key == "keep-this-key"
    assert saved.base_url == "https://new.example/v1"
    assert saved.model == "new-model"


def test_noninteractive_unknown_profile_fails() -> None:
    result = runner.invoke(app, ["--profile", "missing"])
    assert result.exit_code == 1
    assert "unknown profile" in result.output


def test_complete_custom_cli_settings_create_default_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    async def capture(model, api_key, *args, **kwargs) -> None:
        captured["model"] = model
        captured["key"] = api_key

    monkeypatch.setattr(cli, "_async_main", capture)
    result = runner.invoke(
        app,
        [
            "--model",
            "vendor-model",
            "--base-url",
            "https://vendor.example/v1",
            "--api-key",
            "vendor-key",
        ],
    )

    assert result.exit_code == 0, result.output
    config = load_config(cli.default_config_path())
    assert config.active_profile == "default"
    assert config.profiles["default"].model == "vendor-model"
    assert config.profiles["default"].api == "openai-completions"
    assert captured["model"].id == "vendor-model"
    assert captured["key"] == "vendor-key"


def test_cancelled_wizard_does_not_write_partial_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    result = runner.invoke(app, ["--configure"], input="work\n")

    assert result.exit_code == 130
    assert "Configuration cancelled" in result.output
    assert not cli.default_config_path().exists()


def test_malformed_config_reports_error_without_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = cli.default_config_path()
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(cli, "_async_main", pytest.fail)

    result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert "could not load" in result.output
    assert path.read_text(encoding="utf-8") == "{broken"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_default_session_path_format(tmp_path) -> None:
    path = default_session_path(now=datetime(2026, 7, 22, 5, 56, 23), base_dir=tmp_path)
    assert path.parent == tmp_path
    name = path.name
    assert name.startswith("20260722-055623-")
    assert name.endswith(".jsonl")
    uuid_part = name[len("20260722-055623-") : -len(".jsonl")]
    assert len(uuid_part) == 8
    int(uuid_part, 16)  # must be hex


def test_default_session_path_is_unique(tmp_path) -> None:
    first = default_session_path(base_dir=tmp_path)
    second = default_session_path(base_dir=tmp_path)
    assert first != second


def test_parse_slash_command() -> None:
    assert parse_slash_command("/help") == ("help", "")
    assert parse_slash_command("/model deepseek-v4-flash") == (
        "model",
        "deepseek-v4-flash",
    )
    assert parse_slash_command("/compact  ") == ("compact", "")
    assert parse_slash_command("/stats extra words") == ("stats", "extra words")
    assert parse_slash_command("hello world") is None
    assert parse_slash_command("/") is None
    assert parse_slash_command("") is None


def test_summarize_tool_args_picks_key_argument() -> None:
    assert summarize_tool_args("bash", {"command": "ls -la"}) == "ls -la"
    assert summarize_tool_args("write", {"path": "a.txt", "content": "x"}) == "a.txt"
    # Fallback: compact JSON of all args.
    assert summarize_tool_args("other", {"n": 1}) == '{"n": 1}'
    # Newlines are collapsed and long values truncated.
    assert summarize_tool_args("bash", {"command": "a\nb"}) == "a b"
    assert len(summarize_tool_args("bash", {"command": "x" * 500})) <= 120


def test_summarize_tool_result_truncates() -> None:
    assert summarize_tool_result("hello world") == "hello world"
    assert summarize_tool_result("line1\nline2") == "line1 line2"
    long = summarize_tool_result("x" * 1000, limit=200)
    assert len(long) == 200
    assert long.endswith("...")


def test_api_key_env_name() -> None:
    assert api_key_env_name(get_model("deepseek-v4-flash")) == "DEEPSEEK_API_KEY"


def test_build_system_prompt_substitutes_cwd(tmp_path) -> None:
    prompt = build_system_prompt(tmp_path)
    assert str(tmp_path) in prompt
    for tool in ("read", "write", "edit", "bash"):
        assert tool in prompt
    assert len(prompt.split()) < 500


# ---------------------------------------------------------------------------
# Compaction kept-tail selection
# ---------------------------------------------------------------------------


def _tiny_model() -> Model:
    """Model with a tiny context window so the 50% budget is easy to exceed."""
    return Model(
        id="tiny",
        name="Tiny",
        api="openai-completions",
        provider="test",
        base_url="http://localhost:9",
        context_window=64,  # kept-tail budget: 32 tokens ~ 128 chars
    )


def _user(text: str) -> UserMessage:
    return UserMessage(content=text)


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)])


def test_find_first_kept_entry_id_keeps_recent_tail_at_user_boundary() -> None:
    session = SessionManager()  # in-memory
    chunk = "x" * 40  # 10 tokens per message by the chars//4 heuristic
    ids = [
        session.append_message(m)
        for m in (
            _user(chunk),  # U1
            _assistant(chunk),  # A1
            _user(chunk),  # U2
            _assistant(chunk),  # A2
            _user(chunk),  # U3
            _assistant(chunk),  # A3
        )
    ]

    kept = find_first_kept_entry_id(session, _tiny_model())

    # Budget 32 tokens: A3(10) + U3(20) + A2(30) fit, U2 would exceed.
    # Cutting at a user boundary drops A2, so the tail starts at U3.
    assert kept == ids[4]


def test_find_first_kept_entry_id_keeps_everything_when_it_fits() -> None:
    session = SessionManager()
    first = session.append_message(_user("hi"))
    session.append_message(_assistant("hello"))
    assert find_first_kept_entry_id(session, get_model("deepseek-v4-flash")) == first


def test_find_first_kept_entry_id_ignores_messages_before_last_compaction() -> None:
    session = SessionManager()
    session.append_message(_user("old stuff"))
    session.append_compaction("summary", first_kept_entry_id=None)
    recent = session.append_message(_user("recent"))
    session.append_message(_assistant("answer"))
    assert find_first_kept_entry_id(session, get_model("deepseek-v4-flash")) == recent


def test_find_first_kept_entry_id_empty_session() -> None:
    assert find_first_kept_entry_id(SessionManager(), _tiny_model()) is None

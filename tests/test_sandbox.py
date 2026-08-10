"""Tests for Pion-owned sandbox policy, host runtime and tool adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pion.config import PionConfig
from pion.sandbox import (
    SandboxCommandResult,
    SandboxRuntime,
    SandboxSettings,
    WorkspaceAccessError,
    WorkspaceGuard,
    build_runtime,
)
from pion.tools import build_default_tools
from pion.tools.bash import BashArgs
from pion.tools.edit import EditArgs
from pion.tools.read import ReadArgs
from pion.tools.write import WriteArgs


def text_of(result) -> str:
    return "".join(c.text for c in result.content if c.type == "text")


class FakeRuntime(SandboxRuntime):
    backend = "docker"

    def __init__(
        self,
        workspace: Path,
        *,
        command_result: SandboxCommandResult | None = None,
    ) -> None:
        settings = SandboxSettings()
        guard = WorkspaceGuard(workspace, [*settings.protect_paths, ".git"])
        super().__init__(workspace, settings, guard)
        self.command_result = command_result or SandboxCommandResult("ok\n", 0)
        self.updates: list[str] = []

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def execute(
        self,
        command: str,
        *,
        timeout_s: int,
        abort: asyncio.Event | None,
        on_update,
        max_output_bytes: int,
    ) -> SandboxCommandResult:
        if on_update is not None:
            on_update("partial\n")
        return self.command_result

    def describe(self) -> dict[str, object]:
        return {"backend": "docker", "containerId": "123456789abc"}


def test_sandbox_settings_secure_defaults() -> None:
    settings = PionConfig().sandbox
    assert settings == SandboxSettings()
    assert settings.backend == "off"
    assert settings.network == "bridge"
    assert settings.memory_mb == 4096
    assert settings.cpus == 2.0
    assert settings.pids_limit == 256
    assert settings.git_write is False
    assert settings.protect_paths == [".env", ".env.*"]


def test_sandbox_settings_migrates_legacy_docker_backend() -> None:
    settings = SandboxSettings.model_validate({"backend": "docker"})
    assert settings.backend == "mcp"


def test_build_runtime_returns_host(tmp_path: Path) -> None:
    host = build_runtime(SandboxSettings(), tmp_path)
    assert host.backend == "host"
    assert host.guard is None


def test_guard_allows_relative_absolute_and_future_workspace_paths(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path, [".env", ".env.*"])
    existing = tmp_path / "src" / "main.py"
    existing.parent.mkdir()
    existing.write_text("pass\n", encoding="utf-8")
    assert guard.resolve("src/main.py") == existing
    assert guard.resolve(str(existing)) == existing
    assert guard.resolve("new/deep/file.txt") == tmp_path / "new" / "deep" / "file.txt"


def test_guard_rejects_outside_and_symlink_escapes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    guard = WorkspaceGuard(workspace)
    with pytest.raises(WorkspaceAccessError, match="outside workspace"):
        guard.resolve("../outside.txt")
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceAccessError, match="outside workspace"):
        guard.resolve("link/secret.txt")


@pytest.mark.parametrize(
    "requested", [".env", ".env.local", "nested/.env", "nested/.env.production"]
)
def test_guard_rejects_protected_secret_names(tmp_path: Path, requested: str) -> None:
    guard = WorkspaceGuard(tmp_path, [".env", ".env.*"])
    with pytest.raises(WorkspaceAccessError, match="protected"):
        guard.resolve(requested)


def test_guard_write_protects_git_metadata(tmp_path: Path) -> None:
    guarded = WorkspaceGuard(tmp_path, write_protect_paths=[".git"])
    assert guarded.resolve(".git/config", "read") == tmp_path / ".git" / "config"
    with pytest.raises(WorkspaceAccessError, match="protected"):
        guarded.resolve(".git/config", "write")
    writable = WorkspaceGuard(tmp_path)
    assert writable.resolve(".git/config", "write") == tmp_path / ".git" / "config"


async def test_runtime_bound_file_tools_enforce_guard_and_report_backend(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(tmp_path)
    read, write, edit, _ = build_default_tools(runtime)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("host secret", encoding="utf-8")
    denied = [
        await read.execute("r", ReadArgs(path=str(outside))),
        await write.execute("w", WriteArgs(path="../outside.txt", content="overwrite")),
        await edit.execute(
            "e", EditArgs(path=".env", old_string="x", new_string="y")
        ),
    ]
    for result in denied:
        assert "denied" in text_of(result)
        assert result.details["backend"] == "docker"
        assert result.details["denied"] is True
    assert outside.read_text(encoding="utf-8") == "host secret"
    allowed = await write.execute("ok", WriteArgs(path="inside.txt", content="ok"))
    assert (tmp_path / "inside.txt").read_text(encoding="utf-8") == "ok"
    assert allowed.details["bytes"] == 2


async def test_file_open_is_safe_when_parent_becomes_symlink_after_resolution(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    parent = workspace / "parent"
    parent.mkdir()
    outside_target = outside / "target.txt"
    outside_target.write_text("must stay unchanged", encoding="utf-8")

    class SwappingGuard(WorkspaceGuard):
        calls = 0

        def resolve(self, requested: str, operation: str = "access") -> Path:
            resolved = super().resolve(requested, operation)
            self.calls += 1
            if self.calls == 2:
                parent.rename(workspace / "original-parent")
                parent.symlink_to(outside, target_is_directory=True)
            return resolved

    runtime = FakeRuntime(workspace)
    runtime.guard = SwappingGuard(workspace)
    write = build_default_tools(runtime)[1]
    result = await write.execute(
        "race", WriteArgs(path="parent/target.txt", content="escaped")
    )
    assert "could not write" in text_of(result)
    assert outside_target.read_text(encoding="utf-8") == "must stay unchanged"


async def test_runtime_bound_bash_reports_structured_result_and_updates(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(
        tmp_path,
        command_result=SandboxCommandResult(output="oops\n", exit_code=7),
    )
    bash = build_default_tools(runtime)[-1]
    updates = []
    result = await bash.execute(
        "b", BashArgs(command="false"), on_update=updates.append
    )
    assert "[exit code 7]" in text_of(result)
    assert result.details["exitCode"] == 7
    assert updates and "partial" in text_of(updates[0])


@pytest.mark.parametrize(
    ("command_result", "message"),
    [
        (SandboxCommandResult("early\n", None, timed_out=True), "timed out"),
        (SandboxCommandResult("early\n", None, aborted=True), "aborted"),
    ],
)
async def test_runtime_bound_bash_timeout_and_abort_details(
    tmp_path: Path,
    command_result: SandboxCommandResult,
    message: str,
) -> None:
    bash = build_default_tools(FakeRuntime(tmp_path, command_result=command_result))[-1]
    result = await bash.execute("b", BashArgs(command="sleep 30", timeout_s=1))
    assert message in text_of(result).lower()
    assert result.details["exitCode"] is None

"""Policy and command-assembly tests for the Pion sandbox."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from pion.config import PionConfig
from pion.sandbox import (
    DockerSandboxRuntime,
    SandboxCommandResult,
    SandboxError,
    SandboxRuntime,
    SandboxSettings,
    SandboxUnavailableError,
    WorkspaceAccessError,
    WorkspaceGuard,
    build_runtime,
)
from pion.sandbox.docker import DEFAULT_DOCKERFILE
from pion.tools import build_default_tools
from pion.tools.bash import BashArgs
from pion.tools.edit import EditArgs
from pion.tools.read import ReadArgs
from pion.tools.write import WriteArgs


def text_of(result) -> str:
    return "".join(c.text for c in result.content if c.type == "text")


class FakeRuntime(SandboxRuntime):
    """Runtime stub for testing tool wiring without Docker."""

    backend = "docker"

    def __init__(
        self,
        workspace: Path,
        *,
        command_result: SandboxCommandResult | None = None,
    ) -> None:
        settings = SandboxSettings()
        guard = WorkspaceGuard(
            workspace,
            [*settings.protect_paths, ".git"],
        )
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


# ---------------------------------------------------------------------------
# Settings and backend selection
# ---------------------------------------------------------------------------


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
    # Sandboxed execution is mounted via `pion mcp`; build_runtime only
    # serves the default unsandboxed mode.
    host = build_runtime(SandboxSettings(), tmp_path)
    assert host.backend == "host"
    assert host.guard is None


async def test_docker_cli_missing_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = DockerSandboxRuntime(tmp_path, SandboxSettings())
    monkeypatch.setattr("pion.sandbox.docker.shutil.which", lambda _: None)

    with pytest.raises(SandboxUnavailableError, match="CLI was not found"):
        await runtime.start()


async def test_docker_daemon_unavailable_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = DockerSandboxRuntime(tmp_path, SandboxSettings())
    monkeypatch.setattr("pion.sandbox.docker.shutil.which", lambda _: "/bin/docker")

    async def fail_info(*args, **kwargs):
        raise SandboxError("connection refused")

    monkeypatch.setattr(runtime, "_run_docker", fail_info)
    with pytest.raises(SandboxUnavailableError, match="daemon is unavailable"):
        await runtime.start()


# ---------------------------------------------------------------------------
# WorkspaceGuard
# ---------------------------------------------------------------------------


def test_guard_allows_relative_absolute_and_future_workspace_paths(
    tmp_path: Path,
) -> None:
    guard = WorkspaceGuard(tmp_path, [".env", ".env.*"])
    existing = tmp_path / "src" / "main.py"
    existing.parent.mkdir()
    existing.write_text("pass\n")

    assert guard.resolve("src/main.py") == existing
    assert guard.resolve(str(existing)) == existing
    assert guard.resolve("new/deep/file.txt") == tmp_path / "new" / "deep" / "file.txt"
    assert guard.resolve("src/../README.md") == tmp_path / "README.md"


def test_guard_rejects_outside_absolute_and_parent_traversal(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    guard = WorkspaceGuard(workspace)

    with pytest.raises(WorkspaceAccessError, match="outside workspace"):
        guard.resolve(str(outside), "read")
    with pytest.raises(WorkspaceAccessError, match="outside workspace"):
        guard.resolve("../outside.txt", "write")


def test_guard_rejects_existing_and_future_symlink_escapes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    guard = WorkspaceGuard(workspace)

    with pytest.raises(WorkspaceAccessError, match="outside workspace"):
        guard.resolve("link/secret.txt")
    with pytest.raises(WorkspaceAccessError, match="outside workspace"):
        guard.resolve("link/not-created-yet.txt")


@pytest.mark.parametrize(
    "requested",
    [".env", ".env.local", "nested/.env", "nested/.env.production"],
)
def test_guard_rejects_protected_secret_names(tmp_path: Path, requested: str) -> None:
    guard = WorkspaceGuard(tmp_path, [".env", ".env.*"])
    with pytest.raises(WorkspaceAccessError, match="protected"):
        guard.resolve(requested)


def test_guard_rejects_git_metadata_when_git_write_is_disabled(
    tmp_path: Path,
) -> None:
    runtime = DockerSandboxRuntime(tmp_path, SandboxSettings(git_write=False))
    assert runtime.guard is not None
    assert runtime.guard.resolve(".git/config", "read") == tmp_path / ".git" / "config"
    with pytest.raises(WorkspaceAccessError, match="protected"):
        runtime.guard.resolve(".git/config", "write")

    writable = DockerSandboxRuntime(tmp_path, SandboxSettings(git_write=True))
    assert writable.guard is not None
    assert writable.guard.resolve(".git/config") == tmp_path / ".git" / "config"


# ---------------------------------------------------------------------------
# Docker command assembly
# ---------------------------------------------------------------------------


def test_default_dockerfile_has_expected_toolchain_and_no_daemon() -> None:
    for token in (
        "python3.12",
        "uv",
        "bash",
        "build-essential",
        "ca-certificates",
        "curl",
        "git",
        "ripgrep",
    ):
        assert token in DEFAULT_DOCKERFILE
    assert "docker" not in DEFAULT_DOCKERFILE.lower()


def test_docker_run_argv_has_limits_and_no_escape_hatches(tmp_path: Path) -> None:
    runtime = DockerSandboxRuntime(
        tmp_path,
        SandboxSettings(network="none", memory_mb=768, cpus=1.5, pids_limit=64),
    )
    args = runtime.build_run_args()
    joined = " ".join(args)

    assert args[:2] == ["run", "--detach"]
    assert ["--user", f"{os.getuid()}:{os.getgid()}"] == args[
        args.index("--user") : args.index("--user") + 2
    ]
    assert ["--network", "none"] == args[
        args.index("--network") : args.index("--network") + 2
    ]
    assert ["--memory", "768m"] == args[
        args.index("--memory") : args.index("--memory") + 2
    ]
    assert ["--cpus", "1.5"] == args[args.index("--cpus") : args.index("--cpus") + 2]
    assert ["--pids-limit", "64"] == args[
        args.index("--pids-limit") : args.index("--pids-limit") + 2
    ]
    assert ["--cap-drop", "ALL"] == args[
        args.index("--cap-drop") : args.index("--cap-drop") + 2
    ]
    assert "no-new-privileges:true" in args
    assert "--init" in args
    assert "--privileged" not in args
    assert "--device" not in args
    assert "--pid" not in args
    assert "--publish" not in args
    assert "-p" not in args
    assert "docker.sock" not in joined
    assert "host" not in args[args.index("--network") + 1]
    assert "--entrypoint" in args

    workspace_mount = f"type=bind,src={tmp_path.resolve()},dst={tmp_path.resolve()}"
    assert workspace_mount in args


def test_regular_repository_git_metadata_is_mounted_read_only(
    tmp_path: Path,
) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    runtime = DockerSandboxRuntime(tmp_path, SandboxSettings())
    args = runtime.build_run_args()
    expected = f"type=bind,src={git_dir},dst={git_dir},readonly"
    assert expected in args
    # Secret masking and Git read-only mounting must not both target .git.
    assert f"{git_dir}:ro" not in args


def test_git_write_opt_in_uses_writable_git_mount(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    runtime = DockerSandboxRuntime(
        tmp_path,
        SandboxSettings(git_write=True),
    )
    args = runtime.build_run_args()
    writable = f"type=bind,src={git_dir},dst={git_dir}"
    assert writable in args
    assert f"{writable},readonly" not in args


def test_worktree_git_file_common_and_private_dirs_are_read_only(
    tmp_path: Path,
) -> None:
    common = tmp_path / "main-repo" / ".git"
    gitdir = common / "worktrees" / "feature"
    workspace = tmp_path / "feature"
    gitdir.mkdir(parents=True)
    workspace.mkdir()
    (workspace / ".git").write_text(f"gitdir: {gitdir}\n")
    (gitdir / "commondir").write_text("../..\n")
    (gitdir / "gitdir").write_text(f"{workspace / '.git'}\n")

    runtime = DockerSandboxRuntime(workspace, SandboxSettings())
    mounts = runtime._git_mounts()
    assert workspace / ".git" in mounts
    assert common in mounts
    # The common directory mount already contains the worktree-private gitdir.
    assert gitdir.is_relative_to(common)

    args = runtime.build_run_args()
    for path in mounts:
        assert f"type=bind,src={path},dst={path},readonly" in args


def test_crafted_git_file_cannot_request_arbitrary_host_mount(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    arbitrary = tmp_path / "private-host-directory"
    workspace.mkdir()
    arbitrary.mkdir()
    (workspace / ".git").write_text(f"gitdir: {arbitrary}\n")
    (arbitrary / "commondir").write_text(".\n")
    (arbitrary / "gitdir").write_text("/some/other/worktree/.git\n")

    runtime = DockerSandboxRuntime(workspace, SandboxSettings())
    assert runtime._git_mounts() == [workspace / ".git"]
    assert str(arbitrary) not in " ".join(runtime.build_run_args())


def test_existing_secret_files_are_masked_in_container_argv(
    tmp_path: Path,
) -> None:
    env = tmp_path / ".env"
    nested = tmp_path / "nested"
    nested.mkdir()
    local_env = nested / ".env.local"
    env.write_text("API_KEY=secret\n")
    local_env.write_text("TOKEN=secret\n")

    runtime = DockerSandboxRuntime(tmp_path, SandboxSettings())
    args = runtime.build_run_args()
    assert f"type=bind,src=/dev/null,dst={env},readonly" in args
    assert f"type=bind,src=/dev/null,dst={local_env},readonly" in args


async def test_orphan_cleanup_removes_only_dead_pion_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = DockerSandboxRuntime(tmp_path, SandboxSettings())
    calls = []

    async def fake_docker(*args, check=True):
        calls.append((args, check))
        if args[:2] == ("ps", "-a"):
            return "dead-container\t100\nlive-container\t200\n"
        return "__PION_EXIT_0__\n"

    monkeypatch.setattr(runtime, "_run_docker", fake_docker)
    monkeypatch.setattr(runtime, "_pid_is_alive", lambda pid: pid == 200)
    await runtime._cleanup_orphans()

    assert (("rm", "-f", "dead-container"), False) in calls
    assert (("rm", "-f", "live-container"), False) not in calls


# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------


async def test_runtime_bound_file_tools_enforce_guard_and_report_backend(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(tmp_path)
    read, write, edit, _ = build_default_tools(runtime)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("host secret")

    denied_read = await read.execute("r", ReadArgs(path=str(outside)))
    denied_write = await write.execute(
        "w", WriteArgs(path="../outside.txt", content="overwrite")
    )
    denied_edit = await edit.execute(
        "e",
        EditArgs(path=".env", old_string="x", new_string="y"),
    )

    for result in (denied_read, denied_write, denied_edit):
        assert "denied" in text_of(result)
        assert result.details["backend"] == "docker"
        assert result.details["containerId"] == "123456789abc"
        assert result.details["denied"] is True
    assert outside.read_text() == "host secret"

    allowed = await write.execute("ok", WriteArgs(path="inside.txt", content="ok"))
    assert (tmp_path / "inside.txt").read_text() == "ok"
    assert allowed.details == {
        "backend": "docker",
        "containerId": "123456789abc",
        "bytes": 2,
    }


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
    outside_target.write_text("must stay unchanged")

    class SwappingGuard(WorkspaceGuard):
        calls = 0

        def resolve(self, requested: str, operation: str = "access") -> Path:
            resolved = super().resolve(requested, operation)
            self.calls += 1
            # WriteTool resolves once for policy details; open_file resolves
            # again immediately before its descriptor-based walk.
            if self.calls == 2:
                parent.rename(workspace / "original-parent")
                parent.symlink_to(outside, target_is_directory=True)
            return resolved

    runtime = FakeRuntime(workspace)
    runtime.guard = SwappingGuard(workspace)
    write = build_default_tools(runtime)[1]
    result = await write.execute(
        "race",
        WriteArgs(path="parent/target.txt", content="escaped"),
    )

    assert "could not write" in text_of(result)
    assert outside_target.read_text() == "must stay unchanged"


async def test_runtime_bound_bash_reports_structured_result_and_updates(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime(
        tmp_path,
        command_result=SandboxCommandResult(
            output="oops\n",
            exit_code=7,
            truncated=False,
        ),
    )
    bash = build_default_tools(runtime)[-1]
    updates = []

    result = await bash.execute(
        "b",
        BashArgs(command="false"),
        on_update=updates.append,
    )

    assert "[exit code 7]" in text_of(result)
    assert result.details == {
        "backend": "docker",
        "containerId": "123456789abc",
        "exitCode": 7,
        "truncated": False,
        "timedOut": False,
        "aborted": False,
    }
    assert updates
    assert "partial" in text_of(updates[0])


@pytest.mark.parametrize(
    ("command_result", "message"),
    [
        (
            SandboxCommandResult("early\n", None, timed_out=True),
            "timed out",
        ),
        (
            SandboxCommandResult("early\n", None, aborted=True),
            "aborted",
        ),
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
    assert result.details["timedOut"] is command_result.timed_out
    assert result.details["aborted"] is command_result.aborted

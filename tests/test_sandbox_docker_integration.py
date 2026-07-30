"""Docker-backed integration checks.

These tests skip automatically when no Docker daemon is reachable.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pion.sandbox import DockerSandboxRuntime, SandboxSettings, WorkspaceAccessError


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon is not available",
)


async def test_workspace_visibility_secrets_and_interrupt_recycling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("outside-host-secret")
    (workspace / ".env").write_text("PION_SECRET=workspace-secret\n")
    monkeypatch.setenv("PION_HOST_API_KEY", "host-process-secret")

    runtime = DockerSandboxRuntime(
        workspace,
        SandboxSettings(network="none", memory_mb=512, cpus=1, pids_limit=64),
    )
    await runtime.start()
    try:
        original_id = runtime.container_id
        updates = []
        write = await runtime.execute(
            "printf sandbox-write | tee generated.txt",
            timeout_s=10,
            abort=None,
            on_update=updates.append,
            max_output_bytes=1024,
        )
        assert write.exit_code == 0
        assert (workspace / "generated.txt").read_text() == "sandbox-write"
        assert any("sandbox-write" in update for update in updates)

        isolation = await runtime.execute(
            f"cat {outside}; cat .env; env; test ! -S /var/run/docker.sock",
            timeout_s=10,
            abort=None,
            on_update=None,
            max_output_bytes=20_000,
        )
        assert "outside-host-secret" not in isolation.output
        assert "workspace-secret" not in isolation.output
        assert "host-process-secret" not in isolation.output

        interrupted = await runtime.execute(
            "echo early; sleep 30",
            timeout_s=1,
            abort=None,
            on_update=None,
            max_output_bytes=1024,
        )
        assert interrupted.timed_out
        assert "early" in interrupted.output
        assert runtime.container_id == original_id

        abort = asyncio.Event()
        abort_task = asyncio.create_task(_set_event_soon(abort))
        aborted = await runtime.execute(
            "echo before-abort; sleep 30",
            timeout_s=10,
            abort=abort,
            on_update=None,
            max_output_bytes=1024,
        )
        await abort_task
        assert aborted.aborted
        assert "before-abort" in aborted.output
        assert runtime.container_id == original_id

        after = await runtime.execute(
            "echo recycled",
            timeout_s=10,
            abort=None,
            on_update=None,
            max_output_bytes=1024,
        )
        assert after.exit_code == 0
        assert "recycled" in after.output
    finally:
        container_id = runtime.container_id
        await runtime.close()

    inspect = subprocess.run(
        ["docker", "inspect", container_id or ""],
        capture_output=True,
        check=False,
    )
    assert inspect.returncode != 0


async def _set_event_soon(event: asyncio.Event) -> None:
    await asyncio.sleep(0.2)
    event.set()


async def test_git_read_only_and_container_security_limits(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    runtime = DockerSandboxRuntime(
        tmp_path,
        SandboxSettings(network="none", memory_mb=512, cpus=1, pids_limit=64),
    )
    await runtime.start()
    try:
        status = await runtime.execute(
            "git status --short",
            timeout_s=10,
            abort=None,
            on_update=None,
            max_output_bytes=1024,
        )
        assert status.exit_code == 0

        write_git = await runtime.execute(
            "touch .git/pion-must-not-write",
            timeout_s=10,
            abort=None,
            on_update=None,
            max_output_bytes=1024,
        )
        assert write_git.exit_code != 0
        assert not (tmp_path / ".git" / "pion-must-not-write").exists()

        assert runtime.guard is not None
        with pytest.raises(WorkspaceAccessError):
            runtime.guard.resolve(".git/config", "write")

        raw = subprocess.run(
            ["docker", "inspect", runtime.container_id or ""],
            capture_output=True,
            text=True,
            check=True,
        )
        inspect = json.loads(raw.stdout)[0]
        host = inspect["HostConfig"]
        assert host["NetworkMode"] == "none"
        assert host["Memory"] == 512 * 1024 * 1024
        assert host["NanoCpus"] == 1_000_000_000
        assert host["PidsLimit"] == 64
        assert host["CapDrop"] == ["ALL"]
        assert "no-new-privileges:true" in host["SecurityOpt"]
        assert "seccomp=unconfined" not in host["SecurityOpt"]
        assert host["Privileged"] is False
        assert host["Devices"] == []
        assert host["PidMode"] == ""
        assert not host["PortBindings"]
        assert inspect["Config"]["User"] not in ("", "0", "root")
        assert all("docker.sock" not in mount for mount in host["Binds"] or [])
    finally:
        await runtime.close()

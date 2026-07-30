"""Fail-closed Docker CLI sandbox backend."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

from .. import __version__
from .base import (
    OutputCallback,
    SandboxCommandResult,
    SandboxError,
    SandboxRuntime,
    SandboxSettings,
    SandboxUnavailableError,
)
from .workspace import WorkspaceGuard

_LABEL_PREFIX = "io.pion"
_MAX_DOCKER_OUTPUT = 100 * 1024

DEFAULT_DOCKERFILE = """\
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
       bash build-essential ca-certificates curl git ripgrep \\
    && rm -rf /var/lib/apt/lists/*

CMD ["sleep", "infinity"]
"""


class DockerSandboxRuntime(SandboxRuntime):
    """One disposable, non-root Docker container per Pion CLI process."""

    backend = "docker"

    def __init__(self, workspace: Path, settings: SandboxSettings) -> None:
        guard = WorkspaceGuard(
            workspace,
            settings.protect_paths,
            write_protect_paths=[] if settings.git_write else [".git"],
        )
        super().__init__(workspace, settings, guard)
        # Git metadata remains readable in the container. Host-side file-tool
        # writes are protected by ``guard`` and Git paths are separately
        # remounted read-only by ``_git_mounts``. Only configured secret paths
        # are masked.
        self._mask_guard = WorkspaceGuard(workspace, settings.protect_paths)
        self.container_id: str | None = None
        self.container_name = f"pion-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.workspace_hash = hashlib.sha256(
            str(self.workspace).encode("utf-8")
        ).hexdigest()[:16]
        self.image = settings.image or f"pion-sandbox:{__version__}"
        self._uses_default_image = settings.image is None
        self._started = False
        self._start_lock = asyncio.Lock()

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "containerId": self.container_id[:12] if self.container_id else None,
        }

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            if shutil.which("docker") is None:
                raise SandboxUnavailableError(
                    "Docker sandbox is enabled, but the 'docker' CLI was not found. "
                    "Install Docker or run with --sandbox off."
                )
            if "," in str(self.workspace):
                raise SandboxUnavailableError(
                    "Docker sandbox does not support a workspace path containing ',': "
                    f"{self.workspace}"
                )

            await self._check_daemon()
            await self._cleanup_orphans()
            await self._ensure_image()
            args = self.build_run_args()
            try:
                output = await self._run_docker(*args)
            except SandboxError:
                await self._remove_failed_start()
                raise
            container_id = next(
                (
                    line.strip()
                    for line in reversed(output.splitlines())
                    if re.fullmatch(r"[0-9a-f]{12,64}", line.strip())
                ),
                "",
            )
            if not container_id:
                await self._remove_failed_start()
                raise SandboxUnavailableError(
                    "Docker created no sandbox container identifier"
                )
            self.container_id = container_id
            self._started = True

    async def close(self) -> None:
        if self.container_id is None:
            self._started = False
            return
        container_id = self.container_id
        self.container_id = None
        self._started = False
        try:
            await self._run_docker("rm", "-f", container_id, check=False)
        except SandboxError:
            # Cleanup is best effort; the next startup reclaims labeled orphans.
            pass

    async def _remove_failed_start(self) -> None:
        try:
            await self._run_docker(
                "rm",
                "-f",
                self.container_name,
                check=False,
            )
        except SandboxError:
            # A labeled leftover is reclaimed by the next successful startup.
            pass

    async def execute(
        self,
        command: str,
        *,
        timeout_s: int,
        abort: Optional[asyncio.Event],
        on_update: Optional[OutputCallback],
        max_output_bytes: int,
    ) -> SandboxCommandResult:
        await self.start()
        assert self.container_id is not None
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                "--workdir",
                str(self.workspace),
                self.container_id,
                "bash",
                "-lc",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise SandboxError(f"could not start docker exec: {exc}") from exc

        buffer = bytearray()
        truncated = False

        async def pump() -> None:
            nonlocal truncated
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                buffer.extend(chunk)
                if len(buffer) > max_output_bytes:
                    del buffer[: len(buffer) - max_output_bytes]
                    truncated = True
                if on_update is not None:
                    on_update(buffer.decode("utf-8", errors="replace"))

        pump_task = asyncio.create_task(pump())
        wait_task = asyncio.create_task(proc.wait())
        abort_task = asyncio.create_task(abort.wait()) if abort is not None else None
        wait_set: set[asyncio.Task] = {wait_task}
        if abort_task is not None:
            wait_set.add(abort_task)

        try:
            done, pending = await asyncio.wait(
                wait_set, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED
            )
            timed_out = not done
            aborted = (
                abort_task is not None
                and abort_task in done
                and wait_task not in done
                and not timed_out
            )
            if timed_out or aborted:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await self._recycle_after_interrupt()

            for task in pending:
                if task is not wait_task:
                    task.cancel()
            await asyncio.gather(wait_task, pump_task, return_exceptions=True)
        except asyncio.CancelledError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await self._recycle_after_interrupt()
            await asyncio.gather(wait_task, pump_task, return_exceptions=True)
            raise
        finally:
            if abort_task is not None:
                abort_task.cancel()

        return SandboxCommandResult(
            output=buffer.decode("utf-8", errors="replace"),
            exit_code=None if timed_out or aborted else proc.returncode,
            truncated=truncated,
            timed_out=timed_out,
            aborted=aborted,
        )

    def build_run_args(self) -> list[str]:
        """Build the security-sensitive ``docker run`` argv."""
        uid = os.getuid() if hasattr(os, "getuid") else 1000
        gid = os.getgid() if hasattr(os, "getgid") else 1000
        args = [
            "run",
            "--detach",
            "--name",
            self.container_name,
            "--label",
            f"{_LABEL_PREFIX}.managed=true",
            "--label",
            f"{_LABEL_PREFIX}.workspace={self.workspace_hash}",
            "--label",
            f"{_LABEL_PREFIX}.pid={os.getpid()}",
            "--workdir",
            str(self.workspace),
            "--user",
            f"{uid}:{gid}",
            "--env",
            "HOME=/tmp",
            "--network",
            self.settings.network,
            "--memory",
            f"{self.settings.memory_mb}m",
            "--cpus",
            str(self.settings.cpus),
            "--pids-limit",
            str(self.settings.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--init",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,mode=1777",
            "--mount",
            self._bind_spec(self.workspace, self.workspace, readonly=False),
        ]

        for source in self._git_mounts():
            args.extend(
                [
                    "--mount",
                    self._bind_spec(
                        source,
                        source,
                        readonly=not self.settings.git_write,
                    ),
                ]
            )

        for protected in self._mask_guard.protected_existing_paths():
            if protected.is_dir():
                args.extend(["--tmpfs", f"{protected}:ro"])
            else:
                args.extend(
                    [
                        "--mount",
                        self._bind_spec(Path("/dev/null"), protected, readonly=True),
                    ]
                )

        args.extend(["--entrypoint", "sleep", self.image, "infinity"])
        return args

    @staticmethod
    def _bind_spec(source: Path, destination: Path, readonly: bool) -> str:
        spec = f"type=bind,src={source},dst={destination}"
        return f"{spec},readonly" if readonly else spec

    def _git_mounts(self) -> list[Path]:
        marker = self.workspace / ".git"
        if not marker.exists():
            return []
        if marker.is_symlink():
            # Never let an untrusted repository turn a .git symlink into an
            # arbitrary read-only host mount.
            return []
        if marker.is_dir():
            return [marker.absolute()]

        mounts: list[Path] = [marker.absolute()]
        try:
            first_line = marker.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, UnicodeError, IndexError):
            return mounts
        if not first_line.startswith("gitdir:"):
            return mounts
        raw_gitdir = first_line.removeprefix("gitdir:").strip()
        gitdir = Path(raw_gitdir)
        if not gitdir.is_absolute():
            gitdir = (marker.parent / gitdir).resolve()
        else:
            gitdir = gitdir.resolve()

        commondir_file = gitdir / "commondir"
        backlink_file = gitdir / "gitdir"
        if not commondir_file.is_file() or not backlink_file.is_file():
            return mounts
        try:
            raw_common = commondir_file.read_text(encoding="utf-8").strip()
            candidate = Path(raw_common)
            common_dir = (
                candidate.resolve()
                if candidate.is_absolute()
                else (gitdir / candidate).resolve()
            )
            raw_backlink = backlink_file.read_text(encoding="utf-8").strip()
            backlink = Path(raw_backlink)
            backlink = (
                backlink.resolve()
                if backlink.is_absolute()
                else (gitdir / backlink).resolve()
            )
        except (OSError, UnicodeError):
            return mounts

        # A registered Git worktree has a private gitdir beneath its common
        # directory and a backlink to this exact workspace's .git file. These
        # checks prevent a crafted .git file from mounting an arbitrary host
        # directory into the sandbox.
        if backlink != marker.resolve() or not gitdir.is_relative_to(common_dir):
            return mounts

        for path in (common_dir, gitdir):
            if path.exists() and not any(
                path == existing or path.is_relative_to(existing) for existing in mounts
            ):
                mounts.append(path)
        return mounts

    async def _check_daemon(self) -> None:
        try:
            await self._run_docker("info", "--format", "{{.ServerVersion}}")
        except SandboxError as exc:
            raise SandboxUnavailableError(
                "Docker sandbox is enabled, but the Docker daemon is unavailable. "
                "Start Docker Desktop/OrbStack or run with --sandbox off. "
                f"Details: {exc}"
            ) from exc

    async def _ensure_image(self) -> None:
        inspect = await self._run_docker("image", "inspect", self.image, check=False)
        if not inspect.startswith("__PION_EXIT_0__"):
            if not self._uses_default_image:
                raise SandboxUnavailableError(
                    f"configured sandbox image is not available locally: {self.image}"
                )
            await self._build_default_image()

    async def _build_default_image(self) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "build",
                "--tag",
                self.image,
                "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise SandboxUnavailableError(
                f"could not start Docker image build: {exc}"
            ) from exc
        output, _ = await proc.communicate(DEFAULT_DOCKERFILE.encode("utf-8"))
        if proc.returncode != 0:
            tail = output[-_MAX_DOCKER_OUTPUT:].decode("utf-8", errors="replace")
            raise SandboxUnavailableError(
                f"failed to build default sandbox image {self.image}:\n{tail}"
            )

    async def _cleanup_orphans(self) -> None:
        output = await self._run_docker(
            "ps",
            "-a",
            "--filter",
            f"label={_LABEL_PREFIX}.managed=true",
            "--format",
            f'{{{{.ID}}}}\t{{{{.Label "{_LABEL_PREFIX}.pid"}}}}',
        )
        for line in output.splitlines():
            if not line.strip():
                continue
            container_id, _, raw_pid = line.partition("\t")
            try:
                pid = int(raw_pid)
            except ValueError:
                pid = -1
            if self._pid_is_alive(pid):
                continue
            await self._run_docker("rm", "-f", container_id, check=False)

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    async def _recycle_after_interrupt(self) -> None:
        if self.container_id is None:
            return
        container_id = self.container_id
        await self._run_docker("kill", container_id, check=False)
        result = await self._run_docker("start", container_id, check=False)
        if not result.startswith("__PION_EXIT_0__"):
            # The stopped container is no longer usable. A later tool call
            # creates a clean replacement instead of silently running on host.
            await self._run_docker("rm", "-f", container_id, check=False)
            self.container_id = None
            self._started = False

    async def _run_docker(
        self,
        *args: str,
        check: bool = True,
    ) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise SandboxError(f"could not execute Docker CLI: {exc}") from exc
        output, _ = await proc.communicate()
        text = output[-_MAX_DOCKER_OUTPUT:].decode("utf-8", errors="replace")
        if check and proc.returncode != 0:
            raise SandboxError(
                f"docker {' '.join(args[:2])} failed with exit code "
                f"{proc.returncode}: {text.strip()}"
            )
        if not check:
            return f"__PION_EXIT_{proc.returncode}__\n{text}"
        return text

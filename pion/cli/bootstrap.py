"""Runtime/MCP/session/agent assembly orchestrated by the pion CLI."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path

import typer
from rich.markup import escape

from ..agent.agent import Agent
from ..config import ConfigError, MCPServerConfig, PionConfig
from ..controller import AgentSessionController
from ..hooks import ExtensionManager
from ..llm.types import Model, UserMessage
from ..mcp import MCPClientManager
from ..sandbox import (
    SandboxError,
    SandboxRuntime,
    SandboxSettings,
)
from ..session import SessionManager, estimate_tokens
from ..tools import build_default_tools
from ._shared import console, err_console


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without network or a REPL)
# ---------------------------------------------------------------------------


def default_session_path(
    now: datetime | None = None, base_dir: Path | None = None
) -> Path:
    """Session file path: <base>/yyyymmdd-HHMMSS-<uuid8>.jsonl.

    `base_dir` defaults to ~/.pion/sessions. The directory is not created
    here; callers create it when they actually persist.
    """
    now = now or datetime.now()
    base = base_dir or (Path.home() / ".pion" / "sessions")
    return base / f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}.jsonl"


def resolve_sandbox_settings(
    config: PionConfig,
    *,
    backend: str | None = None,
    image: str | None = None,
    network: str | None = None,
    git_write: bool = False,
) -> SandboxSettings:
    """Apply CLI sandbox overrides on top of the persisted v1 config."""
    updates: dict[str, object] = {}
    if backend is not None:
        if backend == "docker":
            err_console.print(
                "[yellow]--sandbox docker is deprecated; the sandbox now runs "
                "through the MCP backend (--sandbox mcp).[/yellow]"
            )
            backend = "mcp"
        if backend not in ("off", "mcp"):
            raise ConfigError("--sandbox must be 'off' or 'mcp'")
        updates["backend"] = backend
    if image is not None:
        if not image.strip():
            raise ConfigError("--sandbox-image cannot be empty")
        updates["image"] = image.strip()
    if network is not None:
        if network not in ("bridge", "none"):
            raise ConfigError("--sandbox-network must be 'bridge' or 'none'")
        updates["network"] = network
    if git_write:
        updates["git_write"] = True
    try:
        return SandboxSettings.model_validate(
            {**config.sandbox.model_dump(mode="python"), **updates}
        )
    except ValueError as exc:
        raise ConfigError(f"invalid sandbox settings: {exc}") from exc


def extension_dirs(include_project: bool = True) -> list[Path]:
    """Directories searched for extensions (user-level, then project-level)."""
    directories = [Path("~/.pion/extensions").expanduser()]
    if include_project:
        directories.append(Path.cwd() / ".pion" / "extensions")
    return directories


def build_system_prompt(cwd: Path) -> str:
    """Terse pi-style coding-agent system prompt with the working directory."""
    return f"""You are pion, a coding agent running in the user's terminal.

Working directory: {cwd} (all relative paths resolve here).

Available tools:
- read: read a UTF-8 text file with line numbers (use offset/limit for large files)
- write: create or overwrite a file (parent directories are created automatically)
- edit: exact text replacement in a file (old_string must match uniquely unless replace_all)
- bash: execute a shell command (combined output, truncated to the last 100KB)

Guidelines:
- Use the tools to act; don't just describe what you would do.
- Prefer the read tool over cat, and edit over rewriting whole files.
- Use bash for file operations like ls, rg, find, and for running builds and tests.
- Be concise: short, direct answers; no preamble, no repetition of what the user said.
- Show file paths clearly when working with files.
- Do what was asked and stop; don't take extra actions the user didn't request."""


def find_first_kept_entry_id(session: SessionManager, model: Model) -> str | None:
    """Session entry id of the earliest message to keep after compaction.

    "Kept" is the newest tail of branch messages totaling roughly <= 50% of
    the model's context window (estimate_tokens), cut at a user-message
    boundary so context never resumes mid tool-cycle. Only messages after
    the newest existing compaction entry are candidates. Returns None when
    nothing should be kept (the summary replaces the whole prior branch).
    """
    entries = []
    node = session.leaf_id
    while node is not None:
        entry = session.get_entry(node)
        entries.append(entry)
        node = entry.parent_id
    entries.reverse()  # root -> leaf

    last_compaction = -1
    for i, entry in enumerate(entries):
        if entry.type == "compaction":
            last_compaction = i
    candidates = [
        entry
        for entry in entries[last_compaction + 1 :]
        if entry.type == "message" and entry.message is not None
    ]

    budget = model.context_window // 2
    kept: list = []
    total = 0
    for entry in reversed(candidates):
        tokens = estimate_tokens([entry.message])
        if kept and total + tokens > budget:
            break
        kept.append(entry)
        total += tokens
    kept.reverse()

    # Cut at a user-message boundary: drop leading non-user messages.
    while kept and not isinstance(kept[0].message, UserMessage):
        kept.pop(0)
    return kept[0].id if kept else None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


async def _async_main(
    model: Model,
    api_key: str,
    session_path: Path | None,
    no_extensions: bool,
    print_prompt: str | None,
    sandbox_settings: SandboxSettings,
    allow_project_extensions: bool,
    mcp_servers: dict[str, MCPServerConfig] | None = None,
    ui_mode: str = "plain",
    theme_name: str = "dark",
) -> None:
    # Resolved through the pion.cli package namespace at call time so tests
    # can monkeypatch pion.cli.{build_runtime, check_docker_available,
    # MCPClientManager, extension_dirs, Repl}.
    from pion import cli as _cli

    sandbox_backend = sandbox_settings.backend
    runtime: SandboxRuntime | None = None
    mcp_manager: MCPClientManager | None = None
    try:
        if sandbox_backend == "mcp":
            try:
                # Fail before constructing the Agent or issuing any model request.
                await _cli.check_docker_available()
            except SandboxError as exc:
                err_console.print(
                    f"[red]Sandbox startup failed:[/red] {escape(str(exc))}",
                    soft_wrap=True,
                )
                raise typer.Exit(1) from exc
            if sandbox_settings.network == "bridge":
                err_console.print(
                    "[yellow]Sandbox notice:[/yellow] Docker bridge networking is enabled; "
                    "it does not prevent source exfiltration. Use "
                    "--sandbox-network none for untrusted repositories.",
                    soft_wrap=True,
                )
        else:
            runtime = _cli.build_runtime(sandbox_settings, Path.cwd())
            try:
                # Fail before constructing the Agent or issuing any model request.
                await runtime.start()
            except SandboxError as exc:
                err_console.print(
                    f"[red]Sandbox startup failed:[/red] {escape(str(exc))}",
                    soft_wrap=True,
                )
                raise typer.Exit(1) from exc

        # Project extensions are host Python code and are disabled by default
        # whenever a sandbox policy is active.
        extensions: ExtensionManager | None = None
        if not no_extensions:
            extensions = ExtensionManager()
            include_project = (
                sandbox_backend == "off" or allow_project_extensions
            )
            if sandbox_backend != "off" and allow_project_extensions:
                err_console.print(
                    "[bold yellow]WARNING:[/bold yellow] project extensions execute "
                    "as trusted Python in the host Pion process and can bypass the sandbox.",
                    soft_wrap=True,
                )
            dirs = [
                path
                for path in _cli.extension_dirs(include_project=include_project)
                if path.is_dir()
            ]
            if dirs:
                await extensions.load(dirs)
            for error in extensions.errors:
                err_console.print(f"[dim yellow]Extension error: {error}[/dim yellow]")

        servers = dict(mcp_servers or {})
        if sandbox_backend == "mcp":
            # Mount pion's own sandbox MCP server: the default tools run in
            # Docker inside the child process instead of on the host.
            env: dict[str, str] = {
                "PION_SANDBOX_NETWORK": sandbox_settings.network,
                "PION_SANDBOX_MEMORY_MB": str(sandbox_settings.memory_mb),
                "PION_SANDBOX_CPUS": str(sandbox_settings.cpus),
            }
            if sandbox_settings.image:
                env["PION_SANDBOX_IMAGE"] = sandbox_settings.image
            if sandbox_settings.git_write:
                env["PION_SANDBOX_GIT_WRITE"] = "1"
            servers["sandbox"] = MCPServerConfig(
                command=sys.executable,
                args=["-m", "pion.cli", "mcp"],
                env=env,
            )
        enabled_mcp = {
            name: server for name, server in servers.items() if server.enabled
        }
        if enabled_mcp:
            if any(name != "sandbox" for name in enabled_mcp):
                err_console.print(
                    "[bold yellow]MCP security notice:[/bold yellow] stdio MCP servers "
                    "run as trusted host processes and are not isolated by the Docker sandbox.",
                    soft_wrap=True,
                )
            mcp_manager = _cli.MCPClientManager(servers)
            default_tools = (
                [] if sandbox_backend == "mcp" else build_default_tools(runtime)
            )
            reserved_names = {tool.name for tool in default_tools}
            if extensions is not None:
                reserved_names.update(tool.name for tool in extensions.tools)
            await mcp_manager.start(reserved_names)
            for error in mcp_manager.errors:
                err_console.print(
                    f"[yellow]MCP server unavailable:[/yellow] {escape(error)}"
                )
            err_console.print(
                f"[dim]MCP: {mcp_manager.connected_server_count} server(s), "
                f"{len(mcp_manager.tools)} tool(s) connected.[/dim]"
            )
        else:
            default_tools = build_default_tools(runtime)

        # Session: resume an existing JSONL file, or start a new one.
        if session_path is not None and session_path.exists():
            try:
                session = SessionManager.load(session_path)
            except Exception as exc:
                err_console.print(
                    f"[red]Error:[/red] could not load session {session_path}: {exc}"
                )
                raise typer.Exit(1)
        else:
            if session_path is None:
                session_path = default_session_path()
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session = SessionManager(session_path)

        agent = Agent(
            model=model,
            tools=[
                *default_tools,
                *(mcp_manager.tools if mcp_manager is not None else []),
            ],
            system_prompt=build_system_prompt(Path.cwd()),
            api_key=api_key,
            extension_manager=extensions,
        )
        agent.messages = session.build_context()

        controller = AgentSessionController(agent, session, session_path, extensions)
        repl = _cli.Repl(controller, console)
        if print_prompt is not None:
            await repl.run_print(print_prompt)
        elif ui_mode == "tui":
            from ..tui import PionTUI, TUIStatus
            from ..tui.theme import load_theme

            try:
                theme = load_theme(theme_name)
            except ValueError:
                err_console.print(
                    f"[yellow]Unknown theme {escape(theme_name)!r}; "
                    "falling back to dark.[/yellow]"
                )
                theme = load_theme("dark")
            tui = PionTUI(
                controller,
                TUIStatus(
                    project=Path.cwd().name,
                    sandbox=sandbox_settings.backend,
                    mcp_servers=(
                        mcp_manager.connected_server_count
                        if mcp_manager is not None
                        else 0
                    ),
                    mcp_tools=len(mcp_manager.tools) if mcp_manager is not None else 0,
                ),
                theme=theme,
            )
            await tui.run_async()
        else:
            await repl.run()
    finally:
        if mcp_manager is not None:
            prior_errors = len(mcp_manager.errors)
            await mcp_manager.close()
            for error in mcp_manager.errors[prior_errors:]:
                err_console.print(
                    f"[dim yellow]MCP shutdown error: {escape(error)}[/dim yellow]"
                )
        if runtime is not None:
            await runtime.close()

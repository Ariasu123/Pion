"""pion command line interface.

`pion` starts the inline TUI (terminal-native scrollback, differential
rendering), `pion --ui plain` starts the legacy REPL, and `pion --print "..."`
runs a single prompt and exits. Sessions persist as JSONL under
~/.pion/sessions (or a file passed via --session), with automatic context
compaction when the conversation approaches the model's context window.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape

from . import __version__
from .agent.agent import Agent
from .agent.events import AgentEvent
from .config import (
    ConfigError,
    MCPServerConfig,
    PionConfig,
    ProfileConfig,
    default_config_path,
    load_config,
    save_config,
)
from .controller import AgentSessionController
from .hooks import ExtensionManager
from .llm.registry import ENV_KEY_NAMES, get_model, resolve_api_key
from .llm.types import (
    AssistantMessage,
    Model,
    TextContent,
    UserMessage,
)
from .mcp import MCPClientManager
from .sandbox import (
    SandboxError,
    SandboxRuntime,
    SandboxSettings,
    build_runtime,
    check_docker_available,
)
from .session import SessionManager, estimate_tokens, should_compact
from .tools import build_default_tools

app = typer.Typer(
    name="pion",
    help="pion — a minimal, hackable coding agent harness.",
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)

DEFAULT_MODEL_ID = "deepseek-v4-flash"


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


def parse_slash_command(text: str) -> tuple[str, str] | None:
    """Parse '/name args' into (name, args); None when not a slash command."""
    stripped = text.strip()
    if not stripped.startswith("/") or stripped == "/":
        return None
    parts = stripped[1:].split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    return name, args


def summarize_tool_args(name: str, args: dict[str, Any]) -> str:
    """One-line summary of a tool call: the most telling argument, single line."""
    for key in ("command", "path", "file_path"):
        if key in args and args[key] is not None:
            summary = str(args[key])
            break
    else:
        summary = json.dumps(args, ensure_ascii=False)
    summary = " ".join(summary.split())
    if len(summary) > 120:
        summary = summary[:117] + "..."
    return summary


def summarize_tool_result(text: str, limit: int = 200) -> str:
    """Short single-line summary of a tool result (first ~`limit` chars)."""
    summary = " ".join(text.split())
    if len(summary) > limit:
        summary = summary[: limit - 3] + "..."
    return summary


def api_key_env_name(model: Model) -> str:
    """Environment variable holding the API key for `model`'s provider."""
    return ENV_KEY_NAMES.get(model.provider, f"{model.provider.upper()}_API_KEY")


def is_interactive() -> bool:
    """Whether startup may safely ask the user for input."""
    return sys.stdin.isatty()


def resolve_ui_mode(ui: str, print_prompt: str | None) -> str:
    """Validate UI selection and enforce terminal requirements."""
    mode = ui.strip().lower()
    if mode not in ("tui", "plain"):
        raise ConfigError("--ui must be 'tui' or 'plain'")
    if print_prompt is None and not is_interactive():
        raise ConfigError(
            "interactive mode requires a TTY; use --print for non-interactive input"
        )
    if print_prompt is None and mode == "tui" and os.environ.get("TERM") == "dumb":
        err_console.print(
            "[yellow]Terminal does not support full-screen UI; using --ui plain.[/yellow]"
        )
        return "plain"
    return mode


def _prompt_nonempty(label: str, default: str | None = None) -> str:
    """Prompt until a non-empty value is entered."""
    while True:
        value = typer.prompt(label, default=default, show_default=default is not None)
        value = str(value).strip()
        if value:
            return value
        err_console.print(f"[yellow]{escape(label)} cannot be empty.[/yellow]")


def _prompt_protocol(existing: ProfileConfig | None = None) -> str:
    default = (
        "anthropic" if existing and existing.api == "anthropic-messages" else "openai"
    )
    while True:
        value = (
            typer.prompt("API protocol (openai/anthropic)", default=default)
            .strip()
            .lower()
        )
        aliases = {
            "openai": "openai-completions",
            "openai-completions": "openai-completions",
            "anthropic": "anthropic-messages",
            "anthropic-messages": "anthropic-messages",
        }
        if value in aliases:
            return aliases[value]
        err_console.print("[yellow]Choose 'openai' or 'anthropic'.[/yellow]")


def configure_profile(
    config: PionConfig, requested_name: str | None = None
) -> tuple[str, ProfileConfig]:
    """Interactively create or update a named profile."""
    default_name = requested_name or config.active_profile or "default"
    name = _prompt_nonempty("Profile name", default_name)
    existing = config.profiles.get(name)
    protocol = _prompt_protocol(existing)
    base_url = _prompt_nonempty(
        "Base URL", existing.base_url if existing is not None else None
    )
    model_id = _prompt_nonempty(
        "Model name", existing.model if existing is not None else None
    )
    entered_key = typer.prompt(
        "API key",
        default="",
        hide_input=True,
        show_default=False,
    ).strip()
    api_key = entered_key or (existing.api_key if existing is not None else "")
    if not api_key:
        raise ConfigError("API key cannot be empty")

    profile = ProfileConfig(
        api=protocol,
        base_url=base_url,
        api_key=api_key,
        model=model_id,
    )
    config.profiles[name] = profile
    config.active_profile = name
    return name, profile


def select_profile(config: PionConfig) -> str:
    """Prompt for one profile by number or name."""
    names = list(config.profiles)
    default_name = config.active_profile or names[0]
    typer.echo("Available profiles:")
    for index, name in enumerate(names, start=1):
        marker = " *" if name == default_name else ""
        typer.echo(f"  {index}. {name}{marker}")
    while True:
        choice = typer.prompt("Profile", default=default_name).strip()
        if choice in config.profiles:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        err_console.print(f"[yellow]Unknown profile: {escape(choice)}[/yellow]")


def _load_user_config() -> tuple[PionConfig, Path]:
    path = default_config_path()
    return load_config(path), path


def resolve_startup(
    model_id: str | None,
    api_key: str | None,
    base_url: str | None,
    profile_name: str | None,
    configure: bool,
) -> tuple[Model, str]:
    """Resolve CLI, profile, environment, and built-in startup settings."""
    config, path = _load_user_config()
    interactive = is_interactive()

    if configure and not interactive:
        raise ConfigError("--configure requires an interactive terminal")

    selected: str | None = None
    if profile_name is not None:
        if profile_name in config.profiles:
            selected = profile_name
        elif not configure:
            raise ConfigError(f"unknown profile {profile_name!r}")
    elif len(config.profiles) > 1 and interactive and not configure:
        selected = select_profile(config)
    elif config.active_profile is not None:
        selected = config.active_profile
    elif len(config.profiles) == 1:
        selected = next(iter(config.profiles))

    if configure:
        selected, configured = configure_profile(config, profile_name or selected)
        save_config(config, path)
        return configured.to_model(), configured.api_key

    if selected is not None:
        saved = config.profiles[selected]
        if model_id is not None and not model_id.strip():
            raise ConfigError("model name cannot be empty")
        if base_url is not None and not base_url.strip():
            raise ConfigError("base URL cannot be empty")
        if api_key is not None and not api_key.strip():
            raise ConfigError("API key cannot be empty")
        updated = saved.model_copy(
            update={
                **({"model": model_id.strip()} if model_id is not None else {}),
                **({"base_url": base_url.strip()} if base_url is not None else {}),
                **({"api_key": api_key} if api_key is not None else {}),
            }
        )
        config.profiles[selected] = updated
        active_changed = config.active_profile != selected
        config.active_profile = selected
        if updated != saved or active_changed:
            save_config(config, path)
        return updated.to_model(), updated.api_key

    requested_model = model_id or DEFAULT_MODEL_ID
    try:
        model = get_model(requested_model)
    except KeyError as exc:
        if model_id is None or base_url is None or api_key is None:
            raise ConfigError(str(exc.args[0])) from exc
        if not model_id.strip() or not base_url.strip() or not api_key.strip():
            raise ConfigError("model, base URL, and API key cannot be empty")
        model = Model(
            id=model_id.strip(),
            name=model_id.strip(),
            api="openai-completions",
            provider="openai",
            baseUrl=base_url.strip(),
            contextWindow=128_000,
            maxTokens=8192,
        )
    if base_url is not None:
        model = model.model_copy(update={"base_url": base_url})
    key = api_key or resolve_api_key(model)

    if key is not None:
        if any(value is not None for value in (model_id, api_key, base_url)):
            new_profile = ProfileConfig(
                api=model.api,
                base_url=model.base_url,
                api_key=key,
                model=model.id,
            )
            config.profiles["default"] = new_profile
            config.active_profile = "default"
            save_config(config, path)
        return model, key

    if not interactive:
        raise ConfigError(
            "no usable profile or API key; run 'pion --configure' in an "
            "interactive terminal or pass --profile/--api-key"
        )

    selected, configured = configure_profile(config)
    save_config(config, path)
    return configured.to_model(), configured.api_key


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
# Streaming renderer
# ---------------------------------------------------------------------------


class StreamRenderer:
    """Render AgentEvents to the console while a run streams.

    Assistant text prints as it arrives; thinking blocks render dim; tool
    calls get a one-line summary, tool results a short dim summary.
    """

    def __init__(self, out: Console) -> None:
        self.console = out
        self._open: str | None = None  # "text" | "thinking" line in progress

    def _close_line(self) -> None:
        if self._open is not None:
            self.console.print()
            self._open = None

    def handle(self, event: AgentEvent) -> None:
        """Subscriber callback for Agent.subscribe (sync variant)."""
        if event.type == "message_update" and event.assistant_event is not None:
            ev = event.assistant_event
            if ev.type == "thinking_delta" and ev.delta:
                if self._open != "thinking":
                    self._close_line()
                    self._open = "thinking"
                self.console.print(
                    ev.delta, end="", style="dim italic", markup=False, highlight=False
                )
            elif ev.type == "text_delta" and ev.delta:
                if self._open != "text":
                    self._close_line()
                    self._open = "text"
                self.console.print(ev.delta, end="", markup=False, highlight=False)
        elif event.type == "message_end":
            if isinstance(event.message, AssistantMessage):
                self._close_line()
        elif event.type == "tool_execution_start":
            self._close_line()
            name = event.tool_name or "tool"
            summary = escape(summarize_tool_args(name, event.args or {}))
            self.console.print(f"[cyan]● {escape(name)}[/cyan] [dim]{summary}[/dim]")
        elif event.type == "tool_execution_end":
            self._close_line()
            text = ""
            if event.result is not None:
                text = "".join(
                    block.text
                    for block in event.result.content
                    if isinstance(block, TextContent)
                )
            summary = escape(summarize_tool_result(text))
            style = "red" if event.is_error else "dim"
            self.console.print(f"  ⎿ {summary}", style=style, markup=False)


# ---------------------------------------------------------------------------
# The REPL / single-shot runner
# ---------------------------------------------------------------------------


class Repl:
    """Interactive session: agent + session store + extensions + console."""

    def __init__(
        self,
        controller: AgentSessionController,
        out: Console,
    ) -> None:
        self.controller = controller
        self.agent = controller.agent
        self.session = controller.session
        self.session_path = controller.session_path
        self.extensions = controller.extensions
        self.console = out

    # ------------------------------------------------------------------
    # Running prompts
    # ------------------------------------------------------------------

    async def _prompt_with_sigint(self, text: str) -> AssistantMessage:
        """Run one prompt; Ctrl-C during the run aborts the agent."""
        loop = asyncio.get_running_loop()
        installed = False
        try:
            loop.add_signal_handler(signal.SIGINT, self.controller.abort)
            installed = True
        except (NotImplementedError, RuntimeError, ValueError):
            # Platform without signal handlers (or not the main thread):
            # Ctrl-C falls back to interrupting the whole process.
            installed = False
        try:
            return await self.controller.prompt(text)
        finally:
            if installed:
                loop.remove_signal_handler(signal.SIGINT)

    async def handle_prompt(self, text: str, render: bool = True) -> AssistantMessage:
        """Run one prompt, persist new messages, auto-compact afterwards."""
        renderer = StreamRenderer(self.console) if render else None
        if renderer is not None:
            self.agent.subscribe(renderer.handle)
        try:
            final = await self._prompt_with_sigint(text)
        finally:
            if renderer is not None:
                self.agent.unsubscribe(renderer.handle)
        if render:
            if final.stop_reason == "error":
                self.console.print(
                    f"[red]Error: {escape(final.error_message or 'unknown error')}[/red]"
                )
            elif final.stop_reason == "aborted":
                self.console.print("[dim]Aborted.[/dim]")

        return final

    async def maybe_compact(self, force: bool = False) -> None:
        """Compact the session when the context crosses the threshold.

        Failures are reported but never crash the REPL.
        """
        if not self.agent.messages:
            if force:
                self.console.print("[dim]Nothing to compact yet.[/dim]")
            return
        if not force and not should_compact(self.agent.messages, self.agent.model):
            return
        try:
            summary = await self.controller.maybe_compact(force=force)
            if summary is None:
                return
            self.console.print(
                f"[dim]Context compacted — summary of {len(summary)} chars, "
                f"{len(self.agent.messages)} messages kept in context.[/dim]"
            )
        except Exception as exc:  # never crash the REPL over compaction
            self.console.print(
                f"[yellow]Compaction failed (continuing without it): "
                f"{escape(str(exc))}[/yellow]"
            )

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _print_help(self) -> None:
        self.console.print("[bold]Commands[/bold]")
        rows = [
            ("/help", "show this help"),
            (
                "/model <id>",
                "switch model (re-resolves the API key); no arg shows current",
            ),
            ("/compact", "force context compaction now"),
            ("/stats", "token usage and cost of the last response"),
            ("/exit", "quit (also Ctrl-D / EOF)"),
        ]
        for name, description in rows:
            self.console.print(f"  [cyan]{name:<14}[/cyan] {description}")
        if self.extensions is not None and self.extensions.commands:
            self.console.print("[bold]Extension commands[/bold]")
            for name in sorted(self.extensions.commands):
                self.console.print(f"  [cyan]/{name}[/cyan]")

    def _print_stats(self) -> None:
        usage = self.controller.last_usage
        if usage is None:
            self.console.print("[dim]No usage yet — run a prompt first.[/dim]")
            return
        cost = usage.cost.total or self.agent.model.compute_cost(usage).total
        self.console.print(
            f"[dim]last usage ({self.agent.model.id}): "
            f"input {usage.input}, output {usage.output}, "
            f"cache read {usage.cache_read}, cache write {usage.cache_write} — "
            f"cost ${cost:.6f}[/dim]"
        )

    def _switch_model(self, args: str) -> None:
        model_id = args.strip()
        if not model_id:
            self.console.print(f"[dim]Current model: {self.agent.model.id}[/dim]")
            return
        try:
            self.controller.switch_model(model_id)
        except KeyError as exc:
            self.console.print(f"[red]{escape(str(exc.args[0]))}[/red]")
            return
        model = self.agent.model
        self.console.print(f"[dim]Switched to {model.id} ({model.name}).[/dim]")
        if self.agent.api_key is None:
            self.console.print(
                f"[yellow]No API key found — set {api_key_env_name(model)} "
                f"or restart with --api-key.[/yellow]"
            )

    async def handle_slash(self, name: str, args: str) -> bool:
        """Handle one slash command. Returns True when the REPL should exit."""
        if name in ("exit", "quit"):
            return True
        if name == "help":
            self._print_help()
        elif name == "model":
            self._switch_model(args)
        elif name == "compact":
            await self.maybe_compact(force=True)
        elif name == "stats":
            self._print_stats()
        elif self.extensions is not None and name in self.extensions.commands:
            command = self.extensions.commands[name]
            result = command()
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                self.console.print(str(result))
        else:
            self.console.print(
                f"[red]Unknown command: /{escape(name)}[/red] — try /help"
            )
        return False

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        self.console.print(
            f"[bold]pion[/bold] {__version__} — model [cyan]{escape(self.agent.model.id)}[/cyan]"
        )
        self.console.print(
            f"[dim]cwd: {Path.cwd()}  session: {self.session_path}[/dim]"
        )
        self.console.print("[dim]/help for commands, /exit or Ctrl-D to quit[/dim]")

    async def run(self) -> None:
        """Interactive input()-based REPL loop."""
        self._print_banner()
        while True:
            try:
                line = input("pion> ")
            except EOFError:
                self.console.print("\n[dim]Bye.[/dim]")
                return
            line = line.strip()
            if not line:
                continue
            parsed = parse_slash_command(line)
            if parsed is not None:
                try:
                    should_exit = await self.handle_slash(*parsed)
                except Exception as exc:
                    self.console.print(f"[red]Command failed: {escape(str(exc))}[/red]")
                    should_exit = False
                if should_exit:
                    return
                continue
            try:
                await self.handle_prompt(line)
            except Exception as exc:  # a failed prompt must not kill the REPL
                self.console.print(f"[red]Error: {escape(str(exc))}[/red]")

    async def run_print(self, text: str) -> None:
        """Single-shot mode: run one prompt, print the final text, exit."""
        final = await self.handle_prompt(text, render=False)
        if final.stop_reason in ("error", "aborted"):
            err_console.print(
                f"[red]Error: {escape(final.error_message or final.stop_reason)}[/red]"
            )
            raise typer.Exit(1)
        sys.stdout.write(final.text() + "\n")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pion {__version__}")
        raise typer.Exit()


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
    sandbox_backend = sandbox_settings.backend
    runtime: SandboxRuntime | None = None
    mcp_manager: MCPClientManager | None = None
    try:
        if sandbox_backend == "mcp":
            try:
                # Fail before constructing the Agent or issuing any model request.
                await check_docker_available()
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
            runtime = build_runtime(sandbox_settings, Path.cwd())
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
                for path in extension_dirs(include_project=include_project)
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
            mcp_manager = MCPClientManager(servers)
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
        repl = Repl(controller, console)
        if print_prompt is not None:
            await repl.run_print(print_prompt)
        elif ui_mode == "tui":
            from .tui import PionTUI, TUIStatus
            from .tui.theme import load_theme

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


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model_id: str | None = typer.Option(
        None, "--model", "-m", help="Model id (overrides and updates the profile)."
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key (default: provider env var, e.g. DEEPSEEK_API_KEY).",
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Override the model's API base URL."
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="Use a saved connection profile."
    ),
    configure: bool = typer.Option(
        False, "--configure", help="Create or update a connection profile."
    ),
    session: Path | None = typer.Option(
        None, "--session", help="Resume (or create) a JSONL session file."
    ),
    no_extensions: bool = typer.Option(
        False, "--no-extensions", help="Do not load any extensions."
    ),
    sandbox: str | None = typer.Option(
        None,
        "--sandbox",
        help="Execution backend: off (default, unrestricted host) or mcp "
        "(Docker sandbox mounted as an MCP server).",
    ),
    sandbox_image: str | None = typer.Option(
        None,
        "--sandbox-image",
        help="Use an existing custom Docker sandbox image.",
    ),
    sandbox_network: str | None = typer.Option(
        None,
        "--sandbox-network",
        help="Docker network mode: bridge (default) or none.",
    ),
    sandbox_git_write: bool = typer.Option(
        False,
        "--sandbox-git-write",
        help="Allow writes to Git metadata from sandboxed tools.",
    ),
    allow_project_extensions: bool = typer.Option(
        False,
        "--allow-project-extensions",
        help="Load project Python extensions on the host even when sandboxed.",
    ),
    ui: str = typer.Option(
        "tui",
        "--ui",
        help="Interactive interface: tui (default) or plain.",
    ),
    print_prompt: str | None = typer.Option(
        None, "--print", "-p", help="Run a single prompt, print the reply, and exit."
    ),
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Print the version and exit.",
    ),
) -> None:
    """pion: a minimal, hackable coding agent. No arguments starts the TUI."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        config, _ = _load_user_config()
        sandbox_settings = resolve_sandbox_settings(
            config,
            backend=sandbox,
            image=sandbox_image,
            network=sandbox_network,
            git_write=sandbox_git_write,
        )
        # Validate terminal/UI requirements before profile resolution so a
        # piped interactive launch reports the actionable TTY error directly.
        ui_mode = resolve_ui_mode(ui, print_prompt)
        model, key = resolve_startup(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            profile_name=profile,
            configure=configure,
        )
    except ConfigError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(1)
    except (typer.Abort, EOFError):
        err_console.print("\n[dim]Configuration cancelled.[/dim]")
        raise typer.Exit(130) from None

    try:
        asyncio.run(
            _async_main(
                model,
                key,
                session,
                no_extensions,
                print_prompt,
                sandbox_settings,
                allow_project_extensions,
                config.mcp_servers,
                ui_mode,
                config.theme,
            )
        )
    except KeyboardInterrupt:
        # Ctrl-C outside a run (e.g. at the input prompt).
        err_console.print("\n[dim]Interrupted.[/dim]")
        raise typer.Exit(130) from None


@app.command("mcp")
def mcp_command() -> None:
    """Run the sandbox MCP server on stdio (bash/read/write/edit in Docker)."""
    from .mcp_server import main as mcp_main

    mcp_main()


if __name__ == "__main__":  # pragma: no cover
    app()

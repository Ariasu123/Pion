"""Typer application object and top-level commands for the pion CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.markup import escape

from ..config import ConfigError
from ._shared import err_console
from .bootstrap import resolve_sandbox_settings
from .profiles import (
    _load_user_config,
    _version_callback,
    resolve_startup,
    resolve_ui_mode,
)

app = typer.Typer(
    name="pion",
    help="pion — a minimal, hackable coding agent harness.",
    add_completion=False,
)


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
    from pion import cli as _cli

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
            _cli._async_main(
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
    from ..mcp.sandbox_server import main as mcp_main

    mcp_main()

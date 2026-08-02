"""Connection-profile interactivity and startup resolution for the pion CLI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.markup import escape

from .. import __version__
from ..config import (
    ConfigError,
    PionConfig,
    ProfileConfig,
    load_config,
    save_config,
)
from ..llm.registry import get_model, resolve_api_key
from ..llm.types import Model
from ._shared import err_console

DEFAULT_MODEL_ID = "deepseek-v4-flash"


def is_interactive() -> bool:
    """Whether startup may safely ask the user for input."""
    return sys.stdin.isatty()


def resolve_ui_mode(ui: str, print_prompt: str | None) -> str:
    """Validate UI selection and enforce terminal requirements."""
    from pion import cli as _cli

    mode = ui.strip().lower()
    if mode not in ("tui", "plain"):
        raise ConfigError("--ui must be 'tui' or 'plain'")
    if print_prompt is None and not _cli.is_interactive():
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
    from pion import cli as _cli

    path = _cli.default_config_path()
    return load_config(path), path


def resolve_startup(
    model_id: str | None,
    api_key: str | None,
    base_url: str | None,
    profile_name: str | None,
    configure: bool,
) -> tuple[Model, str]:
    """Resolve CLI, profile, environment, and built-in startup settings."""
    from pion import cli as _cli

    config, path = _load_user_config()
    interactive = _cli.is_interactive()

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


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"pion {__version__}")
        raise typer.Exit()

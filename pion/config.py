"""Persistent user configuration for Pion connection profiles."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .llm.types import Model
from .sandbox.base import SandboxSettings

ApiProtocol = Literal["openai-completions", "anthropic-messages"]


class ConfigError(ValueError):
    """Raised when the user configuration cannot be loaded or saved."""


class ProfileConfig(BaseModel):
    """One named LLM endpoint configuration."""

    model_config = ConfigDict(extra="forbid")

    api: ApiProtocol
    base_url: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    model: str = Field(min_length=1)

    def to_model(self) -> Model:
        """Build the runtime model descriptor for this profile."""
        provider = "anthropic" if self.api == "anthropic-messages" else "openai"
        return Model(
            id=self.model,
            name=self.model,
            api=self.api,
            provider=provider,
            baseUrl=self.base_url,
            contextWindow=128_000,
            maxTokens=8192,
        )


class MCPServerConfig(BaseModel):
    """One trusted host-side stdio MCP server."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    timeout_seconds: float = Field(default=30.0, gt=0)

    @field_validator("command")
    @classmethod
    def command_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("MCP command cannot be blank")
        return value


class PionConfig(BaseModel):
    """Versioned collection of named connection profiles."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    active_profile: str | None = None
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def active_profile_exists(self) -> "PionConfig":
        if self.active_profile is not None and self.active_profile not in self.profiles:
            raise ValueError(
                f"active profile {self.active_profile!r} is not present in profiles"
            )
        invalid = [
            name
            for name in self.mcp_servers
            if not name
            or any(
                not (char.isascii() and (char.isalnum() or char in "_-"))
                for char in name
            )
        ]
        if invalid:
            raise ValueError(
                "MCP server names may contain only ASCII letters, digits, '_' and '-': "
                + ", ".join(repr(name) for name in invalid)
            )
        return self


def default_config_path() -> Path:
    """Location of the per-user Pion configuration."""
    return Path.home() / ".pion" / "config.json"


def load_config(path: Path | None = None) -> PionConfig:
    """Load configuration, returning an empty config when the file is absent."""
    target = path or default_config_path()
    if not target.exists():
        return PionConfig()
    try:
        return PionConfig.model_validate_json(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"could not load {target}: {exc}") from exc


def save_config(config: PionConfig, path: Path | None = None) -> None:
    """Atomically persist configuration with owner-only permissions."""
    target = path or default_config_path()
    parent = target.parent
    temporary: Path | None = None
    try:
        parent.mkdir(parents=True, exist_ok=True)
        os.chmod(parent, 0o700)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(
                json.dumps(
                    config.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    except OSError as exc:
        raise ConfigError(f"could not save {target}: {exc}") from exc
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass

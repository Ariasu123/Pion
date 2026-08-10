"""Compatibility adapter for the extracted ``sandbox-docker-mcp`` package."""

from __future__ import annotations

from pathlib import Path

from sandbox_docker_mcp import (
    DockerSandboxRuntime as ExternalDockerSandboxRuntime,
)
from sandbox_docker_mcp import SandboxSettings as ExternalSandboxSettings
from sandbox_docker_mcp import (
    SandboxUnavailableError as ExternalSandboxUnavailableError,
)
from sandbox_docker_mcp import check_docker_available as external_docker_preflight
from sandbox_docker_mcp.runtime import default_dockerfile

from .base import SandboxSettings, SandboxUnavailableError

DEFAULT_DOCKERFILE = default_dockerfile()


def to_external_settings(settings: SandboxSettings) -> ExternalSandboxSettings:
    """Convert Pion's persisted policy to the standalone package model."""

    return ExternalSandboxSettings.model_validate(
        settings.model_dump(mode="python", exclude={"backend"})
    )


class DockerSandboxRuntime(ExternalDockerSandboxRuntime):
    """Legacy import path backed entirely by ``sandbox-docker-mcp``."""

    def __init__(self, workspace: Path, settings: SandboxSettings) -> None:
        super().__init__(workspace, to_external_settings(settings))


async def check_docker_available() -> None:
    """Run the external preflight while preserving Pion's error contract."""

    try:
        await external_docker_preflight()
    except ExternalSandboxUnavailableError as exc:
        raise SandboxUnavailableError(str(exc)) from exc


__all__ = [
    "DEFAULT_DOCKERFILE",
    "DockerSandboxRuntime",
    "check_docker_available",
    "to_external_settings",
]

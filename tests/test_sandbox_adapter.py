"""Compatibility tests between Pion and ``sandbox-docker-mcp``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

# The Docker sandbox adapter lives in the optional `sandbox` extra; skip the
# whole module (rather than fail at collection) when it isn't installed — CI's
# default env is extra-free on purpose, to regression-test the decoupling.
pytest.importorskip("sandbox_docker_mcp")

from sandbox_docker_mcp import (
    DockerSandboxRuntime as ExternalDockerSandboxRuntime,
)
from sandbox_docker_mcp import (
    SandboxUnavailableError as ExternalSandboxUnavailableError,
)

from pion.mcp import sandbox_server
from pion.sandbox import SandboxSettings, SandboxUnavailableError
from pion.sandbox.docker import (
    DockerSandboxRuntime,
    check_docker_available,
    to_external_settings,
)


def test_policy_conversion_preserves_all_external_fields() -> None:
    settings = SandboxSettings(
        backend="mcp",
        image="example:sandbox",
        network="none",
        memory_mb=768,
        cpus=1.5,
        pids_limit=64,
        git_write=True,
        protect_paths=[".env", "credentials.json"],
    )
    external = to_external_settings(settings)
    assert external.model_dump(mode="python") == {
        "image": "example:sandbox",
        "network": "none",
        "memory_mb": 768,
        "cpus": 1.5,
        "pids_limit": 64,
        "git_write": True,
        "protect_paths": [".env", "credentials.json"],
    }


def test_legacy_runtime_path_is_external_implementation(tmp_path: Path) -> None:
    runtime = DockerSandboxRuntime(tmp_path, SandboxSettings(network="none"))
    assert isinstance(runtime, ExternalDockerSandboxRuntime)
    assert runtime.settings.network == "none"


async def test_preflight_translates_external_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail() -> None:
        raise ExternalSandboxUnavailableError("no daemon")

    monkeypatch.setattr("pion.sandbox.docker.external_docker_preflight", fail)
    with pytest.raises(SandboxUnavailableError, match="no daemon"):
        await check_docker_available()


def test_legacy_server_environment_overrides_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = SandboxSettings(
        pids_limit=96,
        protect_paths=[".env", "credentials.json"],
    )
    monkeypatch.setattr(
        sandbox_server, "load_config", lambda: SimpleNamespace(sandbox=saved)
    )
    monkeypatch.setenv("PION_SANDBOX_IMAGE", "example:cli")
    monkeypatch.setenv("PION_SANDBOX_NETWORK", "none")
    monkeypatch.setenv("PION_SANDBOX_MEMORY_MB", "512")
    monkeypatch.setenv("PION_SANDBOX_CPUS", "1.25")
    monkeypatch.setenv("PION_SANDBOX_GIT_WRITE", "1")
    resolved = sandbox_server.resolve_server_settings()
    assert resolved.image == "example:cli"
    assert resolved.network == "none"
    assert resolved.memory_mb == 512
    assert resolved.cpus == 1.25
    assert resolved.git_write is True
    assert resolved.pids_limit == 96
    assert resolved.protect_paths == [".env", "credentials.json"]


def test_invalid_sandbox_env_value_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bogus PION_SANDBOX_* value must abort rather than be silently ignored:
    # this process executes sandboxed commands, so a misconfigured policy is a
    # fail-closed condition, not a downgrade to defaults.
    monkeypatch.setattr(
        sandbox_server, "load_config", lambda: SimpleNamespace(sandbox=SandboxSettings())
    )
    monkeypatch.setenv("PION_SANDBOX_NETWORK", "wide-open")
    with pytest.raises(ValueError, match="PION_SANDBOX_NETWORK"):
        sandbox_server.resolve_server_settings()

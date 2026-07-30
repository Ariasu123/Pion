"""Tests for persistent Pion connection profiles."""

from __future__ import annotations

import json
import stat

import pytest

from pion.config import ConfigError, PionConfig, ProfileConfig, load_config, save_config
from pion.sandbox import SandboxSettings


def _profile(api: str = "openai-completions") -> ProfileConfig:
    return ProfileConfig(
        api=api,
        base_url="https://example.com/v1",
        api_key="secret",
        model="example-model",
    )


def test_load_missing_config_returns_empty(tmp_path) -> None:
    config = load_config(tmp_path / "missing.json")
    assert config.profiles == {}
    assert config.active_profile is None
    assert config.sandbox == SandboxSettings()


def test_v1_config_accepts_and_round_trips_sandbox_settings(tmp_path) -> None:
    path = tmp_path / "config.json"
    config = PionConfig(
        sandbox=SandboxSettings(
            image="example/pion-sandbox:local",
            network="none",
            memory_mb=1024,
            cpus=1.25,
            pids_limit=96,
            git_write=True,
            protect_paths=[".env", "credentials.json"],
        )
    )
    save_config(config, path)
    assert load_config(path).sandbox == config.sandbox


def test_old_v1_config_without_sandbox_field_gets_secure_defaults(
    tmp_path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"version": 1, "active_profile": None, "profiles": {}}),
        encoding="utf-8",
    )
    assert load_config(path).sandbox == SandboxSettings()


@pytest.mark.parametrize("protected", ["", "../outside", "/absolute"])
def test_config_rejects_unsafe_protected_path_patterns(
    tmp_path, protected: str
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {},
                "sandbox": {"protect_paths": [protected]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="protected paths"):
        load_config(path)


def test_save_and_load_config_with_owner_only_permissions(tmp_path) -> None:
    path = tmp_path / "nested" / "config.json"
    config = PionConfig(active_profile="main", profiles={"main": _profile()})

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_load_rejects_malformed_config_without_changing_it(tmp_path) -> None:
    path = tmp_path / "config.json"
    original = "{not-json"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ConfigError, match="could not load"):
        load_config(path)

    assert path.read_text(encoding="utf-8") == original


def test_load_rejects_missing_active_profile(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"version": 1, "active_profile": "missing", "profiles": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="active profile"):
        load_config(path)


@pytest.mark.parametrize(
    ("api", "provider"),
    [
        ("openai-completions", "openai"),
        ("anthropic-messages", "anthropic"),
    ],
)
def test_profile_builds_dynamic_model(api: str, provider: str) -> None:
    profile = _profile(api)
    model = profile.to_model()
    assert model.id == "example-model"
    assert model.api == api
    assert model.provider == provider
    assert model.base_url == "https://example.com/v1"
    assert model.context_window == 128_000
    assert model.max_tokens == 8192

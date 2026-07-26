"""Tests for persistent Pion connection profiles."""

from __future__ import annotations

import json
import stat

import pytest

from pion.config import ConfigError, PionConfig, ProfileConfig, load_config, save_config


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

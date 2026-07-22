"""Tests for the built-in model registry and key resolution."""

from __future__ import annotations

import pytest

from pion.llm import registry
from pion.llm.types import Model


def test_get_model_openai_api_model() -> None:
    model = registry.get_model("deepseek-chat")
    assert model.id == "deepseek-chat"
    assert model.api == "openai-completions"
    assert model.provider == "deepseek"
    assert model.base_url == "https://api.deepseek.com"
    assert model.context_window == 64000
    assert model.max_tokens == 8192

    reasoner = registry.get_model("deepseek-reasoner")
    assert reasoner.reasoning is True


def test_get_model_anthropic_api_model() -> None:
    model = registry.get_model("claude-sonnet-4-5")
    assert model.api == "anthropic-messages"
    assert model.provider == "anthropic"
    assert model.base_url == "https://api.anthropic.com"
    assert model.context_window == 200000
    assert model.reasoning is True

    opus = registry.get_model("claude-opus-4-1")
    assert opus.api == "anthropic-messages"


def test_get_model_other_providers() -> None:
    assert registry.get_model("kimi-k2-0905-preview").base_url == "https://api.moonshot.cn/v1"
    assert registry.get_model("glm-4.6").base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert (
        registry.get_model("qwen3-max").base_url
        == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def test_get_model_unknown_id_raises_with_available_ids() -> None:
    with pytest.raises(KeyError) as exc_info:
        registry.get_model("no-such-model")
    message = str(exc_info.value)
    assert "no-such-model" in message
    assert "Available" in message
    assert "deepseek-chat" in message


def test_list_models() -> None:
    ids = {model.id for model in registry.list_models()}
    for expected in (
        "deepseek-chat",
        "deepseek-reasoner",
        "kimi-k2-0905-preview",
        "glm-4.6",
        "qwen3-max",
        "claude-sonnet-4-5",
        "claude-opus-4-1",
    ):
        assert expected in ids


def test_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://proxy.example.com/v1")
    model = registry.get_model("deepseek-chat")
    assert model.base_url == "https://proxy.example.com/v1"
    # The registered entry itself must not be mutated.
    assert registry.KNOWN_MODELS["deepseek-chat"].base_url == "https://api.deepseek.com"


def test_resolve_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    assert registry.resolve_api_key(registry.get_model("deepseek-chat")) == "sk-deepseek"

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert registry.resolve_api_key(registry.get_model("deepseek-chat")) is None


def test_env_key_names_cover_registered_providers() -> None:
    for model in registry.list_models():
        assert model.provider in registry.ENV_KEY_NAMES


def test_register_model_custom_endpoint() -> None:
    custom = Model(
        id="my-local-model",
        provider="openai",
        api="openai-completions",
        baseUrl="http://localhost:8000/v1",
    )
    try:
        registry.register_model(custom)
        assert registry.get_model("my-local-model") is custom
        assert custom in registry.list_models()
    finally:
        registry._REGISTRY.pop("my-local-model", None)
    with pytest.raises(KeyError):
        registry.get_model("my-local-model")

"""Built-in model registry and API key resolution.

A small, hackable catalog of well-known models plus helpers to resolve
credentials from the environment. Custom/self-hosted endpoints can be added
at runtime with `register_model`.
"""

from __future__ import annotations

import os

from .types import Model, ModelCost

#: Environment variable holding the API key for each provider.
ENV_KEY_NAMES: dict[str, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "zhipu": "ZHIPU_API_KEY",
    "alibaba": "DASHSCOPE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# NOTE: token prices below are approximate public list prices in USD per
# million tokens, captured at the time of writing — verify against the
# providers' current pricing pages before relying on cost numbers.
KNOWN_MODELS: dict[str, Model] = {
    "deepseek-v4-flash": Model(
        id="deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        api="openai-completions",
        provider="deepseek",
        baseUrl="https://api.deepseek.com",
        contextWindow=64000,
        maxTokens=8192,
        # Approximate list prices (cache write = regular input).
        cost=ModelCost(input=0.27, output=1.10, cacheRead=0.07),
    ),
    "deepseek-v4-pro": Model(
        id="deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        api="openai-completions",
        provider="deepseek",
        baseUrl="https://api.deepseek.com",
        contextWindow=64000,
        maxTokens=8192,
        reasoning=True,
        # Approximate list prices (cache write = regular input).
        cost=ModelCost(input=0.27, output=1.10, cacheRead=0.07),
    ),
    "kimi-k2-0905-preview": Model(
        id="kimi-k2-0905-preview",
        name="Kimi K2 (0905 preview)",
        api="openai-completions",
        provider="moonshot",
        baseUrl="https://api.moonshot.cn/v1",
        contextWindow=131072,
        maxTokens=8192,
    ),
    "glm-4.6": Model(
        id="glm-4.6",
        name="GLM 4.6",
        api="openai-completions",
        provider="zhipu",
        baseUrl="https://open.bigmodel.cn/api/paas/v4",
        contextWindow=131072,
        maxTokens=8192,
    ),
    "qwen3-max": Model(
        id="qwen3-max",
        name="Qwen3 Max",
        api="openai-completions",
        provider="alibaba",
        baseUrl="https://dashscope.aliyuncs.com/compatible-mode/v1",
        contextWindow=131072,
        maxTokens=8192,
    ),
    "claude-sonnet-4-5": Model(
        id="claude-sonnet-4-5",
        name="Claude Sonnet 4.5",
        api="anthropic-messages",
        provider="anthropic",
        baseUrl="https://api.anthropic.com",
        contextWindow=200000,
        maxTokens=8192,
        reasoning=True,
        # Approximate list prices.
        cost=ModelCost(input=3.0, output=15.0, cacheRead=0.30, cacheWrite=3.75),
    ),
    "claude-opus-4-1": Model(
        id="claude-opus-4-1",
        name="Claude Opus 4.1",
        api="anthropic-messages",
        provider="anthropic",
        baseUrl="https://api.anthropic.com",
        contextWindow=200000,
        maxTokens=8192,
        reasoning=True,
        # Approximate list prices.
        cost=ModelCost(input=15.0, output=75.0, cacheRead=1.50, cacheWrite=18.75),
    ),
}

# Mutable runtime registry; starts as a copy of the built-in catalog.
_REGISTRY: dict[str, Model] = dict(KNOWN_MODELS)


def register_model(model: Model) -> None:
    """Add (or replace) a model, e.g. a custom/self-hosted endpoint."""
    _REGISTRY[model.id] = model


def list_models() -> list[Model]:
    """All registered models, built-in and custom."""
    return list(_REGISTRY.values())


def get_model(model_id: str) -> Model:
    """Look up a model by id.

    Honors a `<PROVIDER>_BASE_URL` environment override (e.g.
    `DEEPSEEK_BASE_URL`) by returning a copy of the model with the
    overridden base URL; the registered entry itself is not mutated.

    Raises:
        KeyError: if the id is unknown; the message lists available ids.
    """
    try:
        model = _REGISTRY[model_id]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown model id: {model_id!r}. Available: {available}") from None

    env_name = _base_url_env_name(model.provider)
    if env_name:
        override = os.environ.get(env_name)
        if override:
            return model.model_copy(update={"base_url": override})
    return model


def resolve_api_key(model: Model) -> str | None:
    """Resolve the API key for `model`'s provider from the environment."""
    env_name = ENV_KEY_NAMES.get(model.provider, f"{model.provider.upper()}_API_KEY")
    return os.environ.get(env_name) or None


def _base_url_env_name(provider: str) -> str | None:
    """`<PROVIDER>_BASE_URL` env var name derived from ENV_KEY_NAMES."""
    if not provider:
        return None
    key_name = ENV_KEY_NAMES.get(provider)
    if key_name and key_name.endswith("_API_KEY"):
        return key_name[: -len("_API_KEY")] + "_BASE_URL"
    return f"{provider.upper()}_BASE_URL"

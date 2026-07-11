"""Chat-model factory - build a KeyPool-wrapped chat model from Settings."""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from pydantic import SecretStr

from clawagent.api_pool import KeyPoolChatModel, get_global_pool
from clawagent.config import Settings
from clawagent.platforms import PLATFORMS

# Dynamic fallback: maps model_provider -> env var name.
# Multiple platforms share model_provider="openai"; the "openai" platform
# entry must come last in PLATFORMS to yield the canonical OPENAI_API_KEY.
_PROVIDER_KEY_ENV: dict[str, str] = {
    preset.model_provider: preset.api_key_env
    for preset in PLATFORMS.values()
}


def _get_api_key(settings: Settings) -> str:
    """Get API key for the current model provider from environment.

    When ``settings.platform`` is set, the platform preset's key env var
    takes priority. Otherwise falls back to the provider-key-env map,
    then to ``settings.anthropic_api_key``.
    """
    preset = PLATFORMS.get(settings.platform)
    if preset:
        key = os.getenv(preset.api_key_env, "")
        if key:
            return key

    env_var = _PROVIDER_KEY_ENV.get(settings.model_provider)
    if env_var:
        key = os.getenv(env_var, "")
        if key:
            return key
    return settings.api_key


def make_model(settings: Settings) -> Any:
    """Build a chat model via init_chat_model, with optional key-pool wrapping.

    When ``settings.platform`` is set, the platform preset overrides
    ``model_provider`` and ``api_base``. When ``settings.api_base`` is
    set directly (without a platform), it takes effect as a custom
    base URL for the configured provider.
    """
    preset = PLATFORMS.get(settings.platform)
    if preset:
        provider: str | None = preset.model_provider
        api_base: str = settings.api_base or preset.api_base
    else:
        provider = settings.model_provider or None
        api_base = settings.api_base

    kwargs: dict[str, Any] = {
        "api_key": SecretStr(_get_api_key(settings)),
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "timeout": settings.request_timeout,
    }
    if api_base:
        kwargs["base_url"] = api_base

    model = init_chat_model(
        model=settings.model_name,
        model_provider=provider,
        **kwargs,
    )

    pool = get_global_pool()
    default_stats = pool.get_pool_stats("default")
    if default_stats.get("total", 0) > 0:
        model = KeyPoolChatModel(
            pool=pool,
            pool_name="default",
            inner=model,
        )

    return model

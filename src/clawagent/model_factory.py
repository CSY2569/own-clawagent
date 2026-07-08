"""Chat-model factory — build a KeyPool-wrapped chat model from Settings."""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from pydantic import SecretStr

from clawagent.api_pool import KeyPoolChatModel, get_global_pool
from clawagent.config import Settings

_PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def _get_api_key(settings: Settings) -> str:
    """Get API key for the current model provider from environment."""
    env_var = _PROVIDER_KEY_ENV.get(settings.model_provider)
    if env_var:
        key = os.getenv(env_var, "")
        if key:
            return key
    return settings.anthropic_api_key


def make_model(settings: Settings) -> Any:
    """Build a chat model via init_chat_model, with optional key-pool wrapping."""
    model = init_chat_model(
        model=settings.model_name,
        model_provider=settings.model_provider or None,
        api_key=SecretStr(_get_api_key(settings)),
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
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

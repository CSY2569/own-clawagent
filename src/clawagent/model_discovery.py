"""Model discovery - fetch available models from platform APIs.

Tries the platform's /models endpoint first. If unavailable (network
error, auth failure, no endpoint), falls back to the platform preset's
``fallback_models`` list. Results are cached with a 5-minute TTL.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from clawagent.config import Settings
from clawagent.platforms import PLATFORMS, PlatformPreset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """A single model returned by a platform's model discovery API."""

    id: str
    owned_by: str = ""
    context_length: int | None = None


_CACHE_TTL: float = 300.0
_model_cache: dict[str, tuple[list[ModelInfo], float]] = {}

_PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "ark": "ARK_API_KEY",
    "opencode-go": "OPENCODE_GO_API_KEY",
}


def _get_platform_config(settings: Settings) -> tuple[str, str, str, PlatformPreset | None]:
    """Return (api_base, api_key, platform_name, preset) for the current platform."""
    preset = PLATFORMS.get(settings.platform)
    if preset:
        api_base = settings.api_base or preset.api_base
        api_key_env = preset.api_key_env
    else:
        api_base = settings.api_base
        api_key_env = _PROVIDER_KEY_ENV.get(settings.model_provider, "CLAWAGENT_API_KEY")

    api_key = os.getenv(api_key_env, "") or settings.api_key
    platform_name = settings.platform or settings.model_provider
    return api_base, api_key, platform_name, preset


def fetch_models(settings: Settings) -> list[ModelInfo]:
    """Fetch available models from the current platform's API.

    Tries GET {api_base}/models first. If that fails, falls back to
    the platform preset's fallback_models list. Results are cached
    with a 5-minute TTL.
    """
    api_base, api_key, platform_name, preset = _get_platform_config(settings)

    cached = _model_cache.get(platform_name)
    if cached and (time.monotonic() - cached[1]) < _CACHE_TTL:
        return cached[0]

    models = _try_fetch_from_api(api_base, api_key, platform_name)

    if not models and preset:
        models = [ModelInfo(id=m, owned_by=platform_name) for m in preset.fallback_models]

    if not models:
        return []

    models.sort(key=lambda m: m.id)
    _model_cache[platform_name] = (models, time.monotonic())
    return models


def _try_fetch_from_api(api_base: str, api_key: str, platform_name: str) -> list[ModelInfo]:
    """Attempt to fetch models from {api_base}/models. Returns [] on any failure."""
    if not api_base:
        return []

    url = f"{api_base.rstrip('/')}/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = httpx.get(url, headers=headers, timeout=10.0)
        if resp.status_code == 401:
            logger.warning("Auth failed fetching models from %s (401)", url)
            return []
        resp.raise_for_status()
    except httpx.TimeoutException:
        logger.warning("Timeout fetching models from %s", url)
        return []
    except Exception as e:
        logger.warning("Failed to fetch models from %s: %s", url, e)
        return []

    try:
        data: Any = resp.json()
    except Exception:
        logger.warning("Invalid JSON from %s", url)
        return []

    models_list = data.get("data", []) if isinstance(data, dict) else []
    models: list[ModelInfo] = []
    for item in models_list:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id", "")
        if not model_id:
            continue
        models.append(ModelInfo(
            id=model_id,
            owned_by=item.get("owned_by", ""),
            context_length=item.get("context_length") or item.get("max_context_length"),
        ))
    return models


def invalidate_cache(platform: str | None = None) -> None:
    """Clear the model cache for a specific platform, or all if None."""
    if platform:
        _model_cache.pop(platform, None)
    else:
        _model_cache.clear()

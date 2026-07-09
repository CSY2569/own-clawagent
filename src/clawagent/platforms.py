"""Platform presets for multi-provider model support.

Each preset bundles the provider type, API base URL, env var name
for the API key, and a fallback model list (used when the platform's
/models endpoint is unavailable).

Supported platforms:
    - deepseek:     OpenAI-compatible, https://api.deepseek.com/v1
    - ark:          OpenAI-compatible, https://ark.cn-beijing.volces.com/api/v3
    - opencode-go:  OpenAI-compatible, https://opencode.ai/zen/go/v1
    - openai:       OpenAI native
    - anthropic:    Anthropic native
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlatformPreset:
    """Bundle of provider + endpoint + key env var + fallback models."""

    model_provider: str
    api_base: str
    api_key_env: str
    display_name: str
    fallback_models: list[str] = field(default_factory=list)


PLATFORMS: dict[str, PlatformPreset] = {
    "deepseek": PlatformPreset(
        model_provider="openai",
        api_base="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        display_name="DeepSeek",
        fallback_models=["deepseek-chat", "deepseek-reasoner", "deepseek-v4-flash", "deepseek-v4-pro"],
    ),
    "ark": PlatformPreset(
        model_provider="openai",
        api_base="https://ark.cn-beijing.volces.com/api/v3",
        api_key_env="ARK_API_KEY",
        display_name="火山方舟 (Volcano Ark)",
        fallback_models=[
            "deepseek-v4-pro-260425",
            "deepseek-v4-flash-260425",
            "doubao-seed-2-0-pro-260215",
            "glm-5-2-260617",
        ],
    ),
    "opencode-go": PlatformPreset(
        model_provider="openai",
        api_base="https://opencode.ai/zen/go/v1",
        api_key_env="OPENCODE_GO_API_KEY",
        display_name="OpenCode Go",
        fallback_models=["glm-5.2", "kimi-k2.7-code", "deepseek-v4-pro", "deepseek-v4-flash"],
    ),
    "openai": PlatformPreset(
        model_provider="openai",
        api_base="",
        api_key_env="OPENAI_API_KEY",
        display_name="OpenAI",
        fallback_models=["gpt-4o", "gpt-4o-mini", "o3-mini"],
    ),
    "anthropic": PlatformPreset(
        model_provider="anthropic",
        api_base="",
        api_key_env="ANTHROPIC_API_KEY",
        display_name="Anthropic",
        fallback_models=["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
    ),
}

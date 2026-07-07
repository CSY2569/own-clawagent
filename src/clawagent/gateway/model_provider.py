"""Model configuration for per-channel model selection.

Reserved interface — Phase 1-4 uses the global Settings model for all channels.
When Direction 2 (Multi-Model Provider Abstraction) is implemented,
these configs will allow each Channel to specify its own model/provider/API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawagent.config import Settings


@dataclass
class ModelConfig:
    """Per-channel model configuration.

    All fields are optional — None means "use the global Settings default".
    When Direction 2 is implemented, this config overrides the global model
    settings for a specific Channel via ``to_settings()``.

    Environment variable format::

        CHANNEL_MODEL_WECHAT=deepseek-v4-pro|anthropic|sk-xxx|https://...|0.5|4096
        CHANNEL_MODEL_CLI=deepseek-v4-flash|anthropic||||

    Fields: model_name | model_provider | api_key | api_base | temperature | max_tokens
    """

    model_name: str | None = None
    model_provider: str | None = None  # "anthropic", "openai", etc.
    api_key: str | None = None
    api_base: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    request_timeout: int | None = None

    def to_settings(self, base: Settings) -> Settings:
        """Merge this config over a base Settings, producing a new Settings.

        Only non-None fields override the base.  Returns a new frozen Settings
        instance suitable for ``create_agent()``.
        """
        from clawagent.config import Settings

        return Settings(
            anthropic_api_key=self.api_key or base.anthropic_api_key,
            model_name=self.model_name or base.model_name,
            model_provider=self.model_provider or base.model_provider,
            max_tokens=self.max_tokens if self.max_tokens is not None else base.max_tokens,
            temperature=(
                self.temperature if self.temperature is not None else base.temperature
            ),
            context_window=base.context_window,
            memory_db_path=base.memory_db_path,
            max_preferences=base.max_preferences,
            agent_id=base.agent_id,
            siliconflow_api_key=base.siliconflow_api_key,
            siliconflow_base_url=base.siliconflow_base_url,
            siliconflow_model=base.siliconflow_model,
            siliconflow_dimensions=base.siliconflow_dimensions,
            compression_strategy=base.compression_strategy,
            compression_max_messages=base.compression_max_messages,
            compression_max_tokens=base.compression_max_tokens,
            compression_keep_recent=base.compression_keep_recent,
            compression_summary_timeout=base.compression_summary_timeout,
            request_timeout=self.request_timeout or base.request_timeout,
        )

    @classmethod
    def from_env_line(cls, value: str) -> ModelConfig:
        """Parse a pipe-delimited env-var value into a ModelConfig.

        Format:``model_name|provider|api_key|api_base|temperature|max_tokens``
        """
        parts = [p.strip() for p in value.split("|")]
        kwargs: dict[str, str | float | int | None] = {}
        field_names = [
            "model_name",
            "model_provider",
            "api_key",
            "api_base",
            "temperature",
            "max_tokens",
        ]
        for i, name in enumerate(field_names):
            if i >= len(parts) or not parts[i]:
                kwargs[name] = None
                continue
            if name == "temperature":
                kwargs[name] = float(parts[i])
            elif name == "max_tokens":
                kwargs[name] = int(parts[i])
            else:
                kwargs[name] = parts[i]
        return cls(**kwargs)  # type: ignore[arg-type]

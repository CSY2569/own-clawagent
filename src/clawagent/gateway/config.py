"""Gateway configuration — platform credentials and session settings.

Loaded from environment variables. Each platform (WeChat, QQ, Feishu)
has its own config block. When a platform is not configured (all fields
empty), its channel is not started.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from clawagent.gateway.model_provider import ModelConfig


@dataclass
class WechatConfig:
    """WeChat iLink Bot configuration (personal WeChat).

    Uses the official iLink protocol via ``weixin-ilink`` SDK.
    QR code login on first run; credentials persisted to disk.

    The old ``token`` / ``app_id`` fields are kept for backward
    compatibility but unused in iLink mode.
    """

    # iLink mode (current)
    ilink_enabled: bool = False
    ilink_creds_file: str = "ilink_creds.json"

    # Official Account mode (legacy, unused)
    token: str = ""
    app_id: str = ""
    app_secret: str = ""
    encoding_aes_key: str = ""

    @property
    def configured(self) -> bool:
        return self.ilink_enabled

    @classmethod
    def from_env(cls) -> WechatConfig:
        return cls(
            ilink_enabled=os.getenv("WECHAT_ILINK_ENABLED", "false").lower() == "true",
            ilink_creds_file=os.getenv("WECHAT_ILINK_CREDS", "ilink_creds.json"),
            # Legacy fields kept for env compatibility
            token=os.getenv("WECHAT_TOKEN", ""),
            app_id=os.getenv("WECHAT_APP_ID", ""),
            app_secret=os.getenv("WECHAT_APP_SECRET", ""),
            encoding_aes_key=os.getenv("WECHAT_ENCODING_AES_KEY", ""),
        )


@dataclass
class QQConfig:
    """QQ bot configuration (OneBot v11 / go-cqhttp)."""

    host: str = "127.0.0.1"
    ws_port: int = 8080
    access_token: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.host)

    @classmethod
    def from_env(cls) -> QQConfig:
        return cls(
            host=os.getenv("QQ_BOT_HOST", "127.0.0.1"),
            ws_port=int(os.getenv("QQ_BOT_WS_PORT", "8080")),
            access_token=os.getenv("QQ_BOT_ACCESS_TOKEN", ""),
        )


@dataclass
class FeishuConfig:
    """Feishu / Lark application configuration."""

    app_id: str = ""
    app_secret: str = ""
    verification_token: str = ""
    encrypt_key: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.app_id and self.app_secret)

    @classmethod
    def from_env(cls) -> FeishuConfig:
        return cls(
            app_id=os.getenv("FEISHU_APP_ID", ""),
            app_secret=os.getenv("FEISHU_APP_SECRET", ""),
            verification_token=os.getenv("FEISHU_VERIFICATION_TOKEN", ""),
            encrypt_key=os.getenv("FEISHU_ENCRYPT_KEY", ""),
        )


def _load_channel_models() -> dict[str, ModelConfig]:
    """Parse CHANNEL_MODEL_* env vars into per-channel ModelConfig entries.

    Format::

        CHANNEL_MODEL_WECHAT=deepseek-v4-pro|anthropic|sk-xxx|https://...|0.5|4096

    Fields are pipe-delimited: model|provider|api_key|api_base|temp|max_tokens.
    Empty fields are treated as None (use global default).
    """
    models: dict[str, ModelConfig] = {}
    prefix = "CHANNEL_MODEL_"
    for key, value in os.environ.items():
        if not key.startswith(prefix) or not value.strip():
            continue
        channel_name = key[len(prefix):].lower()
        models[channel_name] = ModelConfig.from_env_line(value)
    return models


@dataclass
class GatewayConfig:
    """Gateway global configuration — loaded from environment variables."""

    wechat: WechatConfig = field(default_factory=WechatConfig)
    qq: QQConfig = field(default_factory=QQConfig)
    feishu: FeishuConfig = field(default_factory=FeishuConfig)
    enable_cli: bool = True
    session_max: int = 50
    session_ttl: int = 1800
    # Reserved: per-channel model configs (Direction 2)
    channel_models: dict[str, ModelConfig] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> GatewayConfig:
        return cls(
            wechat=WechatConfig.from_env(),
            qq=QQConfig.from_env(),
            feishu=FeishuConfig.from_env(),
            enable_cli=os.getenv("GATEWAY_ENABLE_CLI", "true").lower() == "true",
            session_max=int(os.getenv("GATEWAY_SESSION_MAX", "50")),
            session_ttl=int(os.getenv("GATEWAY_SESSION_TTL", "1800")),
            channel_models=_load_channel_models(),
        )

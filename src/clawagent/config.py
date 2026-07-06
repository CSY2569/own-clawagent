"""Configuration and environment loading for clawagent."""

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root directory (own-clawagent/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_env() -> None:
    """Load .env from project root."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


_load_env()


@dataclass(frozen=True)
class Settings:
    """Application settings sourced from environment variables."""

    anthropic_api_key: str
    model_name: str = "deepseek-v4-flash"
    model_provider: str = "anthropic"
    max_tokens: int = 4096
    temperature: float = 0.0
    context_window: int = 1_000_000
    memory_db_path: str = "memories/sessions.db"
    max_preferences: int = 5
    agent_id: str = "wenbao"
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1/embeddings"
    siliconflow_model: str = "Qwen/Qwen3-VL-Embedding-8B"
    siliconflow_dimensions: int = 768
    compression_strategy: str = "trim"
    compression_max_messages: int = 40
    compression_max_tokens: int = 80_000
    compression_keep_recent: int = 6
    compression_summary_timeout: int = 30
    request_timeout: int = 120

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )
        context_window_raw = os.getenv("CLAWAGENT_CONTEXT_WINDOW", "1000000")
        try:
            context_window = int(context_window_raw)
        except ValueError:
            context_window = 1_000_000
        return cls(
            anthropic_api_key=api_key,
            model_name=os.getenv("CLAWAGENT_MODEL", "deepseek-v4-flash"),
            model_provider=os.getenv("CLAWAGENT_MODEL_PROVIDER", "anthropic"),
            context_window=context_window,
            memory_db_path=os.getenv("CLAWAGENT_MEMORY_DB", "memories/sessions.db"),
            max_preferences=int(os.getenv("CLAWAGENT_MAX_PREFERENCES", "5")),
            agent_id=os.getenv("CLAWAGENT_AGENT_ID", "wenbao"),
            siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            siliconflow_base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1/embeddings"),
            siliconflow_model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-VL-Embedding-8B"),
            siliconflow_dimensions=int(os.getenv("SILICONFLOW_DIMENSIONS", "768")),
            compression_strategy=os.getenv("COMPRESSION_STRATEGY", "trim"),
            compression_max_messages=int(os.getenv("COMPRESSION_MAX_MESSAGES", "40")),
            compression_max_tokens=int(os.getenv("COMPRESSION_MAX_TOKENS", "80000")),
            compression_keep_recent=int(os.getenv("COMPRESSION_KEEP_RECENT", "6")),
            compression_summary_timeout=int(os.getenv("COMPRESSION_SUMMARY_TIMEOUT", "30")),
            request_timeout=int(os.getenv("CLAWAGENT_REQUEST_TIMEOUT", "120")),
        )


@dataclass
class PriceConfig:
    """Per-model pricing (CNY per 1M tokens)."""

    input_per_1m: float = 0.0
    cache_hit_per_1m: float = 0.0
    output_per_1m: float = 0.0


@dataclass
class PriceBook:
    """Price lookup keyed by model name."""

    models: dict[str, PriceConfig] = field(default_factory=dict)

    def get(self, model_name: str) -> PriceConfig:
        return self.models.get(model_name, PriceConfig())


def load_price_book(price_path: str | Path | None = None) -> PriceBook:
    """Parse price config and return a PriceBook.

    Prefers price.toml (TOML format). Falls back to price.txt (legacy).
    """
    if price_path is None:
        toml_path = PROJECT_ROOT / "price.toml"
        txt_path = PROJECT_ROOT / "price.txt"
    else:
        price_path = Path(price_path)
        if price_path.suffix == ".toml":
            toml_path = price_path
            txt_path = price_path.with_suffix(".txt")
        else:
            toml_path = price_path.with_suffix(".toml")
            txt_path = price_path

    if toml_path.exists():
        return _load_price_toml(toml_path)

    if txt_path.exists():
        return _load_price_txt(txt_path)

    return PriceBook()


def _load_price_toml(path: Path) -> PriceBook:
    """Parse TOML-format price config."""
    data = tomllib.loads(path.read_text("utf-8"))
    models: dict[str, PriceConfig] = {}
    for name, prices in data.items():
        models[name] = PriceConfig(
            input_per_1m=float(prices.get("input_per_1m", 0)),
            cache_hit_per_1m=float(prices.get("cache_hit_per_1m", 0)),
            output_per_1m=float(prices.get("output_per_1m", 0)),
        )
    return PriceBook(models=models)


def _load_price_txt(path: Path) -> PriceBook:
    """Parse legacy price.txt format (Chinese table from DeepSeek docs)."""
    text = path.read_text(encoding="utf-8")

    model_match = re.search(
        r"^\s*模型\s+(deepseek[-_]\S+)\s+(deepseek[-_]\S+)",
        text,
        re.MULTILINE,
    )
    if not model_match:
        return PriceBook()

    model_a, model_b = model_match.group(1), model_match.group(2)

    cache_hit = _extract_price(text, r"缓存命中[）)]?\s*([\d.]+)元\s+([\d.]+)元")
    cache_miss = _extract_price(text, r"缓存未命中[）)]?\s*([\d.]+)元\s+([\d.]+)元")
    output = _extract_price(text, r"百万tokens输出\s+([\d.]+)元\s+([\d.]+)元")

    if len(cache_hit) < 2 or len(cache_miss) < 2 or len(output) < 2:
        return PriceBook()

    return PriceBook(models={
        model_a: PriceConfig(
            input_per_1m=cache_miss[0],
            cache_hit_per_1m=cache_hit[0],
            output_per_1m=output[0],
        ),
        model_b: PriceConfig(
            input_per_1m=cache_miss[1],
            cache_hit_per_1m=cache_hit[1],
            output_per_1m=output[1],
        ),
    })


def _extract_price(text: str, pattern: str) -> list[float]:
    """Extract price numbers matching a pattern, returning them in table order.

    Supports multi-group patterns — values from all groups are flattened.
    """
    matches = re.findall(pattern, text)
    return [
        float(x) for m in matches for x in (m if isinstance(m, tuple) else (m,))
    ]

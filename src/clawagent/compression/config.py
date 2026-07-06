"""Compression configuration."""

from dataclasses import dataclass


@dataclass
class CompressionConfig:
    """Context compression configuration.

    Attributes:
        strategy:       Compression strategy ("trim", "token_trim", "summarize")
        max_messages:   Level 1: maximum messages to keep
        max_tokens:     Level 2: token threshold before trimming
        keep_recent:    Level 3: most recent raw messages to preserve
        summary_timeout: Timeout in seconds for LLM-based summarization
    """

    strategy: str = "trim"
    max_messages: int = 40
    max_tokens: int = 80_000
    keep_recent: int = 6
    summary_timeout: int = 30


def load_compression_config() -> CompressionConfig:
    """Load compression configuration from environment variables."""
    import os

    return CompressionConfig(
        strategy=os.getenv("COMPRESSION_STRATEGY", "trim"),
        max_messages=int(os.getenv("COMPRESSION_MAX_MESSAGES", "40")),
        max_tokens=int(os.getenv("COMPRESSION_MAX_TOKENS", "80000")),
        keep_recent=int(os.getenv("COMPRESSION_KEEP_RECENT", "6")),
    )

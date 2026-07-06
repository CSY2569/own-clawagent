"""Runtime context compression for LangGraph agents."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import BaseMessage

from clawagent.compression.config import CompressionConfig, load_compression_config
from clawagent.compression.strategies import summarize_by_llm, trim_by_count, trim_by_tokens

__all__ = [
    "CompressionConfig",
    "compress_state",
    "load_compression_config",
    "make_state_modifier",
    "summarize_by_llm",
    "trim_by_count",
    "trim_by_tokens",
]


def compress_state(
    messages: list[BaseMessage],
    config: CompressionConfig | None = None,
    model: Any = None,
) -> list[BaseMessage]:
    """Unified compression entry point.

    Selects strategy based on config.strategy.
    """
    if config is None:
        config = load_compression_config()

    if config.strategy == "trim":
        return trim_by_count(messages, config.max_messages)
    elif config.strategy == "token_trim":
        return trim_by_tokens(messages, config.max_tokens)
    elif config.strategy == "summarize":
        if model is None:
            raise ValueError("summarize strategy requires an LLM model reference")
        return summarize_by_llm(
            messages, model, config.max_messages, config.keep_recent,
            timeout=config.summary_timeout,
        )
    else:
        raise ValueError(f"Unknown compression strategy: {config.strategy}")


def make_state_modifier(
    config: CompressionConfig | None = None,
    model: Any = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a pre_model_hook function for create_react_agent.

    Usage:
        agent = create_react_agent(
            model=model,
            tools=tools,
            pre_model_hook=make_state_modifier(model=model),
            ...
        )
    """
    if config is None:
        config = load_compression_config()

    def modifier(state: dict[str, Any]) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        trimmed = compress_state(messages, config, model)
        return {"messages": trimmed}

    return modifier

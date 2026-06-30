"""TokenCounter callback for precise per-request token tracking."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class TokenCounter(BaseCallbackHandler):
    """Records the exact token usage for a single LLM call.

    Usage:
        counter = TokenCounter()
        model.invoke("hello", callbacks=[counter])
        # counter.input_tokens, counter.output_tokens now contain values
    """

    def __init__(self) -> None:
        self.input_tokens: int = 0
        self.output_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if not response.generations:
            return
        gen_list = response.generations[0]
        if not gen_list:
            return

        info = gen_list[0].generation_info or {}
        usage: dict[str, Any] = info.get("usage", {}) or info.get("token_usage", {}) or {}
        self.input_tokens = usage.get("input_tokens", 0)
        self.output_tokens = usage.get("output_tokens", 0)

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0


def _is_retryable_error(exc: Exception) -> tuple[bool, int]:
    """Check if an exception indicates a retryable API error.

    Returns (retryable, status_code).
    """
    msg = str(exc).lower()
    if "429" in msg or "rate" in msg or "too many" in msg:
        return True, 429
    if "401" in msg or "unauthorized" in msg or "invalid api key" in msg:
        return True, 401
    if "500" in msg or "502" in msg or "503" in msg or "server error" in msg:
        return True, 503
    return False, 0

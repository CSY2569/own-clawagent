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

    Returns (retryable, status_code). Prefers structured attributes
    (status_code / response.status_code) over string matching, which
    is ambiguous (e.g. "error 42942" falsely matches "429").
    """
    code = _extract_status_code(exc)
    if code is not None:
        if code == 429:
            return True, 429
        if code == 401:
            return True, 401
        if code in (500, 502, 503, 504):
            return True, 503
        return False, 0

    return _match_by_message(exc)


def _extract_status_code(exc: Exception) -> int | None:
    """Try to read status_code from structured exception attributes."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code

    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code

    return None


def _match_by_message(exc: Exception) -> tuple[bool, int]:
    """Fallback: match by semantic keywords, not bare numbers.

    Bare-number matching is ambiguous: "error 42942" would falsely match
    "429". Keyword matching avoids this while still catching common
    provider error messages.
    """
    msg = str(exc).lower()
    if "rate" in msg or "too many" in msg:
        return True, 429
    if "unauthorized" in msg or "invalid api key" in msg:
        return True, 401
    if "server error" in msg or "bad gateway" in msg or "service unavailable" in msg:
        return True, 503
    return False, 0

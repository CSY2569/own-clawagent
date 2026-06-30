"""KeyPoolChatModel — transparent BaseChatModel wrapper for API key pooling."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from clawagent.api_pool.callbacks import TokenCounter, _is_retryable_error
from clawagent.api_pool.pool import ApiKeyPool
from clawagent.api_pool.transport import KeyPoolTransport


def _inject_key(inner: Any, api_key: str, api_base: str = "") -> None:
    """Inject an API key into the inner ChatAnthropic / Anthropic client."""
    if hasattr(inner, "anthropic_api_key"):
        inner.anthropic_api_key = api_key
    if api_base and hasattr(inner, "anthropic_api_url"):
        inner.anthropic_api_url = api_base


class KeyPoolChatModel(BaseChatModel):
    """Transparent wrapper: injects a key from the pool before each LLM call.

    On 429/401 errors, automatically switches to the next available key and retries.
    Tracks per-key token usage via TokenCounter callback.

    The inner model must be a ChatAnthropic instance (Anthropic provider only).
    """

    model_config: ClassVar[dict[str, Any]] = {"arbitrary_types_allowed": True}

    pool: ApiKeyPool
    pool_name: str
    inner: Any

    def _get_key_or_raise(self, last_error: Exception | None = None) -> Any:
        key = self.pool.get_key(self.pool_name)
        if key is None:
            msg = f"[KeyPool] pool '{self.pool_name}' has no available keys"
            raise RuntimeError(msg) from last_error
        return key

    # ── Core: _generate ──────────────────────────────────────────

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_error: Exception | None = None

        for attempt in range(4):
            key = self._get_key_or_raise(last_error)
            _inject_key(self.inner, key.api_key, key.api_base)
            self._inject_transport(key)

            counter = TokenCounter()
            try:
                result = self.inner._generate(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    callbacks=[counter],
                    **kwargs,
                )
                self.pool.record_usage(key, counter.input_tokens, counter.output_tokens)
                self.pool.mark_success(key)
                return result
            except Exception as e:
                last_error = e
                retryable, status_code = _is_retryable_error(e)
                if retryable and attempt < 3:
                    self.pool.mark_error(key, status_code)
                    continue
                raise

        raise RuntimeError(
            f"[KeyPool] pool '{self.pool_name}' all keys exhausted"
        ) from last_error

    # ── Streaming: _stream ───────────────────────────────────────

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        last_error: Exception | None = None

        for attempt in range(4):
            key = self._get_key_or_raise(last_error)
            _inject_key(self.inner, key.api_key, key.api_base)
            self._inject_transport(key)

            counter = TokenCounter()
            try:
                chunks: list[ChatGenerationChunk] = []
                for chunk in self.inner._stream(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    callbacks=[counter],
                    **kwargs,
                ):
                    chunks.append(chunk)
                    yield chunk
                self.pool.record_usage(key, counter.input_tokens, counter.output_tokens)
                self.pool.mark_success(key)
                return
            except Exception as e:
                last_error = e
                retryable, status_code = _is_retryable_error(e)
                if retryable and attempt < 3:
                    self.pool.mark_error(key, status_code)
                    continue
                raise

        raise RuntimeError(
            f"[KeyPool] pool '{self.pool_name}' all keys exhausted during streaming"
        ) from last_error

    # ── Transport injection (best-effort) ────────────────────────

    def _inject_transport(self, key: Any) -> None:
        """Inject KeyPoolTransport into the inner Anthropic client's httpx chain.

        This is best-effort: if the internal client structure doesn't match
        expectations, the wrapper still works (key injection alone handles errors).
        """
        try:
            anthropic_client = getattr(self.inner, "_client", None)
            if anthropic_client is None:
                return
            http_client = getattr(anthropic_client, "_client", None)
            if http_client is None:
                return
            current_transport = getattr(http_client, "_transport", None)
            if current_transport is None:
                return
            # Only inject once
            if isinstance(current_transport, KeyPoolTransport):
                current_transport.set_key(key.api_key)
                return
            kt = KeyPoolTransport(current_transport, self.pool, self.pool_name)
            kt.set_key(key.api_key)
            kt._current_key_record = key
            http_client._transport = kt
        except Exception:
            pass  # Best-effort

    # ── Required BaseChatModel properties ────────────────────────

    @property
    def _llm_type(self) -> str:
        return f"keypool_{self.inner._llm_type}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "pool_name": self.pool_name,
            "inner_type": self.inner._llm_type,
            "inner_params": self.inner._identifying_params,
        }

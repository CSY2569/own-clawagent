"""httpx Transport interceptor — catches 429/401 at the HTTP layer and retries with a new key."""

from __future__ import annotations

from typing import Any

import httpx


class KeyPoolTransport(httpx.BaseTransport):
    """Wraps an httpx transport, intercepting 429/401 responses to switch API keys.

    On 429 (rate limit) or 401 (auth failure), this transport:
    1. Notifies the pool to mark the current key as errored
    2. Gets the next available key from the pool
    3. Updates the request's x-api-key header
    4. Retries the request (up to 3 times)

    The pool reference and current key are injected via closure from the
    KeyPoolChatModel wrapper, keeping the transport decoupled from pool logic.
    """

    def __init__(
        self,
        next_transport: httpx.BaseTransport,
        pool: Any = None,
        pool_name: str = "default",
        max_retries: int = 3,
    ) -> None:
        self._next = next_transport
        self._pool = pool
        self._pool_name = pool_name
        self._max_retries = max_retries
        self._current_key_api_key: str | None = None
        self._current_key_record: Any = None

    def set_key(self, api_key: str) -> None:
        """Set the current key for header injection."""
        self._current_key_api_key = api_key

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        last_response: httpx.Response | None = None

        for attempt in range(self._max_retries + 1):
            # Inject current key into headers
            if self._current_key_api_key:
                request.headers["x-api-key"] = self._current_key_api_key

            response = self._next.handle_request(request)

            if response.status_code in (429, 401) and self._pool and attempt < self._max_retries:
                # Mark error on current key
                if hasattr(self, "_current_key_record") and self._current_key_record:
                    self._pool.mark_error(self._current_key_record, response.status_code)

                # Get next key
                next_key = self._pool.get_key(self._pool_name)
                if next_key is None:
                    return response

                self._current_key_api_key = next_key.api_key
                self._current_key_record = next_key

                # Re-read the request body for retry (stream was consumed)
                if hasattr(request, "read"):
                    body = request.read()
                    request = httpx.Request(
                        method=request.method,
                        url=request.url,
                        headers=dict(request.headers),
                        content=body,
                    )

                last_response = response
                continue

            return response

        return last_response or response

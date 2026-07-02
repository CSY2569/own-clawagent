"""CancelToken — cooperative cancellation via SIGINT flag."""

from __future__ import annotations

import signal
from types import FrameType


class CancelToken:
    """Cooperative cancellation context manager driven by SIGINT.

    Usage:
        with CancelToken() as cancel:
            for chunk in stream:
                cancel.check()   # raises KeyboardInterrupt if SIGINT received
                process(chunk)

    After the block (or interruption), the previous SIGINT handler is restored.
    """

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._old_handler: object = signal.SIG_DFL

    # ── Context manager ──────────────────────────────────────

    def __enter__(self) -> CancelToken:
        self._cancelled = False
        self._old_handler = signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, *args: object) -> None:
        signal.signal(signal.SIGINT, self._old_handler)

    # ── Signal handler (must be simple — signal-safe) ───────

    def _handler(self, signum: int, frame: FrameType | None) -> None:
        self._cancelled = True

    # ── Public API ───────────────────────────────────────────

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check(self) -> None:
        """Raise KeyboardInterrupt if SIGINT was received since entering."""
        if self._cancelled:
            raise KeyboardInterrupt()

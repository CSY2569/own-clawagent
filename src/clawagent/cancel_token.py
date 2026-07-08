"""CancelToken — cooperative cancellation via SIGINT flag."""

from __future__ import annotations

import signal
import warnings
from types import FrameType
from typing import Any, ClassVar


class CancelToken:
    """Cooperative cancellation context manager driven by SIGINT.

    Usage:
        with CancelToken() as cancel:
            for chunk in stream:
                cancel.check()   # raises KeyboardInterrupt if SIGINT received
                process(chunk)

    After the block (or interruption), the previous SIGINT handler is restored.

    Limitation: SIGINT is process-global, so only ONE CancelToken should be
    active at a time. Nested instances corrupt handler restoration — the
    inner __exit__ restores the outer's handler, leaving the outer instance
    unable to receive subsequent SIGINT. A RuntimeWarning is emitted if a
    second instance enters while another is active.
    """

    _active_count: ClassVar[int] = 0

    def __init__(self) -> None:
        self._cancelled: bool = False
        self._old_handler: Any = signal.SIG_DFL

    def __enter__(self) -> CancelToken:
        if CancelToken._active_count > 0:
            warnings.warn(
                "CancelToken already active in this process. Nested instances "
                "corrupt SIGINT handler state — only one should be active at a time.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._cancelled = False
        self._old_handler = signal.signal(signal.SIGINT, self._handler)
        CancelToken._active_count += 1
        return self

    def __exit__(self, *args: object) -> None:
        signal.signal(signal.SIGINT, self._old_handler)
        if CancelToken._active_count > 0:
            CancelToken._active_count -= 1

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

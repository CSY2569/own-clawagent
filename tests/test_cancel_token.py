"""Tests for CancelToken — cooperative cancellation via SIGINT flag."""

from __future__ import annotations

import signal

import pytest

from clawagent.cancel_token import CancelToken


class TestCancelToken:
    def test_no_cancel_normal_exit(self) -> None:
        with CancelToken() as cancel:
            assert not cancel.cancelled
            cancel.check()  # should not raise

    def test_cancel_check_raises(self) -> None:
        with CancelToken() as cancel:
            cancel._cancelled = True
            with pytest.raises(KeyboardInterrupt):
                cancel.check()

    def test_cancelled_property(self) -> None:
        with CancelToken() as cancel:
            assert not cancel.cancelled
            cancel._cancelled = True
            assert cancel.cancelled

    def test_context_manager_restores_handler(self) -> None:
        old = signal.getsignal(signal.SIGINT)
        with CancelToken():
            new = signal.getsignal(signal.SIGINT)
            assert new != old  # our handler is in place
        assert signal.getsignal(signal.SIGINT) == old  # restored

    def test_nested_instance_warns(self) -> None:
        with (
            CancelToken(),
            pytest.warns(RuntimeWarning, match="already active"),
            CancelToken(),
        ):
            pass

    def test_active_count_resets_on_exit(self) -> None:
        assert CancelToken._active_count == 0
        with CancelToken():
            assert CancelToken._active_count == 1
        assert CancelToken._active_count == 0

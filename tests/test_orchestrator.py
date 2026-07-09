"""Tests for clawagent.orchestrator.delegator — make_delegate_task closure."""

# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock

from clawagent.orchestrator.delegator import make_delegate_task
from clawagent.worker.factory import WorkerFactory


class TestDelegateUnknownRole:
    """不存在角色的处理。"""

    def test_returns_error_for_unknown(self):
        factory = MagicMock(spec=WorkerFactory)
        factory.create.side_effect = ValueError("Unconfigured worker role: bad_role")
        delegate = make_delegate_task(factory)

        result = delegate.invoke({"role": "bad_role", "task": "test"})
        assert "错误" in result
        assert "bad_role" in result


class TestDelegateSuccess:
    """成功委托。"""

    def test_returns_worker_output(self):
        factory = MagicMock(spec=WorkerFactory)

        mock_worker = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run.return_value.text = "Worker result: done!"

        factory.create.return_value = mock_worker
        mock_worker.spawn.return_value = mock_agent
        delegate = make_delegate_task(factory)

        result = delegate.invoke({"role": "coder", "task": "write code"})
        assert "Worker result: done!" in result

    def test_truncates_long_result(self):
        factory = MagicMock(spec=WorkerFactory)

        mock_worker = MagicMock()
        mock_agent = MagicMock()
        long_text = "x" * 60_000  # 超过 50K 截断阈值
        mock_agent.run.return_value.text = long_text

        factory.create.return_value = mock_worker
        mock_worker.spawn.return_value = mock_agent
        delegate = make_delegate_task(factory)

        result = delegate.invoke({"role": "coder", "task": "write"})
        assert len(result) < 55_000
        assert "截断" in result

    def test_calls_cleanup_after_success(self):
        """无论成功失败都调用 worker.cleanup()。"""
        factory = MagicMock(spec=WorkerFactory)

        mock_worker = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run.return_value.text = "ok"

        factory.create.return_value = mock_worker
        mock_worker.spawn.return_value = mock_agent
        delegate = make_delegate_task(factory)

        delegate.invoke({"role": "coder", "task": "hi"})
        mock_worker.cleanup.assert_called_once()


class TestDelegateWithRuntimeSettings:
    """运行时 settings 传递。"""

    def test_passes_current_settings_to_spawn(self):
        factory = MagicMock(spec=WorkerFactory)
        factory.get_settings.return_value = "settings_obj"

        mock_worker = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run.return_value.text = "ok"

        factory.create.return_value = mock_worker
        mock_worker.spawn.return_value = mock_agent
        delegate = make_delegate_task(factory)

        delegate.invoke({"role": "coder", "task": "test"})
        _call_args, call_kwargs = mock_worker.spawn.call_args
        assert call_kwargs.get("settings") == "settings_obj"


class TestDelegateFailure:
    """Worker 执行失败时的容错行为。"""

    def test_calls_cleanup_after_failure(self):
        """spawn 抛异常时 finally 仍调用 cleanup()。"""
        factory = MagicMock(spec=WorkerFactory)

        mock_worker = MagicMock()
        mock_worker.spawn.side_effect = RuntimeError("boom")
        factory.create.return_value = mock_worker
        delegate = make_delegate_task(factory)

        delegate.invoke({"role": "coder", "task": "bad"})
        mock_worker.cleanup.assert_called_once()

    def test_returns_error_message_on_failure(self):
        """异常信息包含在返回值中。"""
        factory = MagicMock(spec=WorkerFactory)

        mock_worker = MagicMock()
        mock_worker.spawn.side_effect = RuntimeError("connection refused")
        factory.create.return_value = mock_worker
        delegate = make_delegate_task(factory)

        result = delegate.invoke({"role": "coder", "task": "bad"})
        assert "执行失败" in result
        assert "connection refused" in result

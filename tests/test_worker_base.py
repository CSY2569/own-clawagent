"""Tests for clawagent.worker.base — BaseWorker lifecycle."""

# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock, patch

import pytest

from clawagent.config import Settings
from clawagent.worker.base import BaseWorker
from clawagent.worker.config import WorkerConfig


class _TestWorker(BaseWorker):
    """Concrete worker for testing the abstract base class."""

    def _get_tools(self):
        from clawagent.tools import read_file, write_file

        return [read_file, write_file]


@pytest.fixture
def worker_config(tmp_path):
    return WorkerConfig(
        role="test_worker",
        memory_db=str(tmp_path / "memory.db"),
    )


@pytest.fixture
def settings():
    return Settings(
        api_key="sk-test-fixture",
        model_name="deepseek-v4-pro",
        temperature=0.5,
        max_tokens=8192,
    )


@pytest.fixture
def worker(worker_config):
    return _TestWorker(worker_config)


@pytest.fixture
def _mock_deps():
    """Mock make_model + create_agent to avoid real LLM calls."""
    with patch("clawagent.worker.base.make_model") as mock_init, patch(
        "clawagent.worker.base.create_agent"
    ) as mock_graph:
        mock_init.return_value = MagicMock()
        mock_graph.return_value = MagicMock()
        yield mock_init, mock_graph


class TestSpawn:
    """BaseWorker.spawn() 的行为。"""

    def test_spawn_returns_agent(self, worker, monkeypatch, _mock_deps):
        """spawn() 返回 Agent 实例。"""
        monkeypatch.setenv("CLAWAGENT_API_KEY", "sk-test")
        agent = worker.spawn("test task")
        from clawagent.agent import Agent

        assert isinstance(agent, Agent)

    def test_spawn_passes_model_params(self, worker, settings, _mock_deps):
        mock_init, _ = _mock_deps
        worker.spawn("task", settings=settings)

        worker_settings = mock_init.call_args.args[0]
        assert worker_settings.api_key == "sk-test-fixture"
        assert worker_settings.temperature == worker.config.temperature
        assert worker_settings.max_tokens == worker.config.max_tokens

    def test_spawn_fallback_to_env(self, worker, monkeypatch, _mock_deps):
        mock_init, _ = _mock_deps
        monkeypatch.setenv("CLAWAGENT_API_KEY", "sk-env-key")
        worker.spawn("task")

        worker_settings = mock_init.call_args.args[0]
        assert worker_settings.api_key == "sk-env-key"

    def test_spawn_creates_separate_db(self, worker, monkeypatch, tmp_path, _mock_deps):
        """每次 spawn 创建独立的 SQLite 数据库文件。"""
        monkeypatch.setenv("CLAWAGENT_API_KEY", "sk-test")
        worker2 = _TestWorker(
            WorkerConfig(
                role="test_worker_2",
                memory_db=str(tmp_path / "memory2.db"),
            )
        )
        worker.spawn("task1")
        worker2.spawn("task2")
        assert (tmp_path / "memory.db").exists()
        assert (tmp_path / "memory2.db").exists()


class TestRun:
    """BaseWorker.run() — spawn → run(task) → cleanup → return text。"""

    def test_run_returns_text(self, worker, monkeypatch, _mock_deps):
        """run() 执行完整生命周期并返回结果。"""
        monkeypatch.setenv("CLAWAGENT_API_KEY", "sk-test")
        result = worker.run("test task")
        assert isinstance(result, str)

    def test_run_cleans_up_after(self, worker, monkeypatch, _mock_deps):
        """run() 返回后 worker 资源已释放。"""
        monkeypatch.setenv("CLAWAGENT_API_KEY", "sk-test")
        worker.run("task")
        assert worker._agent is None


class TestBuildPrompt:
    """BaseWorker.build_prompt() 的行为。"""

    def test_prompt_includes_current_task(self, worker):
        """prompt 末尾包含 ## Current Task 节。"""
        prompt = worker.build_prompt("review this code")
        assert "## Current Task" in prompt
        assert "review this code" in prompt

    def test_prompt_includes_role(self, worker):
        """prompt 中包含 role 标识。"""
        prompt = worker.build_prompt("task")
        assert "test_worker" in prompt


class TestCleanup:
    """BaseWorker.cleanup() 释放资源。"""

    def test_cleanup_releases_resources(self, worker, monkeypatch, _mock_deps):
        """cleanup() 后 _agent 和 _conn 均为 None。"""
        monkeypatch.setenv("CLAWAGENT_API_KEY", "sk-test")
        worker.spawn("task")
        worker.cleanup()
        assert worker._agent is None
        assert worker._conn is None

    def test_cleanup_idempotent(self, worker):
        """多次调用 cleanup() 不报错。"""
        worker.cleanup()
        worker.cleanup()  # 第二次调用不应抛异常

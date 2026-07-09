"""Tests for clawagent.worker.factory — WorkerFactory with real instances."""

# mypy: disable-error-code="no-untyped-def"

import os

import pytest

from clawagent.worker.base import BaseWorker
from clawagent.worker.factory import WorkerFactory


def _clean_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("WORKER_"):
            monkeypatch.delenv(key, raising=False)


class TestWorkerFactoryInit:
    """WorkerFactory 初始化。"""

    def test_fallback_to_builtins_when_no_vars(self, monkeypatch):
        """环境变量未设置时 available_roles 回退到 BUILTIN_WORKER_ROLES。"""
        _clean_worker_env(monkeypatch)
        factory = WorkerFactory()
        assert set(factory.available_roles) == {"coder", "researcher", "critic", "writer"}

    def test_discovers_roles_from_env(self, monkeypatch):
        """设置 WORKER_CODER_MODEL 后 available_roles 包含 coder。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_CODER_MODEL", "deepseek-v4-flash")
        factory = WorkerFactory()
        assert "coder" in factory.available_roles

    def test_set_settings(self, monkeypatch):
        """set_settings 记录运行时 Settings。"""
        _clean_worker_env(monkeypatch)
        from clawagent.config import Settings

        factory = WorkerFactory()
        settings = Settings(api_key="sk-test-settings")
        factory.set_settings(settings)
        assert factory._current_settings is settings


class TestWorkerFactoryCreate:
    """WorkerFactory.create() — 实例化真实 Worker。"""

    def test_create_returns_worker(self, monkeypatch):
        """create() 返回 BaseWorker 子类实例。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_CODER_MODEL", "m")
        factory = WorkerFactory()
        worker = factory.create("coder")
        assert isinstance(worker, BaseWorker)
        assert worker.config.role == "coder"

    def test_create_unknown_role_raises(self, monkeypatch):
        """未配置的 role 抛出 ValueError 并列出可用角色。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_CODER_MODEL", "m")
        factory = WorkerFactory()
        with pytest.raises(ValueError) as exc:
            factory.create("nonexistent")
        assert "coder" in str(exc.value)

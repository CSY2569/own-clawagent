"""Tests for clawagent.worker.registry."""

# mypy: disable-error-code="no-untyped-def, arg-type, comparison-overlap"

import pytest

from clawagent.worker.registry import WORKER_REGISTRY, get_worker_class, register_worker


@pytest.fixture(autouse=True)
def _backup_restore_registry():
    """每个测试前后备份/恢复 WORKER_REGISTRY，防止相互污染。"""
    saved = dict(WORKER_REGISTRY)
    yield
    WORKER_REGISTRY.clear()
    WORKER_REGISTRY.update(saved)


class TestRegisterWorker:
    """@register_worker 装饰器的行为。"""

    def test_register_and_get(self):
        """注册后 get_worker_class 返回正确的类。"""

        @register_worker("test_role_a")
        class _TestWorkerA:
            pass

        cls = get_worker_class("test_role_a")
        assert cls is _TestWorkerA

    def test_register_multiple_roles(self):
        """同一个类可以注册多个角色名。"""

        @register_worker("test_role_b1")
        @register_worker("test_role_b2")
        class _TestWorkerB:
            pass

        assert get_worker_class("test_role_b1") is _TestWorkerB
        assert get_worker_class("test_role_b2") is _TestWorkerB


class TestGetWorkerClass:
    """get_worker_class 的查找行为。"""

    def test_get_unknown_role_raises(self):
        """获取未注册的 role 抛出 ValueError 并列出已注册角色。"""
        WORKER_REGISTRY.clear()

        @register_worker("existing_role")
        class _Dummy:
            pass

        with pytest.raises(ValueError) as exc:
            get_worker_class("nonexistent")
        assert "existing_role" in str(exc.value)

    def test_duplicate_register_overwrites(self):
        """相同 role 重复注册，后面的类覆盖前面的。"""

        @register_worker("test_overwrite")
        class _First:
            pass

        @register_worker("test_overwrite")
        class _Second:
            pass

        assert get_worker_class("test_overwrite") is _Second


class TestAllWorkersRegistered:
    """完整模块导入后 4 个 Worker 应全部在 registry 中。"""

    def test_four_workers_present(self):
        """导入 worker 包后检查 4 个角色齐全。"""
        import clawagent.worker.coder  # noqa: F401
        import clawagent.worker.critic  # noqa: F401
        import clawagent.worker.researcher  # noqa: F401
        import clawagent.worker.writer  # noqa: F401

        for role in ("coder", "researcher", "critic", "writer"):
            cls = get_worker_class(role)
            assert cls is not None, f"Worker {role} 未注册"

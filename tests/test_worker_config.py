"""Tests for clawagent.worker.config."""

# mypy: disable-error-code="no-untyped-def"

import os
from typing import Any

import pytest

from clawagent.worker.config import WorkerConfig, load_worker_configs


def _clean_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """移除所有 WORKER_ 相关环境变量。monkeypatch 会在测试后自动恢复。"""
    for key in list(os.environ):
        if key.startswith("WORKER_"):
            monkeypatch.delenv(key, raising=False)


class TestLoadWorkerConfigs:
    """load_worker_configs 的环境变量发现逻辑。"""

    def test_fallback_to_builtins_when_no_vars(self, monkeypatch):
        """没有 WORKER_*_MODEL 时回退到 BUILTIN_WORKER_ROLES。"""
        _clean_worker_env(monkeypatch)
        configs = load_worker_configs()
        assert set(configs.keys()) == {"coder", "researcher", "critic", "writer"}

    # ── 角色发现 ─────────────────────────────────

    def test_discover_by_model(self, monkeypatch):
        """WORKER_CODER_MODEL=xxx → 角色包含 coder。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_CODER_MODEL", "deepseek-v4-flash")
        configs = load_worker_configs()
        assert "coder" in configs

    def test_discover_by_api_key(self, monkeypatch):
        """WORKER_CODER_API_KEY=xxx 也应发现角色（bug19 验证）。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_CODER_API_KEY", "sk-test-key")
        configs = load_worker_configs()
        assert "coder" in configs

    def test_discover_by_api_base(self, monkeypatch):
        """WORKER_CODER_API_BASE=xxx 也应发现角色。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_CODER_API_BASE", "https://api.test.com")
        configs = load_worker_configs()
        assert "coder" in configs

    # ── 回退逻辑 ─────────────────────────────────

    def test_role_override_common(self, monkeypatch):
        """角色专项设置覆盖 WORKER_COMMON_*。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_COMMON_MODEL", "default-model")
        monkeypatch.setenv("WORKER_CODER_MODEL", "custom-model")
        configs = load_worker_configs()
        assert configs["coder"].model == "custom-model"

    def test_common_fallback_max_tokens(self, monkeypatch):
        """只设 WORKER_COMMON_MAX_TOKENS，role 不设专项，应使用通用值。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_CODER_MODEL", "m")  # 仅触发发现
        monkeypatch.setenv("WORKER_COMMON_MAX_TOKENS", "8888")
        configs = load_worker_configs()
        assert configs["coder"].max_tokens == 8888

    def test_common_fallback_temperature(self, monkeypatch):
        """WORKER_COMMON_TEMPERATURE 作为 role 的默认值。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_CODER_MODEL", "m")
        monkeypatch.setenv("WORKER_COMMON_TEMPERATURE", "0.7")
        configs = load_worker_configs()
        assert configs["coder"].temperature == 0.7

    def test_model_provider_default(self, monkeypatch):
        """未设置 model_provider 时默认为空字符串。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_CODER_MODEL", "m")
        configs = load_worker_configs()
        assert configs["coder"].model_provider == ""

    def test_model_provider_fallback(self, monkeypatch):
        """WORKER_COMMON_MODEL_PROVIDER 回退到 role 的 model_provider。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_CODER_MODEL", "m")
        monkeypatch.setenv("WORKER_COMMON_MODEL_PROVIDER", "ollama")
        configs = load_worker_configs()
        assert configs["coder"].model_provider == "ollama"

    def test_role_name_lowercase(self, monkeypatch):
        """环境变量 CASE → 角色名全小写。"""
        _clean_worker_env(monkeypatch)
        monkeypatch.setenv("WORKER_CODER_MODEL", "m")
        configs = load_worker_configs()
        assert "coder" in configs
        assert "CODER" not in configs


class TestWorkerConfig:
    """WorkerConfig 数据类的行为。"""

    def test_default_model(self):
        """未指定 model 时使用默认值。"""
        cfg = WorkerConfig(role="test")
        assert cfg.model == "deepseek-v4-flash"

    def test_default_temperature(self):
        assert WorkerConfig(role="test").temperature == 0.0

    def test_custom_values(self):
        cfg = WorkerConfig(
            role="test",
            model="gpt-4",
            max_tokens=8192,
            temperature=0.5,
        )
        assert cfg.model == "gpt-4"
        assert cfg.max_tokens == 8192
        assert cfg.temperature == 0.5

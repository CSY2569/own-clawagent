"""Tests for clawagent.model_factory."""

# mypy: disable-error-code="no-untyped-def"

import os
from unittest.mock import MagicMock, patch

from clawagent.config import Settings
from clawagent.model_factory import _PROVIDER_KEY_ENV, _get_api_key, make_model
from clawagent.platforms import PLATFORMS


class TestProviderKeyEnv:
    def test_openai_mapped(self):
        assert _PROVIDER_KEY_ENV.get("openai") == "OPENAI_API_KEY"

    def test_anthropic_mapped(self):
        assert _PROVIDER_KEY_ENV.get("anthropic") == "ANTHROPIC_API_KEY"

    def test_derived_from_platforms(self):
        expected_keys = {p.model_provider for p in PLATFORMS.values()}
        assert set(_PROVIDER_KEY_ENV.keys()) == expected_keys


class TestGetApiKey:
    def _make_settings(self, **kwargs):
        defaults = {
            "api_key": "fallback-key",
            "model_name": "test-model",
            "model_provider": "openai",
            "platform": "deepseek",
        }
        defaults.update(kwargs)
        return Settings(**defaults)

    def test_platform_key_takes_priority(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds-key"}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            settings = self._make_settings()
            assert _get_api_key(settings) == "ds-key"

    def test_provider_key_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "oai-key"}, clear=False):
            os.environ.pop("DEEPSEEK_API_KEY", None)
            settings = self._make_settings(platform="")
            assert _get_api_key(settings) == "oai-key"

    def test_settings_api_key_final_fallback(self):
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        settings = self._make_settings(platform="")
        assert _get_api_key(settings) == "fallback-key"

    def test_anthropic_platform(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-key"}, clear=False):
            settings = self._make_settings(platform="anthropic", model_provider="anthropic")
            assert _get_api_key(settings) == "ant-key"


class TestMakeModel:
    def _make_settings(self, **kwargs):
        defaults = {
            "api_key": "test-key",
            "model_name": "test-model",
            "model_provider": "openai",
            "platform": "deepseek",
        }
        defaults.update(kwargs)
        return Settings(**defaults)

    def test_calls_init_chat_model(self):
        settings = self._make_settings()
        mock_model = MagicMock()
        with (
            patch("clawagent.model_factory.init_chat_model", return_value=mock_model) as mock_init,
            patch("clawagent.model_factory.get_global_pool") as mock_pool,
        ):
            mock_pool.return_value.get_pool_stats.return_value = {"total": 0}
            make_model(settings)
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args
        assert call_kwargs.kwargs["model"] == "test-model"

    def test_api_base_from_platform(self):
        settings = self._make_settings()
        with (
            patch("clawagent.model_factory.init_chat_model") as mock_init,
            patch("clawagent.model_factory.get_global_pool") as mock_pool,
        ):
            mock_pool.return_value.get_pool_stats.return_value = {"total": 0}
            make_model(settings)
        call_kwargs = mock_init.call_args
        assert call_kwargs.kwargs["base_url"] == "https://api.deepseek.com/v1"

    def test_custom_api_base_overrides_platform(self):
        settings = self._make_settings(api_base="https://custom.api.com/v1")
        with (
            patch("clawagent.model_factory.init_chat_model") as mock_init,
            patch("clawagent.model_factory.get_global_pool") as mock_pool,
        ):
            mock_pool.return_value.get_pool_stats.return_value = {"total": 0}
            make_model(settings)
        call_kwargs = mock_init.call_args
        assert call_kwargs.kwargs["base_url"] == "https://custom.api.com/v1"

    def test_no_platform_uses_provider(self):
        settings = self._make_settings(platform="", model_provider="openai")
        with (
            patch("clawagent.model_factory.init_chat_model") as mock_init,
            patch("clawagent.model_factory.get_global_pool") as mock_pool,
        ):
            mock_pool.return_value.get_pool_stats.return_value = {"total": 0}
            make_model(settings)
        call_kwargs = mock_init.call_args
        assert call_kwargs.kwargs["model_provider"] == "openai"

    def test_keypool_wrapping_when_pool_has_keys(self):
        settings = self._make_settings()
        mock_model = MagicMock()
        mock_wrapped = MagicMock()
        with (
            patch("clawagent.model_factory.init_chat_model", return_value=mock_model),
            patch("clawagent.model_factory.get_global_pool") as mock_pool,
            patch("clawagent.model_factory.KeyPoolChatModel", return_value=mock_wrapped),
        ):
            mock_pool.return_value.get_pool_stats.return_value = {"total": 3}
            result = make_model(settings)
            assert result is mock_wrapped

    def test_no_keypool_when_pool_empty(self):
        settings = self._make_settings()
        mock_model = MagicMock()
        with (
            patch("clawagent.model_factory.init_chat_model", return_value=mock_model),
            patch("clawagent.model_factory.get_global_pool") as mock_pool,
        ):
            mock_pool.return_value.get_pool_stats.return_value = {"total": 0}
            result = make_model(settings)
            assert result is mock_model

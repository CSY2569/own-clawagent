"""Tests for clawagent.api_pool — API key pooling and failover."""

# mypy: disable-error-code="no-untyped-def"

import os
import time
from unittest.mock import MagicMock

import httpx
import pytest

from clawagent.api_pool import (
    ApiKeyPool,
    KeyPoolChatModel,
    KeyRecord,
    KeyStatus,
    PoolConfig,
    PoolStrategy,
    get_global_pool,
    init_global_pool,
)
from clawagent.api_pool.callbacks import TokenCounter, _is_retryable_error
from clawagent.api_pool.loader import load_pools_from_env
from clawagent.api_pool.pool import ApiKeyPool as Pool
from clawagent.api_pool.transport import KeyPoolTransport

# ── Models ──────────────────────────────────────────────────────────


class TestKeyStatus:
    def test_enum_values(self):
        assert KeyStatus.ACTIVE.value == "active"
        assert KeyStatus.DEGRADED.value == "degraded"
        assert KeyStatus.EXHAUSTED.value == "exhausted"
        assert KeyStatus.DISABLED.value == "disabled"


class TestPoolStrategy:
    def test_enum_values(self):
        assert PoolStrategy.ROUND_ROBIN.value == "round_robin"
        assert PoolStrategy.RANDOM.value == "random"
        assert PoolStrategy.LEAST_ERRORS.value == "least_errors"


class TestKeyRecord:
    def test_defaults(self):
        r = KeyRecord(name="test", api_key="sk-abc")
        assert r.api_base == ""
        assert r.provider == "anthropic"
        assert r.pool_name == "default"
        assert r.status == KeyStatus.ACTIVE
        assert r.error_count == 0
        assert r.last_error_at == 0.0
        assert r.cooldown_seconds == 30.0
        assert r.total_input_tokens == 0
        assert r.total_output_tokens == 0

    def test_custom_fields(self):
        r = KeyRecord(
            name="k1", api_key="sk-xyz", api_base="https://api.example.com",
            provider="openai", pool_name="main",
        )
        assert r.api_base == "https://api.example.com"
        assert r.provider == "openai"
        assert r.pool_name == "main"


class TestPoolConfig:
    def test_defaults(self):
        cfg = PoolConfig(name="default")
        assert cfg.strategy == PoolStrategy.ROUND_ROBIN
        assert cfg.keys == []
        assert cfg.max_cooldown == 600.0

    def test_with_keys(self):
        keys = [KeyRecord(name="k1", api_key="sk-a"), KeyRecord(name="k2", api_key="sk-b")]
        cfg = PoolConfig(name="main", strategy=PoolStrategy.RANDOM, keys=keys)
        assert len(cfg.keys) == 2
        assert cfg.strategy == PoolStrategy.RANDOM


# ── Pool manager ────────────────────────────────────────────────────


class TestApiKeyPool:
    @pytest.fixture
    def pool(self):
        return Pool()

    @pytest.fixture
    def pool_with_keys(self, pool):
        pool.add_pool(PoolConfig(
            name="default",
            keys=[
                KeyRecord(name="k1", api_key="sk-aaa"),
                KeyRecord(name="k2", api_key="sk-bbb"),
                KeyRecord(name="k3", api_key="sk-ccc"),
            ],
        ))
        return pool

    # ── add_pool / get_key ──

    def test_get_key_empty_pool_returns_none(self, pool):
        assert pool.get_key("nonexistent") is None

    def test_get_key_round_robin(self, pool_with_keys):
        k1 = pool_with_keys.get_key("default")
        k2 = pool_with_keys.get_key("default")
        k3 = pool_with_keys.get_key("default")
        k4 = pool_with_keys.get_key("default")
        assert k1.name == "k1"
        assert k2.name == "k2"
        assert k3.name == "k3"
        assert k4.name == "k1"  # wraps around

    def test_get_key_random(self):
        pool = Pool()
        keys = [KeyRecord(name=f"k{i}", api_key=f"sk-{i}") for i in range(20)]
        pool.add_pool(PoolConfig(name="rnd", strategy=PoolStrategy.RANDOM, keys=keys))
        names = {pool.get_key("rnd").name for _ in range(50)}
        assert len(names) >= 2  # probabilistic: should hit multiple keys

    def test_get_key_least_errors(self):
        pool = Pool()
        k1 = KeyRecord(name="k1", api_key="sk-a", error_count=3)
        k2 = KeyRecord(name="k2", api_key="sk-b", error_count=1)
        k3 = KeyRecord(name="k3", api_key="sk-c", error_count=2)
        pool.add_pool(PoolConfig(name="le", strategy=PoolStrategy.LEAST_ERRORS, keys=[k1, k2, k3]))
        selected = pool.get_key("le")
        assert selected.name == "k2"

    def test_get_key_skips_degraded(self, pool_with_keys):
        pool_with_keys._pools["default"].keys[0].status = KeyStatus.DEGRADED
        pool_with_keys._pools["default"].keys[0].last_error_at = time.time()
        selected = pool_with_keys.get_key("default")
        assert selected.name == "k2"

    def test_get_key_skips_disabled(self, pool_with_keys):
        pool_with_keys._pools["default"].keys[0].status = KeyStatus.DISABLED
        pool_with_keys._pools["default"].keys[1].status = KeyStatus.DISABLED
        selected = pool_with_keys.get_key("default")
        assert selected.name == "k3"

    def test_get_key_all_degraded_returns_none(self, pool_with_keys):
        for k in pool_with_keys._pools["default"].keys:
            k.status = KeyStatus.DEGRADED
            k.last_error_at = time.time()
        assert pool_with_keys.get_key("default") is None

    def test_get_key_recovers_after_cooldown(self, pool_with_keys):
        k = pool_with_keys._pools["default"].keys[0]
        k.status = KeyStatus.DEGRADED
        k.last_error_at = time.time() - 999  # well past cooldown
        k.cooldown_seconds = 0.1
        selected = pool_with_keys.get_key("default")
        assert selected.name == "k1"
        assert k.status == KeyStatus.ACTIVE
        assert k.error_count == 0

    # ── mark_error ──

    def test_mark_error_429_degraded(self, pool_with_keys):
        k = pool_with_keys._pools["default"].keys[0]
        pool_with_keys.mark_error(k, 429)
        assert k.status == KeyStatus.DEGRADED
        assert k.error_count == 1
        assert k.last_error_at > 0

    def test_mark_error_401_disabled(self, pool_with_keys):
        k = pool_with_keys._pools["default"].keys[0]
        pool_with_keys.mark_error(k, 401)
        assert k.status == KeyStatus.DISABLED

    def test_mark_error_exponential_backoff(self, pool_with_keys):
        k = pool_with_keys._pools["default"].keys[0]
        initial = k.cooldown_seconds  # 30.0
        pool_with_keys.mark_error(k, 429)
        assert k.cooldown_seconds == min(initial * 2, 600.0)
        second = k.cooldown_seconds
        pool_with_keys.mark_error(k, 429)
        assert k.cooldown_seconds == min(second * 2, 600.0)

    def test_mark_error_caps_at_max_cooldown(self):
        pool = Pool()
        k = KeyRecord(name="k1", api_key="sk-a", pool_name="d")
        pool.add_pool(PoolConfig(name="d", keys=[k], max_cooldown=45.0))
        k.cooldown_seconds = 30.0
        pool.mark_error(k, 429)  # 30 * 2 = 60, capped at 45
        assert k.cooldown_seconds == 45.0

    # ── mark_success ──

    def test_mark_success_resets(self, pool_with_keys):
        k = pool_with_keys._pools["default"].keys[0]
        k.error_count = 5
        k.cooldown_seconds = 120.0
        pool_with_keys.mark_success(k)
        assert k.error_count == 0
        assert k.cooldown_seconds == 30.0

    # ── record_usage ──

    def test_record_usage(self, pool_with_keys):
        k = pool_with_keys._pools["default"].keys[0]
        pool_with_keys.record_usage(k, 100, 50)
        assert k.total_input_tokens == 100
        assert k.total_output_tokens == 50
        pool_with_keys.record_usage(k, 30, 10)
        assert k.total_input_tokens == 130
        assert k.total_output_tokens == 60

    # ── stats ──

    def test_get_pool_stats(self, pool_with_keys):
        stats = pool_with_keys.get_pool_stats("default")
        assert stats["name"] == "default"
        assert stats["total"] == 3
        assert stats["active"] == 3
        assert stats["degraded"] == 0
        assert stats["disabled"] == 0

    def test_get_pool_stats_with_mixed_status(self, pool_with_keys):
        keys = pool_with_keys._pools["default"].keys
        keys[0].status = KeyStatus.DEGRADED
        keys[1].status = KeyStatus.DISABLED
        keys[0].total_input_tokens = 100
        keys[2].total_output_tokens = 50
        stats = pool_with_keys.get_pool_stats("default")
        assert stats["active"] == 1
        assert stats["degraded"] == 1
        assert stats["disabled"] == 1
        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 50

    def test_get_pool_stats_nonexistent(self, pool):
        assert pool.get_pool_stats("nope") == {}

    def test_get_all_stats(self, pool):
        pool.add_pool(PoolConfig(name="a", keys=[KeyRecord(name="k1", api_key="sk-a")]))
        pool.add_pool(PoolConfig(name="b", keys=[KeyRecord(name="k2", api_key="sk-b")]))
        all_stats = pool.get_all_stats()
        assert set(all_stats.keys()) == {"a", "b"}

    # ── Thread safety ──

    def test_concurrent_access(self, pool_with_keys):
        import threading

        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(100):
                    key = pool_with_keys.get_key("default")
                    if key:
                        pool_with_keys.record_usage(key, 1, 1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    # ── Add pool with duplicate name overwrites ──

    def test_add_pool_overwrites(self, pool_with_keys):
        pool_with_keys.add_pool(PoolConfig(
            name="default",
            keys=[KeyRecord(name="new", api_key="sk-new")],
        ))
        stats = pool_with_keys.get_pool_stats("default")
        assert stats["total"] == 1


# ── Loader ──────────────────────────────────────────────────────────


class TestLoadPoolsFromEnv:
    def test_empty(self, monkeypatch):
        for k in list(os.environ):
            if k.startswith("API_POOL_"):
                monkeypatch.delenv(k, raising=False)
        pools = load_pools_from_env()
        assert pools == []

    def test_single_pool_two_keys(self, monkeypatch):
        monkeypatch.setenv("API_POOL_MAIN_STRATEGY", "round_robin")
        monkeypatch.setenv("API_POOL_MAIN_KEY_1", "sk-key1-xxx")
        monkeypatch.setenv("API_POOL_MAIN_KEY_2", "sk-key2-yyy")

        pools = load_pools_from_env()
        assert len(pools) == 1
        cfg = pools[0]
        assert cfg.name == "main"
        assert cfg.strategy == PoolStrategy.ROUND_ROBIN
        assert len(cfg.keys) == 2
        assert cfg.keys[0].name == "main-key1"
        assert cfg.keys[0].api_key == "sk-key1-xxx"
        assert cfg.keys[1].name == "main-key2"
        assert cfg.keys[1].api_key == "sk-key2-yyy"

    def test_with_api_base(self, monkeypatch):
        monkeypatch.setenv("API_POOL_DS_STRATEGY", "random")
        monkeypatch.setenv("API_POOL_DS_KEY_1", "sk-ds")
        monkeypatch.setenv("API_POOL_DS_KEY_1_BASE", "https://api.deepseek.com/anthropic")

        pools = load_pools_from_env()
        assert len(pools) == 1
        assert pools[0].keys[0].api_base == "https://api.deepseek.com/anthropic"

    def test_multiple_pools(self, monkeypatch):
        monkeypatch.setenv("API_POOL_A_STRATEGY", "round_robin")
        monkeypatch.setenv("API_POOL_A_KEY_1", "sk-a")
        monkeypatch.setenv("API_POOL_B_STRATEGY", "least_errors")
        monkeypatch.setenv("API_POOL_B_KEY_1", "sk-b")

        pools = load_pools_from_env()
        assert len(pools) == 2
        names = {p.name for p in pools}
        assert names == {"a", "b"}

    def test_invalid_strategy_falls_back(self, monkeypatch):
        monkeypatch.setenv("API_POOL_X_STRATEGY", "garbage")
        monkeypatch.setenv("API_POOL_X_KEY_1", "sk-x")
        pools = load_pools_from_env()
        assert pools[0].strategy == PoolStrategy.ROUND_ROBIN

    def test_skips_empty_keys(self, monkeypatch):
        monkeypatch.setenv("API_POOL_T_STRATEGY", "round_robin")
        monkeypatch.setenv("API_POOL_T_KEY_1", "sk-t")
        monkeypatch.setenv("API_POOL_T_KEY_2", "   ")  # whitespace = empty
        pools = load_pools_from_env()
        assert len(pools[0].keys) == 1

    def test_default_strategy_when_unset(self, monkeypatch):
        monkeypatch.setenv("API_POOL_DEF_KEY_1", "sk-def")
        pools = load_pools_from_env()
        assert pools[0].strategy == PoolStrategy.ROUND_ROBIN

    def test_no_pool_without_keys(self, monkeypatch):
        monkeypatch.setenv("API_POOL_EMPTY_STRATEGY", "round_robin")
        # No KEY_* vars
        pools = load_pools_from_env()
        assert all(p.name != "empty" for p in pools)


# ── TokenCounter callback ───────────────────────────────────────────


class TestTokenCounter:
    @pytest.fixture
    def _msg(self):
        from langchain_core.messages import AIMessage
        return AIMessage(content="test")

    def test_defaults_zero(self):
        tc = TokenCounter()
        assert tc.input_tokens == 0
        assert tc.output_tokens == 0

    def test_on_llm_end_extracts_usage(self, _msg):
        from langchain_core.outputs import ChatGeneration, LLMResult

        tc = TokenCounter()
        gen = ChatGeneration(
            text="hello",
            generation_info={"usage": {"input_tokens": 150, "output_tokens": 80}},
            message=_msg,
        )
        tc.on_llm_end(LLMResult(generations=[[gen]]))
        assert tc.input_tokens == 150
        assert tc.output_tokens == 80

    def test_on_llm_end_handles_token_usage_key(self, _msg):
        from langchain_core.outputs import ChatGeneration, LLMResult

        tc = TokenCounter()
        gen = ChatGeneration(
            text="hello",
            generation_info={"token_usage": {"input_tokens": 10, "output_tokens": 20}},
            message=_msg,
        )
        tc.on_llm_end(LLMResult(generations=[[gen]]))
        assert tc.input_tokens == 10
        assert tc.output_tokens == 20

    def test_on_llm_end_empty_generations(self):
        tc = TokenCounter()
        tc.on_llm_end(MagicMock(generations=[]))
        assert tc.input_tokens == 0
        assert tc.output_tokens == 0

    def test_on_llm_end_no_usage(self, _msg):
        from langchain_core.outputs import ChatGeneration, LLMResult

        tc = TokenCounter()
        gen = ChatGeneration(text="hello", generation_info={}, message=_msg)
        tc.on_llm_end(LLMResult(generations=[[gen]]))
        assert tc.input_tokens == 0
        assert tc.output_tokens == 0

    def test_reset(self):
        tc = TokenCounter()
        tc.input_tokens = 100
        tc.output_tokens = 50
        tc.reset()
        assert tc.input_tokens == 0
        assert tc.output_tokens == 0


class TestIsRetryableError:
    def test_429(self):
        retryable, code = _is_retryable_error(Exception("Error 429 rate limit exceeded"))
        assert retryable
        assert code == 429

    def test_401(self):
        retryable, code = _is_retryable_error(Exception("401 Unauthorized"))
        assert retryable
        assert code == 401

    def test_500(self):
        retryable, code = _is_retryable_error(Exception("500 Internal Server Error"))
        assert retryable
        assert code == 503

    def test_non_retryable(self):
        retryable, code = _is_retryable_error(Exception("Something else"))
        assert not retryable
        assert code == 0

    def test_by_status_code_attr(self):
        """Exception with status_code attribute — preferred over message."""
        exc = Exception("some error")
        exc.status_code = 429  # type: ignore[attr-defined]
        retryable, code = _is_retryable_error(exc)
        assert retryable
        assert code == 429

    def test_by_response_status_code(self):
        """Exception with .response.status_code — preferred over message."""
        response = MagicMock()
        response.status_code = 401
        exc = Exception("some error")
        exc.response = response  # type: ignore[attr-defined]
        retryable, code = _is_retryable_error(exc)
        assert retryable
        assert code == 401

    def test_no_false_positive_on_unrelated_number(self):
        """Message containing '42942' must NOT match 429 via string fallback
        when no status_code attribute is present."""
        exc = Exception("error code 42942 something")
        retryable, code = _is_retryable_error(exc)
        assert not retryable
        assert code == 0

    def test_504_mapped_to_503(self):
        exc = Exception("gateway timeout")
        exc.status_code = 504  # type: ignore[attr-defined]
        retryable, code = _is_retryable_error(exc)
        assert retryable
        assert code == 503


# ── Transport ───────────────────────────────────────────────────────


class TestKeyPoolTransport:
    @pytest.fixture
    def pool(self):
        p = Pool()
        p.add_pool(PoolConfig(
            name="default",
            keys=[
                KeyRecord(name="k1", api_key="sk-good", pool_name="default"),
                KeyRecord(name="k2", api_key="sk-fallback", pool_name="default"),
            ],
        ))
        return p

    @pytest.fixture
    def mock_next(self):
        return MagicMock(spec=httpx.BaseTransport)

    def _make_request(self, url="https://api.anthropic.com/v1/messages", method="POST"):
        return httpx.Request(method=method, url=url, headers={})

    def test_passthrough_success(self, pool, mock_next):
        mock_next.handle_request.return_value = httpx.Response(200, json={"ok": True})
        transport = KeyPoolTransport(mock_next, pool, "default")
        transport.set_key("sk-good")

        response = transport.handle_request(self._make_request())
        assert response.status_code == 200

    def test_injects_api_key_header(self, pool, mock_next):
        called_with: httpx.Request | None = None

        def capture(req):
            nonlocal called_with
            called_with = req
            return httpx.Response(200)

        mock_next.handle_request.side_effect = capture
        transport = KeyPoolTransport(mock_next, pool, "default")
        transport.set_key("sk-injected")

        transport.handle_request(self._make_request())
        assert called_with is not None
        assert called_with.headers.get("x-api-key") == "sk-injected"

    def test_retry_on_429_switches_key(self, pool, mock_next):
        call_count = [0]

        def handler(req):
            call_count[0] += 1
            if call_count[0] == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json={"ok": True})

        mock_next.handle_request.side_effect = handler
        transport = KeyPoolTransport(mock_next, pool, "default")
        transport.set_key("sk-good")
        transport._current_key_record = pool._pools["default"].keys[0]

        response = transport.handle_request(self._make_request())
        assert response.status_code == 200
        assert call_count[0] == 2

    def test_retry_on_401_disables_key(self, pool, mock_next):
        call_count = [0]

        def handler(req):
            call_count[0] += 1
            if call_count[0] == 1:
                return httpx.Response(401, json={"error": "unauthorized"})
            return httpx.Response(200, json={"ok": True})

        mock_next.handle_request.side_effect = handler
        transport = KeyPoolTransport(mock_next, pool, "default")
        transport.set_key("sk-good")
        transport._current_key_record = pool._pools["default"].keys[0]

        response = transport.handle_request(self._make_request())
        assert response.status_code == 200
        assert pool._pools["default"].keys[0].status == KeyStatus.DISABLED

    def test_exhausts_retries(self, pool, mock_next):
        mock_next.handle_request.return_value = httpx.Response(429, json={"error": "rate limited"})
        transport = KeyPoolTransport(mock_next, pool, "default", max_retries=3)
        transport.set_key("sk-good")
        transport._current_key_record = pool._pools["default"].keys[0]

        response = transport.handle_request(self._make_request())
        assert response.status_code == 429

    def test_no_pool_no_retry(self, mock_next):
        mock_next.handle_request.return_value = httpx.Response(429)
        transport = KeyPoolTransport(mock_next, None, "default", max_retries=3)
        transport.set_key("sk-good")

        response = transport.handle_request(self._make_request())
        assert response.status_code == 429

    def test_original_request_headers_not_mutated(self, pool, mock_next):
        """Retry must not mutate the caller's original request headers."""
        mock_next.handle_request.return_value = httpx.Response(200)
        transport = KeyPoolTransport(mock_next, pool, "default")
        transport.set_key("sk-injected")

        request = self._make_request()
        original_headers = dict(request.headers)
        transport.handle_request(request)

        assert dict(request.headers) == original_headers


# ── KeyPoolChatModel wrapper ───────────────────────────────────────


class TestKeyPoolChatModel:
    @pytest.fixture
    def _msg(self):
        from langchain_core.messages import AIMessage
        return AIMessage(content="test")

    @pytest.fixture
    def _msg_chunk(self):
        from langchain_core.messages import AIMessageChunk
        return AIMessageChunk(content="test")

    @pytest.fixture
    def pool(self):
        p = Pool()
        p.add_pool(PoolConfig(
            name="default",
            keys=[KeyRecord(name="k1", api_key="sk-wrapper-test", pool_name="default")],
        ))
        return p

    @pytest.fixture
    def inner_model(self):
        m = MagicMock()
        m._llm_type = "anthropic"
        m._identifying_params = {"model": "claude-sonnet-4-6"}
        return m

    @pytest.fixture
    def wrapper(self, pool, inner_model):
        return KeyPoolChatModel(pool=pool, pool_name="default", inner=inner_model)

    # ── _llm_type / _identifying_params ──

    def test_llm_type(self, wrapper):
        assert wrapper._llm_type == "keypool_anthropic"

    def test_identifying_params(self, wrapper):
        params = wrapper._identifying_params
        assert params["pool_name"] == "default"
        assert params["inner_type"] == "anthropic"

    # ── _generate: success path ──

    def test_generate_injects_key(self, wrapper, pool, inner_model, _msg):
        from langchain_core.outputs import ChatGeneration, ChatResult

        inner_model.anthropic_api_key = None
        inner_model._generate.return_value = ChatResult(
            generations=[ChatGeneration(text="response", message=_msg)],
        )

        wrapper._generate([_msg])
        assert inner_model.anthropic_api_key == "sk-wrapper-test"

    def test_generate_records_usage(self, wrapper, pool, inner_model, _msg):
        """TokenCounter callback not triggered by MagicMock, so usage stays 0."""
        from langchain_core.outputs import ChatGeneration, ChatResult

        inner_model.anthropic_api_key = None
        inner_model._generate.return_value = ChatResult(
            generations=[ChatGeneration(
                text="r",
                generation_info={"usage": {"input_tokens": 50, "output_tokens": 25}},
                message=_msg,
            )],
        )

        wrapper._generate([_msg])
        key = pool._pools["default"].keys[0]
        # MagicMock doesn't invoke TokenCounter callbacks, so tokens stay 0
        assert key.total_input_tokens == 0
        assert key.total_output_tokens == 0

    def test_generate_marks_success(self, wrapper, pool, inner_model, _msg):
        from langchain_core.outputs import ChatGeneration, ChatResult

        key = pool._pools["default"].keys[0]
        key.error_count = 3
        key.cooldown_seconds = 120.0
        inner_model._generate.return_value = ChatResult(
            generations=[ChatGeneration(text="ok", message=_msg)],
        )

        wrapper._generate([_msg])
        assert key.error_count == 0
        assert key.cooldown_seconds == 30.0

    # ── _generate: retry on error ──

    def test_generate_retry_on_429(self, wrapper, pool, inner_model, _msg):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        inner_model.anthropic_api_key = None
        key1 = pool._pools["default"].keys[0]
        key2 = KeyRecord(name="k2", api_key="sk-backup", pool_name="default")
        pool._pools["default"].keys.append(key2)

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("429 Too Many Requests")
            return ChatResult(
                generations=[ChatGeneration(text="retry ok", message=AIMessage(content="retry ok"))],
            )

        inner_model._generate.side_effect = side_effect

        result = wrapper._generate([_msg])
        assert call_count[0] == 2
        assert key1.status == KeyStatus.DEGRADED
        assert result.generations[0].text == "retry ok"

    def test_generate_401_disables_key(self, wrapper, pool, inner_model, _msg):
        from langchain_core.outputs import ChatGeneration, ChatResult

        key1 = pool._pools["default"].keys[0]
        key2 = KeyRecord(name="k2", api_key="sk-backup2", pool_name="default")
        pool._pools["default"].keys.append(key2)

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("401 Unauthorized")
            return ChatResult(
                generations=[ChatGeneration(text="ok", message=_msg)],
            )

        inner_model._generate.side_effect = side_effect

        wrapper._generate([_msg])
        assert key1.status == KeyStatus.DISABLED

    def test_generate_exhausts_all_keys(self, wrapper, pool, inner_model, _msg):
        pool._pools["default"].keys.clear()
        with pytest.raises(RuntimeError, match="no available keys"):
            wrapper._generate([_msg])

    def test_generate_raises_non_retryable(self, wrapper, inner_model, _msg):
        inner_model._generate.side_effect = ValueError("unexpected error")
        with pytest.raises(ValueError):
            wrapper._generate([_msg])

    # ── _stream: success path ──

    def test_stream_yields_chunks(self, wrapper, inner_model, _msg):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        inner_model._stream.return_value = iter([
            ChatGenerationChunk(text="Hello", message=AIMessageChunk(content="Hello")),
            ChatGenerationChunk(text=" world", message=AIMessageChunk(content=" world")),
        ])

        chunks = list(wrapper._stream([_msg]))
        assert len(chunks) == 2
        assert chunks[0].text == "Hello"
        assert chunks[1].text == " world"

    def test_stream_records_usage_after(self, wrapper, pool, inner_model, _msg, _msg_chunk):
        from langchain_core.outputs import ChatGenerationChunk

        inner_model._stream.return_value = iter([
            ChatGenerationChunk(text="hi", message=_msg_chunk),
        ])

        list(wrapper._stream([_msg]))
        key = pool._pools["default"].keys[0]
        # Usage comes from TokenCounter callback, not chunk metadata
        assert key.total_input_tokens == 0
        assert key.total_output_tokens == 0

    def test_stream_marks_success(self, wrapper, pool, inner_model, _msg, _msg_chunk):
        from langchain_core.outputs import ChatGenerationChunk

        key = pool._pools["default"].keys[0]
        key.error_count = 2
        key.cooldown_seconds = 90
        inner_model._stream.return_value = iter([ChatGenerationChunk(text="x", message=_msg_chunk)])

        list(wrapper._stream([_msg]))
        assert key.error_count == 0
        assert key.cooldown_seconds == 30.0

    # ── _stream: retry on error ──

    def test_stream_retry_on_429(self, wrapper, pool, inner_model, _msg):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        key2 = KeyRecord(name="k2", api_key="sk-stream-backup", pool_name="default")
        pool._pools["default"].keys.append(key2)

        call_count = [0]

        def stream_gen(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("429 rate limited")
            yield ChatGenerationChunk(text="recovered", message=AIMessageChunk(content="recovered"))

        inner_model._stream.side_effect = stream_gen

        chunks = list(wrapper._stream([_msg]))
        assert call_count[0] == 2
        assert chunks[0].text == "recovered"

    # ── _inject_transport ──

    def test_inject_transport_creates_wrapper(self, wrapper, pool, inner_model):
        mock_client = MagicMock()
        mock_client._transport = MagicMock()
        inner_model._client = MagicMock()
        inner_model._client._client = mock_client

        key = pool._pools["default"].keys[0]
        wrapper._inject_transport(key)
        from clawagent.api_pool.transport import KeyPoolTransport

        assert isinstance(mock_client._transport, KeyPoolTransport)

    def test_inject_transport_noop_on_second_call(self, wrapper, pool, inner_model):

        mock_client = MagicMock()
        mock_client._transport = MagicMock()
        inner_model._client = MagicMock()
        inner_model._client._client = mock_client

        key = pool._pools["default"].keys[0]
        wrapper._inject_transport(key)
        first_transport = mock_client._transport
        wrapper._inject_transport(key)
        assert mock_client._transport is first_transport

    def test_inject_transport_silent_on_missing_client(self, wrapper):
        inner = MagicMock(spec=[])  # no _client attr
        wrapper.inner = inner
        key = KeyRecord(name="k", api_key="sk")
        wrapper._inject_transport(key)  # should not raise


# ── Global pool singleton ───────────────────────────────────────────


class TestGlobalPool:
    def test_init_global_pool_returns_pool(self):
        pool = init_global_pool()
        assert isinstance(pool, ApiKeyPool)

    def test_get_global_pool_returns_same_instance(self):
        import clawagent.api_pool.__init__ as mod
        mod._GLOBAL_POOL = None
        a = get_global_pool()
        b = get_global_pool()
        assert a is b

    def test_global_pool_no_env_config_returns_empty_pool(self, monkeypatch):
        import clawagent.api_pool.__init__ as mod
        mod._GLOBAL_POOL = None
        for k in list(os.environ):
            if k.startswith("API_POOL_"):
                monkeypatch.delenv(k, raising=False)
        pool = init_global_pool()
        stats = pool.get_all_stats()
        assert stats == {}

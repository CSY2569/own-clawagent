"""Tests for conversation_log.py — JSONL per-session logging."""

import json
from pathlib import Path
from typing import Any

import pytest

from clawagent.agent import Usage
from clawagent.config import Settings
from clawagent.conversation_log import ConversationLogger, _settings_dict, _usage_dict


@pytest.fixture
def logger(tmp_path: Path) -> ConversationLogger:
    return ConversationLogger(log_dir=str(tmp_path / "logs"))


@pytest.fixture
def settings() -> Settings:
    return Settings(api_key="sk-test", model_name="deepseek-v4-pro")


@pytest.fixture
def usage() -> Usage:
    return Usage(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=30,
        cache_creation_input_tokens=10,
        prompt_cache_hit_tokens=500,
        prompt_cache_miss_tokens=200,
    )


class TestSettingsDict:
    def test_excludes_api_keys(self, settings: Settings) -> None:
        d = _settings_dict(settings)
        assert "api_key" not in d
        assert "siliconflow_api_key" not in d

    def test_includes_model_params(self, settings: Settings) -> None:
        d = _settings_dict(settings)
        assert d["model_name"] == "deepseek-v4-pro"
        assert d["model_provider"] == "openai"
        assert d["max_tokens"] == 4096
        assert d["temperature"] == 0.0


class TestUsageDict:
    def test_all_fields_present(self, usage: Usage) -> None:
        d = _usage_dict(usage)
        assert d == {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 10,
            "prompt_cache_hit_tokens": 500,
            "prompt_cache_miss_tokens": 200,
        }

    def test_zero_usage(self) -> None:
        d = _usage_dict(Usage())
        assert d["input_tokens"] == 0
        assert d["output_tokens"] == 0


class TestConversationLogger:
    def test_log_session_start_writes_jsonl(
        self, logger: ConversationLogger, settings: Settings, tmp_path: Path
    ) -> None:
        logger.log_session_start("abc123", settings)
        lines = _read_log(tmp_path, "abc123")
        assert len(lines) == 1
        assert lines[0]["type"] == "session_start"
        assert lines[0]["thread_id"] == "abc123"
        assert "timestamp" in lines[0]
        assert lines[0]["settings"]["model_name"] == "deepseek-v4-pro"

    def test_log_turn_writes_full_record(
        self,
        logger: ConversationLogger,
        settings: Settings,
        usage: Usage,
        tmp_path: Path,
    ) -> None:
        logger.log_turn("abc123", 1, "hello", "hi there", usage, settings)
        lines = _read_log(tmp_path, "abc123")
        assert len(lines) == 1
        rec = lines[0]
        assert rec["type"] == "turn"
        assert rec["round"] == 1
        assert rec["user_message"] == "hello"
        assert rec["response_preview"] == "hi there"
        assert rec["usage"]["input_tokens"] == 100
        assert rec["usage"]["output_tokens"] == 50
        assert rec["model_name"] == "deepseek-v4-pro"
        assert rec["temperature"] == 0.0
        assert rec["error"] is None

    def test_log_turn_with_error(
        self,
        logger: ConversationLogger,
        settings: Settings,
        usage: Usage,
        tmp_path: Path,
    ) -> None:
        logger.log_turn("abc123", 2, "q", "", usage, settings, error="RuntimeError: boom")
        rec = _read_log(tmp_path, "abc123")[0]
        assert rec["error"] == "RuntimeError: boom"

    def test_log_turn_truncates_long_response(
        self,
        logger: ConversationLogger,
        settings: Settings,
        usage: Usage,
        tmp_path: Path,
    ) -> None:
        long_response = "x" * 500
        logger.log_turn("abc123", 1, "q", long_response, usage, settings)
        rec = _read_log(tmp_path, "abc123")[0]
        assert len(rec["response_preview"]) == 200

    def test_log_settings_change(
        self, logger: ConversationLogger, tmp_path: Path
    ) -> None:
        logger.log_settings_change("abc123", "model_name", "flash", "pro")
        rec = _read_log(tmp_path, "abc123")[0]
        assert rec["type"] == "settings_change"
        assert rec["field"] == "model_name"
        assert rec["old_value"] == "flash"
        assert rec["new_value"] == "pro"

    def test_log_session_end(
        self, logger: ConversationLogger, usage: Usage, tmp_path: Path
    ) -> None:
        logger.log_session_end("abc123", 5, usage)
        rec = _read_log(tmp_path, "abc123")[0]
        assert rec["type"] == "session_end"
        assert rec["total_rounds"] == 5
        assert rec["cumulative_usage"]["input_tokens"] == 100
        assert rec["cumulative_usage"]["output_tokens"] == 50

    def test_multiple_calls_append_to_same_file(
        self,
        logger: ConversationLogger,
        settings: Settings,
        usage: Usage,
        tmp_path: Path,
    ) -> None:
        logger.log_session_start("abc123", settings)
        logger.log_turn("abc123", 1, "q", "a", usage, settings)
        logger.log_session_end("abc123", 1, usage)
        lines = _read_log(tmp_path, "abc123")
        assert len(lines) == 3
        assert lines[0]["type"] == "session_start"
        assert lines[1]["type"] == "turn"
        assert lines[2]["type"] == "session_end"

    def test_different_threads_create_different_files(
        self,
        logger: ConversationLogger,
        settings: Settings,
        usage: Usage,
        tmp_path: Path,
    ) -> None:
        logger.log_session_start("aaa", settings)
        logger.log_session_start("bbb", settings)
        log_dir = tmp_path / "logs"
        files = sorted(p.name for p in log_dir.iterdir())
        assert files == ["aaa.jsonl", "bbb.jsonl"]

    def test_creates_log_dir_if_missing(
        self, logger: ConversationLogger, settings: Settings, tmp_path: Path
    ) -> None:
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()
        logger.log_session_start("t1", settings)
        assert log_dir.exists()


def _read_log(tmp_path: Path, thread_id: str) -> list[dict[str, Any]]:
    """Read a session JSONL file and return parsed records."""
    path = tmp_path / "logs" / f"{thread_id}.jsonl"
    records = []
    for line in path.read_text("utf-8").strip().split("\n"):
        if line:
            records.append(json.loads(line))
    return records

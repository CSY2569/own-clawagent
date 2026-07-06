"""Integration tests — full agent pipeline with mocked LLM responses."""

# mypy: disable-error-code="no-untyped-def"

import json
from unittest.mock import MagicMock

import pytest

from clawagent.agent import Agent, AgentResponse, Usage
from clawagent.config import Settings
from clawagent.conversation_log import ConversationLogger


def _mock_agent_response(text: str = "Hello from mock", tokens: int = 50):
    """Return a MagicMock that behaves like an Agent with .run() and .close()."""
    agent = MagicMock(spec=Agent)
    agent.thread_id = "test1234"
    agent.run.return_value = AgentResponse(
        text=text,
        usage=Usage(input_tokens=tokens, output_tokens=20),
    )
    agent.close = MagicMock()
    return agent


class TestConversationLogger:
    """Session JSONL logging integration."""

    def test_writes_session_log(self, tmp_path):
        logger = ConversationLogger(log_dir=str(tmp_path / "logs"))
        logger.log_session_start(
            "sess-01",
            Settings(anthropic_api_key="sk-test"),
        )

        log_path = tmp_path / "logs" / "sess-01.jsonl"
        assert log_path.exists()

        line = json.loads(log_path.read_text().strip())
        assert line["type"] == "session_start"
        assert line["thread_id"] == "sess-01"
        assert "settings" in line
        assert "anthropic_api_key" not in str(line["settings"])

    def test_logs_turn_with_usage(self, tmp_path):
        logger = ConversationLogger(log_dir=str(tmp_path / "logs"))
        usage = Usage(input_tokens=100, output_tokens=50,
                      cache_read_input_tokens=10, cache_creation_input_tokens=5,
                      prompt_cache_hit_tokens=8, prompt_cache_miss_tokens=2)

        logger.log_turn(
            "sess-01", 1, "hello", "hi there", usage,
            Settings(anthropic_api_key="sk-test"),
        )

        log_path = tmp_path / "logs" / "sess-01.jsonl"
        content = log_path.read_text().strip()
        record = json.loads(content)

        assert record["type"] == "turn"
        assert record["round"] == 1
        assert record["user_message"] == "hello"
        assert record["response_preview"] == "hi there"
        assert record["usage"]["input_tokens"] == 100
        assert record["usage"]["prompt_cache_hit_tokens"] == 8

    def test_logs_session_end_with_cumulative(self, tmp_path):
        logger = ConversationLogger(log_dir=str(tmp_path / "logs"))
        logger.log_session_start(
            "sess-02",
            Settings(anthropic_api_key="sk-test"),
        )
        usage = Usage(input_tokens=500, output_tokens=200)
        logger.log_session_end("sess-02", 10, usage)

        log_path = tmp_path / "logs" / "sess-02.jsonl"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        end_record = json.loads(lines[1])
        assert end_record["type"] == "session_end"
        assert end_record["total_rounds"] == 10
        assert end_record["cumulative_usage"]["input_tokens"] == 500


class TestAgentResponse:
    """Agent response structure and usage extraction."""

    def test_full_response_with_all_fields(self):
        usage = Usage(
            input_tokens=200,
            output_tokens=80,
            cache_read_input_tokens=40,
            cache_creation_input_tokens=20,
            prompt_cache_hit_tokens=30,
            prompt_cache_miss_tokens=10,
        )
        resp = AgentResponse(text="result text", usage=usage)

        assert resp.text == "result text"
        assert resp.usage.input_tokens == 200
        assert resp.usage.prompt_cache_hit_tokens == 30
        assert resp.usage.prompt_cache_miss_tokens == 10

    def test_default_usage_is_zero(self):
        usage = Usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.prompt_cache_hit_tokens == 0


class TestSettingsSerialization:
    """Settings serialization for logging (no API keys)."""

    def test_settings_dict_excludes_api_key(self):
        from clawagent.conversation_log import _settings_dict

        settings = Settings(anthropic_api_key="secret-key-123")
        serialized = _settings_dict(settings)

        assert "model_name" in serialized
        assert "anthropic_api_key" not in serialized
        assert "secret-key" not in str(serialized)


class TestStreamEventStructure:
    """StreamEvent structure and event kind validation."""

    def test_event_kinds_exist(self):
        from clawagent.stream_events import StreamEvent

        kinds = {"think_start", "think_end", "tool_call", "tool_result", "token", "error", "done"}
        for kind in kinds:
            evt = StreamEvent(kind=kind, content="test")
            assert evt.kind == kind
            assert evt.content == "test"


class TestCancelToken:
    """CancelToken cooperative cancellation."""

    def test_not_cancelled_initially(self):
        from clawagent.cancel_token import CancelToken

        with CancelToken() as cancel:
            assert not cancel.cancelled
            cancel.check()  # does not raise

    def test_cancelled_after_sigint(self):
        import signal

        from clawagent.cancel_token import CancelToken

        with CancelToken() as cancel:
            signal.raise_signal(signal.SIGINT)
            assert cancel.cancelled
            with pytest.raises(KeyboardInterrupt):
                cancel.check()


class TestMemoryDB:
    """SQLite memory database integration."""

    def test_creates_tables_on_connect(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        from clawagent.memory.summarizer import (
            close_all_cached,
            get_summary,
            list_summaries,
            save_summary,
        )

        save_summary(db_path, "thread-1", "Test Title", "Summary text", 3)
        summary = get_summary(db_path, "thread-1")

        assert summary is not None
        assert summary["title"] == "Test Title"
        assert summary["summary"] == "Summary text"

        all_summaries = list_summaries(db_path)
        assert len(all_summaries) == 1

        close_all_cached()

    def test_preferences_store(self, tmp_path):
        db_path = str(tmp_path / "prefs.db")

        from clawagent.memory.preferences import (
            load_top_preferences,
            save_preference,
        )

        save_preference(db_path, "language", "english", "sess-1", "user wrote in English", 0.8)
        save_preference(db_path, "response_style", "concise", "sess-1", "user said 'be brief'", 0.9)

        prefs = load_top_preferences(db_path, limit=5)
        assert len(prefs) == 2
        assert prefs[0]["key"] == "response_style"
        assert prefs[0]["value"] == "concise"

"""Tests for clawagent.compression — trim, token_trim, summarize strategies."""

# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from clawagent.compression import compress_state
from clawagent.compression.config import CompressionConfig


class TestTrimByCount:
    """trim strategy: prune oldest messages beyond a count limit."""

    def test_no_trim_when_under_limit(self):
        msgs = [HumanMessage(content=f"msg{i}") for i in range(5)]
        config = CompressionConfig(strategy="trim", max_messages=10)
        result = compress_state(msgs, config)
        assert len(result) == 5

    def test_trims_oldest_non_system(self):
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
            HumanMessage(content="q3"),
            AIMessage(content="a3"),
        ]
        config = CompressionConfig(strategy="trim", max_messages=4)
        result = compress_state(msgs, config)
        assert len(result) == 4
        assert result[0].content == "sys"
        assert result[-1].content == "a3"

    def test_preserves_system_message(self):
        """system message is always retained, oldest non-system trimmed."""
        msgs = [
            SystemMessage(content="identity"),
            HumanMessage(content="old1"),
            AIMessage(content="old2"),
            HumanMessage(content="recent"),
            AIMessage(content="latest"),
        ]
        config = CompressionConfig(strategy="trim", max_messages=3)
        result = compress_state(msgs, config)
        assert result[0].content == "identity"
        assert len(result) == 3
        assert result[-1].content == "latest"


class TestTrimByTokens:
    """token_trim strategy: prune by estimated token count."""

    def test_no_trim_when_under_limit(self):
        msgs = [HumanMessage(content="short") for _ in range(3)]
        config = CompressionConfig(strategy="token_trim", max_tokens=100_000)
        result = compress_state(msgs, config)
        assert len(result) == 3

    def test_trims_until_within_limit(self):
        msgs = [HumanMessage(content="x" * 200) for _ in range(20)]
        config = CompressionConfig(strategy="token_trim", max_tokens=50)
        result = compress_state(msgs, config)
        assert len(result) < 20
        assert len(result) >= 2


class TestSummarizeByLLM:
    """summarize strategy: LLM summary compression."""

    def test_no_summarize_when_under_limit(self):
        msgs = [HumanMessage(content="q"), AIMessage(content="a")]
        config = CompressionConfig(strategy="summarize", max_messages=10)
        model = MagicMock()
        result = compress_state(msgs, config, model=model)
        assert len(result) == 2
        model.invoke.assert_not_called()

    def test_summarize_overflow_messages(self):
        msgs = [HumanMessage(content=f"question{i}") for i in range(10)]
        config = CompressionConfig(
            strategy="summarize", max_messages=4, keep_recent=2
        )

        model = MagicMock()
        model.invoke.return_value = AIMessage(content="summary content")

        result = compress_state(msgs, config, model=model)
        assert len(result) == 3
        assert "[对话历史摘要]" in result[0].content
        model.invoke.assert_called_once()

    def test_summarize_timeout_falls_back_to_trim(self):
        """LLM call exceeding timeout → fall back to trim_by_count."""
        import time as time_mod

        msgs = [HumanMessage(content=f"question{i}") for i in range(10)]
        config = CompressionConfig(
            strategy="summarize", max_messages=4, keep_recent=2,
            summary_timeout=1,
        )

        model = MagicMock()
        model.invoke.side_effect = lambda *a, **kw: time_mod.sleep(5)

        result = compress_state(msgs, config, model=model)
        assert len(result) <= 4

    def test_summarize_llm_error_falls_back_to_trim(self):
        """LLM call raising exception → fall back to trim_by_count."""
        msgs = [HumanMessage(content=f"question{i}") for i in range(10)]
        config = CompressionConfig(
            strategy="summarize", max_messages=4, keep_recent=2,
            summary_timeout=30,
        )

        model = MagicMock()
        model.invoke.side_effect = RuntimeError("LLM unavailable")

        result = compress_state(msgs, config, model=model)
        assert len(result) <= 4

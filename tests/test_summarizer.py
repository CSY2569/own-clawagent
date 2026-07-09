"""Tests for clawagent.memory.summarizer."""

# mypy: disable-error-code="no-untyped-def,index"

from unittest.mock import MagicMock

from clawagent.memory.summarizer import (
    _heuristic_summary,
    generate_session_summary,
    get_summary,
    list_summaries,
    load_messages,
    save_messages,
    save_summary,
)


class TestSaveAndGetSummary:
    def test_save_and_get(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_summary(db, "s1", "Test Title", "Test content", 5)
        result = get_summary(db, "s1")
        assert result is not None
        assert result["title"] == "Test Title"
        assert result["summary"] == "Test content"
        assert result["message_count"] == 5

    def test_get_nonexistent(self, tmp_path):
        db = str(tmp_path / "test.db")
        assert get_summary(db, "nonexistent") is None

    def test_get_no_db_file(self, tmp_path):
        db = str(tmp_path / "nope" / "test.db")
        assert get_summary(db, "s1") is None

    def test_update_existing(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_summary(db, "s1", "Old", "Old content", 1)
        save_summary(db, "s1", "New", "New content", 2)
        result = get_summary(db, "s1")
        assert result is not None
        assert result["title"] == "New"
        assert result["summary"] == "New content"
        assert result["message_count"] == 2


class TestListSummaries:
    def test_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        assert list_summaries(db) == []

    def test_no_db_file(self, tmp_path):
        db = str(tmp_path / "nope" / "test.db")
        assert list_summaries(db) == []

    def test_multiple(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_summary(db, "s1", "First", "Sum 1", 1)
        save_summary(db, "s2", "Second", "Sum 2", 2)
        results = list_summaries(db)
        assert len(results) == 2
        ids = {r["thread_id"] for r in results}
        assert ids == {"s1", "s2"}


class TestMessages:
    def test_save_and_load(self, tmp_path):
        db = str(tmp_path / "test.db")
        msgs = [("user", "hello"), ("assistant", "hi")]
        save_messages(db, "s1", msgs)
        loaded = load_messages(db, "s1")
        assert len(loaded) == 2
        assert loaded[0]["role"] == "user"
        assert loaded[0]["content"] == "hello"
        assert loaded[1]["role"] == "assistant"
        assert loaded[1]["content"] == "hi"

    def test_load_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        assert load_messages(db, "s1") == []

    def test_load_no_db_file(self, tmp_path):
        db = str(tmp_path / "nope" / "test.db")
        assert load_messages(db, "s1") == []

    def test_multiple_batches(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_messages(db, "s1", [("user", "hi")])
        save_messages(db, "s1", [("assistant", "hello")])
        loaded = load_messages(db, "s1")
        assert len(loaded) == 2


class TestGenerateSummary:
    def test_heuristic_fallback(self):
        _, summary = generate_session_summary("hello\nworld", model=None)
        assert isinstance(summary, str)
        assert "hello" in _ or "hello" in summary

    def test_heuristic_empty(self):
        title, _ = _heuristic_summary("")
        assert title == "Conversation"

    def test_heuristic_counts_lines(self):
        _, summary = _heuristic_summary("a\nb\nc")
        assert "3 exchanges" in summary

    def test_with_model(self):
        mock = MagicMock()
        mock.invoke.return_value.content = "TITLE: My Title\nSUMMARY: My summary text."
        title, summary = generate_session_summary("hi", mock)
        assert title == "My Title"
        assert summary == "My summary text."

    def test_model_fallback_on_error(self):
        mock = MagicMock()
        mock.invoke.side_effect = ValueError("API error")
        title, _ = generate_session_summary("hello", mock)
        assert isinstance(title, str)

    def test_no_title_marker(self):
        mock = MagicMock()
        mock.invoke.return_value.content = "Plain response without markers."
        title, summary = generate_session_summary("hi", mock)
        assert title == "Conversation"
        assert summary == "Plain response without markers."

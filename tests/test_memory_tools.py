"""Tests for clawagent.tools.memory_tools."""

# mypy: disable-error-code="no-untyped-def"

from clawagent.memory.summarizer import save_messages, save_summary
from clawagent.tools.memory_tools import create_memory_tools


def _tools(db_path: str):
    """Create memory tools bound to a specific database path."""
    tools = create_memory_tools(db_path)
    return {t.name: t for t in tools}


class TestListSessions:
    def test_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        tools = _tools(db)
        result = tools["list_sessions"].invoke({})
        assert "暂无" in result

    def test_with_data(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_summary(db, "s1", "Test", "content", 3)
        result = _tools(db)["list_sessions"].invoke({})
        assert "s1" in result
        assert "Test" in result

    def test_multiple(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_summary(db, "s1", "First", "c1", 1)
        save_summary(db, "s2", "Second", "c2", 2)
        result = _tools(db)["list_sessions"].invoke({})
        assert "s1" in result
        assert "s2" in result


class TestRecallSession:
    def test_not_found(self, tmp_path):
        db = str(tmp_path / "test.db")
        result = _tools(db)["recall_session"].invoke({"session_id": "nonexistent"})
        assert "未找到" in result

    def test_summary_only(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_summary(db, "s1", "My Title", "My content", 3)
        result = _tools(db)["recall_session"].invoke({"session_id": "s1", "summary_only": True})
        assert "My Title" in result
        assert "My content" in result

    def test_full_recall(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_messages(db, "s1", [("user", "hello"), ("assistant", "world")])
        result = _tools(db)["recall_session"].invoke({"session_id": "s1", "summary_only": False})
        assert "[user]" in result
        assert "hello" in result
        assert "[assistant]" in result
        assert "world" in result


class TestSummarizeSession:
    def test_no_session_id(self, tmp_path):
        db = str(tmp_path / "test.db")
        result = _tools(db)["summarize_session"].invoke({"session_id": ""})
        assert "请提供" in result

    def test_no_messages(self, tmp_path):
        db = str(tmp_path / "test.db")
        result = _tools(db)["summarize_session"].invoke({"session_id": "s1"})
        assert "未找到" in result

    def test_generates_summary(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_messages(db, "s1", [("user", "hello"), ("assistant", "world")])
        result = _tools(db)["summarize_session"].invoke({"session_id": "s1"})
        # With no model, uses heuristic fallback
        assert "摘要已生成" in result

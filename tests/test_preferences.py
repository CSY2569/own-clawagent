"""Tests for clawagent.memory.preferences."""

# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock

from clawagent.memory.preferences import (
    _extract_patterns_basic,
    extract_preferences_from_messages,
    load_top_preferences,
    save_preference,
)


class TestPreferences:
    def test_save_and_load(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_preference(db, "lang", "chinese", "s1", "evidence", 0.8)
        prefs = load_top_preferences(db)
        assert len(prefs) == 1
        assert prefs[0]["key"] == "lang"
        assert prefs[0]["value"] == "chinese"

    def test_load_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        assert load_top_preferences(db) == []

    def test_load_no_db_file(self, tmp_path):
        db = str(tmp_path / "nope" / "test.db")
        assert load_top_preferences(db) == []

    def test_limit(self, tmp_path):
        db = str(tmp_path / "test.db")
        for i in range(10):
            save_preference(db, f"k{i}", f"v{i}", "s1", "", 0.5 + i * 0.05)
        assert len(load_top_preferences(db, limit=3)) == 3

    def test_dedup_by_max_confidence(self, tmp_path):
        db = str(tmp_path / "test.db")
        # Two entries with same key+value, different confidence
        save_preference(db, "lang", "chinese", "s1", "", 0.3)
        save_preference(db, "lang", "chinese", "s2", "", 0.9)
        prefs = load_top_preferences(db)
        assert len(prefs) == 1

    def test_multiple_keys(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_preference(db, "a", "1", "s1", "", 0.5)
        save_preference(db, "b", "2", "s1", "", 0.5)
        assert len(load_top_preferences(db)) == 2

    def test_different_values_same_key(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_preference(db, "lang", "chinese", "s1", "", 0.5)
        save_preference(db, "lang", "english", "s1", "", 0.5)
        prefs = load_top_preferences(db)
        assert len(prefs) == 2  # Both retained (different values)


class TestExtractPatternsBasic:
    def test_chinese_detected(self):
        result = _extract_patterns_basic("你好世界，这是一段中文对话。")
        assert any(p["key"] == "language" and p["value"] == "chinese_priority" for p in result)

    def test_english_not_detected(self):
        result = _extract_patterns_basic("Hello world, this is English.")
        assert not any(p["key"] == "language" for p in result)


class TestExtractFromMessages:
    def test_no_model(self, tmp_path):
        db = str(tmp_path / "test.db")
        result = extract_preferences_from_messages("你好世界", "s1", db, model=None)
        assert len(result) >= 1
        assert load_top_preferences(db)  # persisted

    def test_with_model(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_model = MagicMock()
        mock_model.invoke.return_value.content = (
            '{"preferences": [{"key": "style", "value": "concise", "evidence": "user said so", "confidence": 0.8, "privacy_level": "sensitive"}], "profile": [], "facts": []}'
        )
        result = extract_preferences_from_messages("keep it short", "s1", db, mock_model)
        assert len(result) == 1
        assert result[0]["key"] == "style"

    def test_model_invalid_json(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_model = MagicMock()
        mock_model.invoke.return_value.content = "NOT JSON"
        result = extract_preferences_from_messages("hi", "s1", db, mock_model)
        assert result == []

    def test_model_exception(self, tmp_path):
        db = str(tmp_path / "test.db")
        mock_model = MagicMock()
        mock_model.invoke.side_effect = ValueError("fail")
        result = extract_preferences_from_messages("hi", "s1", db, mock_model)
        assert result == []

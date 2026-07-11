"""Tests for clawagent.memory.facts."""

# mypy: disable-error-code="no-untyped-def"

from clawagent.memory.facts import (
    clear_facts,
    load_facts,
    load_vectorizable_facts,
    save_fact,
    save_facts_batch,
)
from clawagent.memory.privacy import PRIVATE, PUBLIC, SENSITIVE
from clawagent.memory.summarizer import close_all_cached


class TestSaveAndLoad:
    def test_save_and_load(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_fact(db, "project is called clawagent")
        facts = load_facts(db)
        assert len(facts) == 1
        assert facts[0]["content"] == "project is called clawagent"
        close_all_cached()

    def test_load_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        assert load_facts(db) == []
        close_all_cached()

    def test_load_no_db_file(self, tmp_path):
        db = str(tmp_path / "nope" / "test.db")
        assert load_facts(db) == []
        close_all_cached()

    def test_multiple_facts(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_fact(db, "fact A", confidence=0.5)
        save_fact(db, "fact B", confidence=0.9)
        save_fact(db, "fact C", confidence=0.7)
        facts = load_facts(db)
        assert len(facts) == 3
        # Ordered by confidence DESC
        assert facts[0]["content"] == "fact B"
        assert facts[1]["content"] == "fact C"
        assert facts[2]["content"] == "fact A"
        close_all_cached()


class TestCategoryFilter:
    def test_filter_by_category(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_fact(db, "tech fact", category="tech")
        save_fact(db, "life fact", category="life")
        tech = load_facts(db, category="tech")
        assert len(tech) == 1
        assert tech[0]["content"] == "tech fact"
        close_all_cached()

    def test_no_category_returns_all(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_fact(db, "a", category="x")
        save_fact(db, "b", category="y")
        assert len(load_facts(db)) == 2
        close_all_cached()


class TestPrivacyFilter:
    def test_private_not_stored(self, tmp_path):
        db = str(tmp_path / "test.db")
        result = save_fact(db, "secret", privacy_level=PRIVATE)
        assert result is False
        assert load_facts(db) == []
        close_all_cached()

    def test_sensitive_stored(self, tmp_path):
        db = str(tmp_path / "test.db")
        result = save_fact(db, "personal", privacy_level=SENSITIVE)
        assert result is True
        assert len(load_facts(db)) == 1
        close_all_cached()

    def test_public_stored(self, tmp_path):
        db = str(tmp_path / "test.db")
        result = save_fact(db, "public info", privacy_level=PUBLIC)
        assert result is True
        assert len(load_facts(db)) == 1
        close_all_cached()

    def test_invalid_level_defaults_to_public(self, tmp_path):
        db = str(tmp_path / "test.db")
        result = save_fact(db, "info", privacy_level="top_secret")
        assert result is True
        close_all_cached()


class TestVectorizableFacts:
    def test_only_public_returned(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_fact(db, "public fact", privacy_level=PUBLIC)
        save_fact(db, "sensitive fact", privacy_level=SENSITIVE)
        result = load_vectorizable_facts(db)
        assert len(result) == 1
        assert result[0] == "public fact"
        close_all_cached()

    def test_empty_db(self, tmp_path):
        db = str(tmp_path / "test.db")
        assert load_vectorizable_facts(db) == []
        close_all_cached()


class TestBatchSave:
    def test_batch_save(self, tmp_path):
        db = str(tmp_path / "test.db")
        entries = [
            {"content": "fact 1", "category": "tech"},
            {"content": "fact 2", "privacy_level": PRIVATE},
            {"content": "fact 3", "confidence": 0.9},
        ]
        count = save_facts_batch(db, entries)
        assert count == 2  # private filtered
        close_all_cached()

    def test_batch_skips_empty_content(self, tmp_path):
        db = str(tmp_path / "test.db")
        entries = [
            {"content": "real fact"},
            {"content": ""},
        ]
        count = save_facts_batch(db, entries)
        assert count == 1
        close_all_cached()


class TestClear:
    def test_clear(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_fact(db, "a")
        save_fact(db, "b")
        clear_facts(db)
        assert load_facts(db) == []
        close_all_cached()

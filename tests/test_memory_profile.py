"""Tests for clawagent.memory.profile."""

# mypy: disable-error-code="no-untyped-def"

from clawagent.memory.profile import (
    clear_profile,
    load_profile,
    save_profile_batch,
    save_profile_entry,
)
from clawagent.memory.summarizer import close_all_cached


class TestSaveAndLoad:
    def test_save_and_load(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_profile_entry(db, "role", "developer", 0.8)
        profile = load_profile(db)
        assert profile == {"role": "developer"}
        close_all_cached()

    def test_load_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        assert load_profile(db) == {}
        close_all_cached()

    def test_load_no_db_file(self, tmp_path):
        db = str(tmp_path / "nope" / "test.db")
        assert load_profile(db) == {}
        close_all_cached()

    def test_multiple_entries(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_profile_entry(db, "role", "developer")
        save_profile_entry(db, "lang", "python")
        save_profile_entry(db, "tz", "UTC+8")
        profile = load_profile(db)
        assert profile == {"role": "developer", "lang": "python", "tz": "UTC+8"}
        close_all_cached()


class TestUpsert:
    def test_higher_confidence_overwrites(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_profile_entry(db, "role", "junior", 0.3)
        save_profile_entry(db, "role", "senior", 0.9)
        profile = load_profile(db)
        assert profile["role"] == "senior"
        close_all_cached()

    def test_lower_confidence_does_not_overwrite(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_profile_entry(db, "role", "senior", 0.9)
        save_profile_entry(db, "role", "junior", 0.3)
        profile = load_profile(db)
        assert profile["role"] == "senior"
        close_all_cached()

    def test_equal_confidence_does_not_overwrite(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_profile_entry(db, "role", "first", 0.5)
        save_profile_entry(db, "role", "second", 0.5)
        profile = load_profile(db)
        assert profile["role"] == "first"
        close_all_cached()


class TestBatchSave:
    def test_batch_save(self, tmp_path):
        db = str(tmp_path / "test.db")
        entries = [
            {"key": "role", "value": "dev", "confidence": 0.8},
            {"key": "lang", "value": "python"},
            {"key": "tz", "value": "UTC", "confidence": 0.6},
        ]
        count = save_profile_batch(db, entries)
        assert count == 3
        profile = load_profile(db)
        assert len(profile) == 3
        close_all_cached()

    def test_batch_skips_empty(self, tmp_path):
        db = str(tmp_path / "test.db")
        entries = [
            {"key": "role", "value": "dev"},
            {"key": "", "value": "empty key"},
            {"key": "noval", "value": ""},
        ]
        count = save_profile_batch(db, entries)
        assert count == 1
        close_all_cached()

    def test_batch_empty_list(self, tmp_path):
        db = str(tmp_path / "test.db")
        assert save_profile_batch(db, []) == 0
        close_all_cached()


class TestClear:
    def test_clear(self, tmp_path):
        db = str(tmp_path / "test.db")
        save_profile_entry(db, "role", "dev")
        save_profile_entry(db, "lang", "py")
        clear_profile(db)
        assert load_profile(db) == {}
        close_all_cached()

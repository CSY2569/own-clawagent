"""Tests for clawagent.memory.privacy."""

# mypy: disable-error-code="no-untyped-def"

from clawagent.memory.privacy import (
    DEFAULT_LEVEL,
    PRIVATE,
    PUBLIC,
    SENSITIVE,
    VALID_LEVELS,
    filter_storeable,
    is_storeable,
    is_vectorizable,
    normalize_level,
)


class TestConstants:
    def test_values(self):
        assert PUBLIC == "public"
        assert SENSITIVE == "sensitive"
        assert PRIVATE == "private"

    def test_valid_levels(self):
        assert frozenset({PUBLIC, SENSITIVE, PRIVATE}) == VALID_LEVELS

    def test_default_level(self):
        assert DEFAULT_LEVEL == PUBLIC


class TestIsStoreable:
    def test_public_storeable(self):
        assert is_storeable(PUBLIC) is True

    def test_sensitive_storeable(self):
        assert is_storeable(SENSITIVE) is True

    def test_private_not_storeable(self):
        assert is_storeable(PRIVATE) is False


class TestIsVectorizable:
    def test_public_vectorizable(self):
        assert is_vectorizable(PUBLIC) is True

    def test_sensitive_not_vectorizable(self):
        assert is_vectorizable(SENSITIVE) is False

    def test_private_not_vectorizable(self):
        assert is_vectorizable(PRIVATE) is False


class TestNormalizeLevel:
    def test_valid_public(self):
        assert normalize_level(PUBLIC) == PUBLIC

    def test_valid_sensitive(self):
        assert normalize_level(SENSITIVE) == SENSITIVE

    def test_valid_private(self):
        assert normalize_level(PRIVATE) == PRIVATE

    def test_none_falls_back_to_default(self):
        assert normalize_level(None) == DEFAULT_LEVEL

    def test_empty_string_falls_back(self):
        assert normalize_level("") == DEFAULT_LEVEL

    def test_invalid_value_falls_back(self):
        assert normalize_level("top_secret") == DEFAULT_LEVEL

    def test_case_sensitive(self):
        assert normalize_level("Public") == DEFAULT_LEVEL


class TestFilterStoreable:
    def test_filters_private(self):
        items = [
            {"key": "a", "privacy_level": PUBLIC},
            {"key": "b", "privacy_level": SENSITIVE},
            {"key": "c", "privacy_level": PRIVATE},
        ]
        result = filter_storeable(items)
        assert len(result) == 2
        assert result[0]["key"] == "a"
        assert result[1]["key"] == "b"

    def test_filters_invalid_level(self):
        items = [
            {"key": "a", "privacy_level": "unknown"},
            {"key": "b", "privacy_level": PRIVATE},
        ]
        result = filter_storeable(items)
        # "unknown" normalizes to PUBLIC (default), so it's storeable
        assert len(result) == 1
        assert result[0]["key"] == "a"

    def test_missing_key_uses_default(self):
        items = [{"key": "a"}, {"key": "b", "privacy_level": PRIVATE}]
        result = filter_storeable(items)
        assert len(result) == 1
        assert result[0]["key"] == "a"

    def test_empty_list(self):
        assert filter_storeable([]) == []

    def test_custom_key(self):
        items = [
            {"name": "a", "level": PUBLIC},
            {"name": "b", "level": PRIVATE},
        ]
        result = filter_storeable(items, key="level")
        assert len(result) == 1
        assert result[0]["name"] == "a"

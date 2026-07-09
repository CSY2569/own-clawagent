"""Tests for SessionManager LRU eviction and TTL cleanup."""

# mypy: disallow-untyped-defs = False

from unittest.mock import MagicMock, patch

from clawagent.gateway.session_manager import SessionManager


def _make_settings() -> MagicMock:
    s = MagicMock()
    s.memory_db_path = "memories/test.db"
    return s


class TestEviction:
    @patch("clawagent.gateway.session_manager.SessionManager._create_agent")
    def test_evict_when_over_capacity_despite_ttl(self, mock_create: MagicMock) -> None:
        """Over capacity → oldest session force-evicted even if within TTL."""
        mock_create.side_effect = lambda *a, **kw: MagicMock(close=MagicMock())

        mgr = SessionManager(_make_settings(), max_sessions=2, session_ttl=3600.0)

        mgr.get_or_create("cli", "user1", "default")
        mgr.get_or_create("cli", "user2", "default")
        assert len(mgr._sessions) == 2

        # Third session — exceeds max=2, oldest (user1) should be evicted
        mgr.get_or_create("cli", "user3", "default")
        assert len(mgr._sessions) == 2
        keys = [e.user_id for e in mgr._sessions.values()]
        assert "user1" not in keys
        assert "user3" in keys

    @patch("clawagent.gateway.session_manager.SessionManager._create_agent")
    def test_lru_order_on_access(self, mock_create: MagicMock) -> None:
        """Accessing a session moves it to most-recently-used position."""
        mock_create.side_effect = lambda *a, **kw: MagicMock(close=MagicMock())

        mgr = SessionManager(_make_settings(), max_sessions=2, session_ttl=3600.0)
        mgr.get_or_create("cli", "user1", "default")
        mgr.get_or_create("cli", "user2", "default")

        # Access user1 → now most-recently-used
        mgr.get_or_create("cli", "user1", "default")

        # Add user3 → user2 (oldest) should be evicted, not user1
        mgr.get_or_create("cli", "user3", "default")
        keys = [e.user_id for e in mgr._sessions.values()]
        assert "user2" not in keys
        assert "user1" in keys


class TestTtlCleanup:
    @patch("clawagent.gateway.session_manager.SessionManager._create_agent")
    def test_cleanup_expired_drops_idle(self, mock_create: MagicMock) -> None:
        mock_create.side_effect = lambda *a, **kw: MagicMock(close=MagicMock())

        mgr = SessionManager(_make_settings(), max_sessions=10, session_ttl=0.0)
        mgr.get_or_create("cli", "user1", "default")
        assert len(mgr._sessions) == 1

        # TTL=0 → all sessions are expired
        evicted = mgr.cleanup_expired()
        assert evicted == 1
        assert len(mgr._sessions) == 0

    @patch("clawagent.gateway.session_manager.SessionManager._create_agent")
    def test_cleanup_expired_keeps_active(self, mock_create: MagicMock) -> None:
        mock_create.side_effect = lambda *a, **kw: MagicMock(close=MagicMock())

        mgr = SessionManager(_make_settings(), max_sessions=10, session_ttl=3600.0)
        mgr.get_or_create("cli", "user1", "default")

        evicted = mgr.cleanup_expired()
        assert evicted == 0
        assert len(mgr._sessions) == 1

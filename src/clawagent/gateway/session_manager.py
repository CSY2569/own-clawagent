"""Session manager — Agent lifecycle with LRU eviction and TTL expiry.

Each logical session (channel_type + user_id + session_id) maps to
an Agent instance. Sessions are evicted on LRU when the pool exceeds
capacity, or aged out after a TTL of inactivity.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clawagent.agent import Agent
    from clawagent.config import Settings
    from clawagent.gateway.model_provider import ModelConfig


@dataclass
class SessionEntry:
    """Metadata for an active agent session."""

    agent: Agent
    channel_type: str
    user_id: str
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active = time.time()


class SessionManager:
    """Manage Agent instances with LRU eviction and TTL expiry.

    Characteristics:
    - LRU eviction when the pool exceeds ``max_sessions``.
    - TTL expiry: sessions idle for ``session_ttl`` seconds are cleaned up.
    - Per-channel model config: when Direction 2 is implemented, each
      channel can specify its own ModelConfig; currently falls back to
      the global Settings.
    """

    def __init__(
        self,
        settings: Settings,
        max_sessions: int = 50,
        session_ttl: float = 1800.0,
    ) -> None:
        self._settings = settings
        self._max = max_sessions
        self._ttl = session_ttl
        self._sessions: OrderedDict[str, SessionEntry] = OrderedDict()
        # Reserved: per-channel ModelConfig (Direction 2)
        self._channel_models: dict[str, ModelConfig | None] = {}

    # ── Public API ──────────────────────────────────────────

    def get_or_create(
        self,
        channel_type: str,
        user_id: str,
        session_id: str = "default",
    ) -> Agent:
        """Return the Agent for a (channel, user, session) triplet.

        Creates a new Agent if no active session exists.
        """
        key = self._make_key(channel_type, user_id, session_id)

        if key in self._sessions:
            entry = self._sessions.pop(key)
            entry.touch()
            self._sessions[key] = entry
            return entry.agent

        agent = self._create_agent(channel_type, user_id, session_id)
        entry = SessionEntry(
            agent=agent,
            channel_type=channel_type,
            user_id=user_id,
            session_id=session_id,
        )
        self._sessions[key] = entry
        self._evict_if_needed()
        return agent

    def set_channel_model(
        self, channel_type: str, model_cfg: ModelConfig | None
    ) -> None:
        """Reserved — set per-channel ModelConfig (Direction 2)."""
        self._channel_models[channel_type] = model_cfg

    def cleanup_expired(self) -> int:
        """Drop sessions idle longer than TTL. Returns count evicted.

        Called periodically by the gateway main loop (not on every access).
        Distinct from capacity-driven eviction in _evict_if_needed.
        """
        now = time.time()
        evicted = 0
        stale_keys = [
            key for key, entry in self._sessions.items()
            if now - entry.last_active >= self._ttl
        ]
        for key in stale_keys:
            entry = self._sessions.pop(key)
            entry.agent.close()
            evicted += 1
        return evicted

    def close_all(self) -> None:
        """Close all managed Agent instances."""
        for entry in self._sessions.values():
            entry.agent.close()
        self._sessions.clear()

    # ── Internal ────────────────────────────────────────────

    def _create_agent(
        self, channel_type: str, user_id: str, session_id: str
    ) -> Agent:
        from clawagent.agent import Agent, create_agent

        # When Direction 2 is active, use the channel-specific model config
        model_cfg = self._channel_models.get(channel_type)
        agent_settings = (
            model_cfg.to_settings(self._settings) if model_cfg else self._settings
        )

        graph, conn, factory, delegate_tool = create_agent(
            agent_settings, channel=channel_type,
        )
        thread_id = f"{channel_type}:{user_id}:{session_id}"

        return Agent(
            graph=graph,
            db_path=agent_settings.memory_db_path,
            conn=conn,
            default_thread_id=thread_id,
            factory=factory,
            delegate_tool=delegate_tool,
            channel=channel_type,
        )

    def _evict_if_needed(self) -> None:
        """LRU eviction — drop oldest sessions when over capacity.

        Force-evicts the least-recently-used session regardless of TTL,
        because exceeding max_sessions is a hard capacity constraint.
        TTL-based cleanup is handled separately by cleanup_expired().
        """
        while len(self._sessions) > self._max:
            key, _ = next(iter(self._sessions.items()))
            popped = self._sessions.pop(key)
            popped.agent.close()

    @staticmethod
    def _make_key(channel: str, user_id: str, session_id: str) -> str:
        return f"{channel}:{user_id}:{session_id}"

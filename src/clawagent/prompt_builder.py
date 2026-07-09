"""Multi-layer system prompt builder.

Assembles the agent system prompt from layered sources:
markdown files on disk, runtime metadata, SQLite user preferences,
and auto-generated tool descriptions.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool


class PromptBuilder:
    """Build a system prompt by layering multiple sources.

    Args:
        prompts_dir: Root directory containing prompts/agent/ and prompts/shared/.
        memory_db_path: Path to SQLite database for user preferences (Layer 5).
        max_preferences: Max number of user preferences to inject.
    """

    _PREF_CACHE_TTL: float = 60.0

    def __init__(
        self,
        prompts_dir: str | Path,
        memory_db_path: str = "",
        max_preferences: int = 5,
    ) -> None:
        self._prompts_dir = Path(prompts_dir)
        self._memory_db_path = memory_db_path
        self._max_preferences = max_preferences
        self._pref_cache: str | None = None
        self._pref_cache_ts: float = 0.0

    def build(
        self,
        agent_id: str = "wenbao",
        source: str = "cli",
        extra_context: str | None = None,
        delegate_tool: BaseTool | None = None,
    ) -> str:
        """Assemble the full system prompt from all layers."""
        layers: list[str] = []

        # Layer 1: Identity
        identity = self._read(f"agents/{agent_id}/identity.md")
        if not identity:
            identity = f"You are {agent_id}, a helpful assistant."
        layers.append(identity)

        # Layer 2: Personality (optional)
        soul = self._read(f"agents/{agent_id}/soul.md")
        if soul:
            layers.append(f"## Personality\n{soul}")

        # Layer 3: Workspace context (optional files)
        context_parts: list[str] = []
        bootstrap = self._read("shared/bootstrap.md")
        if bootstrap:
            context_parts.append(bootstrap)

        # Workers don't need the agent roster (they can't delegate_task)
        if source != "worker":
            agents_md = self._read("shared/agents.md")
            if agents_md:
                context_parts.append(agents_md)

        search_rules = self._read("shared/search-rules.md")
        if search_rules:
            context_parts.append(search_rules)

        # Role-specific context for workers (e.g. shared/worker-coder.md)
        worker_ctx = self._read(f"shared/worker-{agent_id}.md")
        if worker_ctx:
            context_parts.append(worker_ctx)

        if context_parts:
            layers.append("\n\n".join(context_parts))

        # Layer 4: Runtime (always)
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        layers.append(
            f"## Runtime\nCurrent time: {now}\nAgent: {agent_id}\nChannel: {source}"
        )

        # Layer 5: Long-term memory from SQLite (preferences + profile + recent sessions)
        memory_text = self._build_long_term_memory()
        if memory_text:
            layers.append(memory_text)

        # Tools: auto-generated from ALL_TOOLS (always)
        layers.append(self._build_tools_section(delegate_tool))

        # Extra context (optional, e.g. RAG results or semantic memory recall)
        if extra_context:
            layers.append(f"## Additional Context\n{extra_context}")

        return "\n\n".join(layers)

    def _read(self, relative_path: str) -> str | None:
        """Read an optional prompt file, returning None if it doesn't exist."""
        path = self._prompts_dir / relative_path
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return None

    def _build_long_term_memory(self) -> str | None:
        """Assemble Layer 5: preferences + profile + recent session summaries."""
        if not self._memory_db_path:
            return None

        import time

        now = time.monotonic()
        if self._pref_cache is not None and (now - self._pref_cache_ts) < self._PREF_CACHE_TTL:
            return self._pref_cache

        sections: list[str] = []

        from clawagent.memory.preferences import load_top_preferences

        prefs = load_top_preferences(self._memory_db_path, self._max_preferences)
        if prefs:
            pref_text = "\n".join(f"- {p['key']}: {p['value']}" for p in prefs)
            sections.append(f"### User Preferences\n{pref_text}")

        try:
            from clawagent.memory.profile import load_profile

            profile = load_profile(self._memory_db_path)
            if profile:
                profile_text = "\n".join(f"- {k}: {v}" for k, v in profile.items())
                sections.append(f"### User Profile\n{profile_text}")
        except Exception:
            pass

        try:
            from clawagent.memory.summarizer import get_recent_summaries

            recent = get_recent_summaries(self._memory_db_path, limit=3)
            if recent:
                lines = []
                for s in recent:
                    title = s.get("title", "Untitled")[:50]
                    summary = s.get("summary", "")[:200]
                    lines.append(f"- [{s.get('thread_id', '?')}] {title}: {summary}")
                sections.append("### Recent Sessions\n" + "\n".join(lines))
        except Exception:
            pass

        if not sections:
            return None

        result = "## Long-Term Memory\n" + "\n\n".join(sections)
        self._pref_cache = result
        self._pref_cache_ts = now
        return result

    @staticmethod
    def _build_tools_section(delegate_tool: BaseTool | None = None) -> str:
        """Auto-generate the tools listing from ALL_TOOLS + optional delegate_task.

        Each @tool-decorated function has .name and .description,
        so the list stays accurate as tools are added or removed.
        """
        from clawagent.tools import ALL_TOOLS  # lazy import, avoids circular deps

        tools: list[Any] = [*ALL_TOOLS]
        if delegate_tool is not None:
            tools.append(delegate_tool)

        lines = ["## Available Tools"]
        for t in tools:
            name = getattr(t, "name", str(t))
            desc = getattr(t, "description", "")
            desc_short = desc.split("\n")[0] if desc else ""
            if desc_short:
                lines.append(f"{name} — {desc_short}")
            else:
                lines.append(name)
        return "\n".join(lines)

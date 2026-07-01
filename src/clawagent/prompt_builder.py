"""Multi-layer system prompt builder.

Assembles the agent system prompt from layered sources:
markdown files on disk, runtime metadata, SQLite user preferences,
and auto-generated tool descriptions.
"""

from pathlib import Path


class PromptBuilder:
    """Build a system prompt by layering multiple sources.

    Args:
        prompts_dir: Root directory containing prompts/agent/ and prompts/shared/.
        memory_db_path: Path to SQLite database for user preferences (Layer 5).
        max_preferences: Max number of user preferences to inject.
    """

    def __init__(
        self,
        prompts_dir: str | Path,
        memory_db_path: str = "",
        max_preferences: int = 5,
    ) -> None:
        self._prompts_dir = Path(prompts_dir)
        self._memory_db_path = memory_db_path
        self._max_preferences = max_preferences

    def build(
        self,
        agent_id: str = "pickle",
        source: str = "cli",
        extra_context: str | None = None,
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
        agents_md = self._read("shared/agents.md")
        if agents_md:
            context_parts.append(agents_md)
        search_rules = self._read("shared/search-rules.md")
        if search_rules:
            context_parts.append(search_rules)
        if context_parts:
            layers.append("\n\n".join(context_parts))

        # Layer 4: Runtime (always)
        layers.append(
            f"## Runtime\nAgent: {agent_id}\nChannel: {source}"
        )

        # Layer 5: User preferences from SQLite (optional)
        prefs = self._load_preferences()
        if prefs:
            pref_text = "\n".join(f"- {p['key']}: {p['value']}" for p in prefs)
            layers.append(f"## User Preferences\n{pref_text}")

        # Tools: auto-generated from ALL_TOOLS (always)
        layers.append(self._build_tools_section())

        # Extra context (optional, e.g. RAG results)
        if extra_context:
            layers.append(f"## Additional Context\n{extra_context}")

        return "\n\n".join(layers)

    def _read(self, relative_path: str) -> str | None:
        """Read an optional prompt file, returning None if it doesn't exist."""
        path = self._prompts_dir / relative_path
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return None

    def _load_preferences(self) -> list[dict[str, str]]:
        """Load user preferences from SQLite (Layer 5)."""
        if not self._memory_db_path:
            return []
        from clawagent.memory.preferences import load_top_preferences

        return load_top_preferences(self._memory_db_path, self._max_preferences)

    @staticmethod
    def _build_tools_section() -> str:
        """Auto-generate the tools listing from ALL_TOOLS + delegate_task.

        Each @tool-decorated function has .name and .description,
        so the list stays accurate as tools are added or removed.
        """
        from clawagent.orchestrator.delegator import delegate_task
        from clawagent.tools import ALL_TOOLS  # lazy import, avoids circular deps

        lines = ["## Available Tools"]
        for t in [*ALL_TOOLS, delegate_task]:
            name = getattr(t, "name", str(t))
            desc = getattr(t, "description", "")
            desc_short = desc.split("\n")[0] if desc else ""
            if desc_short:
                lines.append(f"{name} — {desc_short}")
            else:
                lines.append(name)
        return "\n".join(lines)

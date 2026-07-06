"""Tests for PromptBuilder — multi-layer system prompt assembly."""

from clawagent.prompt_builder import PromptBuilder


class TestLayerIdentity:
    """Layer 1: Identity from agents/<id>/identity.md."""

    def test_reads_identity_file(self, tmp_path):
        agent_dir = tmp_path / "agents" / "testbot"
        agent_dir.mkdir(parents=True)
        (agent_dir / "identity.md").write_text("You are TestBot, a test assistant.")

        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="testbot")

        assert "You are TestBot" in prompt

    def test_fallback_when_identity_missing(self, tmp_path):
        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="missing")

        assert "You are missing" in prompt
        assert "helpful assistant" in prompt


class TestLayerPersonality:
    """Layer 2: Personality from agents/<id>/soul.md."""

    def test_includes_soul_when_present(self, tmp_path):
        agent_dir = tmp_path / "agents" / "testbot"
        agent_dir.mkdir(parents=True)
        (agent_dir / "identity.md").write_text("Identity")
        (agent_dir / "soul.md").write_text("Be concise and witty.")

        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="testbot")

        assert "Personality" in prompt
        assert "Be concise and witty" in prompt

    def test_skips_soul_when_absent(self, tmp_path):
        agent_dir = tmp_path / "agents" / "testbot"
        agent_dir.mkdir(parents=True)
        (agent_dir / "identity.md").write_text("Identity")

        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="testbot")

        assert "Personality" not in prompt


class TestLayerRuntime:
    """Layer 4: Runtime metadata (always present)."""

    def test_includes_runtime_info(self, tmp_path):
        agent_dir = tmp_path / "agents" / "wenbao"
        agent_dir.mkdir(parents=True)
        (agent_dir / "identity.md").write_text("Identity")

        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="wenbao", source="cli")

        assert "Runtime" in prompt
        assert "wenbao" in prompt
        assert "cli" in prompt
        assert "UTC" in prompt


class TestToolSection:
    """Auto-generated tools section."""

    def test_lists_core_tools(self, tmp_path):
        agent_dir = tmp_path / "agents" / "wenbao"
        agent_dir.mkdir(parents=True)
        (agent_dir / "identity.md").write_text("Identity")

        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="wenbao")

        assert "Available Tools" in prompt
        assert "read_file" in prompt
        assert "write_file" in prompt
        assert "run_command" in prompt
        assert "search_documents" in prompt

    def test_includes_delegate_tool(self, tmp_path):
        from unittest.mock import MagicMock

        agent_dir = tmp_path / "agents" / "wenbao"
        agent_dir.mkdir(parents=True)
        (agent_dir / "identity.md").write_text("Identity")

        mock_tool = MagicMock()
        mock_tool.name = "delegate_task"
        mock_tool.description = "Delegate a task to a worker"

        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="wenbao", delegate_tool=mock_tool)

        assert "delegate_task" in prompt
        assert "Delegate a task" in prompt


class TestWorkerSource:
    """Worker source excludes agent roster."""

    def test_worker_skips_agent_roster(self, tmp_path):
        agent_dir = tmp_path / "agents" / "coder"
        shared_dir = tmp_path / "shared"
        agent_dir.mkdir(parents=True)
        shared_dir.mkdir(parents=True)
        (agent_dir / "identity.md").write_text("You are a coder.")
        (shared_dir / "agents.md").write_text("Agent roster")

        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="coder", source="worker")

        assert "Agent roster" not in prompt

    def test_cli_source_includes_agent_roster(self, tmp_path):
        agent_dir = tmp_path / "agents" / "wenbao"
        shared_dir = tmp_path / "shared"
        agent_dir.mkdir(parents=True)
        shared_dir.mkdir(parents=True)
        (agent_dir / "identity.md").write_text("Identity")
        (shared_dir / "agents.md").write_text("Agent roster")

        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="wenbao", source="cli")

        assert "Agent roster" in prompt

    def test_includes_worker_context_file(self, tmp_path):
        agent_dir = tmp_path / "agents" / "coder"
        shared_dir = tmp_path / "shared"
        agent_dir.mkdir(parents=True)
        shared_dir.mkdir(parents=True)
        (agent_dir / "identity.md").write_text("You are a coder.")
        (shared_dir / "worker-coder.md").write_text("Coder-specific rules")

        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="coder", source="worker")

        assert "Coder-specific rules" in prompt


class TestSharedFiles:
    """Optional shared workspace files."""

    def test_includes_bootstrap(self, tmp_path):
        agent_dir = tmp_path / "agents" / "wenbao"
        shared_dir = tmp_path / "shared"
        agent_dir.mkdir(parents=True)
        shared_dir.mkdir(parents=True)
        (agent_dir / "identity.md").write_text("Identity")
        (shared_dir / "bootstrap.md").write_text("Workspace context here")

        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="wenbao")

        assert "Workspace context here" in prompt

    def test_includes_search_rules(self, tmp_path):
        agent_dir = tmp_path / "agents" / "wenbao"
        shared_dir = tmp_path / "shared"
        agent_dir.mkdir(parents=True)
        shared_dir.mkdir(parents=True)
        (agent_dir / "identity.md").write_text("Identity")
        (shared_dir / "search-rules.md").write_text("Search rules here")

        builder = PromptBuilder(prompts_dir=tmp_path)
        prompt = builder.build(agent_id="wenbao")

        assert "Search rules here" in prompt

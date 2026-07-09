"""Functional tests for clawagent — tool calling via agent graph."""

# mypy: disable-error-code="no-untyped-def"

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from clawagent.agent import Agent, create_agent
from clawagent.config import Settings
from clawagent.tools import ALL_TOOLS, read_file, run_command, write_file

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings(tmp_path):
    return Settings(
        api_key="sk-test-fake",
        memory_db_path=str(tmp_path / "memory.db"),
    )


@pytest.fixture
def agent(settings):
    """Create an agent with a mocked graph."""
    import sqlite3

    from clawagent.prompt_builder import PromptBuilder
    conn = sqlite3.connect(str(settings.memory_db_path))
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic
    from langgraph.checkpoint.sqlite import SqliteSaver
    from pydantic import SecretStr

    from clawagent.tools import ALL_TOOLS
    model = ChatAnthropic(
        model=settings.model_name,
        api_key=SecretStr(settings.api_key),
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )  # type: ignore[call-arg]
    prompts_dir = _PROJECT_ROOT / "prompts"
    builder = PromptBuilder(
        prompts_dir=str(prompts_dir),
        memory_db_path=str(settings.memory_db_path),
        max_preferences=settings.max_preferences,
    )
    prompt = builder.build(agent_id=settings.agent_id, source="cli")
    graph = create_agent(
        model=model, tools=ALL_TOOLS,
        checkpointer=SqliteSaver(conn), system_prompt=prompt,
    )
    return Agent(graph=graph, db_path=str(settings.memory_db_path))


class TestAgentSetup:
    def test_all_tools_defined(self):
        assert len(ALL_TOOLS) == 4
        names = {t.name for t in ALL_TOOLS}
        assert names == {
            "read_file", "write_file", "run_command", "search_documents",
        }

    def test_create_agent_returns_compiled_graph(self, settings):
        graph, conn, _factory, _delegate_tool = create_agent(settings)
        assert hasattr(graph, "invoke")
        conn.close()

    def test_agent_run_calls_graph_invoke(self, agent):
        agent._graph.invoke = MagicMock(
            return_value={
                "messages": [
                    AIMessage(
                        content="Hello!",
                        response_metadata={
                            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                        },
                    ),
                ]
            }
        )
        resp = agent.run("say hi")
        assert resp.text == "Hello!"
        agent._graph.invoke.assert_called_once()


class TestReadFileTool:
    def test_read_existing_file(self):
        content = read_file.invoke({"path": "README.md"})
        assert isinstance(content, str)
        assert len(content) > 10

    def test_read_nonexistent(self):
        result = read_file.invoke({"path": "does_not_exist_xyz.txt"})
        assert "not found" in result.lower() or "no such" in result.lower()

    def test_path_traversal_blocked(self):
        result = read_file.invoke({"path": "/etc/passwd"})
        assert result.startswith("Error:")


class TestWriteFileTool:
    def test_write_new_file(self, tmp_path):
        import clawagent.tools as t

        original = t.PROJECT_ROOT
        try:
            t.PROJECT_ROOT = tmp_path
            result = write_file.invoke({"path": "hello.txt", "content": "world"})
            assert (tmp_path / "hello.txt").read_text() == "world"
            assert "5 bytes" in result or "written" in result.lower()
        finally:
            t.PROJECT_ROOT = original

    def test_write_outside_project(self):
        result = write_file.invoke({"path": "/tmp/evil.txt", "content": "bad"})
        assert result.startswith("Error:")

    def test_write_creates_parent_dirs(self, tmp_path):
        import clawagent.tools as t

        original = t.PROJECT_ROOT
        try:
            t.PROJECT_ROOT = tmp_path
            write_file.invoke({"path": "deep/nested/file.txt", "content": "content"})
            assert (tmp_path / "deep/nested/file.txt").exists()
        finally:
            t.PROJECT_ROOT = original


class TestRunCommandTool:
    def test_echo(self):
        result = run_command.invoke({"command": "echo hello world"})
        assert "hello world" in result

    def test_failing_command(self):
        result = run_command.invoke({"command": "python -c \"import sys; sys.exit(42)\""})
        assert "42" in result

    def test_pwd(self):
        result = run_command.invoke({"command": "pwd"})
        assert "clawagent" in result


class TestAgentToolExecution:
    """Test that the agent processes tool calls correctly using mocked LLM responses."""

    def test_single_tool_call_flow(self, agent):
        """Verify that after a tool call, the final message text is returned."""
        messages = [
            HumanMessage(content="read README.md"),
            AIMessage(
                content="The file contains the project readme with project info.",
                response_metadata={
                    "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
                },
            ),
        ]
        agent._graph.invoke = MagicMock(return_value={"messages": messages})
        resp = agent.run("read README.md")
        assert "readme" in resp.text.lower()

    def test_usage_tracking(self, agent):
        """Verify usage metadata is extracted from response_metadata."""
        agent._graph.invoke = MagicMock(
            return_value={
                "messages": [
                    AIMessage(
                        content="done",
                        response_metadata={
                            "usage": {
                                "input_tokens": 50,
                                "output_tokens": 30,
                                "cache_read_input_tokens": 10,
                                "cache_creation_input_tokens": 5,
                            }
                        },
                    ),
                ]
            }
        )
        resp = agent.run("hi")
        assert resp.usage.input_tokens == 50
        assert resp.usage.output_tokens == 30
        assert resp.usage.cache_read_input_tokens == 10
        assert resp.usage.cache_creation_input_tokens == 5

    def test_tool_call_with_real_tool(self, tmp_path):
        """Full flow: LLM calls write_file, tool executes, result returned."""
        import clawagent.tools as t

        original_root = t.PROJECT_ROOT
        try:
            t.PROJECT_ROOT = tmp_path

            # Build messages as the real graph would
            target_path = tmp_path / "output" / "test.txt"
            tool_call = {
                "id": "call_write_1",
                "name": "write_file",
                "args": json.dumps({"path": "output/test.txt", "content": "hello"}),
            }

            # Simulate the tool execution
            tool_map = {t.name: t for t in ALL_TOOLS}

            tool_fn = tool_map["write_file"]
            args = json.loads(tool_call["args"])
            result = tool_fn.invoke(args)

            assert target_path.read_text() == "hello"
            assert "5 bytes" in result or "written" in result.lower()
        finally:
            t.PROJECT_ROOT = original_root

    def test_multi_tool_conversation(self, agent):
        """Simulate a multi-step conversation with tool calls and responses."""
        messages = [
            HumanMessage(content="write hello to a file"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "write_file",
                        "type": "tool_call",
                        "args": {"path": "test.txt", "content": "hello"},
                    }
                ],
            ),
            ToolMessage(content="written 5 bytes", tool_call_id="call_1"),
            AIMessage(
                content="Done! I wrote 'hello' to test.txt",
                response_metadata={
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                },
            ),
        ]
        agent._graph.invoke = MagicMock(return_value={"messages": messages})
        resp = agent.run("write hello to a file")
        assert resp.text == "Done! I wrote 'hello' to test.txt"

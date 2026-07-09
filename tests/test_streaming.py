"""Tests for clawagent Agent.stream_events — streaming event generation."""

# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk, ToolMessage

from clawagent.agent import Agent


def _make_agent_with_events(*event_pairs):
    """Create an Agent whose mock graph yields the given (msg, metadata) pairs."""
    graph = MagicMock()
    graph.stream.return_value = iter(event_pairs)
    return Agent(graph, db_path="", conn=None, default_thread_id="test")


def _aichunk(content="", tool_call_chunks=None, usage=None):
    """Build an AIMessageChunk with optional tool_call_chunks and usage metadata."""
    meta = {}
    if usage:
        meta["usage"] = usage
    return AIMessageChunk(
        content=content,
        tool_call_chunks=tool_call_chunks or [],
        response_metadata=meta,
    )


def _tmsg(content="", name=""):
    """Build a ToolMessage for the tools node."""
    return ToolMessage(content=content, name=name, tool_call_id="call_1")


def _meta(node):
    """Shorthand for LangGraph metadata dict."""
    return {"langgraph_node": node}


class TestStreamEvents:
    def test_yields_token_events(self):
        agent = _make_agent_with_events(
            (_aichunk(content="Hello "), _meta("agent")),
            (_aichunk(content="world"), _meta("agent")),
        )
        events = list(agent.stream_events("hi"))
        tokens = [e for e in events if e.kind == "token"]
        assert "".join(t.content for t in tokens) == "Hello world"

    def test_yields_done_event(self):
        agent = _make_agent_with_events(
            (_aichunk(content="Done"), _meta("agent")),
        )
        events = list(agent.stream_events("hi"))
        done = [e for e in events if e.kind == "done"]
        assert len(done) == 1
        assert done[0].content == "Done"

    def test_done_contains_usage(self):
        usage = {"input_tokens": 50, "output_tokens": 10}
        agent = _make_agent_with_events(
            (_aichunk(content="text", usage=usage), _meta("agent")),
        )
        events = list(agent.stream_events("hi"))
        done = next(e for e in events if e.kind == "done")
        assert done.metadata.get("input_tokens") == 50
        assert done.metadata.get("output_tokens") == 10

    def test_yields_tool_call_and_result(self):
        agent = _make_agent_with_events(
            (
                _aichunk(
                    tool_call_chunks=[
                        {"id": "call_1", "name": "read_file", "args": '{"path": "test.py"}'}
                    ]
                ),
                _meta("agent"),
            ),
            (_tmsg(content="file content", name="read_file"), _meta("tools")),
        )
        events = list(agent.stream_events("hi"))
        kinds = [e.kind for e in events]
        assert "tool_call" in kinds
        assert "tool_result" in kinds

    def test_token_before_done(self):
        agent = _make_agent_with_events(
            (_aichunk(content="abc"), _meta("agent")),
        )
        events = list(agent.stream_events("hi"))
        token_idx = next(i for i, e in enumerate(events) if e.kind == "token")
        done_idx = next(i for i, e in enumerate(events) if e.kind == "done")
        assert token_idx < done_idx

    def test_error_event_on_stream_failure(self):
        graph = MagicMock()
        graph.stream.side_effect = RuntimeError("boom")
        agent = Agent(graph, db_path="", conn=None, default_thread_id="test")
        events = list(agent.stream_events("hi"))
        assert any(e.kind == "error" for e in events)

"""Tests for Agent.stream_events() — message-level streaming (Phase 2)."""

# mypy: disable-error-code="no-untyped-def, method-assign, attr-defined"

import sqlite3
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from clawagent.agent import Agent
from clawagent.stream_events import StreamEvent


@pytest.fixture
def agent(tmp_path: Any) -> Agent:
    db_path = str(tmp_path / "test_stream.db")
    conn = sqlite3.connect(db_path)
    agent = Agent(graph=MagicMock(), db_path=db_path, conn=conn)
    agent._graph.stream = MagicMock()
    return agent


# ── Message-mode mock helpers ──


def _agent_chunk(
    content: Any = "",
    tool_call_chunks: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
) -> AIMessageChunk:
    """Build an AIMessageChunk for messages-mode mock."""
    msg = AIMessageChunk(content=content)
    if tool_call_chunks:
        msg.tool_call_chunks = tool_call_chunks  # type: ignore[assignment]
    if usage:
        msg.response_metadata = {"usage": usage}
    return msg


def _tool_msg(name: str = "read_file", content: str = "file contents here") -> ToolMessage:
    """Build a ToolMessage for messages-mode mock."""
    return ToolMessage(content=content, name=name, tool_call_id="call_1")


def _tcc(name: str, args: str, tc_id: str, index: int = 0) -> dict[str, Any]:
    """Build a tool_call_chunk dict."""
    return {"name": name, "args": args, "id": tc_id, "index": index}


# ── Phase 1 tests (adapted to messages mode) ──


class TestStreamEventsDirectReply:
    """LLM responds directly without using any tools."""

    def test_yields_only_done_event(self, agent: Agent) -> None:
        agent._graph.stream.return_value = [
            (_agent_chunk(content="Hello"), {"langgraph_node": "agent"}),
        ]
        events = list(agent.stream_events("hi"))
        kinds = [e.kind for e in events]
        assert "done" in kinds
        done = events[-1]
        assert done.kind == "done"
        assert done.content == "Hello"

    def test_yields_token_then_done(self, agent: Agent) -> None:
        """Multiple text chunks produce token events followed by done."""
        agent._graph.stream.return_value = [
            (_agent_chunk(content="Hello"), {"langgraph_node": "agent"}),
            (_agent_chunk(content=" world"), {"langgraph_node": "agent"}),
        ]
        events = list(agent.stream_events("hi"))
        kinds = [e.kind for e in events]
        assert "token" in kinds
        assert kinds[-1] == "done"
        assert events[-1].content == "Hello world"

    def test_done_event_includes_usage(self, agent: Agent) -> None:
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 10,
        }
        agent._graph.stream.return_value = [
            (_agent_chunk(content="ok", usage=usage), {"langgraph_node": "agent"}),
        ]
        events = list(agent.stream_events("hi"))
        done = events[-1]
        assert done.metadata["input_tokens"] == 100
        assert done.metadata["output_tokens"] == 50
        assert done.metadata["cache_read_input_tokens"] == 30
        assert done.metadata["cache_creation_input_tokens"] == 10

    def test_done_event_zero_usage_when_no_metadata(self, agent: Agent) -> None:
        agent._graph.stream.return_value = [
            (_agent_chunk(content="hi"), {"langgraph_node": "agent"}),
        ]
        # get_state must also return no usage to avoid MagicMock leakage
        agent._graph.get_state = MagicMock(return_value=None)
        events = list(agent.stream_events("hi"))
        done = events[-1]
        assert done.metadata["input_tokens"] == 0
        assert done.metadata["output_tokens"] == 0
        assert done.metadata["cache_read_input_tokens"] == 0
        assert done.metadata["cache_creation_input_tokens"] == 0


class TestStreamEventsSingleTool:
    """Single tool call: agent → tools → agent (messages mode)."""

    def test_yields_tool_call_then_result_then_done(self, agent: Agent) -> None:
        agent._graph.stream.return_value = [
            (_agent_chunk(content="Let me check."), {"langgraph_node": "agent"}),
            (
                _agent_chunk(
                    content="",
                    tool_call_chunks=[_tcc("read_file", '{"path": "test.py"}', "call_1")],
                ),
                {"langgraph_node": "agent"},
            ),
            (_tool_msg(name="read_file", content="print('hello')"), {"langgraph_node": "tools"}),
            (_agent_chunk(content="The file contains a print statement."), {"langgraph_node": "agent"}),
        ]
        events = list(agent.stream_events("read test.py"))
        kinds = [e.kind for e in events]
        assert kinds == ["token", "tool_call", "tool_result", "token", "done"]
        assert events[1].content == "read_file"
        assert events[1].metadata["args"] == {"path": "test.py"}
        assert events[2].content == "read_file"
        assert events[2].metadata["preview"] == "print('hello')"
        assert events[-1].content == "The file contains a print statement."


class TestStreamEventsMultiTool:
    """Multiple consecutive tool calls: dedup and sequence."""

    def test_dedup_tool_calls_by_id(self, agent: Agent) -> None:
        agent._graph.stream.return_value = [
            (_agent_chunk(content="Checking..."), {"langgraph_node": "agent"}),
            (
                _agent_chunk(
                    content="",
                    tool_call_chunks=[
                        _tcc("read_file", '{"path": "a.py"}', "call_1"),
                        _tcc("read_file", '{"path": "b.py"}', "call_2"),
                    ],
                ),
                {"langgraph_node": "agent"},
            ),
            (_tool_msg(name="read_file", content="content a"), {"langgraph_node": "tools"}),
            (_tool_msg(name="read_file", content="content b"), {"langgraph_node": "tools"}),
            (_agent_chunk(content="Done."), {"langgraph_node": "agent"}),
        ]
        events = list(agent.stream_events("read a.py and b.py"))
        tool_calls = [e for e in events if e.kind == "tool_call"]
        assert len(tool_calls) == 2

    def test_final_text_is_only_post_tool_text(self, agent: Agent) -> None:
        """C2 semantics in messages mode: pre-tool filler text is discarded."""
        agent._graph.stream.return_value = [
            (_agent_chunk(content="Looking up..."), {"langgraph_node": "agent"}),
            (
                _agent_chunk(
                    content="",
                    tool_call_chunks=[_tcc("read_file", '{"path": "f.py"}', "call_t1")],
                ),
                {"langgraph_node": "agent"},
            ),
            (_tool_msg(name="read_file", content="file contents"), {"langgraph_node": "tools"}),
            (_agent_chunk(content="The current time is 10:00 AM."), {"langgraph_node": "agent"}),
        ]
        events = list(agent.stream_events("what time is it"))
        done = events[-1]
        assert done.content == "The current time is 10:00 AM."
        assert "Looking up" not in done.content


class TestStreamEventsThinkingBlock:
    """Content in list[dict] format (DeepSeek thinking blocks)."""

    def test_extracts_text_from_content_blocks(self, agent: Agent) -> None:
        agent._graph.stream.return_value = [
            (
                _agent_chunk(
                    content=[
                        {"type": "thinking", "thinking": "Hmm..."},
                        {"type": "text", "text": "I'll help with that."},
                    ]
                ),
                {"langgraph_node": "agent"},
            ),
        ]
        events = list(agent.stream_events("help"))
        kinds = [e.kind for e in events]
        assert kinds == ["think_start", "think_end", "token", "done"]
        assert events[-1].content == "I'll help with that."

    def test_multiple_text_blocks_joined(self, agent: Agent) -> None:
        agent._graph.stream.return_value = [
            (
                _agent_chunk(
                    content=[
                        {"type": "text", "text": "Part 1. "},
                        {"type": "text", "text": "Part 2."},
                    ]
                ),
                {"langgraph_node": "agent"},
            ),
        ]
        events = list(agent.stream_events("test"))
        tokens = [e.content for e in events if e.kind == "token"]
        assert "".join(tokens) == "Part 1. Part 2."


class TestStreamEventsError:
    """Exception during streaming yields error event."""

    def test_error_event_on_stream_failure(self, agent: Agent) -> None:
        agent._graph.stream.side_effect = RuntimeError("connection lost")
        events = list(agent.stream_events("crash test"))
        assert len(events) == 2
        assert events[0].kind == "error"
        assert "RuntimeError" in events[0].content
        assert "connection lost" in events[0].content
        assert events[1].kind == "done"


class TestStreamEventsPersistence:
    """stream_events() writes to DB like Agent.run() does."""

    def test_saves_messages_and_session(self, agent: Agent, tmp_path: Any) -> None:
        agent._graph.stream.return_value = [
            (_agent_chunk(content="Response text."), {"langgraph_node": "agent"}),
        ]
        list(agent.stream_events("hello"))

        from clawagent.memory.summarizer import load_messages

        tid = agent._thread_id
        msgs = load_messages(agent._db_path, tid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Response text."

    def test_extracts_preferences(self, agent: Agent, tmp_path: Any) -> None:
        agent._turn_count = 9
        agent._graph.stream.return_value = [
            (_agent_chunk(content="I like Python."), {"langgraph_node": "agent"}),
        ]
        with patch(
            "clawagent.memory.preferences.extract_memories_from_messages"
        ) as mock_extract:
            list(agent.stream_events("I like Python"))
            mock_extract.assert_called_once()

    def test_no_persistence_when_no_db_path(self) -> None:
        agent = Agent(graph=MagicMock())
        agent._graph.stream = MagicMock(
            return_value=[
                (_agent_chunk(content="Ok."), {"langgraph_node": "agent"}),
            ]
        )
        events = list(agent.stream_events("test"))
        assert events[-1].kind == "done"
        assert events[-1].content == "Ok."


# ── Phase 2: Token-level streaming tests ──


class TestStreamEventsTokenLevel:
    """Token events between tool calls."""

    def test_token_events_between_tool_calls(self, agent: Agent) -> None:
        agent._graph.stream.return_value = [
            (_agent_chunk(content="Let me"), {"langgraph_node": "agent"}),
            (_agent_chunk(content=" read"), {"langgraph_node": "agent"}),
            (
                _agent_chunk(
                    content="",
                    tool_call_chunks=[_tcc("read_file", '{"path": "f.py"}', "call_1")],
                ),
                {"langgraph_node": "agent"},
            ),
            (_tool_msg(name="read_file", content="content"), {"langgraph_node": "tools"}),
            (_agent_chunk(content="Result:"), {"langgraph_node": "agent"}),
            (_agent_chunk(content=" found."), {"langgraph_node": "agent"}),
        ]
        events = list(agent.stream_events("read f.py"))
        tokens = [e for e in events if e.kind == "token"]
        # Pre-tool tokens are yielded (typewriter shows everything) but
        # only post-tool text survives in final_text (C2 semantics)
        assert len(tokens) == 4
        assert events[-1].content == "Result: found."
        assert "Let me" not in events[-1].content

    def test_token_only_no_tools(self, agent: Agent) -> None:
        """All tokens preserved when no tool calls occur."""
        agent._graph.stream.return_value = [
            (_agent_chunk(content="Hello"), {"langgraph_node": "agent"}),
            (_agent_chunk(content=" world"), {"langgraph_node": "agent"}),
            (_agent_chunk(content="!"), {"langgraph_node": "agent"}),
        ]
        events = list(agent.stream_events("hi"))
        tokens = [e for e in events if e.kind == "token"]
        assert len(tokens) == 3
        assert events[-1].content == "Hello world!"


# ── Phase 2: Thinking block tests ──


class TestStreamEventsThinkingBlocks:
    """Thinking block detection and boundary events."""

    def test_yields_think_start_and_end(self, agent: Agent) -> None:
        agent._graph.stream.return_value = [
            (
                _agent_chunk(
                    content=[
                        {"type": "thinking", "thinking": "Analyzing query..."},
                    ]
                ),
                {"langgraph_node": "agent"},
            ),
            (
                _agent_chunk(
                    content=[
                        {"type": "text", "text": "Here is the answer."},
                    ]
                ),
                {"langgraph_node": "agent"},
            ),
        ]
        events = list(agent.stream_events("complex question"))
        kinds = [e.kind for e in events]
        assert kinds == ["think_start", "think_end", "token", "done"]
        assert events[-1].content == "Here is the answer."

    def test_thinking_text_not_in_final(self, agent: Agent) -> None:
        agent._graph.stream.return_value = [
            (
                _agent_chunk(
                    content=[
                        {"type": "thinking", "thinking": "Hmm let me think..."},
                    ]
                ),
                {"langgraph_node": "agent"},
            ),
            (
                _agent_chunk(
                    content=[
                        {"type": "text", "text": "42 is the answer."},
                    ]
                ),
                {"langgraph_node": "agent"},
            ),
        ]
        events = list(agent.stream_events("what is the answer"))
        assert events[-1].content == "42 is the answer."
        # thinking content never appears as tokens or in final
        tokens = [e.content for e in events if e.kind == "token"]
        assert "Hmm let me think..." not in "".join(tokens)

    def test_think_end_on_tool_result(self, agent: Agent) -> None:
        """If thinking was active when tools run, think_end is yielded."""
        agent._graph.stream.return_value = [
            (
                _agent_chunk(
                    content=[{"type": "thinking", "thinking": "Need to check..."}]
                ),
                {"langgraph_node": "agent"},
            ),
            (
                _agent_chunk(
                    content="",
                    tool_call_chunks=[_tcc("read_file", '{"path": "x.py"}', "call_1")],
                ),
                {"langgraph_node": "agent"},
            ),
            (_tool_msg(name="read_file", content="data"), {"langgraph_node": "tools"}),
            (_agent_chunk(content="Done."), {"langgraph_node": "agent"}),
        ]
        events = list(agent.stream_events("check x.py"))
        kinds = [e.kind for e in events]
        # think_start from thinking block, think_end from tools node
        assert "think_start" in kinds
        assert "think_end" in kinds


# ── Phase 2: Usage fallback tests ──


class TestStreamEventsUsageFallback:
    """Usage extraction via chunk metadata and get_state fallback."""

    def test_usage_from_chunk_metadata(self, agent: Agent) -> None:
        usage = {"input_tokens": 50, "output_tokens": 30}
        agent._graph.stream.return_value = [
            (_agent_chunk(content="Hi", usage=usage), {"langgraph_node": "agent"}),
        ]
        events = list(agent.stream_events("hello"))
        done = events[-1]
        assert done.metadata["input_tokens"] == 50
        assert done.metadata["output_tokens"] == 30

    def test_usage_fallback_get_state(self, agent: Agent) -> None:
        """When chunk has no usage, get_state() provides fallback."""
        agent._graph.stream.return_value = [
            (_agent_chunk(content="Response."), {"langgraph_node": "agent"}),
        ]
        agent._graph.get_state = MagicMock(
            return_value=MagicMock(
                values={
                    "messages": [
                        MagicMock(
                            usage_metadata=None,
                            response_metadata={
                                "usage": {
                                    "input_tokens": 200,
                                    "output_tokens": 100,
                                }
                            }
                        )
                    ]
                }
            )
        )
        events = list(agent.stream_events("test"))
        done = events[-1]
        assert done.metadata["input_tokens"] == 200
        assert done.metadata["output_tokens"] == 100

    def test_usage_fallback_silent_on_error(self, agent: Agent) -> None:
        """get_state failure is silent, usage defaults to 0."""
        agent._graph.stream.return_value = [
            (_agent_chunk(content="Response."), {"langgraph_node": "agent"}),
        ]
        agent._graph.get_state = MagicMock(side_effect=RuntimeError("state error"))
        events = list(agent.stream_events("test"))
        done = events[-1]
        assert done.metadata["input_tokens"] == 0
        assert done.metadata["output_tokens"] == 0

    def test_all_cache_token_fields_in_done(self, agent: Agent) -> None:
        """F2 fix: all token fields preserved in done metadata, including DeepSeek."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 300,
            "prompt_cache_hit_tokens": 500,
            "prompt_cache_miss_tokens": 400,
        }
        agent._graph.stream.return_value = [
            (_agent_chunk(content="test", usage=usage), {"langgraph_node": "agent"}),
        ]
        events = list(agent.stream_events("test"))
        done = events[-1]
        assert done.metadata["cache_read_input_tokens"] == 200
        assert done.metadata["cache_creation_input_tokens"] == 300
        assert done.metadata["prompt_cache_hit_tokens"] == 500
        assert done.metadata["prompt_cache_miss_tokens"] == 400


# ── StreamDisplay tests ──


class TestStreamDisplayToken:
    """Phase 2+3: StreamDisplay handles token, thinking, and stats."""

    def test_token_appends_to_buffer(self) -> None:
        from clawagent.ui_stream import StreamDisplay

        display = StreamDisplay()
        display.handle(StreamEvent(kind="token", content="Hello"))
        display.handle(StreamEvent(kind="token", content=" world"))
        assert display._token_buffer == "Hello world"

    def test_token_does_not_refresh_too_fast(self) -> None:
        """Throttle: _refresh not called within 50ms."""
        from clawagent.ui_stream import StreamDisplay

        display = StreamDisplay()
        display._refresh = MagicMock()
        display._last_refresh = 1_000_000.0  # far in the future
        display.handle(StreamEvent(kind="token", content="hi"))
        display._refresh.assert_not_called()

    def test_think_start_sets_status(self) -> None:
        from clawagent.ui_stream import StreamDisplay

        display = StreamDisplay()
        display._refresh = MagicMock()
        display.handle(StreamEvent(kind="think_start"))
        assert "思考中" in display._status_text

    def test_think_end_clears_status(self) -> None:
        from clawagent.ui_stream import StreamDisplay

        display = StreamDisplay()
        display.handle(StreamEvent(kind="think_start"))
        display.handle(StreamEvent(kind="think_end"))
        assert display._status_text == ""

    def test_done_captures_stats(self) -> None:
        from clawagent.ui_stream import StreamDisplay

        display = StreamDisplay()
        display._refresh = MagicMock()
        display.handle(
            StreamEvent(
                kind="done",
                content="final",
                metadata={"input_tokens": 500, "output_tokens": 200},
            )
        )
        assert display._stats["input_tokens"] == 500
        assert display._stats["output_tokens"] == 200

    def test_done_shows_stats_in_render(self) -> None:
        from rich.console import Group as RichGroup

        from clawagent.ui_stream import StreamDisplay

        display = StreamDisplay()
        display.handle(
            StreamEvent(
                kind="done",
                content="final",
                metadata={"input_tokens": 500, "output_tokens": 200},
            )
        )
        result = display._render()
        assert isinstance(result, RichGroup)
        # Walk the Group's children to find the stats line
        texts = [str(c) for c in result.renderables if hasattr(c, "markup")]
        combined = " ".join(texts)
        assert "500" in combined
        assert "200" in combined

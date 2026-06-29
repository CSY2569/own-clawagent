"""Agent creation and invocation logic."""

import contextlib
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr

# Ensure worker classes are registered before WorkerFactory is created
import clawagent.worker.coder
import clawagent.worker.critic
import clawagent.worker.researcher
import clawagent.worker.writer  # noqa: F401
from clawagent.compression import CompressionConfig, make_state_modifier
from clawagent.config import _PROJECT_ROOT, Settings
from clawagent.memory.summarizer import ensure_session_entry
from clawagent.memory.summarizer import save_messages as _save_messages
from clawagent.prompt_builder import PromptBuilder
from clawagent.stream_events import StreamEvent
from clawagent.tools import ALL_TOOLS
from clawagent.tools.memory_tools import configure as _configure_memory_tools

_PROMPTS_DIR = _PROJECT_ROOT / "prompts"


def _ensure_memory_dir(path: str) -> str:
    """Ensure the directory for the memory database exists."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path.resolve())


def _make_model(settings: Settings) -> Any:
    """Build a chat model via init_chat_model (supports any provider)."""
    return init_chat_model(
        model=settings.model_name,
        model_provider=settings.model_provider or None,
        api_key=SecretStr(settings.anthropic_api_key),
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )


def _make_sys_prompt(settings: Settings, memory_db_path: str) -> str:
    """Build the system prompt from layered prompt files and preferences."""
    return PromptBuilder(
        prompts_dir=_PROMPTS_DIR,
        memory_db_path=memory_db_path,
        max_preferences=settings.max_preferences,
    ).build(agent_id=settings.agent_id, source="cli")


def _make_all_tools() -> list[Any]:
    """Return all tools including delegate_task for worker delegation."""
    from clawagent.orchestrator.delegator import delegate_task

    return [*ALL_TOOLS, delegate_task]


def _make_compression_config(settings: Settings) -> CompressionConfig:
    """Build CompressionConfig from Settings."""
    return CompressionConfig(
        strategy=settings.compression_strategy,
        max_messages=settings.compression_max_messages,
        max_tokens=settings.compression_max_tokens,
        keep_recent=settings.compression_keep_recent,
    )


def create_agent(settings: Settings) -> tuple[CompiledStateGraph[Any], sqlite3.Connection]:
    """Build a tool-calling ReAct agent backed by Anthropic Claude.

    Returns (graph, db_connection) tuple. The caller must close the connection
    when done.
    """
    model = _make_model(settings)
    sys_prompt = _make_sys_prompt(settings, settings.memory_db_path)

    db_path = _ensure_memory_dir(settings.memory_db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)

    _configure_memory_tools(db_path, model)

    # ─── Initialize WorkerFactory ────────────────────
    from clawagent.orchestrator.delegator import configure_worker_factory
    from clawagent.worker.factory import WorkerFactory

    factory = WorkerFactory()
    factory.set_settings(settings)
    configure_worker_factory(factory)

    compression_config = _make_compression_config(settings)

    graph = create_react_agent(
        model=model,
        tools=_make_all_tools(),
        checkpointer=SqliteSaver(conn),
        prompt=sys_prompt,
        pre_model_hook=make_state_modifier(config=compression_config, model=model),
    )
    return graph, conn


def rebuild_graph(
    settings: Settings, db_path: str, conn: sqlite3.Connection
) -> CompiledStateGraph[Any]:
    """Rebuild agent graph with new model settings, reusing existing DB connection.

    Use this for hot-reloading model parameters (model name, temperature, max_tokens)
    without losing conversation state stored in the checkpointer.
    """
    model = _make_model(settings)
    sys_prompt = _make_sys_prompt(settings, db_path)

    _configure_memory_tools(db_path, model)

    compression_config = _make_compression_config(settings)

    return create_react_agent(
        model=model,
        tools=_make_all_tools(),
        checkpointer=SqliteSaver(conn),
        prompt=sys_prompt,
        pre_model_hook=make_state_modifier(config=compression_config, model=model),
    )


@dataclass
class Usage:
    """Token usage for a single agent invocation."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_response_metadata(cls, metadata: dict[str, Any]) -> Usage:
        usage = metadata.get("usage", {})
        if not usage:
            return cls()
        return cls(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        )


@dataclass
class AgentResponse:
    """Result of a single agent invocation."""

    text: str
    usage: Usage


def _extract_text(content: Any) -> str:
    """Extract readable text from an AI message content.

    Handles both plain strings and content blocks (e.g. DeepSeek's
    thinking/text blocks in Anthropic API format).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content) if content is not None else ""


class Agent:
    """Wrapper around the compiled LangGraph agent graph."""

    def __init__(
        self,
        graph: CompiledStateGraph[Any],
        db_path: str = "",
        conn: sqlite3.Connection | None = None,
        default_thread_id: str | None = None,
    ) -> None:
        self._graph = graph
        self._db_path = db_path
        self._conn = conn
        self._thread_id = default_thread_id or uuid4().hex[:8]

    @property
    def thread_id(self) -> str:
        return self._thread_id

    def reconfigure(self, settings: Settings) -> None:
        """Hot-reload model settings without losing conversation state."""
        if not self._conn:
            return
        self._graph = rebuild_graph(settings, self._db_path, self._conn)

    def close(self) -> None:
        """Release resources held by this agent, particularly the SQLite connection."""
        if self._conn:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None

    def run(self, message: str, thread_id: str | None = None) -> AgentResponse:
        """Run the agent synchronously and return the response with usage."""
        tid = thread_id or self._thread_id
        result = self._graph.invoke(
            {"messages": [("user", message)]},
            config={"configurable": {"thread_id": tid}},
        )
        messages = result["messages"]
        last_msg = messages[-1]
        text = _extract_text(last_msg.content)
        metadata = getattr(last_msg, "response_metadata", None) or {}
        usage = Usage.from_response_metadata(metadata)

        # Save messages to conversation log and ensure session is discoverable
        if self._db_path:
            with contextlib.suppress(Exception):
                _save_messages(self._db_path, tid, [("user", message), ("assistant", text)])
                ensure_session_entry(self._db_path, tid, message)
                from clawagent.memory.preferences import extract_preferences_from_messages
                extract_preferences_from_messages(
                    messages_text=message + "\n" + text,
                    session_id=tid,
                    db_path=self._db_path,
                )

        return AgentResponse(text=text, usage=usage)

    def stream_events(
        self, message: str, thread_id: str | None = None
    ) -> Iterator[StreamEvent]:
        """Stream agent execution at message-chunk granularity.

        Uses stream_mode="messages" for token-level typewriter effect.
        Yields: think_start, think_end, token, tool_call, tool_result, error, done.
        """
        import json

        tid = thread_id or self._thread_id
        all_text: list[str] = []
        tool_call_accum: dict[str, dict[str, str]] = {}
        usage = Usage()
        in_thinking: bool = False
        tool_calls_emitted: set[str] = set()

        try:
            for msg_chunk, metadata in self._graph.stream(
                {"messages": [("user", message)]},
                config={"configurable": {"thread_id": tid}},
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "") if isinstance(metadata, dict) else ""

                if node == "agent":
                    # ── Text chunk ──
                    chunk_text = getattr(msg_chunk, "content", None)
                    if chunk_text:
                        if isinstance(chunk_text, list):
                            for block in chunk_text:
                                if not isinstance(block, dict):
                                    continue
                                btype = block.get("type", "")
                                text = block.get("text", "") or block.get("thinking", "")
                                if btype == "thinking" and text:
                                    if not in_thinking:
                                        in_thinking = True
                                        yield StreamEvent(kind="think_start")
                                elif btype == "text" and text:
                                    if in_thinking:
                                        in_thinking = False
                                        yield StreamEvent(kind="think_end")
                                    all_text.append(text)
                                    yield StreamEvent(kind="token", node="agent", content=text)
                        elif isinstance(chunk_text, str) and chunk_text:
                            all_text.append(chunk_text)
                            yield StreamEvent(kind="token", node="agent", content=chunk_text)

                    # ── Tool call chunks (incremental) ──
                    tcc_list = getattr(msg_chunk, "tool_call_chunks", None)
                    if tcc_list:
                        all_text.clear()  # C2: discard pre-tool filler text
                        for tcc in tcc_list:
                            tc_id = tcc.get("id", "")
                            if not tc_id:
                                continue
                            if tc_id not in tool_call_accum:
                                tool_call_accum[tc_id] = {"name": "", "args": ""}
                            name_piece = tcc.get("name", "") or ""
                            args_piece = tcc.get("args", "") or ""
                            if name_piece:
                                tool_call_accum[tc_id]["name"] += name_piece
                            if args_piece:
                                tool_call_accum[tc_id]["args"] += args_piece

                    # ── Usage extraction ──
                    meta = getattr(msg_chunk, "response_metadata", None) or {}
                    if meta.get("usage"):
                        usage = Usage.from_response_metadata(meta)

                elif node == "tools":
                    if in_thinking:
                        in_thinking = False
                        yield StreamEvent(kind="think_end")

                    for tc_id, acc in tool_call_accum.items():
                        if tc_id in tool_calls_emitted:
                            continue
                        name = acc["name"]
                        if not name:
                            continue
                        try:
                            args = json.loads(acc["args"]) if acc["args"] else {}
                        except (json.JSONDecodeError, ValueError):
                            args = {"raw": acc["args"]}
                        tool_calls_emitted.add(tc_id)
                        yield StreamEvent(
                            kind="tool_call",
                            node="agent",
                            content=name,
                            metadata={"args": args},
                        )

                    preview = str(getattr(msg_chunk, "content", ""))[:200]
                    yield StreamEvent(
                        kind="tool_result",
                        node="tools",
                        content=getattr(msg_chunk, "name", ""),
                        metadata={"preview": preview},
                    )

        except Exception as e:
            yield StreamEvent(kind="error", content=f"{type(e).__name__}: {e}")
            return

        # ── Usage fallback via get_state ──
        if usage.input_tokens == 0 and usage.output_tokens == 0:
            try:
                final_state = self._graph.get_state({"configurable": {"thread_id": tid}})
                if final_state and final_state.values:
                    msgs = final_state.values.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        meta = getattr(last, "response_metadata", None) or {}
                        if meta.get("usage"):
                            usage = Usage.from_response_metadata(meta)
            except Exception:
                pass

        final_text = "".join(all_text)

        # Persist conversation
        if self._db_path and final_text:
            with contextlib.suppress(Exception):
                _save_messages(self._db_path, tid, [
                    ("user", message),
                    ("assistant", final_text),
                ])
                ensure_session_entry(self._db_path, tid, message)
                from clawagent.memory.preferences import extract_preferences_from_messages
                extract_preferences_from_messages(
                    messages_text=message + "\n" + final_text,
                    session_id=tid,
                    db_path=self._db_path,
                )

        yield StreamEvent(
            kind="done",
            content=final_text,
            metadata={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            },
        )

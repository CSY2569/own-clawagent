"""Agent creation and invocation logic."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain.agents import create_agent as _create_agent
from langchain.agents.middleware import before_model
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

try:
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
except ImportError:  # pragma: no cover
    AnthropicPromptCachingMiddleware = None  # type: ignore

import clawagent.worker  # noqa: F401  # ensures worker classes are registered
from clawagent.compression import CompressionConfig, make_state_modifier
from clawagent.config import PROJECT_ROOT, Settings
from clawagent.memory.summarizer import ensure_session_entry
from clawagent.memory.summarizer import save_messages as _save_messages
from clawagent.model_factory import make_model as _make_model
from clawagent.prompt_builder import PromptBuilder
from clawagent.stream_events import StreamEvent
from clawagent.stream_processor import (
    StreamState,
    emit_tool_events,
    process_text_chunk,
    process_tool_call_chunks,
)
from clawagent.stream_processor import (
    extract_usage as _extract_usage,
)
from clawagent.tools import ALL_TOOLS
from clawagent.tools.memory_tools import create_memory_tools
from clawagent.types import AgentResponse, Usage
from clawagent.utils import extract_text

if TYPE_CHECKING:
    from clawagent.worker.factory import WorkerFactory

logger = logging.getLogger(__name__)

_PROMPTS_DIR = PROJECT_ROOT / "prompts"

_confirm_fn: Any | None = None


def set_confirm_fn(fn: Any | None) -> None:
    """Set the interactive confirmation callback for permission-controlled tools.

    Called by the REPL before ``create_agent``. The callback receives
    (tool_name, args) and returns True to approve, False to deny.
    """
    global _confirm_fn
    _confirm_fn = fn

__all__ = [
    "Agent",
    "AgentResponse",
    "Usage",
    "create_agent",
    "rebuild_graph",
]


def _ensure_memory_dir(path: str) -> str:
    """Ensure the directory for the memory database exists."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path.resolve())


def _make_sys_prompt(
    settings: Settings,
    memory_db_path: str,
    delegate_tool: BaseTool | None = None,
    channel: str = "cli",
) -> str:
    """Build the system prompt from layered prompt files and preferences."""
    extra = ""
    if channel != "cli":
        extra = (
            "## File Sharing\n"
            "When the user asks you to send a file, use write_file to save it, "
            "then include [FILE:path] in your response. Use the SAME relative path "
            "you passed to write_file (e.g. 'output/report.pdf'). "
            "Example: 'Here is the report. [FILE:output/report.pdf]'"
        )
    return PromptBuilder(
        prompts_dir=_PROMPTS_DIR,
        memory_db_path=memory_db_path,
        max_preferences=settings.max_preferences,
    ).build(
        agent_id=settings.agent_id,
        source=channel,
        delegate_tool=delegate_tool,
        extra_context=extra,
    )


def _make_all_tools(
    delegate_tool: BaseTool | None, memory_tools: list[BaseTool] | None = None
) -> list[BaseTool]:
    """Return all tools including the given delegate_task closure and memory tools."""
    tools: list[BaseTool] = [*ALL_TOOLS]
    if delegate_tool is not None:
        tools.append(delegate_tool)
    if memory_tools:
        tools.extend(memory_tools)
    return tools


def _make_compression_config(settings: Settings) -> CompressionConfig:
    """Build CompressionConfig from Settings."""
    return CompressionConfig(
        strategy=settings.compression_strategy,
        max_messages=settings.compression_max_messages,
        max_tokens=settings.compression_max_tokens,
        keep_recent=settings.compression_keep_recent,
        summary_timeout=settings.compression_summary_timeout,
    )


def _is_anthropic_provider(settings: Settings) -> bool:
    """Check whether the current platform/provider is Anthropic-native.

    Only Anthropic-native models support prompt caching middleware.
    OpenAI-compatible providers (deepseek/ark/opencode-go/openai)
    should skip it to avoid unnecessary overhead.
    """
    if settings.platform:
        return settings.platform == "anthropic"
    return settings.model_provider == "anthropic"


def _build_middleware(
    compression_config: CompressionConfig, model: Any, settings: Settings
) -> list[Any]:
    """Build middleware list: prompt caching (Anthropic only) + context compression."""
    middleware: list[Any] = []
    if AnthropicPromptCachingMiddleware is not None and _is_anthropic_provider(settings):
        middleware.append(
            AnthropicPromptCachingMiddleware(
                type="ephemeral",
                ttl="5m",
                unsupported_model_behavior="ignore",
            )
        )
    middleware.append(
        before_model(
            lambda state, runtime: make_state_modifier(config=compression_config, model=model)(
                state
            )
        )
    )
    return middleware


def _make_secured_tools(
    delegate_tool: BaseTool | None,
    memory_tools: list[BaseTool] | None,
    settings: Settings,
    thread_id: str = "",
    confirm_fn: Any = None,
) -> list[BaseTool]:
    """Build tool list with permission middleware wrapping."""
    from clawagent.security import AuditLogger, PermissionConfig, wrap_tools

    raw_tools = _make_all_tools(delegate_tool, memory_tools)
    perm_config = PermissionConfig()
    audit = AuditLogger(settings.audit_log_path)
    return wrap_tools(
        raw_tools,
        perm_config,
        audit,
        thread_id=thread_id,
        auto_confirm=settings.auto_confirm,
        confirm_fn=confirm_fn or _confirm_fn,
    )


def create_agent(
    settings: Settings,
    channel: str = "cli",
) -> tuple[CompiledStateGraph[Any], sqlite3.Connection, WorkerFactory, BaseTool]:
    """Build a tool-calling ReAct agent backed by Anthropic Claude.

    Args:
        settings: Application settings.
        channel: Channel identifier for prompt context ("cli", "wechat", etc.).

    Returns (graph, db_connection, worker_factory, delegate_tool) tuple.
    The caller must close the connection when done.
    """
    model = _make_model(settings)

    from clawagent.orchestrator.delegator import make_delegate_task
    from clawagent.worker.base import BaseWorker
    from clawagent.worker.factory import WorkerFactory

    BaseWorker.set_agent_class(Agent)

    factory = WorkerFactory()
    factory.set_settings(settings)
    delegate_tool = make_delegate_task(factory)

    db_path = _ensure_memory_dir(settings.memory_db_path)
    sys_prompt = _make_sys_prompt(settings, db_path, delegate_tool, channel=channel)

    conn = sqlite3.connect(db_path, check_same_thread=False)

    memory_tools = create_memory_tools(db_path, model)

    compression_config = _make_compression_config(settings)
    middleware = _build_middleware(compression_config, model, settings)

    graph = _create_agent(
        model=model,
        tools=_make_secured_tools(delegate_tool, memory_tools, settings),
        checkpointer=SqliteSaver(conn),
        system_prompt=sys_prompt,
        middleware=middleware,
    )
    return graph, conn, factory, delegate_tool


def rebuild_graph(
    settings: Settings,
    db_path: str,
    conn: sqlite3.Connection,
    delegate_tool: BaseTool | None,
    channel: str = "cli",
) -> CompiledStateGraph[Any]:
    """Rebuild agent graph with new model settings, reusing existing DB connection.

    Use this for hot-reloading model parameters (model name, temperature, max_tokens)
    without losing conversation state stored in the checkpointer.
    """
    model = _make_model(settings)
    sys_prompt = _make_sys_prompt(settings, db_path, delegate_tool, channel=channel)

    memory_tools = create_memory_tools(db_path, model)

    compression_config = _make_compression_config(settings)
    middleware = _build_middleware(compression_config, model, settings)

    return _create_agent(
        model=model,
        tools=_make_secured_tools(delegate_tool, memory_tools, settings),
        checkpointer=SqliteSaver(conn),
        system_prompt=sys_prompt,
        middleware=middleware,
    )


class Agent:
    """Wrapper around the compiled LangGraph agent graph."""

    def __init__(
        self,
        graph: CompiledStateGraph[Any],
        db_path: str = "",
        conn: sqlite3.Connection | None = None,
        default_thread_id: str | None = None,
        factory: WorkerFactory | None = None,
        delegate_tool: BaseTool | None = None,
        channel: str = "cli",
    ) -> None:
        self._graph = graph
        self._db_path = db_path
        self._conn = conn
        self._thread_id = default_thread_id or uuid4().hex[:8]
        self._factory = factory
        self._delegate_tool = delegate_tool
        self._channel = channel
        self._turn_count: int = 0

    @property
    def thread_id(self) -> str:
        return self._thread_id

    def reconfigure(self, settings: Settings) -> None:
        """Hot-reload model settings without losing conversation state.

        Also propagates settings to WorkerFactory so subsequently spawned
        workers see the new configuration.
        """
        if not self._conn:
            return
        if self._factory is not None:
            self._factory.set_settings(settings)
        self._graph = rebuild_graph(
            settings,
            self._db_path,
            self._conn,
            self._delegate_tool,
            channel=self._channel,
        )

    def _persist_turn(self, thread_id: str, user_msg: str, assistant_msg: str) -> None:
        """Save turn to conversation log, session index, and preference store."""
        if not self._db_path or not assistant_msg:
            return
        try:
            _save_messages(
                self._db_path,
                thread_id,
                [
                    ("user", user_msg),
                    ("assistant", assistant_msg),
                ],
            )
            ensure_session_entry(self._db_path, thread_id, user_msg)

            self._turn_count += 1
            if self._turn_count % 10 == 0:
                import threading

                threading.Thread(
                    target=self._extract_memories_async,
                    args=(thread_id, user_msg, assistant_msg),
                    daemon=True,
                ).start()
        except Exception:
            logger.exception("Failed to persist turn thread_id=%s", thread_id)

    def _extract_memories_async(self, thread_id: str, user_msg: str, assistant_msg: str) -> None:
        """Background memory extraction - preferences, profile, and facts."""
        try:
            from clawagent.memory.preferences import extract_memories_from_messages

            extract_memories_from_messages(
                messages_text=user_msg + "\n" + assistant_msg,
                session_id=thread_id,
                db_path=self._db_path,
            )
        except Exception:
            logger.exception("Memory extraction failed thread_id=%s", thread_id)

    def close(self) -> None:
        """Release resources held by this agent, particularly the SQLite connection."""
        from clawagent.memory.summarizer import close_all_cached

        if self._conn:
            try:
                self._conn.close()
            except Exception:
                logger.exception("Failed to close SQLite connection")
            self._conn = None
        try:
            close_all_cached()
        except Exception:
            logger.exception("Failed to close cached connections")

    def run(self, message: str, thread_id: str | None = None) -> AgentResponse:
        """Run the agent synchronously and return the response with usage."""
        tid = thread_id or self._thread_id
        result = self._graph.invoke(
            {"messages": [("user", message)]},
            config={"configurable": {"thread_id": tid}},
        )
        messages = result["messages"]
        last_msg = messages[-1]
        text = extract_text(last_msg.content)
        usage = _extract_usage(last_msg)

        self._persist_turn(tid, message, text)

        return AgentResponse(text=text, usage=usage)

    def stream_events(self, message: str, thread_id: str | None = None) -> Iterator[StreamEvent]:
        """Stream agent execution at message-chunk granularity."""
        tid = thread_id or self._thread_id
        state = StreamState()
        current_node = "agent"

        try:
            for msg_chunk, metadata in self._graph.stream(
                {"messages": [("user", message)]},
                config={"configurable": {"thread_id": tid}},
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node", "") if isinstance(metadata, dict) else ""
                if node:
                    current_node = node

                try:
                    if node in ("agent", "model"):
                        chunk_text = getattr(msg_chunk, "content", None)
                        if chunk_text:
                            yield from process_text_chunk(chunk_text, state)
                        process_tool_call_chunks(msg_chunk, state)

                        chunk_usage = _extract_usage(msg_chunk)
                        if chunk_usage.input_tokens > 0 or chunk_usage.output_tokens > 0:
                            state.usage = chunk_usage

                    elif node == "tools":
                        yield from emit_tool_events(msg_chunk, state)
                except Exception as e:
                    logger.exception("Error processing chunk at node=%s", current_node)
                    yield StreamEvent(
                        kind="error",
                        node=current_node,
                        content=f"{type(e).__name__}: {e}",
                    )

        except Exception as e:
            logger.exception("Stream failed at node=%s thread_id=%s", current_node, tid)
            yield StreamEvent(
                kind="error",
                node=current_node,
                content=f"{type(e).__name__}: {e}",
            )

        if state.usage.input_tokens == 0 and state.usage.output_tokens == 0:
            state.usage = self._extract_usage_fallback(tid)

        final_text = "".join(state.all_text)
        self._persist_turn(tid, message, final_text)

        yield StreamEvent(
            kind="done",
            content=final_text,
            metadata={
                "input_tokens": state.usage.input_tokens,
                "output_tokens": state.usage.output_tokens,
                "cache_read_input_tokens": state.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": state.usage.cache_creation_input_tokens,
                "prompt_cache_hit_tokens": state.usage.prompt_cache_hit_tokens,
                "prompt_cache_miss_tokens": state.usage.prompt_cache_miss_tokens,
            },
        )

    def _extract_usage_fallback(self, thread_id: str) -> Usage:
        """Attempt to extract usage from graph state as fallback."""
        try:
            final_state = self._graph.get_state({"configurable": {"thread_id": thread_id}})
            if final_state and final_state.values:
                msgs = final_state.values.get("messages", [])
                if msgs:
                    return _extract_usage(msgs[-1])
        except Exception:
            logger.exception("Failed to extract usage fallback thread_id=%s", thread_id)
        return Usage()

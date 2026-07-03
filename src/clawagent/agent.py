"""Agent creation and invocation logic."""

import contextlib
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain.agents import create_agent as _create_agent
from langchain.agents.middleware import before_model
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

try:
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
except ImportError:  # pragma: no cover
    AnthropicPromptCachingMiddleware = None  # type: ignore

import clawagent.worker  # noqa: F401  # ensures worker classes are registered
from clawagent.api_pool import KeyPoolChatModel, get_global_pool
from clawagent.compression import CompressionConfig, make_state_modifier
from clawagent.config import PROJECT_ROOT, Settings
from clawagent.memory.summarizer import ensure_session_entry
from clawagent.memory.summarizer import save_messages as _save_messages
from clawagent.prompt_builder import PromptBuilder
from clawagent.stream_events import StreamEvent
from clawagent.tools import ALL_TOOLS
from clawagent.tools.memory_tools import create_memory_tools

if TYPE_CHECKING:
    from clawagent.worker.factory import WorkerFactory

_PROMPTS_DIR = PROJECT_ROOT / "prompts"

_PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def _get_api_key(settings: Settings) -> str:
    """Get API key for the current model provider from environment."""
    env_var = _PROVIDER_KEY_ENV.get(settings.model_provider)
    if env_var:
        key = os.getenv(env_var, "")
        if key:
            return key
    return settings.anthropic_api_key


def _ensure_memory_dir(path: str) -> str:
    """Ensure the directory for the memory database exists."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path.resolve())


def _make_model(settings: Settings) -> Any:
    """Build a chat model via init_chat_model, with optional key-pool wrapping."""
    model = init_chat_model(
        model=settings.model_name,
        model_provider=settings.model_provider or None,
        api_key=SecretStr(_get_api_key(settings)),
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
    )

    # Wrap with KeyPoolChatModel if a pool with keys is configured
    pool = get_global_pool()
    default_stats = pool.get_pool_stats("default")
    if default_stats.get("total", 0) > 0:
        model = KeyPoolChatModel(
            pool=pool,
            pool_name="default",
            inner=model,
        )

    return model


def _make_sys_prompt(settings: Settings, memory_db_path: str, delegate_tool: BaseTool | None = None) -> str:
    """Build the system prompt from layered prompt files and preferences."""
    return PromptBuilder(
        prompts_dir=_PROMPTS_DIR,
        memory_db_path=memory_db_path,
        max_preferences=settings.max_preferences,
    ).build(agent_id=settings.agent_id, source="cli", delegate_tool=delegate_tool)


def _make_all_tools(delegate_tool: BaseTool | None, memory_tools: list[BaseTool] | None = None) -> list[BaseTool]:
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
    )


def _build_middleware(compression_config: CompressionConfig, model: Any) -> list[Any]:
    """Build middleware list: prompt caching + context compression."""
    middleware: list[Any] = []
    if AnthropicPromptCachingMiddleware is not None:
        middleware.append(
            AnthropicPromptCachingMiddleware(
                type="ephemeral",
                ttl="5m",
                unsupported_model_behavior="ignore",
            )
        )
    middleware.append(
        before_model(
            lambda state, runtime: make_state_modifier(
                config=compression_config, model=model
            )(state)
        )
    )
    return middleware


def create_agent(settings: Settings) -> tuple[CompiledStateGraph[Any], sqlite3.Connection, WorkerFactory, BaseTool]:
    """Build a tool-calling ReAct agent backed by Anthropic Claude.

    Returns (graph, db_connection, worker_factory, delegate_tool) tuple.
    The caller must close the connection when done.
    """
    model = _make_model(settings)

    # ─── Initialize WorkerFactory ────────────────────
    from clawagent.orchestrator.delegator import make_delegate_task
    from clawagent.worker.base import BaseWorker
    from clawagent.worker.factory import WorkerFactory

    BaseWorker.set_agent_class(Agent)

    factory = WorkerFactory()
    factory.set_settings(settings)
    delegate_tool = make_delegate_task(factory)

    sys_prompt = _make_sys_prompt(settings, settings.memory_db_path, delegate_tool)

    db_path = _ensure_memory_dir(settings.memory_db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)

    memory_tools = create_memory_tools(db_path, model)

    compression_config = _make_compression_config(settings)
    middleware = _build_middleware(compression_config, model)

    graph = _create_agent(
        model=model,
        tools=_make_all_tools(delegate_tool, memory_tools),
        checkpointer=SqliteSaver(conn),
        system_prompt=sys_prompt,
        middleware=middleware,
    )
    return graph, conn, factory, delegate_tool


def rebuild_graph(
    settings: Settings, db_path: str, conn: sqlite3.Connection, delegate_tool: BaseTool | None
) -> CompiledStateGraph[Any]:
    """Rebuild agent graph with new model settings, reusing existing DB connection.

    Use this for hot-reloading model parameters (model name, temperature, max_tokens)
    without losing conversation state stored in the checkpointer.
    """
    model = _make_model(settings)
    sys_prompt = _make_sys_prompt(settings, db_path, delegate_tool)

    memory_tools = create_memory_tools(db_path, model)

    compression_config = _make_compression_config(settings)
    middleware = _build_middleware(compression_config, model)

    return _create_agent(
        model=model,
        tools=_make_all_tools(delegate_tool, memory_tools),
        checkpointer=SqliteSaver(conn),
        system_prompt=sys_prompt,
        middleware=middleware,
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


def _extract_usage(msg: Any) -> Usage:
    """Extract Usage from message, checking usage_metadata first (ChatAnthropic)."""
    usage_dict = getattr(msg, "usage_metadata", None)
    if usage_dict:
        return Usage(
            input_tokens=usage_dict.get("input_tokens", 0),
            output_tokens=usage_dict.get("output_tokens", 0),
            cache_read_input_tokens=usage_dict.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=usage_dict.get("cache_creation_input_tokens", 0),
        )
    meta = getattr(msg, "response_metadata", None) or {}
    return Usage.from_response_metadata(meta)


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
        factory: WorkerFactory | None = None,
        delegate_tool: BaseTool | None = None,
    ) -> None:
        self._graph = graph
        self._db_path = db_path
        self._conn = conn
        self._thread_id = default_thread_id or uuid4().hex[:8]
        self._factory = factory
        self._delegate_tool = delegate_tool

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
        self._graph = rebuild_graph(settings, self._db_path, self._conn, self._delegate_tool)

    def _persist_turn(self, thread_id: str, user_msg: str, assistant_msg: str) -> None:
        """Save turn to conversation log, session index, and preference store."""
        if not self._db_path or not assistant_msg:
            return
        with contextlib.suppress(Exception):
            _save_messages(self._db_path, thread_id, [
                ("user", user_msg), ("assistant", assistant_msg),
            ])
            ensure_session_entry(self._db_path, thread_id, user_msg)
            from clawagent.memory.preferences import extract_preferences_from_messages
            extract_preferences_from_messages(
                messages_text=user_msg + "\n" + assistant_msg,
                session_id=thread_id,
                db_path=self._db_path,
            )

    def close(self) -> None:
        """Release resources held by this agent, particularly the SQLite connection."""
        from clawagent.memory.summarizer import close_all_cached

        if self._conn:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None
        close_all_cached()

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
        usage = _extract_usage(last_msg)

        self._persist_turn(tid, message, text)

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

                if node in ("agent", "model"):
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
                    chunk_usage = _extract_usage(msg_chunk)
                    if chunk_usage.input_tokens > 0 or chunk_usage.output_tokens > 0:
                        usage = chunk_usage

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

        # ── Usage fallback via get_state ──
        if usage.input_tokens == 0 and usage.output_tokens == 0:
            try:
                final_state = self._graph.get_state({"configurable": {"thread_id": tid}})
                if final_state and final_state.values:
                    msgs = final_state.values.get("messages", [])
                    if msgs:
                        usage = _extract_usage(msgs[-1])
            except Exception:
                pass

        final_text = "".join(all_text)

        self._persist_turn(tid, message, final_text)

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

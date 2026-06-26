"""Agent creation and invocation logic."""

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr

from clawagent.config import Settings
from clawagent.memory.summarizer import ensure_session_entry
from clawagent.memory.summarizer import save_messages as _save_messages
from clawagent.prompt_builder import PromptBuilder
from clawagent.tools import ALL_TOOLS
from clawagent.tools.memory_tools import configure as _configure_memory_tools

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPTS_DIR = _PROJECT_ROOT / "prompts"


def _ensure_memory_dir(path: str) -> str:
    """Ensure the directory for the memory database exists."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path.resolve())


def create_agent(settings: Settings) -> tuple[CompiledStateGraph[Any], sqlite3.Connection]:
    """Build a tool-calling ReAct agent backed by Anthropic Claude.

    Returns (graph, db_connection) tuple. The caller must close the connection
    when done.
    """
    model = ChatAnthropic(
        model=settings.model_name,
        api_key=SecretStr(settings.anthropic_api_key),
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )  # type: ignore[call-arg]

    builder = PromptBuilder(
        prompts_dir=_PROMPTS_DIR,
        memory_db_path=settings.memory_db_path,
        max_preferences=settings.max_preferences,
    )
    sys_prompt = builder.build(agent_id=settings.agent_id, source="cli")

    db_path = _ensure_memory_dir(settings.memory_db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)

    # Configure memory tools with database path and model
    _configure_memory_tools(db_path, model)

    graph = create_react_agent(
        model=model,
        tools=ALL_TOOLS,
        checkpointer=saver,
        prompt=sys_prompt,
    )
    return graph, conn


def rebuild_graph(
    settings: Settings, db_path: str, conn: sqlite3.Connection
) -> CompiledStateGraph[Any]:
    """Rebuild agent graph with new model settings, reusing existing DB connection.

    Use this for hot-reloading model parameters (model name, temperature, max_tokens)
    without losing conversation state stored in the checkpointer.
    """
    model = ChatAnthropic(
        model=settings.model_name,
        api_key=SecretStr(settings.anthropic_api_key),
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )  # type: ignore[call-arg]

    sys_prompt = PromptBuilder(
        prompts_dir=_PROMPTS_DIR,
        memory_db_path=db_path,
        max_preferences=settings.max_preferences,
    ).build(agent_id=settings.agent_id, source="cli")
    saver = SqliteSaver(conn)

    _configure_memory_tools(db_path, model)

    return create_react_agent(
        model=model,
        tools=ALL_TOOLS,
        checkpointer=saver,
        prompt=sys_prompt,
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
    return str(content)


_DB_PATH: str = ""  # module-level ref for Agent to save messages


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
        self._db_path = db_path or _DB_PATH
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
            _save_messages(self._db_path, tid, [("user", message), ("assistant", text)])
            ensure_session_entry(self._db_path, tid, message)

        return AgentResponse(text=text, usage=usage)

    def stream(self, message: str) -> Iterator[str]:
        """Stream the agent's response token-by-token."""
        for chunk in self._graph.stream(
            {"messages": [("user", message)]},
            config={"configurable": {"thread_id": self._thread_id}},
            stream_mode="values",
        ):
            if chunk.get("messages"):
                last_msg = chunk["messages"][-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    yield str(last_msg.content)

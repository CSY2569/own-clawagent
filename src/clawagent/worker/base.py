"""Base worker — each worker is a temporary Agent with isolated context."""

from __future__ import annotations

import contextlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

from clawagent.config import Settings
from clawagent.prompt_builder import PromptBuilder
from clawagent.worker.config import WorkerConfig

if TYPE_CHECKING:
    from clawagent.agent import Agent


class BaseWorker(ABC):
    """Worker base class.

    Each worker instance = a temporary Agent with:
    - Independent PromptBuilder (loaded from prompts/agents/<role>/)
    - Independent model (configured via WorkerConfig)
    - Independent tool set (only tools within its responsibility)
    - Independent SQLite memory database
    - Independent thread_id

    Subclasses implement _get_tools() to return tool list,
    and optionally _customize_prompt() to adjust the prompt.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._agent: Any = None
        self._conn: sqlite3.Connection | None = None

    @abstractmethod
    def _get_tools(self) -> list[Any]:
        """Subclasses return the tool list available to this worker."""
        ...

    def _customize_prompt(self, prompt: str, task: str) -> str:
        """Subclasses may override to inject additional context into the prompt.

        Default: append the task description at the end.
        """
        return f"{prompt}\n\n## Current Task\n{task}"

    def build_prompt(self, task: str) -> str:
        """Assemble the worker's system prompt."""
        prompts_dir = self.config.prompts_dir or str(
            Path(__file__).resolve().parent.parent.parent.parent / "prompts"
        )
        builder = PromptBuilder(
            prompts_dir=prompts_dir,
            memory_db_path=self.config.memory_db,
        )
        base_prompt = builder.build(
            agent_id=self.config.role,
            source="worker",
        )
        return self._customize_prompt(base_prompt, task)

    def spawn(self, task: str, settings: Settings | None = None) -> Agent:
        """Create a worker agent instance.

        Each call creates a fresh Agent with independent:
        - LLM model instance (via init_chat_model, supports any provider)
        - PromptBuilder + task-specific prompt
        - Independent SQLite checkpointer
        - Independent thread_id

        Returns the Agent wrapper (not a CompiledStateGraph).
        """
        from clawagent.agent import Agent

        if settings is None:
            settings = Settings.from_env()

        api_key = self.config.api_key
        if not api_key:
            api_key = settings.anthropic_api_key

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "request_timeout": self.config.request_timeout,
        }
        if self.config.api_base:
            kwargs["base_url"] = self.config.api_base

        model = init_chat_model(
            model=self.config.model,
            model_provider=self.config.model_provider or None,
            **kwargs,
        )

        sys_prompt = self.build_prompt(task)

        db_path = self._ensure_memory_db()
        conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn = conn
        saver = SqliteSaver(conn)

        graph = create_react_agent(
            model=model,
            tools=self._get_tools(),
            checkpointer=saver,
            prompt=sys_prompt,
        )

        agent = Agent(graph=graph, db_path=db_path, conn=conn, default_thread_id=uuid4().hex[:8])
        self._agent = agent
        return agent

    def run(self, task: str) -> str:
        """Create worker → execute task → return result. Cleans up after."""
        agent = self.spawn(task)
        try:
            response = agent.run(task)
            return response.text
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Destroy worker: close DB connection, release resources."""
        if self._agent is not None:
            with contextlib.suppress(Exception):
                self._agent.close()
            self._agent = None
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None

    def _ensure_memory_db(self) -> str:
        """Ensure the memory database directory exists, return absolute path."""
        db_path = self.config.memory_db or f"memories/{self.config.role}.db"
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())

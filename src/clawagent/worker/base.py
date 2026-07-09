"""Base worker — each worker is a temporary Agent with isolated context."""

from __future__ import annotations

import importlib
import logging
import sqlite3
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver

from clawagent.api_pool import KeyPoolChatModel, get_global_pool
from clawagent.config import Settings
from clawagent.prompt_builder import PromptBuilder
from clawagent.worker.config import _WORKER_TOOL_MAP, WORKER_TOOLS, WorkerConfig

logger = logging.getLogger(__name__)


def _resolve_tools(tool_names: list[str]) -> list[Any]:
    """Resolve tool name strings to actual tool objects via lazy import."""
    tools: list[Any] = []
    for name in tool_names:
        module_path = _WORKER_TOOL_MAP.get(name)
        if module_path is None:
            continue
        module = importlib.import_module(module_path)
        tool = getattr(module, name, None)
        if tool is not None:
            tools.append(tool)
    return tools


class BaseWorker:
    """Worker base class.

    Each worker instance = a temporary Agent with:
    - Independent PromptBuilder (loaded from prompts/agents/<role>/)
    - Independent model (configured via WorkerConfig)
    - Independent tool set (only tools within its responsibility)
    - Independent SQLite memory database
    - Independent thread_id

    Subclasses may define _TOOLS class attribute to declare their tool list.
    If _TOOLS is empty, tools are resolved from WORKER_TOOLS config by role name.
    """

    _agent_class: ClassVar[Any | None] = None

    @classmethod
    def set_agent_class(cls, agent_cls: Any) -> None:
        """Inject Agent class reference to break circular import with agent.py."""
        cls._agent_class = agent_cls

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._agent: Any = None
        self._conn: sqlite3.Connection | None = None

    def _get_tools(self) -> list[Any]:
        """Return the tool list for this worker.

        Resolves from _TOOLS class attribute first, falls back to
        WORKER_TOOLS config by role name.
        """
        tool_names = getattr(self, "_TOOLS", None) or WORKER_TOOLS.get(self.config.role, [])
        return _resolve_tools(tool_names)

    def _customize_prompt(self, prompt: str, task: str, prompts_dir: str = "") -> str:
        """Subclasses may override to inject additional context into the prompt.

        Default: load task-template.md if available, otherwise append the task.
        """
        template = self._load_task_template(prompts_dir)
        if template:
            return f"{prompt}\n\n{template.replace('{task}', task)}"
        return f"{prompt}\n\n## Current Task\n{task}"

    def _load_task_template(self, prompts_dir: str) -> str | None:
        """Load a role-specific task template file, returning None if absent."""
        if not prompts_dir:
            return None
        path = Path(prompts_dir) / "agents" / self.config.role / "task-template.md"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return None

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
        return self._customize_prompt(base_prompt, task, prompts_dir)

    def spawn(self, task: str, settings: Settings | None = None) -> Any:
        """Create a worker agent instance.

        Each call creates a fresh Agent with independent:
        - LLM model instance (via init_chat_model, supports any provider)
        - PromptBuilder + task-specific prompt
        - Independent SQLite checkpointer
        - Independent thread_id

        Returns the Agent wrapper (not a CompiledStateGraph).
        """
        if self._agent_class is None:
            raise RuntimeError(
                "Agent class not injected. Call BaseWorker.set_agent_class() during startup."
            )

        if settings is None:
            settings = Settings.from_env()

        api_key = self.config.api_key
        if not api_key:
            api_key = settings.api_key

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "timeout": self.config.request_timeout,
        }
        if self.config.api_base:
            kwargs["base_url"] = self.config.api_base

        model = init_chat_model(
            model=self.config.model,
            model_provider=self.config.model_provider or None,
            **kwargs,
        )

        # Wrap with KeyPoolChatModel if a pool is configured for this worker
        pool_name = self.config.api_pool or "default"
        pool = get_global_pool()
        pool_stats = pool.get_pool_stats(pool_name)
        if pool_stats.get("total", 0) > 0:
            model = KeyPoolChatModel(
                pool=pool,
                pool_name=pool_name,
                inner=model,
            )

        sys_prompt = self.build_prompt(task)

        db_path = self._ensure_memory_db()
        conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn = conn
        saver = SqliteSaver(conn)

        graph = create_agent(
            model=model,
            tools=self._get_tools(),
            checkpointer=saver,
            system_prompt=sys_prompt,
        )

        agent = self._agent_class(
            graph=graph, db_path=db_path, conn=conn, default_thread_id=uuid4().hex[:8]
        )
        self._agent = agent
        return agent

    def run(self, task: str) -> Any:
        """Create worker → execute task → return result. Cleans up after."""
        agent: Any = self.spawn(task)
        try:
            response: Any = agent.run(task)
            return response.text
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Destroy worker: close DB connection, release resources."""
        if self._agent is not None:
            try:
                self._agent.close()
            except Exception:
                logger.exception("Failed to close worker agent role=%s", self.config.role)
            self._agent = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                logger.exception("Failed to close worker DB role=%s", self.config.role)
            self._conn = None

    def _ensure_memory_db(self) -> str:
        """Ensure the memory database directory exists, return absolute path."""
        db_path = self.config.memory_db or f"memories/{self.config.role}.db"
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())

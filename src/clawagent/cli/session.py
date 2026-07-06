"""Session initialization — wire up agent, RAG, logger, splash display."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console as RichConsole

from clawagent.agent import Agent, create_agent
from clawagent.api_pool import init_global_pool
from clawagent.config import PriceConfig, Settings, load_price_book
from clawagent.conversation_log import ConversationLogger
from clawagent.rag.bootstrap import bootstrap_rag
from clawagent.tools.rag_tool import configure_hybrid_search
from clawagent.ui import ConversationStats, render_splash

if TYPE_CHECKING:
    import sqlite3


@dataclass
class AgentRef:
    """Mutable reference to the current Agent instance.

    Used by slash commands (``/new``, ``/load``) to replace the agent
    when creating or switching sessions.
    """

    agent: Agent


@dataclass
class SessionContext:
    """All components needed for a REPL session."""

    agent_ref: AgentRef
    settings: Settings
    pricing: PriceConfig
    logger: ConversationLogger
    conn: sqlite3.Connection
    bm25_ready_signal: list[bool]
    stats: ConversationStats
    worker_roles: list[str]
    pool_all_stats: dict[str, dict[str, int]]


def init_session() -> SessionContext:
    """Initialize all components for a REPL session.

    Loads settings from environment, creates the agent, bootstraps RAG,
    and displays the splash screen.  Returns a ``SessionContext`` with
    everything the interactive loop needs.
    """
    settings = Settings.from_env()

    pool = init_global_pool()

    graph, conn, factory, delegate_tool = create_agent(settings)
    agent = Agent(
        graph,
        db_path=settings.memory_db_path,
        conn=conn,
        factory=factory,
        delegate_tool=delegate_tool,
    )
    agent_ref = AgentRef(agent=agent)

    logger = ConversationLogger()
    logger.log_session_start(agent.thread_id, settings)

    bm25_signal: list[bool] = []
    rag_ctx = bootstrap_rag(settings, configure_hybrid_search)
    if rag_ctx:
        bm25_signal = rag_ctx.bm25_ready_signal

    pricing = load_price_book().get(settings.model_name)

    from clawagent.worker.config import load_worker_configs

    worker_configs = load_worker_configs()
    worker_roles = list(worker_configs.keys())

    console = RichConsole()
    all_stats = pool.get_all_stats()
    render_splash(settings, pricing, console, pool_stats=all_stats, worker_roles=worker_roles)
    if rag_ctx:
        console.print("  [dim]BM25 索引后台构建中，搜索将临时使用纯向量检索...[/dim]")

    return SessionContext(
        agent_ref=agent_ref,
        settings=settings,
        pricing=pricing,
        logger=logger,
        conn=conn,
        bm25_ready_signal=bm25_signal,
        stats=ConversationStats(start_time=time.monotonic()),
        worker_roles=worker_roles,
        pool_all_stats=all_stats,
    )

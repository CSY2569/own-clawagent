"""delegate_task — Orchestrator tool for invoking worker agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from clawagent.config import Settings
    from clawagent.worker.factory import WorkerFactory

# Module-level state, set by agent.py on startup
_worker_factory: WorkerFactory | None = None


def configure_worker_factory(factory: WorkerFactory) -> None:
    """Set the module-level WorkerFactory for use by delegate_task."""
    global _worker_factory
    _worker_factory = factory


def update_worker_settings(settings: Settings) -> None:
    """Update runtime settings on the configured WorkerFactory for hot-reload.

    Call this after /model, /temp, /max-tokens, or /compress so that
    subsequently spawned workers see the new settings.
    """
    if _worker_factory is not None:
        _worker_factory.set_settings(settings)


@tool
def delegate_task(role: str, task: str) -> str:
    """将子任务委托给指定的 Worker Agent 执行。

    根据 role 创建对应 Worker → 传入 task → 执行 → 返回结果 → 销毁。
    每次调用都是独立的临时 Agent，不会保留状态到下一次调用。

    Args:
        role: Worker 角色名称，可选值及职责:
            - coder: 编写/修改代码文件、运行命令、调试错误
            - researcher: 搜索互联网获取最新信息（web_search）+ 搜索本地知识库（search_documents），适合查实时/新闻/技术动态
            - critic: 审查代码质量、发现 bug、提供改进建议
            - writer: 创作长文、小说、技术文档等内容
        task: 要交给 Worker 执行的具体任务描述（尽量详细）

    Returns:
        Worker 执行结果文本
    """
    if _worker_factory is None:
        return "错误: WorkerFactory 未初始化，请先调用 configure_worker_factory()"

    # Import locally to get live available roles from factory
    from clawagent.worker.base import BaseWorker

    try:
        worker: BaseWorker = _worker_factory.create(role)
    except ValueError as e:
        return f"错误: {e}"

    try:
        agent = worker.spawn(task, settings=_worker_factory._current_settings)
        result = agent.run(task).text
        max_len = 50_000
        if len(result) > max_len:
            result = result[:max_len] + f"\n\n...（结果已截断，共 {len(result)} 字符）"
        return result
    except Exception as e:
        return f"Worker [{role}] 执行失败: {type(e).__name__}: {e}"
    finally:
        worker.cleanup()

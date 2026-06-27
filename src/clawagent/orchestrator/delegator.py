"""delegate_task — Orchestrator tool for invoking worker agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from clawagent.worker.factory import WorkerFactory

# Module-level state, set by agent.py on startup
_worker_factory: WorkerFactory | None = None


def configure_worker_factory(factory: WorkerFactory) -> None:
    """Set the module-level WorkerFactory for use by delegate_task."""
    global _worker_factory
    _worker_factory = factory


@tool
def delegate_task(role: str, task: str) -> str:
    """将子任务委托给指定的 Worker Agent 执行。

    根据 role 创建对应 Worker → 传入 task → 执行 → 返回结果 → 销毁。
    每次调用都是独立的临时 Agent，不会保留状态到下一次调用。

    Args:
        role: Worker 角色名称，可选值: coder, researcher, critic, writer
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
        result = worker.run(task)
        max_len = 50_000
        if len(result) > max_len:
            result = result[:max_len] + f"\n\n...（结果已截断，共 {len(result)} 字符）"
        return result
    except Exception as e:
        return f"Worker [{role}] 执行失败: {type(e).__name__}: {e}"
    finally:
        worker.cleanup()

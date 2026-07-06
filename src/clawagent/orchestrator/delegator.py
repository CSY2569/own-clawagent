"""delegate_task — Orchestrator tool for invoking worker agents.

Each Agent gets its own delegate_task closure via make_delegate_task(factory).
No module-level global state — the factory follows the Agent instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

if TYPE_CHECKING:
    from clawagent.worker.factory import WorkerFactory


def make_delegate_task(factory: WorkerFactory) -> Any:
    """Create a delegate_task tool bound to *factory*.

    Returns a new @tool closure each call. Each Agent holds its own
    closure → its own factory → no cross-instance state leakage.
    """

    @tool
    def delegate_task(role: str, task: str) -> str:
        """将子任务委托给指定的 Worker Agent 执行。

        根据 role 创建对应 Worker → 传入 task → 执行 → 返回结果 → 销毁。
        每次调用都是独立的临时 Agent，不会保留状态到下一次调用。

        Args:
            role: Worker 角色名称，可选值及职责:
                - coder: 编写/修改代码文件、运行命令、调试错误
                - researcher: 搜索互联网获取最新信息（web_search）+ 搜索本地知识库（search_documents）
                - critic: 审查代码质量、发现 bug、提供改进建议
                - writer: 创作长文、小说、技术文档等内容
            task: 要交给 Worker 执行的具体任务描述（尽量详细）

        Returns:
            Worker 执行结果文本
        """
        from clawagent.worker.base import BaseWorker

        try:
            worker: BaseWorker = factory.create(role)
        except ValueError as e:
            return f"错误: {e}"

        try:
            agent = worker.spawn(task, settings=factory.get_settings())
            result: str = agent.run(task).text
            max_len = 50_000
            if len(result) > max_len:
                result = result[:max_len] + f"\n\n...（结果已截断，共 {len(result)} 字符）"
            return result
        except Exception as e:
            return f"Worker [{role}] 执行失败: {type(e).__name__}: {e}"
        finally:
            worker.cleanup()

    return delegate_task

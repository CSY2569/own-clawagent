"""Append remaining MCP document content to the existing file."""

content = """

## 工具适配层

将 MCP Server 的工具转换为 LangChain 可调用的 `@tool` 函数。

```python
# src/clawagent/mcp/adapter.py

from __future__ import annotations

from langchain_core.tools import tool

from clawagent.mcp.models import McpToolSchema
from clawagent.mcp.registry import McpRegistry


class McpToolAdapter:
    \"\"\"将 MCP Server 的工具适配为 LangChain @tool。\"\"\"

    def __init__(self, registry: McpRegistry) -> None:
        self._registry = registry

    def build_tools(self, worker_role: str | None = None) -> list:
        \"\"\"构建 LangChain 工具列表，可按 Worker 角色过滤。\"\"\"
        schemas = (
            self._registry.get_tools_for_worker(worker_role)
            if worker_role
            else self._registry.get_tools()
        )
        tools = []
        for schema in schemas:
            fn = self._make_tool_fn(schema)
            fn.__name__ = schema.name
            fn.__doc__ = schema.description
            tools.append(fn)
        return tools

    def _make_tool_fn(self, schema: McpToolSchema):
        \"\"\"为单个 MCP 工具创建 LangChain tool 包装函数。\"\"\"
        tool_to_server = {}
        for conn in self._registry._connections.values():
            for t in conn.tools:
                tool_to_server[t.name] = conn.config.name

        server_name = tool_to_server.get(schema.name, "")

        @tool
        def _wrapper(**kwargs) -> str:
            \"\"\"调用 MCP Server 上的工具。\"\"\"
            import json
            conn = self._registry.get_connection(server_name)
            if conn is None:
                return f"错误: MCP Server '{server_name}' 未连接"
            try:
                from clawagent.mcp.client import McpClient
                from clawagent.mcp.transport.stdio import StdioTransport

                transport = StdioTransport(
                    conn.config.command, conn.config.args,
                    conn.config.env, conn.config.timeout,
                )
                client = McpClient(transport)
                result = client.call_tool(schema.name, kwargs)
                transport.close()

                content = result.get("content", [])
                texts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "\\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)
            except Exception as e:
                return f"工具调用失败: {type(e).__name__}: {e}"

        _wrapper.__dict__["mcp_schema"] = schema.input_schema
        return _wrapper
```

注意：上述实现中每次工具调用都新建 Transport，这是为了简化设计。
**优化方向（二期）**：让 McpClient 保持长连接，避免反复创建子进程的开销。

---

## 与现有架构的集成

### 入口初始化

在 `src/clawagent/mcp/__init__.py` 中导出核心类：

```python
from clawagent.mcp.registry import McpRegistry
from clawagent.mcp.adapter import McpToolAdapter

__all__ = ["McpRegistry", "McpToolAdapter"]
```

### 集成到 Session 初始化

修改 `src/clawagent/cli/session.py`，在 `init_session()` 中加入 MCP 初始化：

```python
# 在 init_session() 函数末尾添加
from clawagent.mcp import McpRegistry, McpToolAdapter

mcp_registry = McpRegistry()
loaded = mcp_registry.load_all()
if loaded:
    console.print(f"  [dim]MCP 已加载: {', '.join(loaded)}[/dim]")

mcp_adapter = McpToolAdapter(mcp_registry)

# 存入 SessionContext
ctx.mcp_registry = mcp_registry
ctx.mcp_adapter = mcp_adapter
```

### 注入到 Agent 的工具列表

在 `src/clawagent/agent.py` 的 `create_agent()` 中：

```python
# 收集 MCP 工具
mcp_tools = []
mcp_registry = getattr(settings, "_mcp_registry", None)
if mcp_registry:
    adapter = McpToolAdapter(mcp_registry)
    mcp_tools = adapter.build_tools()

# 合并到工具列表
all_tools = ALL_TOOLS + memory_tools + [delegate_tool] + mcp_tools
```

### Worker 级别过滤

在 WorkerFactory 中：

```python
mcp_tools = []
if self._mcp_registry:
    adapter = McpToolAdapter(self._mcp_registry)
    mcp_tools = adapter.build_tools(worker_role=role)
# 最终工具集 = 核心工具 + MCP 工具
```

---

## REPL 命令

新增 `/mcp` 斜杠命令体系：

| 命令 | 功能 |
|------|------|
| `/mcp list` | 列出所有配置的 Server 及状态 |
| `/mcp load <name>` | 手动加载一个 Server |
| `/mcp unload <name>` | 卸载一个 Server |
| `/mcp reload <name>` | 重启一个 Server |
| `/mcp tools` | 列出当前所有可用的 MCP 工具 |

### 实现示例

```python
# src/clawagent/mcp/commands.py

from rich.console import Console
from rich.table import Table
from rich import box
from clawagent.mcp.registry import McpRegistry


def handle_mcp_command(args: str, registry: McpRegistry, console: Console) -> None:
    parts = args.strip().split()
    subcmd = parts[0] if parts else "list"

    if subcmd == "list":
        _cmd_list(registry, console)
    elif subcmd == "load" and len(parts) >= 2:
        ok = registry.load(parts[1])
        console.print(f"[{'green' if ok else 'red'}]{'已加载' if ok else '加载失败'}: {parts[1]}[/]")
    elif subcmd == "unload" and len(parts) >= 2:
        ok = registry.unload(parts[1])
        console.print(f"[{'green' if ok else 'red'}]{'已卸载' if ok else '未找到'}: {parts[1]}[/]")
    elif subcmd == "reload" and len(parts) >= 2:
        ok = registry.reload(parts[1])
        console.print(f"[{'green' if ok else 'red'}]{'已重启' if ok else '重启失败'}: {parts[1]}[/]")
    elif subcmd == "tools":
        _cmd_tools(registry, console)
    else:
        console.print("[yellow]用法: /mcp <list|load|unload|reload|tools> [name][/yellow]")


def _cmd_list(registry: McpRegistry, console: Console) -> None:
    table = Table(title="MCP Servers", box=box.SIMPLE)
    table.add_column("名称", style="cyan")
    table.add_column("传输", style="blue")
    table.add_column("状态")
    table.add_column("工具数", justify="right")
    table.add_column("描述")
    for conn in registry._connections.values():
        style = {"connected": "green", "connecting": "yellow",
                 "error": "red", "disconnected": "dim"}.get(conn.status.value, "white")
        table.add_row(conn.config.name, conn.config.transport.value,
                      f"[{style}]{conn.status.value}[/{style}]",
                      str(len(conn.tools)), conn.config.description)
    console.print(table)


def _cmd_tools(registry: McpRegistry, console: Console) -> None:
    tools = registry.get_tools()
    if not tools:
        console.print("[dim]当前没有已加载的 MCP 工具[/dim]")
        return
    table = Table(title=f"MCP 工具 ({len(tools)} 个)", box=box.SIMPLE)
    table.add_column("工具名", style="cyan")
    table.add_column("描述")
    table.add_column("来源 Server", style="blue")
    for t in tools:
        server_name = ""
        for conn in registry._connections.values():
            if any(tt.name == t.name for tt in conn.tools):
                server_name = conn.config.name
                break
        table.add_row(t.name, t.description[:60], server_name)
    console.print(table)
```

---

## 实施计划

### Phase 1 — 基础设施（预估 2-3 天）

| 任务 | 产出 | 依赖 |
|------|------|------|
| 1.1 创建 `mcp/` 模块目录结构 | 文件骨架 | 无 |
| 1.2 实现 `models.py` | 数据结构定义 | 无 |
| 1.3 实现 `loader.py` | JSON 配置加载 + 环境变量解析 | 1.2 |
| 1.4 实现 `transport/stdio.py` | subprocess 通信层 | 无 |
| 1.5 实现 `client.py` | JSON-RPC 协议封装 | 1.4 |
| 1.6 实现 `registry.py` | 连接生命周期管理 | 1.2-1.5 |
| 1.7 编写单元测试 | 测试配置加载、协议编解码、进程管理 | 1.2-1.6 |

### Phase 2 — 集成（预估 2 天）

| 任务 | 产出 | 依赖 |
|------|------|------|
| 2.1 实现 `adapter.py` | MCP -> LangChain 工具适配 | 1.6 |
| 2.2 集成到 `session.py` | McpRegistry 随 Session 初始化 | 2.1 |
| 2.3 集成到 `agent.py` | MCP 工具注入 Agent 工具列表 | 2.1 |
| 2.4 集成到 Worker Factory | 按角色过滤 MCP 工具 | 2.1 |
| 2.5 实现 commands + 注册斜杠命令 | /mcp 系列命令 | 2.2 |

### Phase 3 — 完善（预估 2 天）

| 任务 | 说明 |
|------|------|
| 3.1 自动重连机制 | 进程崩溃后自动重启 |
| 3.2 长连接优化 | McpClient 保持持久连接 |
| 3.3 SSE 远程传输 | 支持远程 MCP Server |
| 3.4 安全沙箱 | 超时、资源限制、白名单 |
| 3.5 集成测试 | 用真实 MCP Server 做 E2E 测试 |

---

## 安全注意事项

1. **子进程隔离**——每个 MCP Server 是独立进程，崩溃不影响主进程
2. **超时保护**——所有工具调用设 30 秒超时，防止 Server 挂死
3. **环境变量引用**——`${VAR}` 避免配置文件中泄露密钥
4. **Worker 权限控制**——`allowed_workers` 限制敏感工具使用范围
5. **沙箱目录**——文件系统类 MCP Server 应限制操作范围

---

## 依赖项

核心 stdio 模式**零外部依赖**，完全基于 Python 标准库实现。

SSE 远程传输（Phase 3）可能需要：
```toml
httpx>=0.28.0
httpx-sse>=0.4.0
```

---

## 测试策略

| 测试层级 | 内容 | 工具 |
|---------|------|------|
| 单元测试 | 配置解析、环境变量替换、JSON-RPC 编解码 | pytest |
| 进程测试 | 启动假 MCP Server 进程，验证握手流程 | subprocess + pytest |
| 集成测试 | 连接真实 MCP Server（如 server-everything） | pytest + npx |
| 适配器测试 | 验证 LangChain tool 的调用链 | pytest + mock |

---

## 附录：MCP 协议要点

MCP 核心是基于 **JSON-RPC 2.0** 的请求-响应协议：

| 方法 | 方向 | 用途 |
|------|------|------|
| `initialize` | Client -> Server | 握手，交换协议版本和能力 |
| `notifications/initialized` | Client -> Server | 通知初始化完成 |
| `tools/list` | Client -> Server | 获取工具列表 |
| `tools/call` | Client -> Server | 调用工具 |
| `resources/list` | Client -> Server | 获取资源列表（可选） |
| `prompts/list` | Client -> Server | 获取提示词模板（可选） |

### 通信流程

```
Client                          Server
  |                               |
  |--- initialize request ------> |
  |<------ initialize response ---|
  |--- notifications/initialized >|
  |                               |
  |--- tools/list request ------> |
  |<----- tools/list response ----|
  |                               |
  |--- tools/call request ------> |  (每次工具调用)
  |<----- tools/call response ----|
  |                               |
```

---

## 总结

MCP 插件化工具方案让 clawagent 从一个**封闭的、硬编码工具集**的系统，进化为一个**开放的、可热插拔的插件生态**。

核心价值：
1. **用户自主权**——用户决定用什么工具，不用等开发者
2. **生态复用**——接入 MCP 社区已有的上百个工具 Server
3. **零耦合**——MCP Server 独立进程，语言无关，协议标准
4. **渐进式**——先从 stdio 本地模式做起，再扩展到 SSE 远程
"""

with open("实施文档/MCP插件化工具方案.md", "a", encoding="utf-8") as f:
    f.write(content)

print("Done - appended successfully")

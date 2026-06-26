# clawagent

[English](README.md) | [中文](README.zh.md)

基于 Anthropic Claude 兼容模型（通过 DeepSeek Anthropic API）的 LangChain/LangGraph 工具调用 agent。

## 功能特性

- **工具调用 agent** — 8 个内置工具：文件读写、Shell 命令、时间查询、问候、会话记忆（列出/回顾/摘要）
- **交互式 REPL** — Rich 驱动的仪表板，实时显示 token 用量、费用估算，支持设置热更新
- **多层系统提示词** — 基于文件的提示词拼装，支持按 agent 区分身份、人格、工作区上下文和运行时元数据，工具列表自动生成
- **对话记忆** — SQLite 持久化会话，自动生成摘要并提取用户偏好
- **设置热更新** — 无需重启即可切换模型、调整 temperature 和最大输出 token
- **中文输入友好** — 基于 prompt_toolkit，正确处理 CJK 字符宽度

## 快速开始

### 环境要求

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
git clone git@github.com:<user>/clawagent.git
cd clawagent
uv sync
cp .env.example .env
# 编辑 .env，设置 ANTHROPIC_API_KEY
```

### 使用

```bash
# 单次问答
uv run clawagent "现在几点了？"

# 交互式 REPL
uv run clawagent
```

## 交互式命令

REPL 支持以下斜杠命令：

| 命令 | 说明 |
|------|------|
| `/sessions` | 列出所有历史会话 |
| `/load <id>` | 加载指定会话 |
| `/new` | 创建新会话 |
| `/model <name>` | 切换模型（如 `deepseek-v4-pro`） |
| `/temp <n>` | 设置 temperature（如 `0.7`） |
| `/max-tokens <n>` | 设置最大输出 token 数（如 `8192`） |
| `/settings` | 显示当前配置 |
| `/help` | 显示帮助 |
| `quit` / `exit` / `q` | 退出 |

输入 `/` 会自动弹出命令补全菜单，附带各命令的说明。

## 配置

环境变量（在 `.env` 中设置）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ANTHROPIC_API_KEY` | *(必填)* | API 密钥 |
| `ANTHROPIC_BASE_URL` | — | API 基础 URL（DeepSeek: `https://api.deepseek.com/anthropic`） |
| `CLAWAGENT_MODEL` | `deepseek-v4-flash` | 模型名称 |
| `CLAWAGENT_AGENT_ID` | `pickle` | Agent 身份标识（读取 `prompts/agents/<id>/` 下的文件） |
| `CLAWAGENT_CONTEXT_WINDOW` | `1000000` | 上下文窗口大小（仅用于显示） |
| `CLAWAGENT_MEMORY_DB` | `memories/sessions.db` | SQLite 记忆数据库路径 |
| `CLAWAGENT_MAX_PREFERENCES` | `5` | 注入提示词的最大用户偏好数量 |

## 多层提示词系统

系统提示词由 `PromptBuilder` 从五个层次拼接而成：

| 层次 | 来源 | 是否必须 |
|------|------|----------|
| 1. 身份 | `prompts/agents/{id}/identity.md` | 是（文件缺失时有 fallback） |
| 2. 人格 | `prompts/agents/{id}/soul.md` | 否 |
| 3. 工作区 | `prompts/shared/bootstrap.md`、`agents.md` | 否 |
| 4. 运行时 | Agent ID、时间戳、渠道（代码自动生成） | 是 |
| 5. 偏好 | SQLite `preferences` 表（自动学习） | 否 |

工具列表从 `ALL_TOOLS` 自动生成 — 在 `src/clawagent/tools/` 中添加新的 `@tool` 即可自动出现在提示词中。

添加新 agent 只需创建 `prompts/agents/<name>/identity.md` 并设置 `CLAWAGENT_AGENT_ID=<name>`。

## 项目结构

```
src/clawagent/
├── config.py          # Settings 数据类、价格表
├── agent.py           # Agent 工厂函数、run/stream/reconfigure
├── prompt_builder.py  # 多层提示词拼装
├── main.py            # CLI 入口、REPL 循环
├── ui.py              # Rich 仪表板、统计、格式化
├── tools/
│   ├── __init__.py    # 核心工具（读写、shell、时间、问候）
│   └── memory_tools.py # 记忆工具（列出/回顾/摘要会话）
└── memory/
    ├── summarizer.py  # 会话摘要与消息持久化
    └── preferences.py # 用户偏好提取与查询

prompts/
├── agents/pickle/     # 默认 agent 提示词文件
│   ├── identity.md    # Agent 身份定义
│   └── soul.md        # 人格与语气
├── shared/            # 共享工作区上下文
│   ├── bootstrap.md
│   └── agents.md
└── README.md
```

## 设计文档

以下设计文档仅存在于本地，未纳入版本管理：

- `多层提示词设计.md` — 多层提示词架设���设计
- `记忆系统实施文档.md` — 记忆系统实施记录
- `记忆训练强化.txt` — 记忆训练笔记

## 开发

```bash
# 安装依赖
uv sync

# 代码检查
uv run ruff check .

# 代码格式化
uv run ruff format .

# 类型检查
uv run mypy src/ tests/

# 运行测试
uv run pytest tests/ -v
```

## 许可证

MIT

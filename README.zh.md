# clawagent

[English](README.md) | [中文](README.zh.md)

基于 Anthropic Claude 兼容模型（DeepSeek Anthropic API）的 LangChain/LangGraph 工具调用 Agent，支持多 Agent 编排、混合 RAG 检索、实时流式显示和对话记忆。

## 功能特性

- **工具调用 Agent** — 10+ 内置工具：文件读写、Shell 命令、时间查询、对话记忆、RAG 文档检索、多 Agent 任务委托
- **多 Agent 协作** — Orchestrator + 4 个专业 Worker（Coder、Researcher、Critic、Writer）。复杂任务自动分解，通过 `delegate_task(role, task)` 委托执行。每个 Worker 拥有独立的模型、工具集和提示词上下文。
- **实时流式显示** — 逐 token 打字机效果 + 旋转动画 Spinner。工具调用和结果实时显示。支持 DeepSeek 思考块（thinking block）检测。
- **RAG 混合检索** — KNN 向量检索（Chroma + SiliconFlow 嵌入）与 BM25 词法匹配（jieba 分词）通过 Reciprocal Rank Fusion (RRF) 融合。BM25 索引在启动时后台构建，瞬间启动无卡顿。
- **多层系统提示词** — 基于文件的提示词拼装，支持按 Agent 区分身份、人格、工作区上下文、运行时元数据和用户偏好，工具列表自动生成
- **对话记忆** — SQLite 持久化会话，自动生成摘要并提取用户偏好
- **上下文压缩** — 3 种策略：按消息数裁剪、按预估 token 数裁剪、LLM 摘要压缩
- **设置热更新** — 无需重启即可切换模型、temperature、最大输出 token 和压缩策略
- **交互式 REPL** — Rich 驱动的仪表板，实时显示 token 用量、费用估算、流式输出
- **204 个测试通过** — 完整测试覆盖，含 Worker 注册、配置、生命周期、工具集和编排器委托

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

REPL 支持斜杠命令（输入 `/` 自动补全）：

| 命令 | 说明 |
|------|------|
| `/sessions` | 列出所有历史会话 |
| `/load <id>` | 加载并切换到指定会话 |
| `/new` | 创建新会话 |
| `/model <name>` | 切换模型（如 `deepseek-v4-pro`） |
| `/temp <n>` | 设置 temperature（如 `0.7`） |
| `/max-tokens <n>` | 设置最大输出 token 数（如 `8192`） |
| `/compress <strategy>` | 切换压缩策略（`trim` / `token_trim` / `summarize`） |
| `/settings` | 显示当前设置 |
| `/rag-search <关键词>` | 直接搜索 RAG 向量库 |
| `/help` | 显示帮助 |
| `quit` / `exit` / `q` | 退出 |

## 多 Agent 协作

复杂任务由 Orchestrator 自动拆解并委托给专业 Worker 执行：

| Worker | 职责 | 可用工具 |
|--------|------|----------|
| `coder` | 代码编写和调试 | `read_file`, `write_file`, `run_command` |
| `researcher` | 信息检索和研究 | `search_documents` (RAG) |
| `critic` | 代码审查和方案评审 | `read_file`, `search_documents` |
| `writer` | 文档编写和内容创作 | `read_file`, `write_file` |

每个 Worker 是独立的临时 Agent，拥有独立的模型配置、身份提示词和工具集。Worker 按需创建，任务完成后自动销毁。运行时配置（模型、temperature）通过热更新自动同步到 Worker。

### 搜索规范（共享）

搜索行为规则（使用原文搜索、不编造答案、交叉验证）从 `prompts/shared/search-rules.md` 加载，通过 PromptBuilder Layer 3 自动应用于所有 Agent。

## 流式显示

REPL 实时显示 Agent 执行过程：

```
  ⠋ 正在调用 search_documents("龙神恩雅")...
  ✓ search_documents (6 行)
  ✓ read_file (152 行)
  ──────────────────────────────────────────
  从《黎明之剑》来看，龙神恩雅并没有彻底消亡...

  In 1,230 · Out 340
```

特性：
- 思考时旋转动画 Spinner
- 实时工具调用追踪（含参数预览）
- 逐 token 打字机输出（Phase 2）
- DeepSeek 思考块检测（`think_start` / `think_end`）
- 完成后显示 token 用量统计

## 配置

环境变量（在 `.env` 中设置）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ANTHROPIC_API_KEY` | *(必填)* | API 密钥 |
| `ANTHROPIC_BASE_URL` | — | API 基础 URL（DeepSeek: `https://api.deepseek.com/anthropic`） |
| `CLAWAGENT_MODEL` | `deepseek-v4-flash` | 模型名称 |
| `CLAWAGENT_MODEL_PROVIDER` | `anthropic` | 模型提供商（`anthropic`、`openai` 等） |
| `CLAWAGENT_AGENT_ID` | `pickle` | Agent 身份（读取 `prompts/agents/<id>/`） |
| `CLAWAGENT_CONTEXT_WINDOW` | `1000000` | 上下文窗口大小（仅用于显示） |
| `CLAWAGENT_MEMORY_DB` | `memories/sessions.db` | SQLite 记忆数据库路径 |
| `CLAWAGENT_MAX_PREFERENCES` | `5` | 注入提示词的最大用户偏好数量 |
| `COMPRESSION_STRATEGY` | `trim` | 上下文压缩策略 |
| `COMPRESSION_MAX_MESSAGES` | `40` | 触发裁剪的最大消息数 |
| `COMPRESSION_MAX_TOKENS` | `80000` | token_trim 的 token 阈值 |
| `COMPRESSION_KEEP_RECENT` | `6` | 保留的最近消息数 |
| `SILICONFLOW_API_KEY` | *(可选)* | SiliconFlow API 密钥（RAG 嵌入用） |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1/embeddings` | 嵌入 API 地址 |
| `SILICONFLOW_MODEL` | `Qwen/Qwen3-VL-Embedding-8B` | 嵌入模型名称 |
| `SILICONFLOW_DIMENSIONS` | `768` | 嵌入向量维度 |

### Worker 配置

每个 Worker 可使用不同的模型/提供商：

```bash
# 通用默认值（fallback）
WORKER_COMMON_MODEL=deepseek-v4-flash

# 可按 Worker 覆盖
WORKER_CODER_MODEL=deepseek-v4-flash
WORKER_RESEARCHER_MODEL=Qwen/Qwen3-235B-A22B
WORKER_RESEARCHER_MODEL_PROVIDER=openai
WORKER_RESEARCHER_API_BASE=https://api.siliconflow.cn/v1
WORKER_CRITIC_MODEL=Qwen/Qwen3-235B-A22B
WORKER_CRITIC_MODEL_PROVIDER=openai
WORKER_CRITIC_API_BASE=https://api.siliconflow.cn/v1
WORKER_WRITER_MODEL=deepseek-v4-flash
```

## 多层提示词系统

系统提示词由 `PromptBuilder` 从五个层次拼装而成：

| 层 | 来源 | 是否必须 |
|----|------|----------|
| 1. 身份 | `prompts/agents/{id}/identity.md` | 是（有 fallback） |
| 2. 人格 | `prompts/agents/{id}/soul.md` | 否 |
| 3. 工作区 | `prompts/shared/bootstrap.md`、`agents.md`、`search-rules.md` | 否 |
| 4. 运行时 | Agent ID、时间戳、渠道（代码自动生成） | 是 |
| 5. 偏好 | SQLite `preferences` 表（自动学习） | 否 |

工具列表从 `ALL_TOOLS + delegate_task` 自动生成——在 `src/clawagent/tools/` 中添加新的 `@tool` 即可自动出现在提示词中，与真实注册的工具集保持同步。

添加新 Agent：创建 `prompts/agents/<name>/identity.md` 并设置 `CLAWAGENT_AGENT_ID=<name>`。

## RAG 检索增强生成

clawagent 支持 Agentic RAG——LLM 自行判断何时通过 `search_documents` 工具检索文档，而非每次对话都自动注入上下文。

**架构：**

- **混合检索**：KNN 向量检索（Chroma）+ BM25 词法匹配（jieba），通过 Reciprocal Rank Fusion (RRF) 融合
- **嵌入**：云端 SiliconFlow API（`Qwen/Qwen3-VL-Embedding-8B`，768 维）
- **向量库**：本地 Chroma（HNSW 索引，持久化至 `chroma_db/`）
- **章节元数据**：入库时自动识别章节标记（`第X章`、`Chapter X` 等）
- **后台 BM25**：启动时后台线程构建 BM25 索引，构建期间自动降级为 KNN-only——启动无延迟

**配置：**

```bash
# 1. 在 .env 中配置 SILICONFLOW_API_KEY

# 2. 入库文档（仅需执行一次）
uv run python -m clawagent.rag.ingest docs/ --chunk-size 512 --overlap 64

# 3. 从 CLI 测试检索
uv run clawagent
You: /rag-search 高文的亲人
```

检索结果附带章节信息：`[1] (相关度: 0.85, 第12章) — 高文·塞西尔是...`

## 项目结构

```
src/clawagent/
├── config.py            # Settings 数据类、价格表
├── agent.py             # Agent 工厂、run/stream_events/reconfigure
├── prompt_builder.py    # 多层提示词拼装
├── main.py              # CLI 入口、REPL 循环
├── ui.py                # Rich 仪表板、统计、格式化
├── ui_stream.py         # 实时流式显示（spinner + token）
├── stream_events.py     # 流式事件类型定义
├── tools/
│   ├── __init__.py      # 核心工具（读写、shell、时间、问候）
│   ├── memory_tools.py  # 记忆工具（列出/回顾/摘要）
│   └── rag_tool.py      # RAG search_documents 工具
├── memory/
│   ├── summarizer.py    # 会话摘要与消息持久化
│   └── preferences.py   # 用户偏好提取与查询
├── orchestrator/
│   └── delegator.py     # delegate_task 工具（Worker 委托）
├── worker/
│   ├── base.py          # BaseWorker 抽象基类
│   ├── factory.py       # WorkerFactory（按角色创建）
│   ├── config.py        # 环境变量配置加载
│   ├── registry.py      # @register_worker 装饰器
│   ├── coder.py         # 代码编写 Worker
│   ├── researcher.py    # 信息检索 Worker
│   ├── critic.py        # 代码审查 Worker
│   └── writer.py        # 文档编写 Worker
├── compression/
│   ├── __init__.py      # 统一入口 + pre_model_hook
│   ├── config.py        # CompressionConfig
│   ├── strategies.py    # trim / token_trim / summarize
│   └── counters.py      # Token 估算
└── rag/
    ├── __init__.py
    ├── embedding.py     # SiliconFlow 云端嵌入客户端
    ├── store.py         # Chroma 向量库
    ├── chunker.py       # 固定窗口文本分块
    ├── bm25.py          # BM25 词法检索（jieba 分词）
    ├── hybrid.py        # KNN + BM25 混合检索 + RRF 融合
    └── ingest.py        # 文档入库 CLI 脚本

prompts/
├── agents/pickle/       # 默认 Agent 提示词文件
│   ├── identity.md
│   └── soul.md
├── shared/              # 共享工作区上下文
│   ├── bootstrap.md
│   ├── agents.md
│   └── search-rules.md
└── README.md

tests/                   # 204 个测试
├── test_worker_*.py     # Worker 注册、配置、生命周期、工具集
├── test_orchestrator.py # delegate_task 委托
├── test_agent.py        # Agent 包装类
├── test_config.py       # 配置 + 价格表
├── test_tools.py        # 核心工具
├── test_memory_tools.py # 记忆工具
├── test_functional.py   # Agent 图集成
└── ...
```

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

# 运行测试（204 个测试）
uv run pytest tests/ -v

# 运行单文件测试
uv run pytest tests/test_worker_registry.py -v
```

## 许可证

MIT

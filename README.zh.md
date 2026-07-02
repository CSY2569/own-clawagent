# clawagent

[English](README.md) | [中文](README.zh.md)

基于 DeepSeek（Anthropic 兼容 API）的 LangChain/LangGraph 工具调用 Agent，支持多 Agent 编排、混合 RAG 检索、流式显示和对话记忆。

## 功能特性

- **工具调用 Agent** — 7 个内置工具：文件读写、Shell 命令、对话记忆（列出/回顾/摘要）、RAG 文档检索、多 Agent 任务委托
- **多 Agent 协作** — Orchestrator + 4 个专业 Worker（Coder、Researcher、Critic、Writer）。复杂任务通过 `delegate_task(role, task)` 自动分解并委托执行，每个 Worker 拥有独立的模型、工具集和提示词上下文
- **实时流式显示** — 逐 token 打字机效果 + Rich Live 仪表板。工具调用和结果实时显示。支持 DeepSeek 思考块检测
- **Ctrl+C 即时中断** — 线程化 producer-consumer 模式，取消响应 ≤100ms，后台优雅清理
- **联网搜索** — Researcher Worker 使用 Bing 直连搜索 + trafilatura 正文提取，并发抓取多页面
- **RAG 混合检索** — KNN 向量检索（Chroma + SiliconFlow 嵌入）+ BM25 词法匹配（jieba 分词），通过 Reciprocal Rank Fusion 融合。BM25 索引后台构建，零启动延迟
- **API Key 池** — 多 Key 故障转移，每 Key 指数退避。支持 `API_POOL_*` 环境变量自动轮换
- **多层系统提示词** — 基于文件的提示词拼装：身份、人格、工作区上下文、运行时元数据、用户偏好、工具列表自动生成
- **对话记忆** — SQLite 持久化会话，自动摘要和用户偏好提取，按会话输出 JSONL 日志
- **上下文压缩** — 3 种策略：按消息数裁剪、按 token 估算裁剪、LLM 摘要压缩
- **设置热更新** — 无需重启即可切换模型、temperature、最大输出 token、压缩策略
- **交互式 REPL** — Rich 驱动仪表板，实时 token 用量和费用统计，斜杠命令自动补全，流式输出
- **304 个测试通过** — 完整覆盖：Agent、Worker、工具、记忆、RAG、压缩、流式、API 池、取消令牌、UI

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

在流式输出过程中按 **Ctrl+C** 可即时中断当前生成，回到输入提示符。

## 架构

### 多 Agent 协作

复杂任务由 Orchestrator 自动拆解并委托给专业 Worker：

| Worker | 职责 | 核心工具 |
|--------|------|----------|
| `coder` | 代码编写和调试 | `read_file`, `write_file`, `run_command` |
| `researcher` | 联网搜索和信息检索 | Bing 搜索, trafilatura 正文提取, `search_documents` (RAG) |
| `critic` | 代码审查和方案评审 | `read_file`, `search_documents` |
| `writer` | 文档编写和内容创作 | `read_file`, `write_file` |

每个 Worker 是独立的临时 Agent，按需创建，任务完成后自动销毁。运行时配置（模型、temperature）通过热更新自动同步到 Worker。

### 流式显示

REPL 实时展示 Agent 执行过程：

```
  ⠋ 正在调用 search_documents("龙神恩雅")...
  ✓ search_documents (6 行)
  ✓ read_file (152 行)
  ──────────────────────────────────────────
  从《黎明之剑》来看，龙神恩雅并没有彻底消亡...

  In 1,230 · Out 340
```

- 思考时旋转动画 Spinner
- 实时工具调用追踪（含参数预览）
- 逐 token 打字机输出
- DeepSeek 思考块检测（`think_start` / `think_end`）
- 完成后显示 token 用量和费用

### 提示词系统

系统提示词由 `PromptBuilder` 从五个层次拼装：

| 层 | 来源 | 是否必须 |
|----|------|----------|
| 1. 身份 | `prompts/agents/{id}/identity.md` | 是（有 fallback） |
| 2. 人格 | `prompts/agents/{id}/soul.md` | 否 |
| 3. 工作区 | `prompts/shared/bootstrap.md`、`agents.md`、`search-rules.md` | 否 |
| 4. 运行时 | Agent ID、时间戳、渠道（自动生成） | 是 |
| 5. 偏好 | SQLite `preferences` 表（自动学习） | 否 |

工具列表从 `ALL_TOOLS + delegate_task` 自动生成 — 添加新的 `@tool` 即可自动出现在提示词中。

添加新 Agent：创建 `prompts/agents/<name>/identity.md` 并设置 `CLAWAGENT_AGENT_ID=<name>`。

### RAG 检索增强生成

Agentic RAG — LLM 自行判断何时通过 `search_documents` 工具检索文档。

- **混合检索**：KNN 向量检索（Chroma HNSW）+ BM25 词法匹配（jieba），RRF 融合
- **嵌入**：云端 SiliconFlow API（`Qwen/Qwen3-VL-Embedding-8B`，768 维）
- **章节元数据**：入库时自动识别章节标记
- **后台 BM25**：启动时后台线程构建索引，构建期间自动降级为纯 KNN 检索

```bash
# 入库文档（仅需执行一次）
uv run python -m clawagent.rag.ingest docs/ --chunk-size 512 --overlap 64

# 从 CLI 测试
uv run clawagent
> /rag-search 关键词
```

### API Key 池

多 Key 故障转移，每 Key 独立指数退避：

```bash
# .env
API_POOL_DEFAULT_KEYS=sk-key1,sk-key2,sk-key3
```

当某个 Key 触发限流或错误时，池自动轮换到下一个可用 Key。失败的 Key 进入冷却期（指数退避）。也支持通过 `API_POOL_DEFAULT_KEY_FILE` 从 JSON 文件加载 Key。

### 联网搜索

Researcher Worker 可执行实时网络搜索：

- **Bing 直连搜索** — 抓取 Bing 搜索结果页
- **trafilatura 正文提取** — 抓取前几条结果的完整正文
- **并发抓取** — 使用 `ThreadPoolExecutor` 并行获取多个页面
- 可配置项：最大搜索结果数、最大深度页面数、页面超时

## 配置

环境变量（在 `.env` 中设置）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ANTHROPIC_API_KEY` | *(必填)* | DeepSeek API 密钥 |
| `ANTHROPIC_BASE_URL` | — | API 基础 URL（DeepSeek: `https://api.deepseek.com/anthropic`） |
| `CLAWAGENT_MODEL` | `deepseek-v4-flash` | 模型名称 |
| `CLAWAGENT_MODEL_PROVIDER` | `anthropic` | 模型提供商 |
| `CLAWAGENT_AGENT_ID` | `wenbao` | Agent 身份（读取 `prompts/agents/<id>/`） |
| `CLAWAGENT_CONTEXT_WINDOW` | `1000000` | 上下文窗口大小（仅用于显示） |
| `CLAWAGENT_MEMORY_DB` | `memories/sessions.db` | SQLite 记忆数据库路径 |
| `CLAWAGENT_MAX_PREFERENCES` | `5` | 注入提示词的最大偏好数量 |
| `CLAWAGENT_REQUEST_TIMEOUT` | `120` | API 请求超时（秒） |
| `COMPRESSION_STRATEGY` | `trim` | 上下文压缩策略 |
| `COMPRESSION_MAX_MESSAGES` | `40` | 触发裁剪的最大消息数 |
| `COMPRESSION_MAX_TOKENS` | `80000` | `token_trim` 的 token 阈值 |
| `COMPRESSION_KEEP_RECENT` | `6` | 保留的最近消息数 |
| `API_POOL_DEFAULT_KEYS` | *(可选)* | 逗号分隔的 API Key 列表 |
| `API_POOL_DEFAULT_KEY_FILE` | *(可选)* | 包含 API Key 的 JSON 文件 |
| `SILICONFLOW_API_KEY` | *(可选)* | SiliconFlow 密钥（RAG 嵌入用） |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1/embeddings` | 嵌入 API 地址 |
| `SILICONFLOW_MODEL` | `Qwen/Qwen3-VL-Embedding-8B` | 嵌入模型 |
| `SILICONFLOW_DIMENSIONS` | `768` | 嵌入向量维度 |

### Worker 配置

每个 Worker 可使用不同的模型/提供商：

```bash
# 通用默认值（fallback）
WORKER_COMMON_MODEL=deepseek-v4-flash

# 各 Worker 独立配置
WORKER_CODER_MODEL=deepseek-v4-flash
WORKER_RESEARCHER_MODEL=Qwen/Qwen3-235B-A22B
WORKER_RESEARCHER_MODEL_PROVIDER=openai
WORKER_RESEARCHER_API_BASE=https://api.siliconflow.cn/v1
WORKER_CRITIC_MODEL=Qwen/Qwen3-235B-A22B
WORKER_CRITIC_MODEL_PROVIDER=openai
WORKER_CRITIC_API_BASE=https://api.siliconflow.cn/v1
WORKER_WRITER_MODEL=deepseek-v4-flash
```

## 项目结构

```
src/clawagent/
├── config.py              # Settings 数据类、价格表
├── agent.py               # Agent 工厂、run/stream_events/reconfigure
├── prompt_builder.py      # 多层提示词拼装
├── main.py                # CLI 入口、REPL 循环
├── ui.py                  # Rich 仪表板、统计、格式化
├── ui_stream.py           # 流式显示（spinner、token、工具日志）
├── stream_events.py       # 流式事件类型定义
├── cancel_token.py        # Ctrl+C 协作式取消
├── conversation_log.py    # 按会话 JSONL 日志
├── api_pool/              # API Key 池 + 故障转移
│   ├── pool.py            #   ApiKeyPool 核心
│   ├── loader.py          #   环境变量加载
│   ├── models.py          #   KeyRecord、PoolConfig、KeyStatus
│   ├── wrapper.py         #   KeyPoolChatModel 包装器
│   ├── transport.py       #   自定义 HTTP 传输层
│   └── callbacks.py       #   流式回调支持
├── tools/
│   ├── __init__.py        #   核心工具（读写、Shell、搜索）
│   ├── memory_tools.py    #   记忆工具（列出/回顾/摘要）
│   └── rag_tool.py        #   RAG search_documents 工具
├── memory/
│   ├── summarizer.py      #   会话摘要与消息持久化
│   └── preferences.py     #   用户偏好提取与查询
├── orchestrator/
│   └── delegator.py       #   delegate_task 工具（Worker 委托）
├── worker/
│   ├── base.py            #   BaseWorker 抽象基类
│   ├── factory.py         #   WorkerFactory（按角色创建）
│   ├── config.py          #   环境变量 Worker 配置
│   ├── registry.py        #   @register_worker 装饰器
│   ├── coder.py           #   代码编写 Worker
│   ├── researcher.py      #   联网搜索 + RAG Worker
│   ├── critic.py          #   代码审查 Worker
│   └── writer.py          #   文档编写 Worker
├── compression/
│   ├── __init__.py        #   统一入口 + pre_model_hook
│   ├── config.py          #   CompressionConfig
│   ├── strategies.py      #   trim / token_trim / summarize
│   └── counters.py        #   Token 估算
└── rag/
    ├── embedding.py       #   SiliconFlow 云端嵌入客户端
    ├── store.py           #   Chroma 向量库
    ├── chunker.py         #   固定窗口文本分块
    ├── bm25.py            #   BM25 词法检索（jieba）
    ├── hybrid.py          #   KNN + BM25 混合检索 + RRF 融合
    └── ingest.py          #   文档入库 CLI 脚本

prompts/
├── agents/wenbao/         # 默认 Agent 提示词文件
│   ├── identity.md
│   └── soul.md
├── shared/                # 共享工作区上下文
│   ├── bootstrap.md
│   ├── agents.md
│   └── search-rules.md
└── README.md

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

# 运行测试（304 个测试）
uv run pytest tests/ -v

# 运行单文件测试
uv run pytest tests/test_worker_registry.py -v
```

## 许可证

MIT

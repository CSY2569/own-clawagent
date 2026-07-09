# clawagent

[English](README.md) | [中文](README.zh.md)

支持多平台（DeepSeek、火山方舟、OpenCode Go、OpenAI、Anthropic）的 LangChain/LangGraph 工具调用 Agent，具备多 Agent 编排、混合 RAG 检索、流式显示和对话记忆。

## 功能特性

- **多平台支持** - 开箱即用 5 个平台：DeepSeek（默认）、火山方舟、OpenCode Go、OpenAI、Anthropic。运行时通过 `/platform` 切换，自动从平台 API 拉取可用模型列表
- **工具调用 Agent** - 8+ 内置工具：4 个核心工具（文件读写、Shell 命令、RAG 文档检索）、3 个记忆工具（会话列出/回顾/摘要，闭包工厂模式）、联网搜索、多 Agent 任务委托
- **多 Agent 协作** - Orchestrator + 4 个专业 Worker（Coder、Researcher、Critic、Writer）。复杂任务通过 `delegate_task(role, task)` 自动分解并委托执行，每个 Worker 拥有独立的模型、工具集和提示词上下文
- **实时流式显示** - 逐 token 打字机效果 + Rich Live 仪表板。工具调用和结果实时显示。支持思考块检测（`think_start` / `think_end`）
- **Ctrl+C 即时中断** - 线程化 producer-consumer 模式，取消响应 ≤100ms，后台优雅清理
- **联网搜索** - Researcher Worker 使用 Bing 直连搜索 + trafilatura 正文提取，并发抓取多页面（线程安全限流器）
- **RAG 混合检索** - KNN 向量检索（Chroma + SiliconFlow 嵌入）+ BM25 词法匹配（jieba 分词），通过 Reciprocal Rank Fusion 融合。BM25 索引缓存（JSON + HMAC 签名）在文档不变时跳过重启分词
- **API Key 池** - 多 Key 故障转移，每 Key 指数退避。支持 `API_POOL_*` 环境变量自动轮换
- **多层系统提示词** - 基于文件的提示词拼装：身份、人格、工作区上下文、运行时元数据、用户偏好（TTL 缓存）、工具列表自动生成
- **对话记忆** - SQLite 持久化会话，自动摘要和用户偏好提取（后台线程），按会话输出 JSONL 日志。统一 WAL 连接缓存跨内存模块共享
- **上下文压缩** - 3 种策略：按消息数裁剪、按 token 估算裁剪、LLM 摘要压缩（真超时回退）
- **网关模式** - 多渠道事件循环（CLI、微信 iLink Bot），asyncio.Queue 桥接真流式
- **设置热更新** - 无需重启即可切换平台、模型、temperature、最大输出 token、压缩策略
- **交互式 REPL** - Rich 驱动仪表板，实时 token 用量和费用统计，斜杠命令自动补全，模型发现对话框，流式输出
- **mypy 零错误** - 71 个源文件全部通过严格模式类型检查
- **379 个测试通过** - 完整覆盖：Agent、Worker、工具、记忆、RAG、压缩、流式、API 池、取消令牌、网关、UI

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
# 编辑 .env，设置 CLAWAGENT_API_KEY
```

### 使用

```bash
# 单次问答
uv run clawagent "现在几点了？"

# 交互式 REPL
uv run clawagent

# 网关模式（微信 + CLI 渠道）
uv run clawagent gateway
```

## 交互式命令

REPL 支持斜杠命令（输入 `/` 自动补全）：

| 命令 | 说明 |
|------|------|
| `/sessions` | 列出所有历史会话 |
| `/load <id>` | 加载并切换到指定会话 |
| `/new` | 创建新会话 |
| `/model` | 切换模型（无参数弹选框，或 `平台:模型名`） |
| `/models` | 列出当前平台可用模型（`/models refresh` 刷新缓存） |
| `/platform` | 切换平台（无参数列出全部，或 `ark` / `deepseek` / `opencode-go` / `openai` / `anthropic`） |
| `/temp <n>` | 设置 temperature（如 `0.7`） |
| `/max-tokens <n>` | 设置最大输出 token 数（如 `8192`） |
| `/compress <strategy>` | 切换压缩策略（`trim` / `token_trim` / `summarize`） |
| `/settings` | 显示当前设置 |
| `/rag-search <关键词>` | 直接搜索 RAG 向量库 |
| `/help` | 显示帮助 |
| `quit` / `exit` / `q` | 退出 |

在流式输出过程中按 **Ctrl+C** 可即时中断当前生成，回到输入提示符。

## 多平台支持

运行时切换 5 个平台，无需重启：

| 平台 | 提供商类型 | 端点 | 密钥环境变量 |
|------|-----------|------|-------------|
| `deepseek`（默认） | OpenAI 兼容 | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| `ark` | OpenAI 兼容 | `https://ark.cn-beijing.volces.com/api/v3` | `ARK_API_KEY` |
| `opencode-go` | OpenAI 兼容 | `https://opencode.ai/zen/go/v1` | `OPENCODE_GO_API_KEY` |
| `openai` | OpenAI 原生 | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | Anthropic 原生 | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |

每个平台有 `fallback_models` 列表，当 `/models` 端点不可用时回退。模型发现从平台 `/models` API 拉取，5 分钟 TTL 缓存。

```bash
# 切换平台（自动弹出模型选择列表）
/platform opencode-go

# 列出可用模型
/models

# 交互式选择模型
/model

# 直接指定模型
/model ark:doubao-seed-2-0-pro-260215
```

## 架构

### 模块结构

```
src/clawagent/
├── config.py              # Settings 数据类、价格表
├── platforms.py           # 平台预设（5 个平台）
├── model_factory.py       # 聊天模型工厂（平台感知）
├── model_discovery.py     # 从平台 API 拉取可用模型
├── agent.py               # Agent 类、create_agent、rebuild_graph
├── types.py               # Usage、AgentResponse 数据类
├── stream_processor.py    # 流式事件处理器（OpenAI + Anthropic 兼容）
├── stream_events.py       # 流式事件类型定义
├── prompt_builder.py      # 多层提示词拼装（TTL 缓存偏好）
├── main.py                # CLI 入口、REPL 循环
├── ui.py                  # Rich 仪表板、统计、格式化
├── ui_stream.py           # 流式显示（spinner、token、工具日志）
├── cancel_token.py        # Ctrl+C 协作式取消
├── conversation_log.py    # 按会话 JSONL 日志
├── cli/                   # CLI 命令和显示辅助
│   ├── commands.py        #   斜杠命令处理（/model /models /platform ...）
│   └── display.py         #   会话列表和格式化
├── gateway/               # 多渠道网关（微信、CLI）
│   ├── server.py          #   asyncio.Queue 真流式消息处理
│   ├── session_manager.py #   LRU + TTL 会话管理
│   └── channels/          #   微信 iLink Bot 渠道
├── api_pool/              # API Key 池 + 故障转移
│   ├── pool.py            #   ApiKeyPool 核心
│   ├── loader.py          #   环境变量加载
│   ├── models.py          #   KeyRecord、PoolConfig、KeyStatus
│   ├── wrapper.py         #   KeyPoolChatModel 包装器（跨提供商）
│   ├── transport.py       #   自定义 HTTP 传输层（深拷贝 headers）
│   └── callbacks.py       #   流式回调支持
├── tools/
│   ├── __init__.py        #   核心工具（读写、Shell、搜索）
│   ├── memory_tools.py    #   记忆工具（列出/回顾/摘要，闭包工厂）
│   ├── rag_tool.py        #   RAG search_documents 工具
│   └── web_search.py      #   Bing 搜索 + trafilatura（线程安全）
├── memory/
│   ├── summarizer.py      #   会话摘要 + 统一 WAL 连接缓存
│   └── preferences.py     #   用户偏好提取与查询
├── orchestrator/
│   └── delegator.py       #   delegate_task 工具（可配置截断）
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
│   ├── strategies.py      #   trim / token_trim / summarize（真超时）
│   └── counters.py        #   Token 估算
└── rag/
    ├── bootstrap.py       #   RAG 系统初始化 + BM25 缓存
    ├── embedding.py       #   SiliconFlow 云端嵌入客户端
    ├── store.py           #   Chroma 向量库
    ├── chunker.py         #   固定窗口文本分块
    ├── bm25.py            #   BM25 词法检索（jieba）+ JSON+HMAC 缓存
    ├── hybrid.py          #   KNN + BM25 混合检索 + RRF 融合（SHA256 去重）
    └── ingest.py          #   文档入库 CLI 脚本
```

### 多 Agent 协作

复杂任务由 Orchestrator 自动拆解并委托给专业 Worker：

| Worker | 职责 | 核心工具 |
|--------|------|----------|
| `coder` | 代码编写和调试 | `read_file`, `write_file`, `run_command` |
| `researcher` | 联网搜索和信息检索 | Bing 搜索, trafilatura 正文提取, `search_documents` (RAG) |
| `critic` | 代码审查和方案评审 | `read_file`, `search_documents` |
| `writer` | 文档编写和内容创作 | `read_file`, `write_file` |

每个 Worker 是独立的临时 Agent，按需创建，任务完成后自动销毁。运行时配置（模型、temperature）通过热更新自动同步到 Worker。

## 配置

环境变量（在 `.env` 中设置）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CLAWAGENT_API_KEY` | *(必填)* | API 密钥（所有平台的 fallback） |
| `CLAWAGENT_PLATFORM` | `deepseek` | 平台：`deepseek` / `ark` / `opencode-go` / `openai` / `anthropic` |
| `CLAWAGENT_MODEL` | `deepseek-v4-flash` | 模型名称 |
| `CLAWAGENT_MODEL_PROVIDER` | `openai` | LangChain 提供商（`openai`、`anthropic`） |
| `CLAWAGENT_API_BASE` | *(跟随平台)* | 自定义 API 基础 URL（覆盖平台预设） |
| `CLAWAGENT_AGENT_ID` | `wenbao` | Agent 身份（读取 `prompts/agents/<id>/`） |
| `CLAWAGENT_CONTEXT_WINDOW` | `1000000` | 上下文窗口大小（仅用于显示） |
| `CLAWAGENT_MEMORY_DB` | `memories/sessions.db` | SQLite 记忆数据库路径 |
| `CLAWAGENT_MAX_PREFERENCES` | `5` | 注入提示词的最大偏好数量 |
| `CLAWAGENT_MAX_RESULT_CHARS` | `50000` | Worker 委托结果最大字符数 |
| `CLAWAGENT_REQUEST_TIMEOUT` | `120` | API 请求超时（秒） |
| `COMPRESSION_STRATEGY` | `trim` | 上下文压缩策略 |
| `COMPRESSION_MAX_MESSAGES` | `40` | 触发裁剪的最大消息数 |
| `COMPRESSION_MAX_TOKENS` | `80000` | `token_trim` 的 token 阈值 |
| `COMPRESSION_KEEP_RECENT` | `6` | 保留的最近消息数 |
| `API_POOL_DEFAULT_KEYS` | *(可选)* | 逗号分隔的 API Key 列表 |
| `SILICONFLOW_API_KEY` | *(可选)* | SiliconFlow 密钥（RAG 嵌入用） |

### 各平台独立密钥

每个平台优先读自己的密钥环境变量，未设置则 fallback 到 `CLAWAGENT_API_KEY`：

| 平台 | 密钥环境变量 |
|------|-------------|
| deepseek | `DEEPSEEK_API_KEY` |
| ark | `ARK_API_KEY` |
| opencode-go | `OPENCODE_GO_API_KEY` |
| openai | `OPENAI_API_KEY` |
| anthropic | `ANTHROPIC_API_KEY` |

### Worker 配置

每个 Worker 可使用不同的模型/提供商：

```bash
# 通用默认值（fallback）
WORKER_COMMON_MODEL=deepseek-v4-flash
WORKER_COMMON_MODEL_PROVIDER=openai

# 各 Worker 独立配置
WORKER_CODER_MODEL=deepseek-v4-flash
WORKER_RESEARCHER_MODEL=Qwen/Qwen3-235B-A22B
WORKER_RESEARCHER_MODEL_PROVIDER=openai
WORKER_RESEARCHER_API_BASE=https://api.siliconflow.cn/v1
```

## 开发

```bash
# 安装依赖
uv sync

# 代码检查
uv run ruff check .

# 类型检查（0 错误，71 个源文件）
uv run mypy src/

# 运行测试（379 个测试）
uv run pytest tests/ -v

# 运行单文件测试
uv run pytest tests/test_rag.py -v
```

## 许可证

MIT

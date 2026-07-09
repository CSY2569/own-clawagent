# clawagent

[English](README.md) | [中文](README.zh.md)

A LangChain/LangGraph tool-calling agent with multi-platform support (DeepSeek, Volcano Ark, OpenCode Go, OpenAI, Anthropic), multi-agent orchestration, hybrid RAG, streaming display, and conversation memory.

## Features

- **Multi-platform support** - 5 platforms out of the box: DeepSeek (default), Volcano Ark, OpenCode Go, OpenAI, Anthropic. Switch at runtime via `/platform`. Auto-fetch available models from each platform's API
- **Tool-calling agent** - 8+ built-in tools: 4 core tools (file I/O, shell commands, RAG document search), 3 memory tools (list/recall/summarize sessions, closure-factory pattern), web search, and multi-agent task delegation
- **Multi-agent orchestration** - Orchestrator + 4 specialized Workers (Coder, Researcher, Critic, Writer). Complex tasks are decomposed and delegated via `delegate_task(role, task)`. Each Worker runs with an independent model, tool set, and prompt context
- **Real-time streaming display** - Token-level typewriter effect with Rich Live dashboard. Tool calls and results are shown as they execute. Supports thinking blocks (`think_start` / `think_end`)
- **Ctrl+C instant interrupt** - Threaded producer-consumer pattern ensures ≤100ms response to cancellation, with graceful background cleanup
- **Web search** - Researcher Worker uses Bing direct search + trafilatura full-text extraction with concurrent page fetching (thread-safe rate limiter)
- **RAG (Retrieval-Augmented Generation)** - Hybrid search: KNN vector retrieval (Chroma, SiliconFlow embeddings) + BM25 lexical matching (jieba tokenizer) fused via Reciprocal Rank Fusion. BM25 index cache (JSON + HMAC signature) skips tokenization on restart
- **API key pool** - Multi-key failover with exponential backoff per key. Supports `API_POOL_*` environment variables for automatic key rotation
- **Multi-layer system prompt** - File-based assembly: identity, personality, workspace context, runtime metadata, user preferences (TTL-cached), and auto-generated tool listing
- **Conversation memory** - SQLite-backed session persistence with automatic summarization, user preference extraction (background thread), and JSONL logging per session. Unified WAL connection cache across memory modules
- **Context compression** - 3 strategies: message-count trim, token-estimate trim, or LLM-based summarization (with real timeout fallback)
- **Gateway mode** - Multi-channel event loop (CLI, WeChat iLink Bot) with real streaming via asyncio.Queue bridge
- **Settings hot-reload** - Switch platform, model, temperature, max tokens, and compression strategy without restart
- **Interactive REPL** - Rich-powered dashboard with live token tracking, cost estimation, slash-command autocomplete, model discovery dialog, and streaming output
- **mypy zero errors** - Full strict-mode type checking across 71 source files
- **379 passing tests** - Full coverage: agents, workers, tools, memory, RAG, compression, streaming, API pool, cancel token, gateway, and UI

## Quick Start

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
git clone git@github.com:<user>/clawagent.git
cd clawagent
uv sync
cp .env.example .env
# Edit .env and set CLAWAGENT_API_KEY
```

### Usage

```bash
# One-shot
uv run clawagent "What time is it?"

# Interactive REPL
uv run clawagent

# Gateway mode (WeChat + CLI channels)
uv run clawagent gateway
```

## Interactive Commands

The REPL supports slash commands with autocomplete (type `/`):

| Command | Description |
|---------|-------------|
| `/sessions` | List all historical sessions |
| `/load <id>` | Load and switch to a session |
| `/new` | Start a new session |
| `/model` | Switch model (no args = interactive dialog, or `platform:model_name`) |
| `/models` | List available models from current platform (`/models refresh` to clear cache) |
| `/platform` | Switch platform (no args = list, or `ark` / `deepseek` / `opencode-go` / `openai` / `anthropic`) |
| `/temp <n>` | Set temperature (e.g. `0.7`) |
| `/max-tokens <n>` | Set max output tokens (e.g. `8192`) |
| `/compress <strategy>` | Switch compression (`trim` / `token_trim` / `summarize`) |
| `/settings` | Show current configuration |
| `/rag-search <query>` | Search RAG vector store directly |
| `/help` | Show help |
| `quit` / `exit` / `q` | Exit |

Press **Ctrl+C** during streaming to instantly cancel the current generation and return to the prompt.

## Multi-Platform Support

Switch between 5 platforms at runtime without restarting:

| Platform | Provider | Endpoint | Key Env Var |
|----------|----------|----------|-------------|
| `deepseek` (default) | OpenAI-compatible | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| `ark` | OpenAI-compatible | `https://ark.cn-beijing.volces.com/api/v3` | `ARK_API_KEY` |
| `opencode-go` | OpenAI-compatible | `https://opencode.ai/zen/go/v1` | `OPENCODE_GO_API_KEY` |
| `openai` | OpenAI native | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | Anthropic native | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |

Each platform has a `fallback_models` list used when the `/models` endpoint is unavailable. Model discovery fetches from the platform's `/models` API with a 5-minute TTL cache.

```bash
# Switch platform (auto-shows model selection dialog)
/platform opencode-go

# List available models
/models

# Switch model interactively
/model

# Switch model directly
/model ark:doubao-seed-2-0-pro-260215
```

## Architecture

### Module Structure

```
src/clawagent/
├── config.py              # Settings dataclass, price book
├── platforms.py           # Platform presets (5 platforms)
├── model_factory.py       # Chat model factory (platform-aware)
├── model_discovery.py     # Fetch available models from platform APIs
├── agent.py               # Agent class, create_agent, rebuild_graph
├── types.py               # Usage, AgentResponse dataclasses
├── stream_processor.py    # Stream event processor (OpenAI + Anthropic compat)
├── stream_events.py       # Stream event type definitions
├── prompt_builder.py      # Multi-layer prompt assembly (TTL-cached prefs)
├── main.py                # CLI entry point, REPL loop
├── ui.py                  # Rich dashboard, stats, formatting
├── ui_stream.py           # Streaming display (spinner, tokens, tool log)
├── cancel_token.py        # Ctrl+C cooperative cancellation
├── conversation_log.py    # Per-session JSONL logging
├── cli/                   # CLI commands and display helpers
│   ├── commands.py        #   Slash-command handling (/model /models /platform ...)
│   └── display.py         #   Session listing and formatting
├── gateway/               # Multi-channel gateway (WeChat, CLI)
│   ├── server.py          #   asyncio.Queue real-streaming message handler
│   ├── session_manager.py #   LRU + TTL session management
│   └── channels/          #   WeChat iLink Bot channel
├── api_pool/              # API key pool with failover
│   ├── pool.py            #   ApiKeyPool core
│   ├── loader.py          #   Environment variable loading
│   ├── models.py          #   KeyRecord, PoolConfig, KeyStatus
│   ├── wrapper.py         #   KeyPoolChatModel wrapper (cross-provider)
│   ├── transport.py       #   Custom HTTP transport (deep-copy headers)
│   └── callbacks.py       #   Streaming callback support
├── tools/
│   ├── __init__.py        #   Core tools (read, write, shell, search)
│   ├── memory_tools.py    #   Session list/recall/summarize (closure factory)
│   ├── rag_tool.py        #   RAG search_documents tool
│   └── web_search.py      #   Bing search + trafilatura (thread-safe)
├── memory/
│   ├── summarizer.py      #   Session summarization + unified WAL connection cache
│   └── preferences.py     #   User preference extraction & querying
├── orchestrator/
│   └── delegator.py       #   delegate_task tool (configurable truncation)
├── worker/
│   ├── base.py            #   BaseWorker abstract class
│   ├── factory.py         #   WorkerFactory (creates workers by role)
│   ├── config.py          #   Worker configuration from env vars
│   ├── registry.py        #   @register_worker decorator
│   ├── coder.py           #   Code writing worker
│   ├── researcher.py      #   Web search + RAG worker
│   ├── critic.py          #   Code review worker
│   └── writer.py          #   Documentation worker
├── compression/
│   ├── __init__.py        #   Entry point + pre_model_hook
│   ├── config.py          #   CompressionConfig
│   ├── strategies.py      #   trim / token_trim / summarize (real timeout)
│   └── counters.py        #   Token estimation
└── rag/
    ├── bootstrap.py       #   RAG system initialization + BM25 cache
    ├── embedding.py       #   SiliconFlow cloud embedding client
    ├── store.py           #   Chroma vector store
    ├── chunker.py         #   Fixed-window text chunking
    ├── bm25.py            #   BM25 lexical retriever (jieba) + JSON+HMAC cache
    ├── hybrid.py          #   KNN + BM25 hybrid with RRF fusion (SHA256 dedup)
    └── ingest.py          #   Document ingestion CLI
```

### Multi-Agent Orchestration

Complex tasks are automatically decomposed by the Orchestrator and delegated to specialized Workers:

| Worker | Responsibility | Key Tools |
|--------|---------------|-----------|
| `coder` | Code writing and debugging | `read_file`, `write_file`, `run_command` |
| `researcher` | Web search and information retrieval | Bing search, trafilatura extraction, `search_documents` (RAG) |
| `critic` | Code review and solution assessment | `read_file`, `search_documents` |
| `writer` | Documentation and content creation | `read_file`, `write_file` |

Each Worker is an independent, temporary Agent with its own model configuration, prompt identity, and tool set. Workers are created on demand and destroyed after task completion. Runtime settings (model, temperature) are automatically propagated to workers via hot-reload.

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAWAGENT_API_KEY` | *(required)* | API key (fallback for all platforms) |
| `CLAWAGENT_PLATFORM` | `deepseek` | Platform: `deepseek` / `ark` / `opencode-go` / `openai` / `anthropic` |
| `CLAWAGENT_MODEL` | `deepseek-v4-flash` | Model name |
| `CLAWAGENT_MODEL_PROVIDER` | `openai` | LangChain provider (`openai`, `anthropic`) |
| `CLAWAGENT_API_BASE` | *(from platform)* | Custom API base URL (overrides platform preset) |
| `CLAWAGENT_AGENT_ID` | `wenbao` | Agent identity (`prompts/agents/<id>/`) |
| `CLAWAGENT_CONTEXT_WINDOW` | `1000000` | Context window size (for display) |
| `CLAWAGENT_MEMORY_DB` | `memories/sessions.db` | SQLite memory database path |
| `CLAWAGENT_MAX_PREFERENCES` | `5` | Max preferences injected into prompt |
| `CLAWAGENT_MAX_RESULT_CHARS` | `50000` | Max chars for worker delegation results |
| `CLAWAGENT_REQUEST_TIMEOUT` | `120` | API request timeout in seconds |
| `COMPRESSION_STRATEGY` | `trim` | Context compression strategy |
| `COMPRESSION_MAX_MESSAGES` | `40` | Max messages before trimming |
| `COMPRESSION_MAX_TOKENS` | `80000` | Token threshold for `token_trim` |
| `COMPRESSION_KEEP_RECENT` | `6` | Recent messages to preserve |
| `API_POOL_DEFAULT_KEYS` | *(optional)* | Comma-separated API keys for pool |
| `SILICONFLOW_API_KEY` | *(optional)* | SiliconFlow key for RAG embedding |

### Per-Platform Keys

Each platform reads its own key env var first, falling back to `CLAWAGENT_API_KEY`:

| Platform | Key Env Var |
|----------|-------------|
| deepseek | `DEEPSEEK_API_KEY` |
| ark | `ARK_API_KEY` |
| opencode-go | `OPENCODE_GO_API_KEY` |
| openai | `OPENAI_API_KEY` |
| anthropic | `ANTHROPIC_API_KEY` |

### Worker Configuration

Each Worker can use a different model/provider:

```bash
# Common defaults (fallback)
WORKER_COMMON_MODEL=deepseek-v4-flash
WORKER_COMMON_MODEL_PROVIDER=openai

# Per-worker overrides
WORKER_CODER_MODEL=deepseek-v4-flash
WORKER_RESEARCHER_MODEL=Qwen/Qwen3-235B-A22B
WORKER_RESEARCHER_MODEL_PROVIDER=openai
WORKER_RESEARCHER_API_BASE=https://api.siliconflow.cn/v1
```

## Development

```bash
# Install dependencies
uv sync

# Lint
uv run ruff check .

# Type check (0 errors, 71 source files)
uv run mypy src/

# Run tests (379 tests)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_rag.py -v
```

## License

MIT

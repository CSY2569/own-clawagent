# clawagent

[English](README.md) | [中文](README.zh.md)

A LangChain/LangGraph tool-calling agent powered by DeepSeek (via Anthropic-compatible API), with multi-agent orchestration, hybrid RAG, streaming display, and conversation memory.

## Features

- **Tool-calling agent** — 8+ built-in tools: 4 core tools (file I/O, shell commands, RAG document search), 3 memory tools (list/recall/summarize sessions, closure-factory pattern), web search, and multi-agent task delegation
- **Multi-agent orchestration** — Orchestrator + 4 specialized Workers (Coder, Researcher, Critic, Writer). Complex tasks are decomposed and delegated via `delegate_task(role, task)`. Each Worker runs with an independent model, tool set, and prompt context. Settings hot-reload propagates automatically to WorkerFactory
- **Real-time streaming display** — Token-level typewriter effect with Rich Live dashboard. Tool calls and results are shown as they execute. Supports DeepSeek thinking blocks (`think_start` / `think_end`)
- **Ctrl+C instant interrupt** — Threaded producer-consumer pattern ensures ≤100ms response to cancellation, with graceful background cleanup
- **Web search** — Researcher Worker uses Bing direct search + trafilatura full-text extraction with concurrent page fetching
- **RAG (Retrieval-Augmented Generation)** — Hybrid search: KNN vector retrieval (Chroma, SiliconFlow embeddings) + BM25 lexical matching (jieba tokenizer) fused via Reciprocal Rank Fusion. BM25 index cache (SHA256-validated pickle) skips tokenization on restart when documents are unchanged
- **API key pool** — Multi-key failover with exponential backoff per key. Supports `API_POOL_*` environment variables for automatic key rotation
- **Multi-layer system prompt** — File-based assembly: identity, personality, workspace context, runtime metadata, user preferences, and auto-generated tool listing
- **Conversation memory** — SQLite-backed session persistence with automatic summarization, user preference extraction, and JSONL logging per session. Connection cache with WAL mode avoids repeated connect/disconnect within a turn
- **Context compression** — 3 strategies: message-count trim, token-estimate trim, or LLM-based summarization
- **Settings hot-reload** — Switch model, temperature, max tokens, and compression strategy without restart
- **Interactive REPL** — Rich-powered dashboard with live token tracking, cost estimation, slash-command autocomplete, and streaming output
- **mypy zero errors** — Full strict-mode type checking across 50 source files
- **327 passing tests** — Full coverage: agents, workers, tools, memory, RAG, compression, streaming, API pool, cancel token, and UI

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
# Edit .env and set ANTHROPIC_API_KEY
```

### Usage

```bash
# One-shot
uv run clawagent "What time is it?"

# Interactive REPL
uv run clawagent
```

## Interactive Commands

The REPL supports slash commands with autocomplete (type `/`):

| Command | Description |
|---------|-------------|
| `/sessions` | List all historical sessions |
| `/load <id>` | Load and switch to a session |
| `/new` | Start a new session |
| `/model <name>` | Switch model (e.g. `deepseek-v4-pro`) |
| `/temp <n>` | Set temperature (e.g. `0.7`) |
| `/max-tokens <n>` | Set max output tokens (e.g. `8192`) |
| `/compress <strategy>` | Switch compression (`trim` / `token_trim` / `summarize`) |
| `/settings` | Show current configuration |
| `/rag-search <query>` | Search RAG vector store directly |
| `/help` | Show help |
| `quit` / `exit` / `q` | Exit |

Press **Ctrl+C** during streaming to instantly cancel the current generation and return to the prompt.

## Architecture

### Multi-Agent Orchestration

Complex tasks are automatically decomposed by the Orchestrator and delegated to specialized Workers:

| Worker | Responsibility | Key Tools |
|--------|---------------|-----------|
| `coder` | Code writing and debugging | `read_file`, `write_file`, `run_command` |
| `researcher` | Web search and information retrieval | Bing search, trafilatura extraction, `search_documents` (RAG) |
| `critic` | Code review and solution assessment | `read_file`, `search_documents` |
| `writer` | Documentation and content creation | `read_file`, `write_file` |

Each Worker is an independent, temporary Agent with its own model configuration, prompt identity, and tool set. Workers are created on demand and destroyed after task completion. Runtime settings (model, temperature) are automatically propagated to workers via hot-reload.

### Streaming Display

The REPL displays agent execution in real-time:

```
  ⠋ Calling search_documents("龙神恩雅")...
  ✓ search_documents (6 lines)
  ✓ read_file (152 lines)
  ──────────────────────────────────────────
  From Dawn of Swords, Enya did not completely vanish...

  In 1,230 · Out 340
```

- Animated spinner while thinking
- Live tool call tracking with argument preview
- Token-level typewriter output
- DeepSeek thinking block detection (`think_start` / `think_end`)
- Token usage and cost stats on completion

### Prompt System

The system prompt is assembled by `PromptBuilder` from five layers:

| Layer | Source | Required |
|-------|--------|----------|
| 1. Identity | `prompts/agents/{id}/identity.md` | Yes (with fallback) |
| 2. Personality | `prompts/agents/{id}/soul.md` | No |
| 3. Workspace | `prompts/shared/bootstrap.md`, `agents.md`, `search-rules.md` | No |
| 4. Runtime | Agent ID, timestamp, channel (auto-generated) | Yes |
| 5. Preferences | SQLite `preferences` table (auto-learned) | No |

Tools are auto-listed from `ALL_TOOLS + delegate_task` — add a new `@tool` and it appears automatically.

To add a new agent: create `prompts/agents/<name>/identity.md` and set `CLAWAGENT_AGENT_ID=<name>`.

### RAG (Retrieval-Augmented Generation)

Agentic RAG — the LLM decides when to search documents via the `search_documents` tool.

- **Hybrid search**: KNN vector retrieval (Chroma HNSW) + BM25 lexical matching (jieba) fused via Reciprocal Rank Fusion
- **Embedding**: SiliconFlow cloud API (`Qwen/Qwen3-VL-Embedding-8B`, 768 dimensions)
- **Chapter metadata**: Auto-detects chapter markers during ingestion
- **Background BM25**: Index builds in background thread; graceful fallback to KNN-only during construction

```bash
# Ingest documents (run once)
uv run python -m clawagent.rag.ingest docs/ --chunk-size 512 --overlap 64

# Test from CLI
uv run clawagent
> /rag-search keyword
```

### API Key Pool

Automatic key failover with per-key exponential backoff:

```bash
# .env
API_POOL_DEFAULT_KEYS=sk-key1,sk-key2,sk-key3
```

When one key hits rate limits or errors, the pool automatically rotates to the next. Failed keys are cooled off with exponential backoff. Keys can also be loaded from a JSON file via `API_POOL_DEFAULT_KEY_FILE`.

### Web Search

The Researcher Worker performs live web searches:

- **Bing direct search** — scrapes Bing search result pages
- **trafilatura extraction** — fetches and extracts clean article text from top results
- **Concurrent fetching** — uses `ThreadPoolExecutor` for parallel page retrieval
- Configurable limits: max search results, max deep pages, page timeout

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | DeepSeek API key |
| `ANTHROPIC_BASE_URL` | — | API base URL (DeepSeek: `https://api.deepseek.com/anthropic`) |
| `CLAWAGENT_MODEL` | `deepseek-v4-flash` | Model name |
| `CLAWAGENT_MODEL_PROVIDER` | `anthropic` | Provider (`anthropic`, `openai`, etc.) |
| `CLAWAGENT_AGENT_ID` | `wenbao` | Agent identity (`prompts/agents/<id>/`) |
| `CLAWAGENT_CONTEXT_WINDOW` | `1000000` | Context window size (for display) |
| `CLAWAGENT_MEMORY_DB` | `memories/sessions.db` | SQLite memory database path |
| `CLAWAGENT_MAX_PREFERENCES` | `5` | Max preferences injected into prompt |
| `CLAWAGENT_REQUEST_TIMEOUT` | `120` | API request timeout in seconds |
| `COMPRESSION_STRATEGY` | `trim` | Context compression strategy |
| `COMPRESSION_MAX_MESSAGES` | `40` | Max messages before trimming |
| `COMPRESSION_MAX_TOKENS` | `80000` | Token threshold for `token_trim` |
| `COMPRESSION_KEEP_RECENT` | `6` | Recent messages to preserve |
| `API_POOL_DEFAULT_KEYS` | *(optional)* | Comma-separated API keys for pool |
| `API_POOL_DEFAULT_KEY_FILE` | *(optional)* | JSON file with API keys |
| `SILICONFLOW_API_KEY` | *(optional)* | SiliconFlow key for RAG embedding |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1/embeddings` | Embedding API URL |
| `SILICONFLOW_MODEL` | `Qwen/Qwen3-VL-Embedding-8B` | Embedding model |
| `SILICONFLOW_DIMENSIONS` | `768` | Embedding dimensions |

### Worker Configuration

Each Worker can use a different model/provider:

```bash
# Common defaults (fallback)
WORKER_COMMON_MODEL=deepseek-v4-flash

# Per-worker overrides
WORKER_CODER_MODEL=deepseek-v4-flash
WORKER_RESEARCHER_MODEL=Qwen/Qwen3-235B-A22B
WORKER_RESEARCHER_MODEL_PROVIDER=openai
WORKER_RESEARCHER_API_BASE=https://api.siliconflow.cn/v1
WORKER_CRITIC_MODEL=Qwen/Qwen3-235B-A22B
WORKER_CRITIC_MODEL_PROVIDER=openai
WORKER_CRITIC_API_BASE=https://api.siliconflow.cn/v1
WORKER_WRITER_MODEL=deepseek-v4-flash
```

## Project Structure

```
src/clawagent/
├── config.py              # Settings dataclass, price book
├── agent.py               # Agent factory, run/stream_events/reconfigure
├── prompt_builder.py      # Multi-layer prompt assembly
├── main.py                # CLI entry point, REPL loop
├── ui.py                  # Rich dashboard, stats, formatting
├── ui_stream.py           # Streaming display (spinner, tokens, tool log)
├── stream_events.py       # Stream event type definitions
├── cancel_token.py        # Ctrl+C cooperative cancellation
├── conversation_log.py    # Per-session JSONL logging
├── cli/                   # CLI commands and display helpers
│   ├── commands.py        #   Slash-command handling
│   └── display.py         #   Session listing and formatting
├── api_pool/              # API key pool with failover
│   ├── pool.py            #   ApiKeyPool core
│   ├── loader.py          #   Environment variable loading
│   ├── models.py          #   KeyRecord, PoolConfig, KeyStatus
│   ├── wrapper.py         #   KeyPoolChatModel wrapper
│   ├── transport.py       #   Custom HTTP transport
│   └── callbacks.py       #   Streaming callback support
├── tools/
│   ├── __init__.py        #   Core tools (read, write, shell, search)
│   ├── memory_tools.py    #   Session list/recall/summarize (closure factory)
│   ├── rag_tool.py        #   RAG search_documents tool
│   └── web_search.py      #   Bing search + trafilatura extraction
├── memory/
│   ├── summarizer.py      #   Session summarization & message persistence
│   └── preferences.py     #   User preference extraction & querying
├── orchestrator/
│   └── delegator.py       #   delegate_task tool for worker delegation
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
│   ├── strategies.py      #   trim / token_trim / summarize
│   └── counters.py        #   Token estimation
└── rag/
    ├── bootstrap.py       #   RAG system initialization + BM25 cache
    ├── embedding.py       #   SiliconFlow cloud embedding client
    ├── store.py           #   Chroma vector store
    ├── chunker.py         #   Fixed-window text chunking
    ├── bm25.py            #   BM25 lexical retriever (jieba) + pickle cache
    ├── hybrid.py          #   KNN + BM25 hybrid with RRF fusion
    └── ingest.py          #   Document ingestion CLI

prompts/
├── agents/wenbao/         # Default agent prompt files
│   ├── identity.md
│   └── soul.md
├── shared/                # Shared workspace context
│   ├── bootstrap.md
│   ├── agents.md
│   └── search-rules.md
└── README.md

tests/                     # 327 tests (22 files)
├── test_agent.py          # Agent wrapper (Usage, AgentResponse)
├── test_api_pool.py       # API key pool
├── test_cancel_token.py   # Ctrl+C cancellation
├── test_compression.py    # Context compression
├── test_config.py         # Settings + price book
├── test_conversation_log.py
├── test_functional.py     # End-to-end agent graph integration
├── test_memory_tools.py   # Memory tool interface (closure factory)
├── test_orchestrator.py   # delegate_task delegation
├── test_preferences.py    # User preference extraction
├── test_rag.py            # RAG: chunker, BM25 (incl. cache), bootstrap
├── test_stream_events.py  # Stream events + UI display
├── test_streaming.py      # Streaming integration
├── test_summarizer.py     # Session summarization
├── test_tools.py          # Core tools
├── test_ui.py             # UI stats/formatting
├── test_web_search.py     # Web search
├── test_worker_base.py    # Worker lifecycle
├── test_worker_config.py  # Worker env var loading
├── test_worker_factory.py # WorkerFactory
├── test_worker_impl.py    # Individual worker behavior
└── test_worker_registry.py
```

## Development

```bash
# Install dependencies
uv sync

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check (0 errors, 50 source files)
uv run mypy src/

# Run tests (327 tests)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_rag.py -v
```

## License

MIT

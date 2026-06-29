# clawagent

[English](README.md) | [中文](README.zh.md)

A LangChain/LangGraph tool-calling agent powered by Anthropic Claude-compatible models (DeepSeek via Anthropic API), with multi-agent orchestration, hybrid RAG, streaming display, and conversation memory.

## Features

- **Tool-calling agent** — 10+ built-in tools: file I/O, shell commands, time query, conversation memory, RAG document search, and multi-agent task delegation
- **Multi-agent orchestration** — Orchestrator + 4 specialized Workers (Coder, Researcher, Critic, Writer). Complex tasks are decomposed and delegated via `delegate_task(role, task)`. Each Worker runs with an independent model, tool set, and prompt context.
- **Real-time streaming display** — Token-level typewriter effect with animated spinner. Tool calls and results are shown live as they execute. Supports DeepSeek thinking blocks.
- **RAG (Retrieval-Augmented Generation)** — Hybrid search combining KNN vector retrieval (Chroma, SiliconFlow embeddings) with BM25 lexical matching (jieba tokenizer) via Reciprocal Rank Fusion (RRF). BM25 index builds in background on startup for instant launch.
- **Multi-layer system prompt** — File-based prompt assembly with per-agent identity, personality, workspace context, runtime metadata, user preferences, and auto-generated tool listing
- **Conversation memory** — SQLite-backed session persistence with automatic summarization and user preference extraction
- **Context compression** — 3 strategies: trim by message count, trim by estimated token count, or LLM-based summarization
- **Settings hot-reload** — Switch model, temperature, max tokens, and compression strategy without restart
- **Interactive REPL** — Rich-powered dashboard with live token tracking, cost estimation, and streaming display
- **204 passing tests** — Full test coverage including worker registry, config, lifecycle, tool sets, and orchestrator delegation

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
# Edit .env and set your ANTHROPIC_API_KEY
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
| `/compress <strategy>` | Switch compression strategy (`trim` / `token_trim` / `summarize`) |
| `/settings` | Show current configuration |
| `/rag-search <query>` | Search the RAG vector store directly |
| `/help` | Show help |
| `quit` / `exit` / `q` | Exit |

## Multi-Agent Orchestration

Complex tasks are automatically decomposed by the Orchestrator and delegated to specialized Workers:

| Worker | Responsibility | Tools |
|--------|---------------|-------|
| `coder` | Code writing and debugging | `read_file`, `write_file`, `run_command` |
| `researcher` | Information retrieval and research | `search_documents` (RAG) |
| `critic` | Code review and solution assessment | `read_file`, `search_documents` |
| `writer` | Documentation and content creation | `read_file`, `write_file` |

Each Worker is an independent, temporary Agent with its own model configuration, prompt identity, and tool set. Workers are created on demand and destroyed after task completion. Runtime settings (model, temperature) are automatically propagated to workers via hot-reload.

### Search Rules (Shared)

Search behavior rules (use exact keywords, never fabricate, cross-validate) are loaded from `prompts/shared/search-rules.md` and applied to all agents automatically via PromptBuilder Layer 3.

## Streaming Display

The REPL displays agent execution in real-time:

```
  ⠋ Calling search_documents("龙神恩雅")...
  ✓ search_documents (6 lines)
  ✓ read_file (152 lines)
  ──────────────────────────────────────────
  从《黎明之剑》来看，龙神恩雅并没有彻底消亡...

  In 1,230 · Out 340
```

Features:
- Animated spinner while thinking
- Live tool call tracking with argument preview
- Token-level typewriter output (Phase 2)
- DeepSeek thinking block detection (`think_start` / `think_end`)
- Usage stats after completion

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | API key |
| `ANTHROPIC_BASE_URL` | — | API base URL (DeepSeek: `https://api.deepseek.com/anthropic`) |
| `CLAWAGENT_MODEL` | `deepseek-v4-flash` | Model name |
| `CLAWAGENT_MODEL_PROVIDER` | `anthropic` | Model provider (`anthropic`, `openai`, etc.) |
| `CLAWAGENT_AGENT_ID` | `pickle` | Agent identity (reads `prompts/agents/<id>/`) |
| `CLAWAGENT_CONTEXT_WINDOW` | `1000000` | Context window size (for display) |
| `CLAWAGENT_MEMORY_DB` | `memories/sessions.db` | Path to SQLite memory database |
| `CLAWAGENT_MAX_PREFERENCES` | `5` | Max user preferences injected into prompt |
| `COMPRESSION_STRATEGY` | `trim` | Context compression strategy |
| `COMPRESSION_MAX_MESSAGES` | `40` | Max messages before trimming |
| `COMPRESSION_MAX_TOKENS` | `80000` | Token threshold for token_trim |
| `COMPRESSION_KEEP_RECENT` | `6` | Recent messages to preserve |
| `SILICONFLOW_API_KEY` | *(optional)* | SiliconFlow API key for RAG embedding |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1/embeddings` | Embedding API URL |
| `SILICONFLOW_MODEL` | `Qwen/Qwen3-VL-Embedding-8B` | Embedding model name |
| `SILICONFLOW_DIMENSIONS` | `768` | Embedding vector dimensions |

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

## Multi-Layer Prompt System

The system prompt is assembled by `PromptBuilder` from five layers:

| Layer | Source | Required |
|-------|--------|----------|
| 1. Identity | `prompts/agents/{id}/identity.md` | Yes (fallback if missing) |
| 2. Personality | `prompts/agents/{id}/soul.md` | No |
| 3. Workspace | `prompts/shared/bootstrap.md`, `agents.md`, `search-rules.md` | No |
| 4. Runtime | Agent ID, timestamp, channel (auto-generated) | Yes |
| 5. Preferences | SQLite `preferences` table (auto-learned) | No |

Tools are auto-listed from `ALL_TOOLS + delegate_task` — add a new `@tool` and it appears automatically. The tool description in the system prompt stays in sync with the actual registered tools.

To add a new agent: create `prompts/agents/<name>/identity.md` and set `CLAWAGENT_AGENT_ID=<name>`.

## RAG (Retrieval-Augmented Generation)

clawagent supports Agentic RAG — the LLM decides when to search documents by calling the `search_documents` tool, rather than injecting context on every conversation turn.

**Architecture:**

- **Hybrid search**: KNN vector retrieval (Chroma) + BM25 lexical matching (jieba) fused via Reciprocal Rank Fusion (RRF)
- **Embedding**: Cloud-based via SiliconFlow API (`Qwen/Qwen3-VL-Embedding-8B`, 768 dimensions)
- **Vector store**: Local Chroma (HNSW index, persisted at `chroma_db/`)
- **Chapter metadata**: Automatically detects chapter markers (`第X章`, `Chapter X`, etc.) during ingestion
- **Background BM25**: BM25 index builds in a background thread on startup, with graceful fallback to KNN-only during construction — no startup delay

**Setup:**

```bash
# 1. Configure SILICONFLOW_API_KEY in .env

# 2. Ingest documents (run once)
uv run python -m clawagent.rag.ingest docs/ --chunk-size 512 --overlap 64

# 3. Test retrieval from CLI
uv run clawagent
You: /rag-search 高文的亲人
```

Searches return chapter info when available: `[1] (相关度: 0.85, 第12章) — 高文·塞西尔是...`

## Project Structure

```
src/clawagent/
├── config.py            # Settings dataclass, price book
├── agent.py             # Agent factory, run/stream_events/reconfigure
├── prompt_builder.py    # Multi-layer prompt assembly
├── main.py              # CLI entry point, REPL loop
├── ui.py                # Rich dashboard, stats, formatting
├── ui_stream.py         # Real-time streaming display (spinner + tokens)
├── stream_events.py     # Typed stream event dataclass
├── tools/
│   ├── __init__.py      # Core tools (read, write, shell, time, greet)
│   ├── memory_tools.py  # Memory tools (list/recall/summarize)
│   └── rag_tool.py      # RAG search_documents tool
├── memory/
│   ├── summarizer.py    # Session summarization & message persistence
│   └── preferences.py   # User preference extraction & querying
├── orchestrator/
│   └── delegator.py     # delegate_task tool for worker delegation
├── worker/
│   ├── base.py          # BaseWorker abstract class
│   ├── factory.py       # WorkerFactory (creates workers by role)
│   ├── config.py        # Worker configuration from env vars
│   ├── registry.py      # @register_worker decorator
│   ├── coder.py         # Code writing worker
│   ├── researcher.py    # Information retrieval worker
│   ├── critic.py        # Code review worker
│   └── writer.py        # Documentation worker
├── compression/
│   ├── __init__.py      # Entry point + pre_model_hook factory
│   ├── config.py        # CompressionConfig
│   ├── strategies.py    # trim / token_trim / summarize
│   └── counters.py      # Token estimation
└── rag/
    ├── __init__.py
    ├── embedding.py     # SiliconFlow cloud embedding client
    ├── store.py         # Chroma vector store
    ├── chunker.py       # Fixed-window text chunking
    ├── bm25.py          # BM25 lexical retriever (jieba)
    ├── hybrid.py        # KNN + BM25 hybrid with RRF fusion
    └── ingest.py        # CLI document ingestion

prompts/
├── agents/pickle/       # Default agent prompt files
│   ├── identity.md
│   └── soul.md
├── shared/              # Shared workspace context
│   ├── bootstrap.md
│   ├── agents.md
│   └── search-rules.md
└── README.md

tests/                   # 204 passing tests
├── test_worker_*.py     # Worker registry, config, lifecycle, tool sets
├── test_orchestrator.py # delegate_task delegation
├── test_agent.py        # Agent wrapper (Usage, extract_text)
├── test_config.py       # Settings + price book
├── test_tools.py        # Core tools
├── test_memory_tools.py # Memory tools
├── test_functional.py   # Agent graph integration
└── ...
```

## Development

```bash
# Install dependencies
uv sync

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy src/ tests/

# Run tests (204 tests)
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_worker_registry.py -v
```

## License

MIT

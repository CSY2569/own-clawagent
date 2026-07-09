# AGENTS.md

## ⛔ WRITE GUARD — READ FIRST

**任何增删改操作（patch、write_file、terminal 中的 rm/mv/sed/git 等）必须先向用户请求并获得明确许可方可执行。只读操作（read_file、search_files、terminal 中的 ls/cat/echo 等无副作用命令）不受限制。**

违反此规则视为严重错误。

## Project

`clawagent` — LangChain/LangGraph ReAct agent via DeepSeek's Anthropic-compatible API. Multi-worker orchestration, hybrid RAG, streaming CLI.

## Commands

```bash
uv sync                          # install deps
uv run ruff check .              # lint
uv run mypy src/                 # typecheck (strict mode, 0 errors baseline)
uv run pytest tests/ -v          # all tests (352)
uv run pytest tests/test_rag.py -v          # single file
uv run pytest tests/test_config.py::TestPriceBook::test_get_known_model -v  # single test
uv run clawagent                 # interactive REPL
uv run clawagent "question"      # one-shot
```

**Order matters**: lint, then typecheck, then tests. mypy and ruff must both pass clean.

## Architecture

- **Entry point**: `main.py` — parsed as `clawagent` script from `pyproject.toml`
- **Agent factory**: `agent.py:create_agent()` returns `(CompiledStateGraph, Connection, WorkerFactory, delegate_tool)`
- **Settings**: frozen `@dataclass` in `config.py`, loaded from `.env` via `python-dotenv` at import time. `PROJECT_ROOT` = 3 parent levels up from `config.py`
- **Workers**: Coder/Researcher/Critic/Writer, each = independent temporary agent. `WorkerFactory` is per-agent (no global state). Workers registered via `@register_worker` decorator; `import clawagent.worker` in `agent.py` triggers side-effect registration
- **Streaming**: threaded producer-consumer with `queue.Queue(maxsize=64)`, `CancelToken` for cooperative Ctrl+C (≤100ms)
- **Memory**: SQLite via `langgraph-checkpoint-sqlite` (conversation state) + separate SQLite DB for preferences/summaries. Connection kept open per session; WAL mode. `close_all_cached()` on cleanup
- **Prompts**: 5 layers assembled by `PromptBuilder`. Files in `prompts/agents/<id>/identity.md` (required), `soul.md` (optional). Tool listing auto-generated from `ALL_TOOLS + delegate_task`

## Critical Gotchas

### Provider packages are NOT auto-installed

The project uses `langchain.chat_models.init_chat_model` which supports any provider at runtime, but each provider requires its own `langchain-*` package. Currently installed:
- `langchain-anthropic` — for `anthropic` provider
- `langchain-openai` — for `openai` provider (Researcher/Critic Workers use SiliconFlow)

If you add a Worker with `model_provider=ollama`, you must add `langchain-ollama` to `pyproject.toml`. Same for `bedrock`, `groq`, etc. The error is `ImportError: Initializing ChatXxx requires the langchain-xxx package`.

### Multi-platform support

The project supports 5 platforms via `platforms.py` presets: `deepseek` (default), `ark`, `opencode-go`, `openai`, `anthropic`. Switch at runtime with `/platform <name>` or set `CLAWAGENT_PLATFORM` in `.env`. Each platform preset bundles `model_provider`, `api_base`, `api_key_env`, and `fallback_models`. Use `/models` to list available models from the current platform (fetched from the platform's `/models` endpoint, with fallback to the preset list). `/model` without arguments shows a numbered selection dialog.

### Worker env var discovery is automatic

`load_worker_configs()` discovers roles by scanning for `WORKER_*_MODEL` env vars. If none are set, it falls back to `BUILTIN_WORKER_ROLES`. To add a new Worker role without env vars, register the class with `@register_worker` and add it to `BUILTIN_WORKER_ROLES`.

### Slash commands are defined in cli/commands.py

New REPL commands follow the `SLASH_COMMANDS` list pattern. Autocomplete works via `_SlashCommandCompleter` in `main.py` that reads this list.

### RAG BM25 background indexing

`bootstrap_rag()` starts BM25 in a background thread. The `bm25_ready_signal` list signals completion. While building, searches fall back to KNN-only. BM25 index is cached as pickle with SHA256 validation.

### Settings is frozen (immutable)

`Settings` is `@dataclass(frozen=True)`. To change a setting, create a new instance. The `handle_command` function in `cli/commands.py` does this for `/model`, `/temp`, etc.

### Prompt files are not validated

If `prompts/agents/<id>/identity.md` is missing, `PromptBuilder` generates a generic fallback. No error is raised. Missing `soul.md` or `shared/*` files are silently skipped.

### price.txt vs price.toml

`load_price_book()` prefers `price.toml` (TOML). Falls back to `price.txt` (legacy regex parsing). If both are missing, returns empty `PriceBook` (cost display shows `--`).

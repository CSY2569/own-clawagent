# clawagent

[English](README.md) | [中文](README.zh.md)

A LangChain/LangGraph tool-calling agent powered by Anthropic Claude-compatible models (DeepSeek via Anthropic API).

## Features

- **Tool-calling agent** — 8 built-in tools: file I/O, shell commands, time query, greeting, and conversation memory (list/recall/summarize)
- **Interactive REPL** — Rich-powered dashboard with live token tracking, cost estimation, and hot-reload settings
- **Multi-layer system prompt** — File-based prompt assembly with per-agent identity, personality, workspace context, runtime metadata, and auto-generated tool listing
- **Conversation memory** — SQLite-backed session persistence with automatic summarization and user preference extraction
- **Settings hot-reload** — Switch model, temperature, and max tokens without restarting the REPL
- **CJK-safe input** — prompt_toolkit-based input with correct CJK character width handling

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

The REPL supports slash commands:

| Command | Description |
|---------|-------------|
| `/sessions` | List all historical sessions |
| `/load <id>` | Load a session by ID |
| `/new` | Start a new session |
| `/model <name>` | Switch model (e.g. `deepseek-v4-pro`) |
| `/temp <n>` | Set temperature (e.g. `0.7`) |
| `/max-tokens <n>` | Set max output tokens (e.g. `8192`) |
| `/settings` | Show current configuration |
| `/help` | Show help |
| `quit` / `exit` / `q` | Exit |

Typing `/` shows an autocomplete menu with all available commands.

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | API key |
| `ANTHROPIC_BASE_URL` | — | API base URL (DeepSeek: `https://api.deepseek.com/anthropic`) |
| `CLAWAGENT_MODEL` | `deepseek-v4-flash` | Model name |
| `CLAWAGENT_AGENT_ID` | `pickle` | Agent identity (reads `prompts/agents/<id>/`) |
| `CLAWAGENT_CONTEXT_WINDOW` | `1000000` | Context window size (for display) |
| `CLAWAGENT_MEMORY_DB` | `memories/sessions.db` | Path to SQLite memory database |
| `CLAWAGENT_MAX_PREFERENCES` | `5` | Max user preferences injected into prompt |

## Multi-Layer Prompt System

The system prompt is assembled by `PromptBuilder` from five layers:

| Layer | Source | Required |
|-------|--------|----------|
| 1. Identity | `prompts/agents/{id}/identity.md` | Yes (fallback if missing) |
| 2. Personality | `prompts/agents/{id}/soul.md` | No |
| 3. Workspace | `prompts/shared/bootstrap.md`, `agents.md` | No |
| 4. Runtime | Agent ID, timestamp, channel (auto-generated) | Yes |
| 5. Preferences | SQLite `preferences` table (auto-learned) | No |

Tools are auto-listed from `ALL_TOOLS` — add a new `@tool` to `src/clawagent/tools/` and it appears automatically.

To add a new agent: create `prompts/agents/<name>/identity.md` and set `CLAWAGENT_AGENT_ID=<name>`.

## Project Structure

```
src/clawagent/
├── config.py          # Settings dataclass, price book
├── agent.py           # Agent factory, run/stream/reconfigure
├── prompt_builder.py  # Multi-layer prompt assembly
├── main.py            # CLI entry point, REPL loop
├── ui.py              # Rich dashboard, stats, formatting
├── tools/
│   ├── __init__.py    # Core tools (read, write, shell, time, greet)
│   └── memory_tools.py # Memory tools (list/recall/summarize)
└── memory/
    ├── summarizer.py  # Session summarization & message persistence
    └── preferences.py # User preference extraction & querying

prompts/
├── agents/pickle/     # Default agent prompt files
│   ├── identity.md    # Agent identity
│   └── soul.md        # Personality and tone
├── shared/            # Shared workspace context
│   ├── bootstrap.md
│   └── agents.md
└── README.md
```

## Design Documents

Additional design documentation exists locally (not committed):

- `多层提示词设计.md` — Multi-layer prompt architecture design
- `记忆系统实施文档.md` — Memory system implementation document
- `记忆训练强化.txt` — Memory training notes

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

# Run tests
uv run pytest tests/ -v
```

## License

MIT

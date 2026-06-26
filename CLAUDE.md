# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`clawagent` — a LangChain/LangGraph tool-calling agent powered by Anthropic Claude (via `langchain-anthropic`).

## Architecture

- `src/clawagent/config.py` — Settings dataclass, automatic `.env` loading
- `src/clawagent/agent.py` — Agent factory using `create_react_agent` from LangGraph, SqliteSaver checkpointer
- `src/clawagent/tools/__init__.py` — Tool definitions using `@tool` from `langchain_core.tools`
- `src/clawagent/tools/memory_tools.py` — Memory tools: list_sessions, recall_session, summarize_session
- `src/clawagent/memory/summarizer.py` — Conversation summarization and message persistence (SQLite)
- `src/clawagent/memory/preferences.py` — User preference extraction and querying (SQLite)
- `src/clawagent/main.py` — CLI entry point (one-shot + interactive REPL with /sessions, /load, /new)

## Commands

```bash
# Install dependencies
uv sync

# Lint & type check
uv run ruff check .
uv run mypy src/ tests/

# Run all tests
uv run pytest tests/ -v

# Run single test file
uv run pytest tests/test_config.py -v

# Run single test
uv run pytest tests/test_config.py::TestPriceBook::test_get_known_model -v

# Run the agent (requires ANTHROPIC_API_KEY in .env)
uv run clawagent "Your question"
uv run clawagent          # interactive mode
```

## CodeGraph

This project has CodeGraph MCP configured. Run `codegraph init -i` to build the index for fast symbol navigation.

## Conventions

- Python 3.14+, `uv` for package management
- `src/` layout (package at `src/clawagent/`)
- Ruff for formatting and linting (100-char lines, double quotes)
- Mypy in strict mode
- Environment variables via `.env` (copy `.env.example`)
- LangChain tools use `@tool` from `langchain_core.tools`
- Agent orchestration via `langgraph.prebuilt.create_react_agent`

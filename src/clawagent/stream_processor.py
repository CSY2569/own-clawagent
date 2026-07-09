"""Streaming event processor — converts raw LangGraph message chunks into StreamEvents."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from clawagent.stream_events import StreamEvent
from clawagent.types import Usage


@dataclass
class StreamState:
    """Mutable state shared across stream event processors."""

    all_text: list[str] = field(default_factory=list)
    tool_call_accum: dict[str, dict[str, str]] = field(default_factory=dict)
    tool_calls_emitted: set[str] = field(default_factory=set)
    in_thinking: bool = False
    usage: Usage = field(default_factory=Usage)


def extract_usage(msg: Any) -> Usage:
    """Extract Usage from message, checking usage_metadata first (ChatAnthropic)."""
    usage_dict = getattr(msg, "usage_metadata", None)
    if usage_dict:
        return Usage(
            input_tokens=usage_dict.get("input_tokens", 0),
            output_tokens=usage_dict.get("output_tokens", 0),
            cache_read_input_tokens=usage_dict.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=usage_dict.get("cache_creation_input_tokens", 0),
            prompt_cache_hit_tokens=usage_dict.get("prompt_cache_hit_tokens", 0),
            prompt_cache_miss_tokens=usage_dict.get("prompt_cache_miss_tokens", 0),
        )
    meta = getattr(msg, "response_metadata", None) or {}
    return Usage.from_response_metadata(meta)


def process_text_chunk(chunk_text: object, state: StreamState) -> Iterator[StreamEvent]:
    """Process a text chunk from the agent/model node."""
    if isinstance(chunk_text, list):
        for block in chunk_text:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            text = block.get("text", "") or block.get("thinking", "")
            if btype == "thinking" and text:
                if not state.in_thinking:
                    state.in_thinking = True
                    yield StreamEvent(kind="think_start")
            elif btype == "text" and text:
                if state.in_thinking:
                    state.in_thinking = False
                    yield StreamEvent(kind="think_end")
                state.all_text.append(text)
                yield StreamEvent(kind="token", node="agent", content=text)
    elif isinstance(chunk_text, str) and chunk_text:
        state.all_text.append(chunk_text)
        yield StreamEvent(kind="token", node="agent", content=chunk_text)


def process_tool_call_chunks(msg_chunk: Any, state: StreamState) -> None:
    """Accumulate incremental tool call name/args chunks.

    Handles both Anthropic and OpenAI streaming formats. OpenAI may
    send the first chunk without an ``id``, using ``index`` instead.
    We fall back to ``index`` as a synthetic ID to avoid dropping
    the initial name/args fragments.
    """
    tcc_list = getattr(msg_chunk, "tool_call_chunks", None)
    if not tcc_list:
        return
    state.all_text.clear()
    for tcc in tcc_list:
        tc_id = tcc.get("id", "") or str(tcc.get("index", ""))
        if not tc_id or tc_id == "None":
            continue
        if tc_id not in state.tool_call_accum:
            state.tool_call_accum[tc_id] = {"name": "", "args": ""}
        name_piece = tcc.get("name", "") or ""
        args_piece = tcc.get("args", "") or ""
        if name_piece:
            state.tool_call_accum[tc_id]["name"] += name_piece
        if args_piece:
            state.tool_call_accum[tc_id]["args"] += args_piece


def emit_tool_events(msg_chunk: Any, state: StreamState) -> Iterator[StreamEvent]:
    """Emit tool_call and tool_result events for a tools-node message."""
    if state.in_thinking:
        state.in_thinking = False
        yield StreamEvent(kind="think_end")

    for tc_id, acc in state.tool_call_accum.items():
        if tc_id in state.tool_calls_emitted:
            continue
        name = acc["name"]
        if not name:
            continue
        try:
            args = json.loads(acc["args"]) if acc["args"] else {}
        except json.JSONDecodeError, ValueError:
            args = {"raw": acc["args"]}
        state.tool_calls_emitted.add(tc_id)
        yield StreamEvent(kind="tool_call", node="agent", content=name, metadata={"args": args})

    preview = str(getattr(msg_chunk, "content", ""))[:200]
    yield StreamEvent(
        kind="tool_result",
        node="tools",
        content=getattr(msg_chunk, "name", ""),
        metadata={"preview": preview},
    )

"""Streaming producer-consumer — threaded event pipeline for REPL."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from clawagent.agent import Agent, Usage
from clawagent.cancel_token import CancelToken
from clawagent.stream_events import StreamEvent
from clawagent.ui_stream import stream_display


@dataclass
class StreamResult:
    """Result of one streaming round."""

    response_text: str
    usage: Usage


def run_streaming_round(
    agent: Agent,
    user_input: str,
) -> StreamResult:
    """Execute one round of streaming agent interaction.

    Spawns a background producer thread, drains the event queue through
    the stream display, and collects the response text and token usage.

    KeyboardInterrupt is forwarded to the producer thread for cooperative
    cancellation.

    Returns:
        ``StreamResult`` with the full response text and cumulative usage.
    """
    response_text = ""
    round_usage = Usage()
    event_queue: queue.Queue[StreamEvent] = queue.Queue(maxsize=64)
    cancel_event = threading.Event()

    def _produce() -> None:
        try:
            for event in agent.stream_events(user_input):
                if cancel_event.is_set():
                    return
                event_queue.put(event, timeout=0.5)
        except queue.Full:
            pass
        except Exception:
            pass

    worker = threading.Thread(target=_produce, daemon=True)
    worker.start()

    with CancelToken() as cancel, stream_display() as display:
        while True:
            try:
                event = event_queue.get(timeout=0.1)
            except queue.Empty:
                cancel.check()
                if not worker.is_alive() and event_queue.empty():
                    break
                continue
            cancel.check()
            display.handle(event)
            if event.kind == "done":
                response_text = event.content
                round_usage = Usage(
                    input_tokens=event.metadata.get("input_tokens", 0),
                    output_tokens=event.metadata.get("output_tokens", 0),
                    cache_read_input_tokens=event.metadata.get("cache_read_input_tokens", 0),
                    cache_creation_input_tokens=event.metadata.get("cache_creation_input_tokens", 0),
                    prompt_cache_hit_tokens=event.metadata.get("prompt_cache_hit_tokens", 0),
                    prompt_cache_miss_tokens=event.metadata.get("prompt_cache_miss_tokens", 0),
                )
                break

    worker.join(timeout=2.0)
    return StreamResult(response_text=response_text, usage=round_usage)

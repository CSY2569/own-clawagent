"""CLI dashboard for the clawagent interactive REPL."""

import time
from dataclasses import dataclass

from rich.console import Console

from clawagent.agent import Usage
from clawagent.config import PriceConfig, Settings

# ── Conversation Stats ─────────────────────────────────────────────────


@dataclass
class ConversationStats:
    """Cumulative conversation statistics across all turns."""

    cumulative_input_tokens: int = 0
    cumulative_output_tokens: int = 0
    cumulative_cache_read_tokens: int = 0
    cumulative_cache_creation_tokens: int = 0
    latest_input_tokens: int = 0
    message_count: int = 0
    start_time: float = 0.0

    def update(self, usage: Usage) -> None:
        """Accumulate a single invocation's usage into the stats."""
        self.cumulative_input_tokens += usage.input_tokens
        self.cumulative_output_tokens += usage.output_tokens
        self.cumulative_cache_read_tokens += usage.cache_read_input_tokens
        self.cumulative_cache_creation_tokens += usage.cache_creation_input_tokens
        self.latest_input_tokens = usage.input_tokens
        self.message_count += 1

    def reset(self) -> None:
        """Reset all cumulative stats for a new session."""
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0
        self.cumulative_cache_read_tokens = 0
        self.cumulative_cache_creation_tokens = 0
        self.latest_input_tokens = 0
        self.message_count = 0
        self.start_time = time.monotonic()

    @property
    def total_tokens(self) -> int:
        return self.cumulative_input_tokens + self.cumulative_output_tokens

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time == 0.0:
            return 0.0
        return time.monotonic() - self.start_time

    def context_usage_pct(self, context_window: int) -> float:
        """Context window usage based on the latest call's input tokens."""
        if context_window <= 0:
            return 0.0
        return (self.latest_input_tokens / context_window) * 100

    def cost(self, pricing: PriceConfig) -> float:
        """Total cost in CNY based on cumulative tokens."""
        return (
            (self.cumulative_input_tokens / 1_000_000) * pricing.input_per_1m
            + (self.cumulative_cache_read_tokens / 1_000_000) * pricing.cache_hit_per_1m
            + (self.cumulative_output_tokens / 1_000_000) * pricing.output_per_1m
        )


# ── Helper formatters ──────────────────────────────────────────────────


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs:02d}s"


def _format_tokens(n: int) -> str:
    """Format a token count to a human-readable string."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}K"
    return f"{n / 1_000_000:.1f}M"


def _format_cost(cny: float) -> str:
    """Format a cost in CNY."""
    if cny < 0.01:
        return "¥0.00"
    return f"¥{cny:.4f}".rstrip("0").rstrip(".")


def _context_color(pct: float) -> str:
    """Return Rich color name based on context usage percentage."""
    if pct < 70:
        return "green"
    if pct < 90:
        return "yellow"
    return "red"


# ── Splash screen ──────────────────────────────────────────────────────


def render_splash(
    settings: Settings,
    pricing: PriceConfig,
    console: Console,
) -> None:
    """Render a compact startup banner."""
    console.print()
    console.print(
        f"  [bold green]clawagent[/bold green]  "
        f"[dim]{settings.model_name}[/dim]  "
        f"[dim]t={settings.temperature}[/dim]  "
        f"[dim]ctx={_format_tokens(settings.context_window)}[/dim]"
    )
    console.print(f"  [dim]{_format_cost(0)} · 0 msg · 0s[/dim]")
    console.print()


# ── Status line ────────────────────────────────────────────────────────


def render_status_line(
    stats: ConversationStats,
    pricing: PriceConfig,
    settings: Settings,
    console: Console,
) -> None:
    """Render a compact right-aligned status line."""
    ctx_pct = stats.context_usage_pct(settings.context_window)
    cost = stats.cost(pricing)

    line = (
        f"[bold]R{stats.message_count + 1}[/bold]  "
        f"[dim]In {_format_tokens(stats.cumulative_input_tokens)}  "
        f"Out {_format_tokens(stats.cumulative_output_tokens)}  "
        f"Ctx [{_context_color(ctx_pct)}]{ctx_pct:.1f}%[/]  "
        f"{_format_duration(stats.elapsed_seconds)}  "
        f"{_format_cost(cost)}[/dim]"
    )
    console.print(line, justify="right")

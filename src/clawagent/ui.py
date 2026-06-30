"""CLI dashboard for the clawagent interactive REPL."""

import time
from dataclasses import dataclass
from typing import Any

import pyfiglet
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
    pool_stats: dict[str, dict[str, Any]] | None = None,
    worker_roles: list[str] | None = None,
) -> None:
    """Render a startup banner with pyfiglet logo, info table, and welcome message."""
    console.print()

    # ── Left column: ASCII logo ─────────────────────────────────
    try:
        logo_text = pyfiglet.Figlet(font="slant", width=30).renderText("clawagent")
    except Exception:
        logo_text = "  clawagent"
    left = Text(logo_text, style="bold green")

    # ── Right column: info table ────────────────────────────────
    table = Table(box=None, padding=(0, 2), show_header=False, show_edge=False)
    table.add_column(style="dim", justify="right")
    table.add_column(style="bold")
    info_rows = [
        ("Agent:", settings.agent_id),
        ("Model:", settings.model_name),
        ("Provider:", settings.model_provider or "auto"),
        ("Context:", _format_tokens(settings.context_window)),
        ("Max Tokens:", str(settings.max_tokens)),
        ("Temperature:", str(settings.temperature)),
        ("Compression:", settings.compression_strategy),
        ("SubAgents:", str(len(worker_roles or []))),
        ("RAG:", "enabled" if settings.siliconflow_api_key else "disabled"),
    ]
    for label, value in info_rows:
        table.add_row(label, value)

    # ── Assemble panel ──────────────────────────────────────────
    panel = Panel(
        Columns([left, table], padding=(0, 4), expand=False),
        title="[bold]OWN ClawAgent[/bold]",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)

    # ── Welcome message ─────────────────────────────────────────
    console.print("  欢迎使用 [bold green]Own ClawAgent[/bold green]，输入消息开始对话，输入 [bold]/help[/bold] 查看可用命令")
    console.print("  [dim]✦ Tip: /model 切换模型 · /compress 调整压缩策略 · /sessions 查看历史会话[/dim]")

    # ── Pool status ─────────────────────────────────────────────
    if pool_stats:
        for name, s in pool_stats.items():
            active = s.get("active", 0)
            total = s.get("total", 0)
            console.print(f"  [dim]API Pool '{name}': {active}/{total} active[/dim]")


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

    # Visual context bar
    bar_width = 10
    filled = int(ctx_pct / 100 * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    line = (
        f"[bold]R{stats.message_count + 1}[/bold]  "
        f"[dim]In {_format_tokens(stats.cumulative_input_tokens)}  "
        f"Out {_format_tokens(stats.cumulative_output_tokens)}  "
        f"Ctx [{_context_color(ctx_pct)}]{bar} {ctx_pct:.1f}%[/]  "
        f"{_format_duration(stats.elapsed_seconds)}  "
        f"{_format_cost(cost)}[/dim]"
    )
    console.print(line, justify="right")

"""Slash command handlers and registry dispatch."""

from dataclasses import replace
from typing import Any

from rich.console import Console
from rich.table import Table

from clawagent.cli.display import load_session, new_session, rag_search, show_sessions
from clawagent.cli.session import AgentRef
from clawagent.config import PriceConfig, Settings, load_price_book
from clawagent.conversation_log import ConversationLogger
from clawagent.ui import ConversationStats

type _CmdResult = tuple[Settings, PriceConfig] | None


def _cmd_sessions(cmd: str, agent_ref: AgentRef, settings: Settings,
                  console: Console, stats: ConversationStats, pricing: PriceConfig,
                  logger: ConversationLogger) -> _CmdResult:
    show_sessions(agent_ref.agent, settings, console)
    return None


def _cmd_load(cmd: str, agent_ref: AgentRef, settings: Settings,
              console: Console, stats: ConversationStats, pricing: PriceConfig,
              logger: ConversationLogger) -> _CmdResult:
    load_session(cmd[6:].strip(), agent_ref.agent, settings, console)
    return None


def _cmd_new(cmd: str, agent_ref: AgentRef, settings: Settings,
             console: Console, stats: ConversationStats, pricing: PriceConfig,
             logger: ConversationLogger) -> _CmdResult:
    new_session(agent_ref, settings, console, stats, logger)
    return None


def _cmd_model(cmd: str, agent_ref: AgentRef, settings: Settings,
               console: Console, stats: ConversationStats, pricing: PriceConfig,
               logger: ConversationLogger) -> _CmdResult:
    agent = agent_ref.agent
    model_name = cmd[7:].strip()
    try:
        new_settings = replace(settings, model_name=model_name)
        agent.reconfigure(new_settings)
        new_pricing = load_price_book().get(model_name)
        logger.log_settings_change(agent.thread_id, "model_name", settings.model_name, model_name)
        console.print(f"[green]模型已切换至: {model_name}[/green]")
        return new_settings, new_pricing
    except Exception as e:
        console.print(f"[red]无效模型名 '{model_name}'：{e}[/red]")
        return settings, pricing


def _cmd_temp(cmd: str, agent_ref: AgentRef, settings: Settings,
              console: Console, stats: ConversationStats, pricing: PriceConfig,
              logger: ConversationLogger) -> _CmdResult:
    agent = agent_ref.agent
    try:
        temp = float(cmd[6:].strip())
    except ValueError:
        console.print("[red]无效的温度值，请输入数字如 0.7[/red]")
        return settings, pricing
    new_settings = replace(settings, temperature=temp)
    agent.reconfigure(new_settings)
    logger.log_settings_change(agent.thread_id, "temperature", settings.temperature, temp)
    console.print(f"[green]温度已设置为: {temp}[/green]")
    return new_settings, pricing


def _cmd_max_tokens(cmd: str, agent_ref: AgentRef, settings: Settings,
                    console: Console, stats: ConversationStats, pricing: PriceConfig,
                    logger: ConversationLogger) -> _CmdResult:
    agent = agent_ref.agent
    try:
        max_tok = int(cmd[12:].strip())
    except ValueError:
        console.print("[red]无效的 token 数，请输入整数如 8192[/red]")
        return settings, pricing
    new_settings = replace(settings, max_tokens=max_tok)
    agent.reconfigure(new_settings)
    logger.log_settings_change(agent.thread_id, "max_tokens", settings.max_tokens, max_tok)
    console.print(f"[green]最大输出 token 数已设置为: {max_tok}[/green]")
    return new_settings, pricing


def _cmd_settings(cmd: str, agent_ref: AgentRef, settings: Settings,
                  console: Console, stats: ConversationStats, pricing: PriceConfig,
                  logger: ConversationLogger) -> _CmdResult:
    from clawagent.ui import _format_tokens

    console.print(
        f"[bold]Model[/] {settings.model_name}  "
        f"[bold]T[/] {settings.temperature}  "
        f"[bold]Tok[/] {settings.max_tokens}  "
        f"[bold]Ctx[/] {_format_tokens(settings.context_window)}  "
        f"[bold]Compress[/] {settings.compression_strategy}"
    )
    return None


def _cmd_compress(cmd: str, agent_ref: AgentRef, settings: Settings,
                  console: Console, stats: ConversationStats, pricing: PriceConfig,
                  logger: ConversationLogger) -> _CmdResult:
    agent = agent_ref.agent
    strategy = cmd[10:].strip()
    valid = {"trim", "token_trim", "summarize"}
    if strategy not in valid:
        console.print(f"[yellow]无效策略: {strategy}。可选: {', '.join(sorted(valid))}[/yellow]")
        return settings, pricing
    new_settings = replace(settings, compression_strategy=strategy)
    agent.reconfigure(new_settings)
    logger.log_settings_change(agent.thread_id, "compression_strategy", settings.compression_strategy, strategy)
    console.print(f"[green]压缩策略已切换至: {strategy}[/green]")
    return new_settings, pricing


def _cmd_rag_search(cmd: str, agent_ref: AgentRef, settings: Settings,
                    console: Console, stats: ConversationStats, pricing: PriceConfig,
                    logger: ConversationLogger) -> _CmdResult:
    query = cmd[12:].strip()
    if not query:
        console.print("[yellow]用法: /rag-search <关键词>[/yellow]")
    else:
        rag_search(query, console)
    return None


def _cmd_help(cmd: str, agent_ref: AgentRef, settings: Settings,
              console: Console, stats: ConversationStats, pricing: PriceConfig,
              logger: ConversationLogger) -> _CmdResult:
    from clawagent.cli import SLASH_COMMANDS

    table = Table(box=None, padding=(0, 2))
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="dim")
    for cmd_name, desc in SLASH_COMMANDS:
        table.add_row(cmd_name, desc)
    table.add_row("quit / q", "退出")
    console.print(table)
    return None


# ── Registry ───────────────────────────────────────────────────────
_CMD_EXACT: dict[str, Any] = {
    "/sessions": _cmd_sessions,
    "/list": _cmd_sessions,
    "/new": _cmd_new,
    "/settings": _cmd_settings,
    "/help": _cmd_help,
}
_CMD_PREFIX: list[tuple[str, Any]] = [
    ("/load ", _cmd_load),
    ("/model ", _cmd_model),
    ("/temp ", _cmd_temp),
    ("/max-tokens ", _cmd_max_tokens),
    ("/compress ", _cmd_compress),
    ("/rag-search ", _cmd_rag_search),
]


def handle_command(
    cmd: str,
    agent_ref: AgentRef,
    settings: Settings,
    console: Console,
    stats: ConversationStats,
    pricing: PriceConfig,
    logger: ConversationLogger,
) -> tuple[Settings, PriceConfig]:
    """Handle slash commands in interactive mode via registry dispatch."""
    cmd_lower = cmd.lower()

    # 1. Exact match
    handler = _CMD_EXACT.get(cmd_lower)
    if handler is not None:
        result = handler(cmd_lower, agent_ref, settings, console, stats, pricing, logger)
        return result if result is not None else (settings, pricing)

    # 2. Prefix match
    for prefix, handler in _CMD_PREFIX:
        if cmd_lower.startswith(prefix):
            result = handler(cmd_lower, agent_ref, settings, console, stats, pricing, logger)
            return result if result is not None else (settings, pricing)

    # 3. Unknown
    console.print(f"[yellow]未知命令: {cmd}。输入 /help 查看可用命令。[/yellow]")
    return settings, pricing

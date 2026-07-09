"""Slash command handlers and registry dispatch."""

from dataclasses import replace
from typing import Any

from rich.console import Console
from rich.table import Table

from clawagent.cli.display import load_session, new_session, rag_search, show_sessions
from clawagent.cli.session import AgentRef
from clawagent.config import PriceConfig, Settings, load_price_book
from clawagent.conversation_log import ConversationLogger
from clawagent.model_discovery import ModelInfo, fetch_models, invalidate_cache
from clawagent.platforms import PLATFORMS
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


def _select_model_dialog(
    models: list[ModelInfo], platform_label: str, console: Console, current_model: str = ""
) -> ModelInfo | None:
    """Show a numbered model list and let the user pick by number."""
    if not models:
        return None

    table = Table(box=None, padding=(0, 2), title=f"选择模型 - {platform_label} ({len(models)})")
    table.add_column("#", style="dim", no_wrap=True)
    table.add_column("Model ID", style="cyan")
    table.add_column("Owner", style="dim")
    for i, m in enumerate(models, 1):
        marker = " ←" if m.id == current_model else ""
        table.add_row(str(i), f"{m.id}{marker}", m.owned_by or "-")
    console.print(table)

    try:
        choice = console.input("[bold]输入编号选择 (q 取消):[/] ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if choice.lower() == "q" or not choice:
        return None
    try:
        idx = int(choice) - 1
    except ValueError:
        return None
    if 0 <= idx < len(models):
        return models[idx]
    return None


def _cmd_models(cmd: str, agent_ref: AgentRef, settings: Settings,
                console: Console, stats: ConversationStats, pricing: PriceConfig,
                logger: ConversationLogger) -> _CmdResult:
    """List available models from the current platform: /models [refresh]"""
    arg = cmd[8:].strip() if len(cmd) > 8 else ""
    if arg == "refresh":
        invalidate_cache(settings.platform or settings.model_provider)
        console.print("[dim]缓存已清除，重新拉取…[/dim]")

    models = fetch_models(settings)
    if not models:
        console.print(
            "[yellow]未能获取模型列表。请检查 API Key 和平台配置，"
            "或用 /model <name> 手动指定。[/yellow]"
        )
        return None

    preset = PLATFORMS.get(settings.platform)
    plat_label = preset.display_name if preset else (settings.platform or settings.model_provider)

    table = Table(box=None, padding=(0, 2), title=f"{plat_label} 可用模型 ({len(models)})")
    table.add_column("#", style="dim", no_wrap=True)
    table.add_column("Model ID", style="cyan")
    table.add_column("Owner", style="dim")
    table.add_column("Context", style="dim")
    for i, m in enumerate(models, 1):
        ctx = f"{m.context_length:,}" if m.context_length else "—"
        marker = " ←" if m.id == settings.model_name else ""
        table.add_row(str(i), f"{m.id}{marker}", m.owned_by or "—", ctx)
    console.print(table)
    console.print("[dim]用 /model <id> 或 /model（无参数）交互选择[/dim]")
    return None


def _cmd_model(cmd: str, agent_ref: AgentRef, settings: Settings,
               console: Console, stats: ConversationStats, pricing: PriceConfig,
               logger: ConversationLogger) -> _CmdResult:
    """Switch model: /model [platform:]model_name  or  /model (interactive)"""
    arg = cmd[7:].strip()

    if not arg:
        models = fetch_models(settings)
        preset = PLATFORMS.get(settings.platform)
        plat_label = preset.display_name if preset else (settings.platform or "default")
        selected = _select_model_dialog(models, plat_label, console, settings.model_name)
        if selected is None:
            console.print("[yellow]未选择模型，保持当前设置。[/yellow]")
            return settings, pricing
        arg = selected.id

    agent = agent_ref.agent
    platform = settings.platform
    model_name = arg

    if ":" in arg:
        parts = arg.split(":", 1)
        candidate = parts[0].strip()
        if candidate in PLATFORMS:
            platform = candidate
            model_name = parts[1].strip()

    try:
        new_settings = replace(settings, model_name=model_name, platform=platform)
        agent.reconfigure(new_settings)
        new_pricing = load_price_book().get(model_name)
        preset = PLATFORMS.get(platform)
        plat_label = preset.display_name if preset else platform or "default"
        logger.log_settings_change(
            agent.thread_id, "model", f"{settings.platform}:{settings.model_name}",
            f"{platform}:{model_name}",
        )
        console.print(f"[green]模型已切换至: {plat_label} / {model_name}[/green]")
        return new_settings, new_pricing
    except Exception as e:
        console.print(f"[red]无效模型 '{arg}'：{e}[/red]")
        return settings, pricing


def _cmd_platform(cmd: str, agent_ref: AgentRef, settings: Settings,
                  console: Console, stats: ConversationStats, pricing: PriceConfig,
                  logger: ConversationLogger) -> _CmdResult:
    """Switch platform: /platform [name]  —  interactive model dialog after switch"""
    platform = cmd[10:].strip()
    if not platform:
        table = Table(box=None, padding=(0, 2), title="可用平台")
        table.add_column("Platform", style="cyan")
        table.add_column("Provider")
        table.add_column("API Base", style="dim")
        table.add_column("Key Env")
        for name, preset in PLATFORMS.items():
            marker = " ← current" if name == settings.platform else ""
            table.add_row(
                f"{name}{marker}",
                preset.model_provider,
                preset.api_base or "(default)",
                preset.api_key_env,
            )
        console.print(table)
        return None

    if platform not in PLATFORMS:
        console.print(
            f"[red]未知平台 '{platform}'。可选: {', '.join(PLATFORMS.keys())}[/red]"
        )
        return settings, pricing

    agent = agent_ref.agent
    preset = PLATFORMS[platform]
    try:
        new_settings = replace(
            settings,
            platform=platform,
            model_provider=preset.model_provider,
            api_base=preset.api_base,
        )
        agent.reconfigure(new_settings)
        logger.log_settings_change(
            agent.thread_id, "platform", settings.platform or "(none)", platform,
        )
        console.print(
            f"[green]平台已切换至: {preset.display_name} "
            f"({preset.api_base or 'default endpoint'})[/green]"
        )
    except Exception as e:
        console.print(f"[red]切换平台失败: {e}[/red]")
        return settings, pricing

    models = fetch_models(new_settings)
    if not models:
        console.print(
            "[yellow]模型列表获取失败。用 /model <name> 手动指定，"
            "或检查 API Key 配置。[/yellow]"
        )
        return new_settings, pricing

    selected = _select_model_dialog(models, preset.display_name, console, new_settings.model_name)
    if selected is None:
        console.print("[dim]未选择模型，保持当前模型。用 /model 重新选择。[/dim]")
        return new_settings, pricing

    try:
        final_settings = replace(new_settings, model_name=selected.id)
        agent.reconfigure(final_settings)
        new_pricing = load_price_book().get(selected.id)
        logger.log_settings_change(
            agent.thread_id, "model", settings.model_name, selected.id,
        )
        console.print(f"[green]模型已切换至: {preset.display_name} / {selected.id}[/green]")
        return final_settings, new_pricing
    except Exception as e:
        console.print(f"[red]模型切换失败: {e}[/red]")
        return new_settings, pricing


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

    platform_label = settings.platform or settings.model_provider
    console.print(
        f"[bold]Platform[/] {platform_label}  "
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
    "/model": _cmd_model,
    "/models": _cmd_models,
    "/platform": _cmd_platform,
}
_CMD_PREFIX: list[tuple[str, Any]] = [
    ("/load ", _cmd_load),
    ("/model ", _cmd_model),
    ("/models ", _cmd_models),
    ("/platform ", _cmd_platform),
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

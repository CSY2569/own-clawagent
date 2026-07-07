"""Entry point for the clawagent CLI.

Usage:
    uv run clawagent "Your question here"   # one-shot
    uv run clawagent                         # interactive REPL
    uv run clawagent gateway                 # multi-platform Gateway server
"""

from __future__ import annotations

import sys


def _setup_gateway_logging() -> None:
    """Configure logging: console INFO+ for operators, file DEBUG for troubleshooting."""
    import logging
    import logging.handlers
    from pathlib import Path

    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # File handler — DEBUG and above, rotated
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "gateway.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root = logging.getLogger("gateway")
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


def _run_gateway() -> None:
    """Start the multi-platform Gateway server."""
    import asyncio

    from clawagent.config import Settings
    from clawagent.gateway.channel import IChannel
    from clawagent.gateway.channels.cli_channel import CliChannel
    from clawagent.gateway.config import GatewayConfig
    from clawagent.gateway.server import run_gateway
    from clawagent.gateway.session_manager import SessionManager

    # Configure logging: console (INFO+) + file (DEBUG, rotating)
    _setup_gateway_logging()

    settings = Settings.from_env()
    gateway_cfg = GatewayConfig.from_env()

    session_mgr = SessionManager(
        settings,
        max_sessions=gateway_cfg.session_max,
        session_ttl=gateway_cfg.session_ttl,
    )

    # Register per-channel model configs (reserved for Direction 2)
    for ch_name, model_cfg in gateway_cfg.channel_models.items():
        session_mgr.set_channel_model(ch_name, model_cfg)

    channels: list[IChannel] = []
    channel_names: list[str] = []

    # WeChat iLink uses QR code on stdout — disable CLI to avoid
    # prompt_toolkit fullscreen mode hiding the QR code.
    cli_enabled = gateway_cfg.enable_cli and not gateway_cfg.wechat.configured
    if cli_enabled:
        channels.append(CliChannel())
        channel_names.append("CLI")
    if gateway_cfg.wechat.configured:
        from clawagent.gateway.channels.wechat_channel import WechatChannel
        channels.append(WechatChannel(creds_file=gateway_cfg.wechat.ilink_creds_file))
        channel_names.append("WeChat (iLink)")

    if not channels:
        print("Error: No channels enabled.", file=sys.stderr)
        print("  Set GATEWAY_ENABLE_CLI=true or WECHAT_ILINK_ENABLED=true in .env", file=sys.stderr)
        sys.exit(1)

    print(f"Gateway starting — channels: {', '.join(channel_names)}")
    if not gateway_cfg.wechat.configured:
        print("  (WeChat not enabled — set WECHAT_ILINK_ENABLED=true in .env to activate)")
    if gateway_cfg.wechat.configured and gateway_cfg.enable_cli:
        print("  (CLI auto-disabled when WeChat is active — QR code needs clean terminal)")

    try:
        asyncio.run(run_gateway(channels, session_mgr))
    finally:
        session_mgr.close_all()


def main() -> None:
    """Run the clawagent from the command line."""
    # Gateway subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "gateway":
        _run_gateway()
        return

    try:
        from clawagent.cli.session import init_session
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    ctx = init_session()

    if len(sys.argv) > 1:
        response = ctx.agent_ref.agent.run(" ".join(sys.argv[1:]))
        print(response.text)
        ctx.conn.close()
        return

    from clawagent.cli.repl import run_repl

    run_repl(ctx)

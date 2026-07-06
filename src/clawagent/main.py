"""Entry point for the clawagent CLI."""

from __future__ import annotations

import sys


def main() -> None:
    """Run the clawagent from the command line.

    Usage: uv run clawagent "Your question here"
    If no argument is given, runs an interactive REPL.
    """
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

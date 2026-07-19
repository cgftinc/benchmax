"""``castform`` CLI — a single argparse tree assembled from command groups.

Each command group lives in its own module exposing ``register(sub)``;
``build_parser`` wires them onto the top-level subparsers and ``main`` dispatches
to the selected handler's ``func``. The ``castform`` distribution exposes the
``castform.cli:main`` entry point. Argparse keeps the base SDK dependency surface
small.
"""

from __future__ import annotations

import argparse
import sys

from castform.cli import (
    _auth,
    control,
    corpus,
    data,
    doctor,
    help,
    launch,
    runs,
    setup,
)

# Re-export auth handlers — tests/unit/test_cli.py imports them as cli._cmd_*.
from castform.cli._auth import _cmd_login, _cmd_logout, _cmd_whoami

__all__ = ["build_parser", "main", "_cmd_login", "_cmd_logout", "_cmd_whoami"]


def build_parser() -> argparse.ArgumentParser:
    """Build the full castform parser. Tests snapshot its ``format_help()``."""
    parser = argparse.ArgumentParser(prog="castform", description="Castform CLI")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    _auth.register(sub)
    runs.register(sub)
    control.register(sub)
    launch.register(sub)
    data.register(sub)
    corpus.register(sub)
    setup.register(sub)
    doctor.register(sub)

    # `guide` renders the getting-started walkthrough (the renderer lives in
    # ``castform.cli.help``). Named `guide`, not `quickstart`, because `setup`
    # is itself the quickstart flow — keep the two from blurring together.
    gp = sub.add_parser("guide", help="Walk through your first run")
    gp.set_defaults(func=help._cmd_help)

    # `help` mirrors `-h`: just list the commands. Users reach for it by habit;
    # the walkthrough is `castform guide`, not here.
    def _list_commands(_args: argparse.Namespace) -> int:
        parser.print_help()
        return 0

    hp = sub.add_parser("help", help="List the available commands")
    hp.set_defaults(func=_list_commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

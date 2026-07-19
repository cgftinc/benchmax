"""Best-effort local browser open — shared by device-login and `data view`.

Pure stdlib (no benchmax imports) so both the CLI (`castform.cli`) and the
platform auth flow can use it without inverting the cli→platform layering.
"""

from __future__ import annotations

import os
import sys
import webbrowser


def maybe_open_browser(url: str) -> None:
    """Best-effort: open *url* in a local browser.

    Skipped when there is no useful local browser to open — a non-interactive
    stdin, an SSH session (the browser would spawn on the remote host, not the
    user's machine), or an explicit ``CASTFORM_NO_BROWSER`` opt-out. Callers
    always print the link/path too, which is the real fallback; opening is just
    convenience and never raises.
    """
    if (
        not sys.stdin.isatty()
        or os.environ.get("SSH_CONNECTION")
        or os.environ.get("CASTFORM_NO_BROWSER")
    ):
        return
    try:
        webbrowser.open(url)
    except Exception:
        pass  # best-effort; the printed link/path is the fallback

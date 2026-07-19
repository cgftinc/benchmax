"""castform doctor — check the local environment is ready to validate/launch.

A fast, read-only preflight a user or agent can run before ``python main.py validate``:
the Python interpreter, the castform install, sign-in state, and which env-dependency
extras are importable — so a missing ``castform[rag]`` / provider SDK surfaces as a
check here rather than as a mid-rollout ``ModuleNotFoundError``. Never installs
anything.
"""

from __future__ import annotations

import argparse
import sys

from castform import config
from castform.cli._client import handle_errors
from castform.cli._output import BLUE, _GREY, paint, print_json
from castform.cli._preflight import _EXTRA_SENTINEL, extra_is_installed
from castform.platform import credentials

_REQUIRED_PY = (3, 12)


def _py_check() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) == _REQUIRED_PY
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if not ok:
        detail += f" (castform requires {_REQUIRED_PY[0]}.{_REQUIRED_PY[1]}.x)"
    return ok, detail


def _version() -> str:
    from importlib.metadata import version

    # Distribution is published as "castform"; "benchmax" covers dev installs.
    for dist in ("castform", "benchmax"):
        try:
            return version(dist)
        except Exception:
            pass
    return "unknown"


def _auth_check() -> tuple[bool, str]:
    """(signed_in, detail). Resolves the same bearer seam validate/launch use."""
    try:
        credentials.platform_bearer()
    except Exception:
        return False, "not signed in — run `castform login`"
    jwt = credentials._session_jwt()
    who = (credentials._jwt_claims(jwt).get("email") if jwt else None) or "your account"
    return True, f"{who} ({config.base_domain()})"


def _collect() -> dict:
    py_ok, py_detail = _py_check()
    auth_ok, auth_detail = _auth_check()
    extras = {name: extra_is_installed(name) for name in _EXTRA_SENTINEL}
    return {
        "python_ok": py_ok,
        "python": py_detail,
        "benchmax_version": _version(),
        "signed_in": auth_ok,
        "auth": auth_detail,
        "extras": extras,
    }


def _row(ok: bool | None, label: str, detail: str = "") -> str:
    symbol = {True: paint("✓", BLUE), False: "✗", None: paint("·", _GREY)}[ok]
    line = f"  {symbol}  {label:<22}"
    return (line + paint(detail, _GREY)).rstrip() if detail else line.rstrip()


@handle_errors
def _cmd_doctor(args: argparse.Namespace) -> int:
    info = _collect()
    if args.json:
        print_json(info)
        # Blocking issues (interpreter / auth) drive the exit code; extras are optional.
        return 0 if (info["python_ok"] and info["signed_in"]) else 1

    print()
    print(paint("  castform doctor", bold=True))
    print()
    print(_row(info["python_ok"], "python", info["python"]))
    print(_row(True, "castform (benchmax)", info["benchmax_version"]))
    print(_row(info["signed_in"], "signed in", info["auth"]))
    print()
    print("  env-dependency extras (optional — install only what your main.py uses)")
    for name, present in info["extras"].items():
        detail = (
            "installed" if present else f"absent — uv pip install 'castform[{name}]'"
        )
        print(_row(None if not present else True, f"[{name}]", detail))
    print()

    blocking = not (info["python_ok"] and info["signed_in"])
    if blocking:
        print("  ✗ not ready — fix the ✗ rows above (interpreter / sign-in).")
    else:
        print("  ✓ ready to run your environment script.")
    return 1 if blocking else 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the top-level `doctor` verb."""
    p = sub.add_parser(
        "doctor", help="Check interpreter, sign-in, and env deps are ready"
    )
    p.add_argument("--json", action="store_true", help="Emit raw JSON")
    p.set_defaults(func=_cmd_doctor)

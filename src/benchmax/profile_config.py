"""User profile config for Castform auth and endpoint resolution.

The file is intentionally small and user-editable:

    ~/.castform.toml

Only the fields modeled here are first-class. Advanced edits can happen
directly in TOML; the CLI only creates/updates profiles during login/logout.
"""

from __future__ import annotations

import json
import os
import tempfile
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_PATH_ENV = "CASTFORM_CONFIG_PATH"
PROFILE_ENV = "CASTFORM_PROFILE"
DEFAULT_PROFILE = "prod"
DEFAULT_DOMAIN = "castform.com"

URL_FIELDS = ("api_url", "llm_url", "auth_url", "app_url")
SESSION_FIELDS = ("access_token", "refresh_token", "expires_at")


@dataclass(frozen=True)
class ProfileSelection:
    name: str
    source: str


def config_path() -> Path:
    override = os.environ.get(CONFIG_PATH_ENV)
    return Path(override).expanduser() if override else Path.home() / ".castform.toml"


def _default_config() -> dict[str, Any]:
    return {
        "default_profile": DEFAULT_PROFILE,
        "profiles": {DEFAULT_PROFILE: {"domain": DEFAULT_DOMAIN}},
    }


def _check_permissions(path: Path) -> None:
    try:
        st = path.stat()
    except OSError:
        return
    if os.name == "posix" and (st.st_mode & 0o077):
        raise RuntimeError(
            f"Ignoring {path}: permissions {oct(st.st_mode & 0o777)} are looser "
            f"than 0600. Run `chmod 600 {path}`."
        )


def _clean_session(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    session: dict[str, Any] = {}
    for key in SESSION_FIELDS:
        if key in value:
            session[key] = value[key]
    return session or None


def _clean_profile(name: str, value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        warnings.warn(f"Ignoring profile {name!r}: expected a table.", stacklevel=3)
        return None
    profile: dict[str, Any] = {}
    domain = value.get("domain")
    if isinstance(domain, str) and domain:
        profile["domain"] = domain
    for key in URL_FIELDS:
        url = value.get(key)
        if isinstance(url, str) and url:
            profile[key] = url
    session = _clean_session(value.get("session"))
    if session:
        profile["session"] = session
    if name == DEFAULT_PROFILE and "domain" not in profile:
        profile["domain"] = DEFAULT_DOMAIN
    return profile


def _normalize_config(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise RuntimeError("Castform config must be a TOML table.")
    default_profile = data.get("default_profile", DEFAULT_PROFILE)
    if not isinstance(default_profile, str) or not default_profile:
        raise RuntimeError("Castform config default_profile must be a non-empty string.")

    raw_profiles = data.get("profiles", {})
    if raw_profiles is None:
        raw_profiles = {}
    if not isinstance(raw_profiles, dict):
        raise RuntimeError("Castform config profiles must be a TOML table.")

    profiles: dict[str, Any] = {}
    for name, raw in raw_profiles.items():
        if not isinstance(name, str) or not name:
            continue
        profile = _clean_profile(name, raw)
        if profile is not None:
            profiles[name] = profile
    profiles.setdefault(DEFAULT_PROFILE, {"domain": DEFAULT_DOMAIN})
    return {"default_profile": default_profile, "profiles": profiles}


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return _default_config()
    _check_permissions(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Could not read Castform config {path}: {exc}") from exc
    return _normalize_config(data)


def _toml_str(value: str) -> str:
    return json.dumps(value)


def _toml_key(value: str) -> str:
    return json.dumps(value)


def _dump_config(data: dict[str, Any]) -> str:
    data = _normalize_config(data)
    lines = [f"default_profile = {_toml_str(data['default_profile'])}", ""]
    for name, profile in data["profiles"].items():
        profile_key = _toml_key(name)
        lines.append(f"[profiles.{profile_key}]")
        if domain := profile.get("domain"):
            lines.append(f"domain = {_toml_str(str(domain))}")
        for key in URL_FIELDS:
            if value := profile.get(key):
                lines.append(f"{key} = {_toml_str(str(value))}")
        if session := profile.get("session"):
            lines.extend(["", f"[profiles.{profile_key}.session]"])
            for key in SESSION_FIELDS:
                if key in session:
                    value = session[key]
                    if isinstance(value, str):
                        lines.append(f"{key} = {_toml_str(value)}")
                    elif isinstance(value, (int, float)):
                        lines.append(f"{key} = {int(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_config(data: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _dump_config(data)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
        os.chmod(path, 0o600)
    finally:
        try:
            Path(tmp_name).unlink()
        except OSError:
            pass


def select_profile(explicit: str | None = None) -> ProfileSelection:
    if explicit:
        return ProfileSelection(explicit, "--profile")
    env = os.environ.get(PROFILE_ENV)
    if env:
        return ProfileSelection(env, PROFILE_ENV)
    cfg = load_config()
    default = cfg.get("default_profile") or DEFAULT_PROFILE
    if isinstance(default, str) and default:
        return ProfileSelection(default, "default_profile")
    return ProfileSelection(DEFAULT_PROFILE, "builtin")


def get_profile(name: str | None = None) -> dict[str, Any] | None:
    cfg = load_config()
    selected = select_profile(name).name if name is not None else select_profile().name
    profile = cfg["profiles"].get(selected)
    if profile is None and selected == DEFAULT_PROFILE:
        return {"domain": DEFAULT_DOMAIN}
    return profile


def upsert_profile(
    name: str,
    *,
    domain: str | None = None,
    api_url: str | None = None,
    llm_url: str | None = None,
    auth_url: str | None = None,
    app_url: str | None = None,
    session: dict[str, Any] | None = None,
) -> None:
    cfg = load_config()
    profile = dict(cfg["profiles"].get(name, {}))
    if domain:
        profile["domain"] = domain
    elif "domain" not in profile:
        if name == DEFAULT_PROFILE:
            profile["domain"] = DEFAULT_DOMAIN
        else:
            raise RuntimeError(
                f"Profile {name!r} does not exist yet. Pass --domain when logging in."
            )
    for key, value in {
        "api_url": api_url,
        "llm_url": llm_url,
        "auth_url": auth_url,
        "app_url": app_url,
    }.items():
        if value:
            profile[key] = value
    if session is not None:
        profile["session"] = session
    cfg["profiles"][name] = profile
    write_config(cfg)


def clear_profile_session(name: str) -> None:
    cfg = load_config()
    profile = cfg["profiles"].get(name)
    if profile:
        profile.pop("session", None)
        write_config(cfg)


def clear_all_sessions() -> None:
    cfg = load_config()
    changed = False
    for profile in cfg["profiles"].values():
        if isinstance(profile, dict) and "session" in profile:
            profile.pop("session", None)
            changed = True
    if changed:
        write_config(cfg)


def stored_profiles() -> dict[str, dict[str, Any]]:
    return load_config()["profiles"]


def session_for_profile(name: str | None = None) -> dict[str, Any] | None:
    profile = get_profile(name)
    if not profile:
        return None
    session = profile.get("session")
    return session if isinstance(session, dict) else None


def selected_profile_name(explicit: str | None = None) -> str:
    return select_profile(explicit).name

"""Named Castform endpoint profiles.

Profiles keep non-secret routing configuration in ``~/.castform/config.toml``.
The corresponding login sessions live separately in the mode-0600
``credentials.json`` managed by :mod:`castform.platform.credentials`.
"""

from __future__ import annotations

import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

CONFIG_PATH_ENV = "CASTFORM_CONFIG_PATH"
PROFILE_ENV = "CASTFORM_PROFILE"
DEFAULT_PROFILE = "prod"
DEFAULT_DOMAIN = "castform.com"
URL_FIELDS = ("platform_url", "llm_url", "auth_url", "app_url")


def config_path() -> Path:
    override = os.environ.get(CONFIG_PATH_ENV)
    return Path(override).expanduser() if override else Path.home() / ".castform" / "config.toml"


def _default_config() -> dict[str, Any]:
    return {
        "active_profile": DEFAULT_PROFILE,
        "profiles": {DEFAULT_PROFILE: {"domain": DEFAULT_DOMAIN}},
    }


def _normalize_profile(name: str, value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Castform profile {name!r} must be a TOML table.")
    profile: dict[str, str] = {}
    domain = value.get("domain")
    if domain is not None:
        if not isinstance(domain, str) or not domain.strip():
            raise RuntimeError(f"Castform profile {name!r} has an invalid domain.")
        profile["domain"] = domain.strip()
    for field in URL_FIELDS:
        url = value.get(field)
        if url is not None:
            if not isinstance(url, str) or not url.strip():
                raise RuntimeError(f"Castform profile {name!r} has an invalid {field}.")
            profile[field] = url.strip().rstrip("/")
    if name == DEFAULT_PROFILE and not profile:
        profile["domain"] = DEFAULT_DOMAIN
    if not profile:
        raise RuntimeError(
            f"Castform profile {name!r} needs a domain or explicit service URLs."
        )
    return profile


def _normalize_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Castform config must be a TOML table.")
    active = value.get("active_profile", DEFAULT_PROFILE)
    if not isinstance(active, str) or not active:
        raise RuntimeError("Castform active_profile must be a non-empty string.")
    raw_profiles = value.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise RuntimeError("Castform profiles must be a TOML table.")
    profiles = {
        name: _normalize_profile(name, profile)
        for name, profile in raw_profiles.items()
        if isinstance(name, str) and name
    }
    profiles.setdefault(DEFAULT_PROFILE, {"domain": DEFAULT_DOMAIN})
    return {"active_profile": active, "profiles": profiles}


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return _default_config()
    try:
        return _normalize_config(tomllib.loads(path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"Could not read Castform config {path}: {error}") from error


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _dump_config(value: dict[str, Any]) -> str:
    config = _normalize_config(value)
    lines = [f"active_profile = {_toml_string(config['active_profile'])}", ""]
    for name, profile in config["profiles"].items():
        lines.append(f"[profiles.{_toml_string(name)}]")
        for field in ("domain", *URL_FIELDS):
            if field in profile:
                lines.append(f"{field} = {_toml_string(profile[field])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_config(value: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(_dump_config(value))
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except OSError:
            pass


def selected_profile_name(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if environment := os.environ.get(PROFILE_ENV):
        return environment
    return str(load_config()["active_profile"])


def get_profile(name: str | None = None) -> dict[str, str] | None:
    selected = selected_profile_name(name)
    return load_config()["profiles"].get(selected)


def stored_profiles() -> dict[str, dict[str, str]]:
    return load_config()["profiles"]


def upsert_profile(
    name: str,
    *,
    domain: str | None = None,
    platform_url: str | None = None,
    llm_url: str | None = None,
    auth_url: str | None = None,
    app_url: str | None = None,
) -> None:
    if not name:
        raise RuntimeError("Profile name must be non-empty.")
    config = load_config()
    profile = dict(config["profiles"].get(name, {}))
    supplied = {
        "domain": domain,
        "platform_url": platform_url,
        "llm_url": llm_url,
        "auth_url": auth_url,
        "app_url": app_url,
    }
    for field, value in supplied.items():
        if value:
            profile[field] = value.strip().rstrip("/")
    if not profile:
        raise RuntimeError(
            f"Profile {name!r} is not configured. Pass --domain or explicit service URLs."
        )
    config["profiles"][name] = _normalize_profile(name, profile)
    write_config(config)


def activate_profile(name: str) -> None:
    config = load_config()
    if name not in config["profiles"]:
        raise RuntimeError(
            f"Profile {name!r} is not configured. Run `castform login --profile {name} --domain "
            f"<domain>`."
        )
    config["active_profile"] = name
    write_config(config)

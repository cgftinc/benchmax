"""Small, explicit training-project contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Repository:
    name: str
    url: str
    revision: str = "main"


@dataclass(frozen=True, slots=True)
class Backend:
    name: str
    model: str
    provider: str


@dataclass(frozen=True, slots=True)
class Project:
    repositories: tuple[Repository, ...]
    backends: tuple[Backend, ...]


def load_project(path: Path) -> Project:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        repositories = tuple(
            Repository(
                name=item["name"],
                url=item["url"],
                revision=item.get("revision", "main"),
            )
            for item in value["repositories"]
        )
        backends = tuple(
            Backend(name=item["name"], model=item["model"], provider=item["provider"])
            for item in value["backends"]
        )
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid training project: {path}") from error
    if not repositories or not backends:
        raise ValueError("a training project needs repositories and backends")
    if len({item.name for item in repositories}) != len(repositories):
        raise ValueError("repository names must be unique")
    if len({item.name for item in backends}) != len(backends):
        raise ValueError("backend names must be unique")
    if any(not item.name or not item.url or not item.revision for item in repositories):
        raise ValueError("repository fields must be non-empty")
    if any(not item.name or not item.model or not item.provider for item in backends):
        raise ValueError("backend fields must be non-empty")
    return Project(repositories=repositories, backends=backends)

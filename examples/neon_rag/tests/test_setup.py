from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

from neon_backend.provision import write_env_file

_SETUP_PATH = Path(__file__).parents[1] / "setup_neon.py"
_SPEC = importlib.util.spec_from_file_location("neon_rag_example_setup", _SETUP_PATH)
assert _SPEC is not None and _SPEC.loader is not None
setup_neon = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = setup_neon
_SPEC.loader.exec_module(setup_neon)


def test_setup_reports_each_missing_input(monkeypatch, capsys) -> None:
    monkeypatch.delenv("NEON_API_KEY", raising=False)
    monkeypatch.delenv("NEON_PROJECT_ID", raising=False)

    assert setup_neon.main() == 2

    error = capsys.readouterr().err
    assert "NEON_API_KEY" in error
    assert "NEON_PROJECT_ID" in error


def test_setup_reuses_existing_connections_for_the_same_project(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env.neon"
    write_env_file(
        str(env_file),
        {
            "NEON_PROJECT_ID": "flat-frost-16947914",
            "NEON_DATA_PREPARATION_DATABASE_URL": "postgresql://prepare:old@db/neondb",
            "NEON_SEARCH_DATABASE_URL": "postgresql://search:old@db/neondb",
        },
    )
    calls: list[dict[str, object]] = []

    def fake_provision(api_key, project_id, **kwargs):
        calls.append({"api_key": api_key, "project_id": project_id, **kwargs})
        return SimpleNamespace(
            project_id=project_id,
            data_preparation_database_url="postgresql://prepare:new@db/neondb",
            search_database_url="postgresql://search:new@db/neondb",
        )

    monkeypatch.setattr(setup_neon, "ENV_FILE", env_file)
    monkeypatch.setattr(setup_neon, "provision", fake_provision)
    monkeypatch.setenv("NEON_API_KEY", "neon-secret")
    monkeypatch.setenv("NEON_PROJECT_ID", "flat-frost-16947914")

    assert setup_neon.run_setup() == 0
    assert calls == [
        {
            "api_key": "neon-secret",
            "project_id": "flat-frost-16947914",
            "existing_data_preparation_database_url": ("postgresql://prepare:old@db/neondb"),
            "existing_search_database_url": "postgresql://search:old@db/neondb",
        }
    ]
    generated = setup_neon._read_generated_env(env_file)
    assert generated == {
        "NEON_PROJECT_ID": "flat-frost-16947914",
        "NEON_DATA_PREPARATION_DATABASE_URL": "postgresql://prepare:new@db/neondb",
        "NEON_SEARCH_DATABASE_URL": "postgresql://search:new@db/neondb",
    }
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_setup_does_not_reuse_connections_from_another_project(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env.neon"
    write_env_file(
        str(env_file),
        {
            "NEON_PROJECT_ID": "old-project",
            "NEON_DATA_PREPARATION_DATABASE_URL": "postgresql://prepare:old@db/neondb",
            "NEON_SEARCH_DATABASE_URL": "postgresql://search:old@db/neondb",
        },
    )
    calls: list[dict[str, object]] = []

    def fake_provision(api_key, project_id, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            project_id=project_id,
            data_preparation_database_url="postgresql://prepare:new@db/neondb",
            search_database_url="postgresql://search:new@db/neondb",
        )

    monkeypatch.setattr(setup_neon, "ENV_FILE", env_file)
    monkeypatch.setattr(setup_neon, "provision", fake_provision)
    monkeypatch.setenv("NEON_API_KEY", "neon-secret")
    monkeypatch.setenv("NEON_PROJECT_ID", "new-project")

    assert setup_neon.run_setup() == 0
    assert calls == [
        {
            "existing_data_preparation_database_url": None,
            "existing_search_database_url": None,
        }
    ]

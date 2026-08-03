from pathlib import Path

import pytest

from convert_to_harbor import DEFAULT_PROFILES_DIR, load_environment_profile


@pytest.mark.parametrize("name", ["pytest", "pydantic", "numpy", "sympy", "xarray"])
def test_builtin_environment_profiles(name: str) -> None:
    profile = load_environment_profile(name)

    assert profile.name == name
    assert profile.repo == name
    assert profile.path == DEFAULT_PROFILES_DIR / f"{name}.toml"
    assert profile.repo_url.startswith("https://github.com/")
    assert profile.base_image == "python:3.12-slim"
    assert profile.install_cmd
    assert profile.build_timeout_sec == 900.0
    assert profile.agent_timeout_sec == 1800.0
    assert profile.verifier_timeout_sec == 600.0


def test_profile_rejects_unknown_keys(tmp_path: Path) -> None:
    profile = tmp_path / "bad.toml"
    profile.write_text(
        """\
schema_version = 1
name = "bad"
repo = "bad"
repo_url = "https://github.com/example/bad"
base_image = "python:3.12-slim"
install_cmd = "pip install -e ."
surprise = true
"""
    )

    with pytest.raises(ValueError, match="unknown profile keys: surprise"):
        load_environment_profile(profile)


def test_profile_name_matches_filename(tmp_path: Path) -> None:
    profile = tmp_path / "expected.toml"
    profile.write_text(
        """\
schema_version = 1
name = "different"
repo = "different"
repo_url = "https://github.com/example/different"
base_image = "python:3.12-slim"
install_cmd = "pip install -e ."
"""
    )

    with pytest.raises(ValueError, match="must match filename"):
        load_environment_profile(profile)


def test_profile_name_is_not_shadowed_by_repo_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "numpy").mkdir()
    monkeypatch.chdir(tmp_path)

    profile = load_environment_profile("numpy")

    assert profile.path == DEFAULT_PROFILES_DIR / "numpy.toml"

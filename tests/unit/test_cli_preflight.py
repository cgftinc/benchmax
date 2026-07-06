"""Unit tests for the preflight install-hint mapping (`castform validate`/`launch`)."""

from __future__ import annotations

from benchmax.cli import _preflight
from benchmax.cli._preflight import (
    extra_is_installed,
    install_hint_for_import_error,
)


def _mnfe(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


def test_provider_module_maps_to_provider_extra():
    hint = install_hint_for_import_error(_mnfe("turbopuffer"))
    assert hint is not None and "benchmax[turbopuffer]" in hint


def test_chromadb_maps_to_chroma_extra():
    # import name (chromadb) differs from the extra name (chroma).
    hint = install_hint_for_import_error(_mnfe("chromadb"))
    assert hint is not None and "benchmax[chroma]" in hint


def test_datagen_module_maps_to_rag_extra():
    # scikit-learn's import name is sklearn; it lives in the [rag] extra.
    hint = install_hint_for_import_error(_mnfe("sklearn"))
    assert hint is not None and "benchmax[rag]" in hint


def test_submodule_resolves_to_top_level_import_name():
    hint = install_hint_for_import_error(_mnfe("turbopuffer.query"))
    assert hint is not None and "benchmax[turbopuffer]" in hint


def test_unknown_module_gets_generic_hint():
    hint = install_hint_for_import_error(_mnfe("some_random_pkg"))
    assert hint is not None
    assert "some_random_pkg" in hint
    assert "benchmax[" not in hint  # not one of our extras


def test_non_import_error_returns_none():
    assert install_hint_for_import_error(ValueError("boom")) is None


def test_hint_follows_cause_chain():
    # load_project wraps the import error: `raise ProjectError(...) from exc`.
    try:
        try:
            raise _mnfe("pinecone")
        except ModuleNotFoundError as exc:
            raise RuntimeError("Failed to import main.py") from exc
    except RuntimeError as wrapped:
        hint = install_hint_for_import_error(wrapped)
    assert hint is not None and "benchmax[pinecone]" in hint


def test_hint_follows_implicit_context_chain():
    # Implicit chaining (no `from`) sets __context__, which we also walk.
    try:
        try:
            raise _mnfe("keybert")
        except ModuleNotFoundError:
            raise RuntimeError("wrapped without from")
    except RuntimeError as wrapped:
        hint = install_hint_for_import_error(wrapped)
    assert hint is not None and "benchmax[rag]" in hint


def test_print_project_error_appends_hint(capsys):
    # A ProjectError-shaped wrapper whose cause is the missing import.
    try:
        raise _mnfe("turbopuffer")
    except ModuleNotFoundError as exc:
        wrapped = RuntimeError(f"Failed to import main.py: {exc}")
        wrapped.__cause__ = exc
    _preflight.print_project_error(wrapped)
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "benchmax[turbopuffer]" in err


def test_extra_is_installed_known_and_unknown():
    # [rag] is installed in the dev env (uv sync --all-extras); a bogus extra is not.
    assert extra_is_installed("rag") is True
    assert extra_is_installed("does-not-exist") is False

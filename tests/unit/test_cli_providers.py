"""Offline: the ``_providers`` source-of-truth stays wired to the CLI surface.

``PROVIDER_PIP`` mirrors the pyproject provider extras; these guard that the
``qa-gen --provider`` choices are derived from it (no drift) and that chroma keeps
the un-guessable ``snowballstemmer`` dep chromadb's BM25 needs.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from benchmax.cli import build_parser
from benchmax.cli._providers import (
    PROVIDER_PIP,
    install_hint,
    provider_choices,
    resolve_pip_dependencies,
)


def _qa_provider_choices() -> list[str]:
    """Drill into the built parser for `data qa-gen`'s `--provider` choices."""
    parser = build_parser()
    for top in parser._actions:
        if isinstance(top, argparse._SubParsersAction):
            data = top.choices["data"]
            for da in data._actions:
                if isinstance(da, argparse._SubParsersAction):
                    for action in da.choices["qa-gen"]._actions:
                        if action.dest == "provider":
                            return list(action.choices)
    raise AssertionError("could not locate qa-gen --provider choices in the parser")


def test_provider_pip_keys_match_qagen_choices():
    # The live argparse choices come from provider_choices() → no hand-kept list to drift.
    assert _qa_provider_choices() == list(PROVIDER_PIP)


def test_provider_choices_helper_is_the_dict_keys():
    assert provider_choices() == list(PROVIDER_PIP)


def test_provider_pip_mirrors_pyproject_extras():
    # The real "single source of truth" guard: PROVIDER_PIP must equal pyproject's
    # per-provider extras, or the sandbox installs a stale pin while the suite stays
    # green. Parse [project.optional-dependencies] and assert byte-equality.
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text("utf-8"))["project"][
        "optional-dependencies"
    ]
    for provider, deps in PROVIDER_PIP.items():
        assert extras[provider] == deps, (
            f"{provider}: PROVIDER_PIP {deps} != pyproject extra {extras.get(provider)}"
        )


def test_chroma_pip_includes_snowballstemmer():
    # chromadb's BM25 embedding function imports snowballstemmer without declaring it;
    # an env that all-zeros for the missing stemmer is the #1 hollow-green footgun.
    assert any("snowballstemmer" in dep for dep in PROVIDER_PIP["chroma"])
    assert any(dep.startswith("chromadb") for dep in PROVIDER_PIP["chroma"])


def test_install_hint_names_the_extra():
    assert install_hint("turbopuffer") == "Install with: pip install benchmax[turbopuffer]"


# --- resolve_pip_dependencies: the merge helper -------------------------------


class _SlotEnv:
    PIP_DEPENDENCIES = ["myorg-search>=2.0", "snowballstemmer>=2.2.0"]


class _PlainEnv:  # no slot at all
    pass


def test_resolve_provider_injects_chroma_sdk():
    # (a) --provider chroma → chromadb + the un-guessable snowballstemmer.
    assert resolve_pip_dependencies(None, _PlainEnv, "chroma") == [
        "chromadb>=1.0.0",
        "snowballstemmer>=2.2.0",
    ]


def test_resolve_reads_self_declared_slot():
    # (b) the env's PIP_DEPENDENCIES class attr is read + merged.
    assert resolve_pip_dependencies(None, _SlotEnv, None) == [
        "myorg-search>=2.0",
        "snowballstemmer>=2.2.0",
    ]


def test_resolve_dedupes_pip_and_slot_order_preserved():
    # (c) overlap between --pip and the slot de-dupes, first-seen order kept.
    assert resolve_pip_dependencies(["snowballstemmer>=2.2.0"], _SlotEnv, None) == [
        "snowballstemmer>=2.2.0",
        "myorg-search>=2.0",
    ]


def test_resolve_no_slot_no_provider_returns_pip_verbatim():
    # (d) neither slot nor --provider → the --pip list verbatim (old `args.pip or
    # None` behaviour); empty/None everything → None.
    assert resolve_pip_dependencies(["turbopuffer", "pinecone>=5"], _PlainEnv, None) == [
        "turbopuffer",
        "pinecone>=5",
    ]
    assert resolve_pip_dependencies(None, _PlainEnv, None) is None
    assert resolve_pip_dependencies(None, None, None) is None
    # empty (not just None) explicit list must also collapse to None, so the
    # dump_bundle channel stays "no deps" rather than "empty deps list".
    assert resolve_pip_dependencies([], _PlainEnv, None) is None


def test_resolve_ignores_non_list_slot():
    # A stringly-typed slot is a user error — guard ignores it rather than splatting
    # the string into chars.
    class _BadEnv:
        PIP_DEPENDENCIES = "chromadb"

    assert resolve_pip_dependencies(["keep"], _BadEnv, None) == ["keep"]

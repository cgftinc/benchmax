"""Preflight: turn a missing-env-dependency ImportError into a copy-paste fix.

``castform validate``/``launch`` import the project's ``run.py`` in-process (see
:func:`benchmax.cli._project.load_project`). A ``run.py`` that uses a corpus-provider
backend (turbopuffer / pinecone / chroma) or the data-generation helpers imports a
package that lives behind a ``castform[...]`` extra, not in base castform — so a base
install fails with ``ModuleNotFoundError`` mid-import. Rather than surface a raw
traceback, map the missing top-level module back to the extra that ships it and print
the exact ``uv pip install`` line.

The default *postgres* rag env imports on base castform (openai + httpx); this is for
the provider backends and the ``castform data qa-gen`` / chunking path, which pull the
heavier deps.
"""

from __future__ import annotations

import importlib.util
import sys

# Import-name → the ``castform[<extra>]`` extra that provides it. Import names differ
# from the pip/extra names (scikit-learn→``sklearn``; chromadb ships under the
# ``[chroma]`` extra), and ``ModuleNotFoundError.name`` reports the *import* name — so
# this is keyed on that. Provider rows mirror ``_providers.PROVIDER_PIP``'s keys; the
# rest are the ``[rag]`` extra's data-generation + chunking deps.
_MODULE_EXTRA: dict[str, str] = {
    # provider search backends — benchmax.rag.corpus.{turbopuffer,pinecone,chroma}
    "turbopuffer": "turbopuffer",
    "pinecone": "pinecone",
    "chromadb": "chroma",
    "snowballstemmer": "chroma",
    # data generation (qa-gen) + chunking / ingest — the [rag] extra
    "sentence_transformers": "rag",
    "sklearn": "rag",
    "keybert": "rag",
    "nest_asyncio": "rag",
    "langchain_text_splitters": "rag",
    "ragas": "rag",
    "ruamel": "rag",
    "tqdm": "rag",
}

# One sentinel import name per extra — used by ``castform doctor`` to report whether an
# extra is installed without importing the (heavy) package itself.
_EXTRA_SENTINEL: dict[str, str] = {
    "rag": "keybert",
    "turbopuffer": "turbopuffer",
    "pinecone": "pinecone",
    "chroma": "chromadb",
}


def _missing_module(exc: BaseException) -> str | None:
    """Top-level name of the first ``ModuleNotFoundError`` in the exception chain."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ModuleNotFoundError) and cur.name:
            return cur.name.split(".")[0]
        cur = cur.__cause__ or cur.__context__
    return None


def install_hint_for_import_error(exc: BaseException) -> str | None:
    """A copy-paste install line if ``exc`` (or its cause chain) is a missing import.

    Returns ``None`` when the error isn't a ``ModuleNotFoundError`` — callers then
    just print the base error and leave it at that.
    """
    module = _missing_module(exc)
    if module is None:
        return None
    extra = _MODULE_EXTRA.get(module)
    if extra:
        return f"→ install the env deps: uv pip install 'castform[{extra}]'"
    # A third-party module the env imports that isn't one of our extras — still nudge
    # toward installing it rather than leaving a bare "no module named …".
    return (
        f"→ missing dependency {module!r}: install it into this environment "
        f"(e.g. uv pip install {module})"
    )


def print_project_error(exc: Exception) -> None:
    """Print a project-load error to stderr, appending an install hint for a
    missing env dependency (so a base-install ImportError is a one-line fix, not a
    traceback). Shared by ``validate`` and ``launch``."""
    print(f"Error: {exc}", file=sys.stderr)
    hint = install_hint_for_import_error(exc)
    if hint:
        print(hint, file=sys.stderr)


def extra_is_installed(extra: str) -> bool:
    """True if ``extra``'s sentinel module resolves (no import executed)."""
    sentinel = _EXTRA_SENTINEL.get(extra)
    if sentinel is None:
        return False
    try:
        return importlib.util.find_spec(sentinel) is not None
    except (ImportError, ValueError):
        # A parent package missing (or a namespace edge case) => treat as absent.
        return False

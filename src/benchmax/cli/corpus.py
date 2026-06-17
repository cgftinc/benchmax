"""castform ``corpus`` command group (RAG fast path).

``corpus ingest <folder>`` chunks a local document folder and uploads it to the
deployed Corpora backend (BM25/lexical search), yielding a named corpus that
``data qa-gen`` and the rag ``run.py`` template resolve by name. Thin wrapper over
``benchmax.rag.corpus.postgres.PostgresChunkSource`` — the rag bits import lazily
so users without the ``[rag]`` extra don't trip at CLI registration.

Corpus creation is forced **non-interactive** (``on_limit="error"``): the lib's
default ``on_limit="prompt"`` would call ``input()`` at the 5-corpus cap, which
hangs a CLI. ``"error"`` instead finds an existing corpus by name, creates one if
under cap, or raises ``CorpusLimitError`` — surfaced here as a clean message.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmax.cli._client import handle_errors

_RAG_INSTALL_HINT = "Install RAG support with: pip install castform[rag]"


@handle_errors  # backstop stray httpx errors + no-credential RuntimeError, like the
# other verbs; the corpus-specific excepts below still run first.
def _cmd_corpus_ingest(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: not a folder: {folder}", file=sys.stderr)
        return 1

    # Lazy import — keep langchain_text_splitters (via the chunker) and the rest of
    # the rag extra out of the base install; only needed when this verb runs.
    try:
        from benchmax.rag.corpus.postgres.exceptions import (
            CorpusAPIError,
            CorpusLimitError,
        )
        from benchmax.rag.corpus.postgres.source import PostgresChunkSource
    except ImportError as exc:
        print(f"Error: {exc}. {_RAG_INSTALL_HINT}", file=sys.stderr)
        return 1

    name = args.name or folder.resolve().name
    source = PostgresChunkSource(corpus_name=name)
    try:
        # on_limit="error" => non-interactive: reuse-by-name, create if under cap,
        # else raise CorpusLimitError. Never reaches the lib's input() prompt.
        source.populate_from_folder(str(folder), on_limit="error")
    except CorpusLimitError as exc:
        have = ", ".join(c.name for c in exc.existing_corpora) or "—"
        print(
            f"Error: at the 5-corpus cap (have: {have}). Delete one, or pass "
            "`--name <existing corpus>` to add to it.",
            file=sys.stderr,
        )
        return 1
    except ImportError as exc:  # chunker dep (langchain_text_splitters) missing
        print(f"Error: {exc}. {_RAG_INSTALL_HINT}", file=sys.stderr)
        return 1
    except CorpusAPIError as exc:
        print(f"Error: corpus API: {exc}", file=sys.stderr)
        return 1

    corpus_id = source.corpus_id
    if args.json:
        from benchmax.cli._output import print_json

        print_json({"corpus_name": name, "corpus_id": corpus_id})
    else:
        print(f"✓ Ingested {folder.name} → corpus '{name}' (id: {corpus_id})")
        print(f"  Next: castform data qa-gen --corpus-name {name} --fast")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the `corpus` group to the top-level subparsers."""
    corpus = sub.add_parser("corpus", help="Corpus utilities for RAG envs")
    corpus_sub = corpus.add_subparsers(
        dest="corpus_command", required=True, metavar="<subcommand>"
    )

    p_ing = corpus_sub.add_parser(
        "ingest",
        help="Chunk a local folder and upload it as a searchable corpus",
    )
    p_ing.add_argument("folder", help="Folder of documents to ingest (.md/.mdx)")
    p_ing.add_argument(
        "--name",
        help="Corpus name (default: folder basename); reuses an existing corpus "
        "of the same name",
    )
    p_ing.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_ing.set_defaults(func=_cmd_corpus_ingest)

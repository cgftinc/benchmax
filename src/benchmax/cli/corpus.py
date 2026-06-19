"""castform ``corpus`` command group (RAG fast path).

``corpus ingest <folder>`` chunks a local document folder and uploads it to the
deployed Corpora backend (BM25/lexical search), yielding a named corpus that
``data qa-gen`` and the rag ``run.py`` template resolve by name. ``corpus list`` /
``corpus delete`` manage them against the 5-corpus-per-user cap (the cap is why
``ingest`` can refuse — delete one to free a slot). Thin wrappers over
``benchmax.rag.corpus.postgres`` — the rag bits import lazily so users without the
``[rag]`` extra don't trip at CLI registration.

Corpus creation is forced **non-interactive** (``on_limit="error"``): the lib's
default ``on_limit="prompt"`` would call ``input()`` at the 5-corpus cap, which
hangs a CLI. ``"error"`` instead finds an existing corpus by name, creates one if
under cap, or raises ``CorpusLimitError`` — surfaced here as a clean message.
``delete`` likewise requires ``--yes`` rather than prompting.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from benchmax import config
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


def _corpus_client():
    """A CorpusClient bound to the configured platform host + credential seam.

    Lazy-imported so the base CLI registers without the ``[rag]`` extra.
    """
    from benchmax.rag.corpus.postgres.client import CorpusClient

    return CorpusClient(base_url=config.platform_url())


@handle_errors
def _cmd_corpus_list(args: argparse.Namespace) -> int:
    try:
        client = _corpus_client()
    except ImportError as exc:
        print(f"Error: {exc}. {_RAG_INSTALL_HINT}", file=sys.stderr)
        return 1
    corpora = client.list_corpora()
    if args.json:
        from benchmax.cli._output import print_json

        print_json(
            [
                {"id": c.id, "name": c.name, "created_at": c.created_at.isoformat()}
                for c in corpora
            ]
        )
        return 0
    if not corpora:
        print("No corpora yet. Create one: castform corpus ingest <folder>")
        return 0
    print(f"{len(corpora)}/5 corpora:")
    for c in corpora:
        print(f"  {c.name}  (id: {c.id})")
    return 0


@handle_errors
def _cmd_corpus_delete(args: argparse.Namespace) -> int:
    try:
        client = _corpus_client()
    except ImportError as exc:
        print(f"Error: {exc}. {_RAG_INSTALL_HINT}", file=sys.stderr)
        return 1
    corpora = client.list_corpora()
    target = next(
        (c for c in corpora if c.id == args.corpus or c.name == args.corpus), None
    )
    if target is None:
        have = ", ".join(c.name for c in corpora) or "(none)"
        print(
            f"Error: no corpus matching {args.corpus!r}. Have: {have}", file=sys.stderr
        )
        return 1
    # Destructive + irreversible — require explicit --yes rather than prompting
    # (an input() would hang a non-interactive/agent run).
    if not args.yes:
        print(
            f"Error: refusing to delete corpus '{target.name}' (id: {target.id}) "
            "without --yes.",
            file=sys.stderr,
        )
        return 1
    if not client.delete_corpus(target.id):
        print(
            f"Error: delete failed for corpus '{target.name}' (id: {target.id}).",
            file=sys.stderr,
        )
        return 1
    if args.json:
        from benchmax.cli._output import print_json

        print_json({"deleted": True, "corpus_name": target.name, "corpus_id": target.id})
    else:
        print(f"✓ Deleted corpus '{target.name}' (id: {target.id})")
    return 0


@handle_errors
def _cmd_corpus_search(args: argparse.Namespace) -> int:
    # Confirm retrieval the same way the rollout does (resolve by name, lexical) —
    # real hits here mean a rag env can actually search; empty/Error means fix the
    # corpus before trusting a green validate.
    try:
        from benchmax.rag.corpus.postgres.search import PostgresSearch
    except ImportError as exc:
        print(f"Error: {exc}. {_RAG_INSTALL_HINT}", file=sys.stderr)
        return 1
    search = PostgresSearch(args.corpus, base_url=config.platform_url())
    hits = search.search(args.query, top_k=args.top_k)
    if args.json:
        from benchmax.cli._output import print_json

        print_json(hits)
        return 0
    if not hits:
        print(f"No results for {args.query!r} in corpus '{args.corpus}'.")
        return 0
    print(f"{len(hits)} hit(s) for {args.query!r} in '{args.corpus}':")
    for h in hits:
        score = h.get("score")
        score_s = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
        content = (h.get("content") or "").replace("\n", " ")[:100]
        print(f"  {score_s}  {h.get('source', '')}  {content}")
    return 0


@handle_errors
def _cmd_corpus_browse(args: argparse.Namespace) -> int:
    # Recover the wizard's "preview the corpus" step: pull the stored chunks and
    # render them in the local HTML viewer (same one `data view` uses).
    from benchmax.cli.dataview import build_view_model, write_html
    from benchmax.platform.browser import maybe_open_browser

    try:
        client = _corpus_client()
    except ImportError as exc:
        print(f"Error: {exc}. {_RAG_INSTALL_HINT}", file=sys.stderr)
        return 1

    corpora = client.list_corpora()
    target = next(
        (c for c in corpora if c.id == args.corpus or c.name == args.corpus), None
    )
    if target is None:
        have = ", ".join(c.name for c in corpora) or "(none)"
        print(f"Error: no corpus matching {args.corpus!r}. Have: {have}", file=sys.stderr)
        return 1

    # Page through chunks up to --limit (0 = all). `more` tracks whether a
    # next-page cursor remained when we stopped, so we can say so honestly.
    want = args.limit
    chunks: list = []
    cursor: str | None = None
    while True:
        if want and want > 0 and len(chunks) >= want:
            break
        page_size = 500 if not (want and want > 0) else min(500, want - len(chunks))
        page, cursor = client.list_corpus_chunks(target.id, limit=page_size, cursor=cursor)
        chunks.extend(page)
        if not cursor or not page:
            break
    more = cursor is not None

    records = [
        {"id": c.id, "content": c.content, "metadata": c.metadata, "score": c.score}
        for c in chunks
    ]
    if args.json:
        from benchmax.cli._output import print_json

        print_json(records)
        return 0

    model = build_view_model(
        records, source=f"{target.name} (chunks)", type_override="chunks", limit=0
    )
    if args.out:
        out = Path(args.out)
    else:
        fd, tmp = tempfile.mkstemp(prefix="castform-chunks-", suffix=".html")
        os.close(fd)
        out = Path(tmp)
    write_html(model, out, view=args.view)

    n = len(records)
    print(f"✓ corpus '{target.name}' — {n} chunk{'' if n == 1 else 's'}")
    print(f"  {out}")
    if more:
        print(f"  (showing first {n}; corpus has more — re-run with --limit 0 for all)")
    if not args.no_open:
        maybe_open_browser(out.resolve().as_uri())
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

    p_ls = corpus_sub.add_parser("list", help="List your corpora (and the 5-corpus cap)")
    p_ls.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_ls.set_defaults(func=_cmd_corpus_list)

    p_del = corpus_sub.add_parser("delete", help="Delete a corpus by name or id")
    p_del.add_argument("corpus", help="Corpus name or id to delete")
    p_del.add_argument(
        "--yes", action="store_true", help="Confirm deletion (required; irreversible)"
    )
    p_del.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_del.set_defaults(func=_cmd_corpus_delete)

    p_se = corpus_sub.add_parser(
        "search", help="Search a corpus to confirm retrieval works (lexical)"
    )
    p_se.add_argument("corpus", help="Corpus name")
    p_se.add_argument("query", help="Query text")
    p_se.add_argument(
        "--top-k", type=int, default=5, help="Max hits to show (default: 5)"
    )
    p_se.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_se.set_defaults(func=_cmd_corpus_search)

    p_br = corpus_sub.add_parser(
        "browse",
        help="Browse a corpus's chunks in an HTML viewer (opens in your browser)",
        description="Page a corpus's stored chunks into the local HTML viewer — recovering "
        "the wizard's 'preview corpus' step. Resolves the corpus by name or id.",
        epilog="Examples:\n"
        "  castform corpus browse my-corpus\n"
        "  castform corpus browse my-corpus --limit 0          # all chunks\n"
        "  castform corpus browse my-corpus --json > chunks.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_br.add_argument("corpus", help="Corpus name or id")
    p_br.add_argument(
        "--limit", type=int, default=200, help="Max chunks to fetch (default: 200; 0 = all)"
    )
    p_br.add_argument(
        "--view",
        choices=["auto", "table", "messages", "raw"],
        default="auto",
        help="Initial display mode (default: auto)",
    )
    p_br.add_argument("--out", help="Write the HTML here (default: a temp file)")
    p_br.add_argument(
        "--no-open", action="store_true", help="Write the file but don't open a browser"
    )
    p_br.add_argument(
        "--json", action="store_true", help="Emit raw chunk JSON instead of the viewer"
    )
    p_br.set_defaults(func=_cmd_corpus_browse)

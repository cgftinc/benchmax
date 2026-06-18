"""castform ``data`` command group (slice 1.6).

``data upload`` is a self-contained thin wrapper over the SDK's StorageClient
(GET /v1/storage/upload-url → SAS PUT). ``qa-gen`` and ``traces`` both acquire a
local ``{train,eval}_dataset.jsonl`` pair for the env loop, wrapping benchmax libs
directly (the same ones the platform wizard codegens), NOT as REST wrappers: qa-gen
over ``benchmax.rag.qa_generation``, traces over ``benchmax.traces``. Their imports
are lazy so the base ``data`` group registers without the ``[rag]`` / ``[traces]``
extras.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from benchmax.cli._client import handle_errors
from benchmax.cli._output import print_json
from benchmax.cli._providers import install_hint, provider_choices
from benchmax.platform.client import StorageClient

_RAG_INSTALL_HINT = "Install RAG support with: pip install castform[rag]"
_TRACES_INSTALL_HINT = "Install traces support with: pip install castform[traces]"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _provider_qa_source_factory(provider: str) -> tuple[Any, str]:
    """Build a ``(source_factory, corpus_label)`` for ``qa-gen --provider``.

    Reads the same ``DATA_*`` env vars the wizard / run.py use (``DATA_api_key`` plus
    per-provider resource fields). qa-gen reads stored chunks directly, so no ``embed_fn``
    is wired here. Raises ``ValueError`` with a clean message on a missing env var or a
    missing provider SDK (the caller turns that into an ``Error:`` line).
    """
    api_key = os.environ.get("DATA_api_key", "")

    def _require(name: str, value: str) -> str:
        if not value:
            raise ValueError(f"--provider {provider} needs {name} (set it as an env var)")
        return value

    if provider == "turbopuffer":
        try:
            from benchmax.rag.corpus.turbopuffer.source import TpufChunkSource
        except ImportError:
            raise ValueError(
                f"turbopuffer not installed. {install_hint('turbopuffer')}"
            ) from None
        _require("DATA_api_key", api_key)
        namespace = _require("DATA_namespace", os.environ.get("DATA_namespace", ""))
        region = os.environ.get("DATA_region") or "aws-us-east-1"
        content_field = os.environ.get("DATA_content_field") or None
        return (
            lambda cfg: TpufChunkSource(
                api_key=api_key,
                namespace=namespace,
                region=region,
                content_field=content_field,
            ),
            namespace,
        )

    if provider == "pinecone":
        try:
            from benchmax.rag.corpus.pinecone.source import PineconeChunkSource
        except ImportError:
            raise ValueError(
                f"pinecone not installed. {install_hint('pinecone')}"
            ) from None
        _require("DATA_api_key", api_key)
        index_name = _require("DATA_index_name", os.environ.get("DATA_index_name", ""))
        index_host = os.environ.get("DATA_index_host") or None
        namespace = os.environ.get("DATA_namespace", "")
        content_field = os.environ.get("DATA_content_field") or None
        return (
            lambda cfg: PineconeChunkSource(
                api_key=api_key,
                index_name=index_name,
                index_host=index_host,
                namespace=namespace,
                content_field=content_field,
            ),
            index_name,
        )

    # chroma — api_key is optional (self-hosted needs none; Chroma Cloud does)
    try:
        from benchmax.rag.corpus.chroma.source import ChromaChunkSource
    except ImportError:
        raise ValueError(
            f"chromadb not installed. {install_hint('chroma')}"
        ) from None
    collection = _require("DATA_collection_name", os.environ.get("DATA_collection_name", ""))
    tenant = os.environ.get("DATA_tenant") or None
    database = os.environ.get("DATA_database") or None
    host = os.environ.get("DATA_host") or None
    return (
        lambda cfg: ChromaChunkSource(
            collection_name=collection,
            api_key=api_key or None,
            tenant=tenant,
            database=database,
            host=host,
        ),
        collection,
    )


@handle_errors
def _cmd_data_upload(args: argparse.Namespace) -> int:
    local = Path(args.file)
    if not local.exists():
        print(f"Error: file not found: {local}", file=sys.stderr)
        return 1
    storage_path = args.path or f"datasets/cli/{local.name}"
    try:
        with StorageClient() as client:
            result = client.upload_local_file(storage_path, str(local))
    except ValueError as exc:  # unsupported file type (_get_mime_type)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print_json(result)
    else:
        print(f"✓ Uploaded {local.name} → {result.get('blobPath', storage_path)}")
    return 0


@handle_errors
def _cmd_data_qa_gen(args: argparse.Namespace) -> int:
    # Lazy import — qa-gen pulls the [rag] extra (langchain_text_splitters et al.);
    # keep them out of base `data` registration.
    try:
        from benchmax.rag.qa_generation.pipeline import run_pipeline
        from benchmax.rag.qa_generation.pipeline_config import (
            CorpusConfig,
            CorpusContextConfig,
            FilteringConfig,
            OutputConfig,
            PipelineConfig,
            PlatformConfig,
            RefinementConfig,
            TargetsConfig,
        )
    except ImportError as exc:
        print(f"Error: {exc}. {_RAG_INSTALL_HINT}", file=sys.stderr)
        return 1

    # Source: a provider corpus (DATA_* env) or an existing CGFT corpus by name/id —
    # never docs_path, which would make the lib's loader fall into the interactive
    # on_limit="prompt" create branch.
    # Lower --min-chunk-chars for small docs: the default 400-char floor rejects
    # short chunks ("No eligible chunks ..."); leave unset to keep the lib default.
    chunk_kw = {} if args.min_chunk_chars is None else {"min_chunk_chars": args.min_chunk_chars}
    source_factory = None
    if args.provider:
        try:
            source_factory, corpus_label = _provider_qa_source_factory(args.provider)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        corpus = CorpusConfig(corpus_name=corpus_label, corpus_id="", **chunk_kw)
    else:
        corpus = CorpusConfig(
            corpus_name=args.corpus_name or CorpusConfig().corpus_name,
            corpus_id=args.corpus_id or "",
            **chunk_kw,
        )

    # --fast: keep only the heuristic quality_gate (skip the 3 LLM-judge filters),
    # disable refinement retries and entity-pattern extraction (keep the linker's
    # profile enabled) → a quick, small dataset without the judge round-trips.
    if args.fast:
        filtering = FilteringConfig(filters=["quality_gate"])
        refinement = RefinementConfig(enabled=False)
        corpus_context = CorpusContextConfig(generate_entity_patterns=False)
    else:
        filtering = FilteringConfig()
        refinement = RefinementConfig()
        corpus_context = CorpusContextConfig()

    cfg = PipelineConfig(
        platform=PlatformConfig(),  # keyless seam → bearer resolved per request
        corpus=corpus,
        corpus_context=corpus_context,
        targets=TargetsConfig(total_samples=args.samples),
        filtering=filtering,
        refinement=refinement,
        # Project-convention filenames so `castform validate`/`launch` find them.
        output=OutputConfig(
            dir=args.out,
            train_jsonl="train_dataset.jsonl",
            eval_jsonl="eval_dataset.jsonl",
        ),
    )

    try:
        result = run_pipeline(cfg, source_factory=source_factory)
    except ValueError as exc:  # e.g. corpus name not found / bad config
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:  # a provider SDK is imported lazily on first use
        hint = f" {install_hint(args.provider)}" if args.provider else ""
        print(f"Error: {exc}.{hint}", file=sys.stderr)
        return 1

    paths = result.get("output_paths", {})
    stats = result.get("stats", {})
    if args.json:
        print_json({"output_paths": paths, "stats": stats})
    else:
        print(
            f"✓ qa-gen wrote {stats.get('train', 0)} train + "
            f"{stats.get('eval', 0)} eval rows"
        )
        print(f"  train: {paths.get('train_jsonl')}")
        print(f"  eval:  {paths.get('eval_jsonl')}")
        print("  Next: castform setup --template rag   (then castform validate)")
    return 0


@handle_errors
def _cmd_data_traces(args: argparse.Namespace) -> int:
    # Lazy import — keep the traces lib (tqdm et al.) out of base `data` registration.
    try:
        from benchmax.traces import TracesPipeline
        from benchmax.traces.braintrust.adapter import BraintrustTraceAdapter
    except ImportError as exc:
        print(f"Error: {exc}. {_TRACES_INSTALL_HINT}", file=sys.stderr)
        return 1

    api_key = args.api_key or os.environ.get("BT_API_KEY", "")
    if not api_key:
        print(
            "Error: no Braintrust API key. Pass --api-key or set BT_API_KEY.",
            file=sys.stderr,
        )
        return 1

    adapter = BraintrustTraceAdapter(api_key)

    # Resolve the project: explicit id wins, then a name lookup, then BT_PROJECT_ID,
    # then the sole project if there's exactly one. (HTTP errors → handle_errors.)
    project_id = args.project_id or os.environ.get("BT_PROJECT_ID", "")
    if not project_id:
        projects = adapter.list_projects()
        if args.project:
            match = [p for p in projects if p.name == args.project]
            if not match:
                names = ", ".join(p.name for p in projects) or "(none)"
                print(
                    f"Error: no Braintrust project named {args.project!r}. "
                    f"Available: {names}",
                    file=sys.stderr,
                )
                return 1
            project_id = match[0].id
        elif len(projects) == 1:
            project_id = projects[0].id
        else:
            avail = ", ".join(f"{p.name} ({p.id})" for p in projects) or "(none)"
            print(
                "Error: specify a project with --project-id / --project or "
                f"BT_PROJECT_ID. Available: {avail}",
                file=sys.stderr,
            )
            return 1

    traces, _ = adapter.fetch_traces(project_id, limit=args.limit)
    if not traces:
        print(f"Error: no traces fetched from project {project_id}.", file=sys.stderr)
        return 1

    pipeline = TracesPipeline(
        traces=traces,
        system_prompt=args.system_prompt,
        max_examples=args.max_examples,
        output_dir=None,  # write the project-convention filenames ourselves (below)
        verbose=not args.json,
    )
    try:
        result = pipeline.run()
    except ValueError as exc:  # too few examples survived filtering
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        # Detect-only: surface the FULL detected prompt + tools so the agent can
        # confirm them before committing a dataset. No files written.
        system_prompt = result.get("system_prompt", "")
        tools = result.get("tools", [])
        tool_names = [t.get("name", "?") for t in tools]
        if args.json:
            print_json({"system_prompt": system_prompt, "tools": tools})
        else:
            print(f"detected system prompt ({len(system_prompt)} chars):")
            print(system_prompt or "(none)")
            print(f"detected tools: {', '.join(tool_names) or '(none)'}")
            print("dry run — no datasets written. re-run without --dry-run to write them.")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_dataset.jsonl"
    eval_path = out_dir / "eval_dataset.jsonl"
    _write_jsonl(train_path, result["train_dataset"])
    _write_jsonl(eval_path, result["eval_dataset"])

    stats = result.get("stats", {})
    system_prompt = result.get("system_prompt", "")
    tools = result.get("tools", [])
    if args.json:
        print_json(
            {
                "output_paths": {
                    "train_jsonl": str(train_path),
                    "eval_jsonl": str(eval_path),
                },
                "stats": stats,
                "system_prompt": system_prompt,
                "tools": tools,
            }
        )
    else:
        print(
            f"✓ traces wrote {stats.get('train_count', 0)} train + "
            f"{stats.get('eval_count', 0)} eval rows"
        )
        print(f"  train: {train_path}")
        print(f"  eval:  {eval_path}")
        if system_prompt:
            preview = (
                system_prompt
                if len(system_prompt) <= 200
                else system_prompt[:200] + "…"
            )
            print(f"  detected system prompt ({len(system_prompt)} chars): {preview}")
        if tools:
            print(f"  detected tools: {', '.join(t.get('name', '?') for t in tools)}")
        # The agent authors run.py's dataset_preprocess to match the pulled rows.
        print(
            "  Rows are {prompt_messages, ground_truth, init_rollout_args} — author "
            "run.py's dataset_preprocess to match."
        )
        print("  Next: castform setup   (then castform validate)")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Attach the `data` group to the top-level subparsers."""
    data = sub.add_parser("data", help="Dataset utilities")
    data_sub = data.add_subparsers(
        dest="data_command", required=True, metavar="<subcommand>"
    )

    p_up = data_sub.add_parser("upload", help="Upload a local dataset file to storage")
    p_up.add_argument("file", help="Local file to upload (.jsonl/.json/.yaml/.pkl)")
    p_up.add_argument("--path", help="Storage path (default: datasets/cli/<filename>)")
    p_up.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_up.set_defaults(func=_cmd_data_upload)

    p_qa = data_sub.add_parser(
        "qa-gen",
        help="Generate a train/eval QA dataset from a corpus (RAG)",
    )
    src = p_qa.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--corpus-name", help="Existing corpus name (from `corpus ingest`)"
    )
    src.add_argument("--corpus-id", help="Existing corpus id")
    src.add_argument(
        "--provider",
        choices=provider_choices(),
        help="Generate from a provider corpus instead of CGFT (reads DATA_* env vars)",
    )
    p_qa.add_argument(
        "--samples", type=int, default=50, help="Target QA pairs (default: 50)"
    )
    p_qa.add_argument(
        "--min-chunk-chars",
        type=int,
        default=None,
        help="Min chars for a chunk to be eligible (default 400; lower it for "
        "small docs that otherwise yield 'No eligible chunks')",
    )
    p_qa.add_argument(
        "--fast",
        action="store_true",
        help="Skip the LLM-judge filters + refinement for a quick small set",
    )
    p_qa.add_argument(
        "--out", default=".", help="Output dir for the datasets (default: cwd)"
    )
    p_qa.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_qa.set_defaults(func=_cmd_data_qa_gen)

    p_tr = data_sub.add_parser(
        "traces",
        help="Build a train/eval dataset from recorded agent traces (Braintrust)",
    )
    p_tr.add_argument(
        "--api-key", help="Braintrust API key (default: BT_API_KEY env)"
    )
    proj = p_tr.add_mutually_exclusive_group()
    proj.add_argument(
        "--project-id", help="Braintrust project id (default: BT_PROJECT_ID env)"
    )
    proj.add_argument("--project", help="Braintrust project name (resolved to its id)")
    p_tr.add_argument(
        "--limit", type=int, help="Max traces to fetch (default: up to 1000)"
    )
    p_tr.add_argument(
        "--system-prompt",
        help="Override the system prompt (default: auto-detected from the traces)",
    )
    p_tr.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="Cap on training examples (default: 1000)",
    )
    p_tr.add_argument(
        "--out", default=".", help="Output dir for the datasets (default: cwd)"
    )
    p_tr.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect-only: print the full detected system prompt + tools and write "
        "nothing — confirm them before generating the dataset",
    )
    p_tr.add_argument("--json", action="store_true", help="Emit raw JSON")
    p_tr.set_defaults(func=_cmd_data_traces)

"""castform ``data`` command group (slice 1.6).

``data upload`` is a self-contained thin wrapper over the SDK's StorageClient
(GET /v1/storage/upload-url → SAS PUT). ``qa-gen`` wraps the benchmax qa-generation
lib (``benchmax.rag.qa_generation`` — the same pipeline the platform wizard
codegens) directly, not as a REST wrapper; its rag imports are lazy so the base
``data`` group registers without the ``[rag]`` extra. ``traces`` is still deferred.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmax.cli._client import handle_errors
from benchmax.cli._output import print_json
from benchmax.platform.client import StorageClient

_RAG_INSTALL_HINT = "Install RAG support with: pip install castform[rag]"


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

    # Resolve an EXISTING corpus by name or id — never docs_path, which would make
    # the lib's loader fall into the interactive on_limit="prompt" create branch.
    corpus = CorpusConfig(
        corpus_name=args.corpus_name or CorpusConfig().corpus_name,
        corpus_id=args.corpus_id or "",
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
        result = run_pipeline(cfg)
    except ValueError as exc:  # e.g. corpus name not found / bad config
        print(f"Error: {exc}", file=sys.stderr)
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
    p_qa.add_argument(
        "--samples", type=int, default=50, help="Target QA pairs (default: 50)"
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

"""castform ``data`` command group (slice 1.6).

``data upload`` is a self-contained thin wrapper over the SDK's StorageClient
(GET /v1/storage/upload-url → SAS PUT). ``qa-gen`` / ``traces`` are deferred:
they should call the benchmax lib pipelines directly (``benchmax.rag.qa_generation``
/ ``benchmax.traces`` — the platform wizard just codegens those same calls), and
land lib-direct alongside the 3.4 corpus-ingest work, not as REST wrappers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmax.cli._client import handle_errors
from benchmax.cli._output import print_json
from benchmax.platform.client import StorageClient


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

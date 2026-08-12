"""Prepare (and optionally launch) a PII-masking SFT run from OpenPII 1M.

Deterministic by construction: the example reads the literal ``train`` split
of the revision-pinned source in storage order, inspects exactly ``--rows``
records (no shuffle, no filtering), and maps each into one
`benchmax-sft-v1` chat row. Preparation streams the corpus and only writes a
validated local ``train.jsonl``; uploading and starting a paid training run
requires the explicit ``--launch`` flag.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from benchmax.sft import SftDataset
from castform.platform import SftTrainingConfig, TrainerClient, upload_sft_assets

SOURCE_DATASET = "ai4privacy/pii-masking-openpii-1m"
SOURCE_REVISION = "ecfdc547f4a0955600cfe6ab98ba2a162207fcc0"
SOURCE_SPLIT = "train"
DEFAULT_ROWS = 256
MAX_ROWS = 4096

SYSTEM_PROMPT = (
    "Replace personal information with the dataset's typed placeholders and "
    "preserve the remaining text."
)


def map_source_row(source: Mapping[str, object]) -> dict[str, object]:
    """Map one OpenPII record into one `benchmax-sft-v1` row."""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": source["source_text"]},
            {"role": "assistant", "content": source["masked_text"]},
        ],
        "metadata": {
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "source_uid": source["uid"],
            "language": source["language"],
        },
    }


def build_rows(source_rows: Iterable[Mapping[str, object]], limit: int) -> list[dict[str, object]]:
    """Map exactly ``limit`` records in source order; fail if fewer exist."""

    if not 1 <= limit <= MAX_ROWS:
        raise ValueError(f"row count must be between 1 and {MAX_ROWS}")
    rows: list[dict[str, object]] = []
    iterator = iter(source_rows)
    while len(rows) < limit:
        try:
            rows.append(map_source_row(next(iterator)))
        except StopIteration:
            raise RuntimeError(
                f"source stream ended after {len(rows)} records; {limit} required"
            ) from None
    return rows


def stream_source_rows() -> Iterator[Mapping[str, object]]:
    """Stream the pinned corpus without downloading it.

    Imported lazily so the offline tests and the package import never touch
    the optional ``datasets`` dependency (installed via the workspace
    ``sft-example`` group) or the network.
    """

    from datasets import load_dataset

    return iter(
        load_dataset(
            SOURCE_DATASET,
            revision=SOURCE_REVISION,
            split=SOURCE_SPLIT,
            streaming=True,
        )
    )


def prepare(output: Path, rows: int, source_rows: Iterable[Mapping[str, object]]) -> SftDataset:
    """Build, validate, and write the canonical dataset to ``output``."""

    dataset = SftDataset.from_rows(build_rows(source_rows, rows))
    output.write_bytes(dataset.to_jsonl_bytes())
    return dataset


# Stated explicitly rather than left to the platform default: the public SDK
# contract accepts ranks 32 and 64, while an unset rank inherits a legacy
# platform value outside that range. A tutorial whose documented path cannot be
# served back is worse than no tutorial.
LORA_RANK = 64


def launch(dataset: SftDataset, run_name: str) -> str:
    """Upload the validated dataset and start a paid SFT run."""

    uploaded = upload_sft_assets(dataset=dataset, run_name=run_name)
    return TrainerClient().launch_sft_run(
        assets=uploaded,
        name=run_name,
        config=SftTrainingConfig(lora_rank=LORA_RANK),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pii_masking",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    def bounded_rows(value: str) -> int:
        rows = int(value)
        if not 1 <= rows <= MAX_ROWS:
            raise argparse.ArgumentTypeError(f"--rows must be between 1 and {MAX_ROWS}")
        return rows

    parser.add_argument(
        "--rows",
        type=bounded_rows,
        default=DEFAULT_ROWS,
        help=f"records to inspect and emit, in source order (default {DEFAULT_ROWS}, "
        f"max {MAX_ROWS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train.jsonl"),
        help="local path for the validated canonical JSONL (default train.jsonl)",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="after preparing, upload the dataset and start a PAID training run",
    )
    parser.add_argument(
        "--run-name",
        help="run name for --launch (required with --launch)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.launch and not args.run_name:
        parser.error("--launch requires --run-name")

    dataset = prepare(args.output, args.rows, stream_source_rows())
    print(f"wrote {len(dataset)} validated rows to {args.output}")

    if not args.launch:
        print("re-run with --launch --run-name <name> to upload and start a paid run")
        return 0

    from castform.platform.exceptions import AuthenticationError, JobLaunchError

    try:
        run_id = launch(dataset, args.run_name)
    except JobLaunchError as exc:
        print(f"launch rejected by the platform: {exc}", file=sys.stderr)
        return 1
    except AuthenticationError as exc:
        print(
            f"not signed in ({exc}); run `castform login` and retry",
            file=sys.stderr,
        )
        return 1
    print(f"launched SFT run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

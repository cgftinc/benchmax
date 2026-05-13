"""View the Python source captured inside a bundle, without unpickling.

The bundler stores plaintext source for the env class's module (and any
``register_pickle_by_value`` modules) inside the metadata JSON under the
``sources`` key. This module reads that JSON directly — no cloudpickle, no
matching Python deps, no code execution.

Usage:
    python -m benchmax.bundle.view path/to/env_meta.json
    python -m benchmax.bundle.view path/to/env_meta.json --module my.env.module
    python -m benchmax.bundle.view path/to/env_meta.json --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional


def read_sources(metadata_path: Path) -> Dict[str, str]:
    data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    sources = data.get("sources")
    if not sources:
        raise KeyError(
            f"{metadata_path} has no 'sources' field. The bundle was likely "
            "produced by an older benchmax that didn't capture sources."
        )
    return sources


def render(sources: Dict[str, str], only: Optional[str] = None) -> str:
    if only is not None:
        if only not in sources:
            raise KeyError(
                f"Module {only!r} not in bundle. Available: {sorted(sources)}"
            )
        return sources[only]
    chunks = []
    for name, src in sources.items():
        header = f"# === {name} " + "=" * max(0, 70 - len(name))
        chunks.append(f"{header}\n{src.rstrip()}\n")
    return "\n".join(chunks)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata_path", type=Path, help="Path to the bundle metadata .json"
    )
    parser.add_argument("--module", help="Only show source for this module name")
    parser.add_argument(
        "--json", action="store_true", help="Emit the raw sources dict as JSON"
    )
    parser.add_argument(
        "--list", action="store_true", help="Just list module names in the bundle"
    )
    args = parser.parse_args(argv)

    sources = read_sources(args.metadata_path)

    if args.list:
        for name in sources:
            print(name)
        return 0
    if args.json:
        print(json.dumps(sources, indent=2, ensure_ascii=False))
        return 0
    print(render(sources, only=args.module))
    return 0


if __name__ == "__main__":
    sys.exit(main())

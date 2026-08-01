#!/usr/bin/env -S uv run --isolated --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "benchmax[rag] @ git+https://github.com/castform-ai/benchmax.git@26ec5c8afc200f3b0d51d13f7f8752c87bacc178",
# ]
# ///
"""Build the Benchmax 0.1 bundle consumed by the pre-Harbor trainer."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import environment
import search_env
from benchmax.bundle import dump_bundle, load_bundle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"


def _canonical_tool(tool: Any) -> dict[str, Any]:
    """Express historical and current tool definitions in OpenAI tool shape."""
    if is_dataclass(tool):
        value = asdict(tool)
    elif hasattr(tool, "model_dump"):
        value = tool.model_dump()
    elif hasattr(tool, "dict"):
        value = tool.dict()
    elif isinstance(tool, Mapping):
        value = dict(tool)
    else:
        raise TypeError(f"unsupported tool definition: {type(tool).__name__}")

    if {"name", "description", "input_schema"} <= value.keys():
        return {
            "type": "function",
            "function": {
                "name": value["name"],
                "description": value["description"],
                "parameters": value["input_schema"],
                "strict": False,
            },
        }
    return value


def main() -> None:
    bundle = dump_bundle(
        environment.HandbookBm25SearchEnv,
        local_modules=[environment, search_env],
    )
    loaded = load_bundle(bundle)
    if not isinstance(loaded, environment.HandbookBm25SearchEnv):
        raise TypeError(f"bundle smoke-load returned {type(loaded).__name__}")
    tools = asyncio.run(loaded.list_tools())
    contract = {
        "system_prompt": loaded.system_prompt,
        "tools": [_canonical_tool(tool) for tool in tools],
        "format_probe": loaded._format_results(
            [
                {
                    "content": "A" * 1700,
                    "source": "content/handbook/example.md",
                    "metadata": {"file": "content/handbook/example.md", "section": "Example"},
                    "score": 1.25,
                }
            ]
        ),
        "source_probe": environment._canonical_source_id(
            "https://gitlab.com/content/handbook/example.md?plain=1"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    cls_path = OUT / "env-cls.pkl"
    metadata_path = OUT / "env-metadata.json"
    cls_path.write_bytes(bundle.pickled)
    metadata_path.write_bytes(bundle.metadata.to_json_bytes())
    contract_path = OUT / "contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "arm": "pre_harbor",
        "castform_ref": "466e947c25eb137e1778d5f8a33c87cc906b729c",
        "benchmax_ref": "26ec5c8afc200f3b0d51d13f7f8752c87bacc178",
        "env_cls_sha256": hashlib.sha256(bundle.pickled).hexdigest(),
        "env_metadata_sha256": hashlib.sha256(bundle.metadata.to_json_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }
    (OUT / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

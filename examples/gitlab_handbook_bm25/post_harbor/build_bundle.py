#!/usr/bin/env -S uv run --isolated --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "benchmax @ git+https://github.com/castform-ai/benchmax.git@c19b4addb767a745bc8f75e7167afd3958d4dfa3#subdirectory=packages/benchmax",
#   "castform @ git+https://github.com/castform-ai/benchmax.git@c19b4addb767a745bc8f75e7167afd3958d4dfa3#subdirectory=packages/castform",
# ]
# ///
"""Build the current Benchmax bundle consumed by the gateway trainer."""

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
CASTFORM_RUNTIME = (
    "castform @ git+https://github.com/castform-ai/benchmax.git"
    "@c19b4addb767a745bc8f75e7167afd3958d4dfa3"
    "#subdirectory=packages/castform"
)


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
        pip_dependencies=[CASTFORM_RUNTIME],
        local_modules=[environment, search_env],
    )
    loaded = load_bundle(bundle)
    if not isinstance(loaded, environment.HandbookBm25SearchEnv):
        raise TypeError(f"bundle smoke-load returned {type(loaded).__name__}")
    if loaded.max_tool_calls != environment.EFFECTIVE_MAX_TOOL_CALLS:
        raise ValueError(
            "bundle lost the pre-Harbor-compatible effective tool budget: "
            f"{loaded.max_tool_calls} != {environment.EFFECTIVE_MAX_TOOL_CALLS}"
        )
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
        "arm": "post_harbor",
        "castform_ref": "1aed982f2621d87438b3ad095818440a4ea930c5",
        "benchmax_ref": "c19b4addb767a745bc8f75e7167afd3958d4dfa3",
        "advertised_max_search_calls": environment.MAX_SEARCH_CALLS,
        "effective_max_tool_calls": environment.EFFECTIVE_MAX_TOOL_CALLS,
        "tool_budget_note": (
            "Current gateway enforcement is set to 3 executed calls to match "
            "the pre-Harbor trainer's effective behavior for its declared limit of 4."
        ),
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

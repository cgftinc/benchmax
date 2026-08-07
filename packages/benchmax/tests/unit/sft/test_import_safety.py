"""`benchmax.sft` must stay importable without optional/heavy dependencies."""

from __future__ import annotations

import subprocess
import sys

_SMOKE = """
import sys
import benchmax.sft

forbidden = [name for name in ("openai", "cloudpickle", "httpx", "harbor") if name in sys.modules]
assert not forbidden, f"benchmax.sft pulled in heavy modules: {forbidden}"

dataset = benchmax.sft.SftDataset.from_rows(
    [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
)
assert dataset.to_jsonl_bytes().endswith(b"\\n")
"""


def test_sft_imports_without_heavy_dependencies() -> None:
    subprocess.run([sys.executable, "-c", _SMOKE], check=True, capture_output=True, timeout=60)

"""Tests for env source capture in dump_bundle.

The wizard builds env classes via ``exec()`` into an in-memory namespace, so
``inspect.getsource`` can't recover their text. ``dump_bundle`` accepts an
``env_class_source`` override for exactly this case; these tests pin both the
override path and the introspection gap it exists to paper over.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from benchmax.bundle import _get_source, dump_bundle
from benchmax.envs.base_env import BaseEnv
from benchmax.envs.types import Messages, ToolDefinition

# Defined at module scope so cloudpickle can pickle it by-value when the test
# module is registered as a local module (dump_bundle enforces this).
_TEST_MODULE = sys.modules[__name__]


class MinimalEnv(BaseEnv):
    """Minimal valid BaseEnv subclass for bundling tests."""

    system_prompt = "test"

    async def list_tools(self) -> List[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args: Any) -> Any:
        return None

    async def compute_reward(
        self,
        rollout_id: str,
        messages: Messages,
        task: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, float]:
        return {"score": 0.0}


def test_source_introspected_by_default() -> None:
    """With no override, source is recovered from the class via inspect."""
    bundle = dump_bundle(MinimalEnv, local_modules=[_TEST_MODULE])
    assert bundle.metadata.env_class_source is not None
    assert "class MinimalEnv" in bundle.metadata.env_class_source


def test_source_override_wins_over_introspection() -> None:
    """An explicit override is recorded verbatim, even when introspection
    would have succeeded."""
    bundle = dump_bundle(
        MinimalEnv,
        local_modules=[_TEST_MODULE],
        env_class_source="# handed in by the caller\nclass Whatever: ...\n",
    )
    assert bundle.metadata.env_class_source == (
        "# handed in by the caller\nclass Whatever: ...\n"
    )


def test_exec_defined_class_has_no_introspectable_source() -> None:
    """Regression: a class produced by exec() (as the wizard does) has no
    source file, so _get_source returns None — which is why the override
    parameter is needed."""
    source_code = (
        "from benchmax.envs.base_env import BaseEnv\n"
        "class GeneratedEnv(BaseEnv):\n"
        "    system_prompt = 'x'\n"
        "    async def list_tools(self): return []\n"
        "    async def run_tool(self, *a, **k): return None\n"
        "    async def compute_reward(self, *a, **k): return {'score': 0.0}\n"
    )
    namespace: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__modal_generated_env__",
    }
    exec(source_code, namespace)
    env_class = namespace["GeneratedEnv"]

    assert _get_source(env_class) is None

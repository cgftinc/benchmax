from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

RewardFunc = Callable[[str, str, Any], float]
OutputParser = Callable[[str], str]


@dataclass
class ToolDefinition:
    name: str
    impl: Callable[..., Any]
    description: str = ""
    mcp_native: bool = True
    parser: Optional[OutputParser] = None
    property_schema: Optional[Dict[str, Any]] = None

    def serialize(self) -> str:
        return f"### {self.name}\n{self.description.strip()}".strip()


class BaseSandbox:
    """Lightweight, framework‑agnostic RL environment for MCP tool use.

    The base class is intentionally *agnostic* about which tools are allowed –
    it simply provides registration helpers and a convenient reward interface.
    Any allow‑list logic should be implemented by subclasses (e.g. `MCPSandbox`).
    """

    def __init__(
        self
    ) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._reward_funcs: List[RewardFunc] = []
        self.state: Dict[str, Any] = {}

    # — Tools ————————————————————————————————————————————————
    def register_tool(
        self,
        name: str,
        impl: Callable[..., Any],
        *,
        description: str = "",
        parser: Optional[OutputParser] = None,
        mcp_native: bool = False,
        property_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Sub‑classes may override this to enforce an allow‑list.
        self._tools[name] = ToolDefinition(name, impl, description, mcp_native, parser, property_schema)

    def run_tool(self, name: str, *args, parse_output: bool = True, rollout_id: Optional[str] = None, **kwargs):
        if name not in self._tools:
            raise KeyError(name)

        # Build the payload that will be forwarded to the concrete tool impl
        if args and isinstance(args[0], dict):
            payload = args[0].copy()
        else:
            payload = kwargs.copy()

        if rollout_id is not None:
            payload["rollout_id"] = rollout_id

        raw = self._tools[name].impl(payload)
        parser = self._tools[name].parser
        return parser(raw) if (parser and parse_output) else raw

    def get_tool_definitions(self) -> str:
        return "\n\n".join(t.serialize() for t in self._tools.values())

    # — Rewards ————————————————————————————————————————————————
    def add_reward_func(self, fn: RewardFunc) -> None:
        self._reward_funcs.append(fn)
    
    def get_reward_funcs(self) -> List[RewardFunc]:
        """Return a list of all registered reward functions."""
        return self._reward_funcs

    def compute_reward(
        self, prompt: str, completion: str, ground_truth: Any, **kw
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for fn in self._reward_funcs:
            try:
                out[fn.__name__] = float(fn(prompt, completion, ground_truth, **kw))
            except Exception as exc:
                out[fn.__name__] = float("nan")
                print(f"[WARN] reward {fn.__name__} failed: {exc}", file=sys.stderr)
        return out

    # — State ————————————————————————————————————————————————————
    def initialize_state(self, **kw) -> None:
        self.state.clear()
        self.state.update(kw)

    def snapshot_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tools={list(self._tools)})"

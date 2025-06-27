"""
Adapter utilities for bridging a lightweight `BaseSandbox` (RL‑style tool environment)
into the **Environment** API used by Verifiers (``ToolEnv``/``MultiTurnEnv``).

The key responsibilities of the adapter are:
1.  **Expose every registered sandbox tool** to the LLM via the XML‑tool calling
    protocol expected by ``ToolEnv``.  Stub functions are generated on‑the‑fly so
    that tool metadata (name & description) flows into the system prompt.
2.  **Propagate a unique *rollout‑id* through the entire call‑stack**.
    *   A context‑local variable (``contextvars.ContextVar``) guarantees thread /
        task‑safety, so multiple concurrent rollouts on the **same** environment
        instance never collide.
    *   ``SandboxVerifierEnv.rollout`` seeds the variable with a fresh UUID **per
        rollout** (or honours a caller‑supplied one).  ``call_tool`` then passes
        that value down to ``sandbox.run_tool``.
3.  **Keep the adapter completely stateless aside from the current context**, so
    a single environment object can safely serve many parallel evaluations.

Usage
-----
```python
from envs.mcp_sandbox import MCPSandbox  # or any other BaseSandbox subclass
from adapters import sandbox_to_verifiers

sandbox = MCPSandbox()
verifiers_env = sandbox_to_verifiers(sandbox)
```
The returned ``SandboxVerifierEnv`` instance is a drop‑in replacement wherever a
``ToolEnv`` / ``MultiTurnEnv`` is expected.
"""

from __future__ import annotations

from envs.base_sandbox import BaseSandbox

from contextvars import ContextVar
import inspect
import uuid
from typing import Any, Callable, List

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.envs.tool_env import ToolEnv
from verifiers import RewardFunc
from verifiers.parsers import XMLParser
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE
from verifiers.rubrics import Rubric

# ----------------------------------------------------------------------------
# Context‑local rollout id
# ----------------------------------------------------------------------------
_CURRENT_ROLLOUT_ID: ContextVar[str | None] = ContextVar("current_rollout_id", default=None)


# ----------------------------------------------------------------------------
# Helper: stub‑function factory
# ----------------------------------------------------------------------------

def _make_tool_stub(name: str, description: str, sandbox_runner: Callable[..., Any]) -> Callable[..., Any]:
    """Return a function that forwards kwargs into *sandbox_runner* and has an
    informative signature / docstring for schema‑inference.
    """

    # Inspect the original impl to copy its signature where possible.  If the
    # impl is not introspectable (e.g. built‑in), fall back to **kwargs only.
    try:
        orig_sig = inspect.signature(sandbox_runner)
    except (TypeError, ValueError):
        orig_sig = inspect.Signature([inspect.Parameter("**kwargs", inspect.Parameter.VAR_KEYWORD)])

    # We prepend no new parameters – just replicate the existing signature.

    def _stub(**kwargs):  # type: ignore[override]
        """Auto‑generated adapter for sandbox tool **{name}**.

        {description}
        """
        # Fetch the rollout‑id from the *current* logical context and forward it.
        rollout_id = _CURRENT_ROLLOUT_ID.get()
        return sandbox_runner(**kwargs, rollout_id=rollout_id)

    _stub.__name__ = name
    _stub.__qualname__ = name  # keeps nice representation in schema docs
    _stub.__signature__ = orig_sig  # type: ignore[attr-defined]
    _stub.__doc__ = description or f"Sandbox tool wrapper for `{name}`."
    return _stub

class SandboxRubric(Rubric):
    """Wrap all reward functions registered in a ``BaseSandbox`` so they are
    compatible with the Verifiers evaluation pipeline.

    By default each sandbox reward gets weight **1.0** and the standard XML
    format‑compliance reward from ``XMLParser`` is added with *format_weight*
    (default **0.2**).
    """

    def __init__(
        self,
        sandbox: BaseSandbox,  # quoted to avoid hard import requirement
        *,
        parser: XMLParser | None = None,
        extra_funcs: List[RewardFunc] | None = None,
        extra_weights: List[float] | None = None,
    ) -> None:
        if parser is None:
            parser = XMLParser(fields=["think", ("tool", "answer")])

        super().__init__(funcs=[], weights=[], parser=parser)

        # Add sandbox‑native rewards ------------------------------------------------
        for fn in sandbox.get_reward_funcs():
            self.add_reward_func(fn, weight=1.0)

        # Add any caller‑supplied extra rewards ------------------------------------
        extra_funcs = extra_funcs or []
        extra_weights = extra_weights or []
        for fn, w in zip(extra_funcs, extra_weights, strict=False):
            self.add_reward_func(fn, weight=w)


# ----------------------------------------------------------------------------
# Main factory
# ----------------------------------------------------------------------------

def get_verifiers_environment(
    sandbox: BaseSandbox,  # quoted‑type‑hint to avoid hard dependency
    *,
    system_prompt: str | None = None,
    format_prompt: bool = True,
    parser: XMLParser = XMLParser(fields=["think", ("tool", "answer")]),
    env_parser: XMLParser = XMLParser(fields=["result"]),
    max_turns: int = 10,
) -> MultiTurnEnv:
    """Create a **ToolEnv**‑compatible environment that forwards tool calls into
    *sandbox*.

    Parameters
    ----------
    sandbox : BaseSandbox
        The RL‑style sandbox instance whose tools & reward functions you want
        to expose.
    system_prompt : str, optional
        Custom system prompt template.  Falls back to
        ``verifiers.prompts.DEFAULT_TOOL_PROMPT_TEMPLATE`` if *None*.
    format_prompt : bool, default True
        Whether to auto‑inject the tool descriptions into *system_prompt*.
    parser / env_parser : XMLParser
        XML parsers for both agent (LLM) messages and environment replies.
    max_turns : int, default 10
        Maximum number of dialogue turns per rollout.
    """

    # ---------------------------------------------------------------------
    # Step 1: build stub functions for *every* registered sandbox tool
    # ---------------------------------------------------------------------
    tool_stubs: List[Callable[..., Any]] = []
    for _tool_name, _tool_def in sandbox._tools.items():
        stub = _make_tool_stub(
            name=_tool_name,
            description=_tool_def.description,
            sandbox_runner=lambda rollout_id=None, _name=_tool_name, **kw: sandbox.run_tool(  # noqa: E501
                _name, **kw, rollout_id=rollout_id
            ),
        )
        tool_stubs.append(stub)

    # ---------------------------------------------------------------------
    # Step 2: dynamic subclass that tweaks ``call_tool`` & ``rollout``
    # ---------------------------------------------------------------------

    class SandboxVerifierEnv(ToolEnv):  # pylint: disable=too-few-public-methods
        """ToolEnv‑compatible adapter around *sandbox*."""

        def __init__(self):
            super().__init__(
                tools=tool_stubs,
                system_prompt=system_prompt or DEFAULT_TOOL_PROMPT_TEMPLATE,
                format_prompt=format_prompt,
                parser=parser,
                env_parser=env_parser,
                max_turns=max_turns,
            )

        # --------------------------------------------------------------
        # Forward tool calls (with rollout_id) directly into sandbox
        # --------------------------------------------------------------
        def call_tool(self, tool_json: str, max_chars: int = 1024, **kwargs: Any) -> str:  # noqa: D401, N803
            import json

            try:
                cmd = json.loads(tool_json)
                if not isinstance(cmd, dict):
                    return "Error: Tool command must be a JSON object."

                tool_name = cmd.get("name")
                if not tool_name:
                    return "Error: Missing 'name' field in tool command."

                args = cmd.get("args", {})
                if not isinstance(args, dict):
                    return f"Error: 'args' for {tool_name} must be a JSON object."

                # Retrieve rollout id from context
                rid = _CURRENT_ROLLOUT_ID.get()
                result = sandbox.run_tool(tool_name, **args, rollout_id=rid)

                result_str = str(result)
                if 0 < max_chars < len(result_str):
                    result_str = result_str[:max_chars] + "..."
                return result_str
            except Exception as exc:  # pylint: disable=broad-except
                return f"Error: {exc}"

        # --------------------------------------------------------------
        # Inject *rollout‑id* into context for every rollout
        # --------------------------------------------------------------
        def rollout(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            rid = kwargs.pop("rollout_id", None) or str(uuid.uuid4())
            token = _CURRENT_ROLLOUT_ID.set(rid)
            try:
                return super().rollout(*args, **kwargs)
            finally:
                _CURRENT_ROLLOUT_ID.reset(token)

    # ---------------------------------------------------------------------
    # Done – return a ready‑to‑use environment instance
    # ---------------------------------------------------------------------
    return SandboxVerifierEnv()

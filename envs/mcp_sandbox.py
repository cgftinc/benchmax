from __future__ import annotations

import contextlib
import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from envs.base_sandbox import BaseSandbox, OutputParser
from utils import AsyncLoopThread


class MCPSandbox(BaseSandbox):
    """Boot one or more MCP servers (via *fastmcp*) and expose their tools **as** an
    `RLSandbox` instance.  All lifetime management is explicit; no `async with`
    is needed by the caller.

    Parameters
    ----------
    mcp_config : str | Path | dict
        Same format you would pass to *fastmcp.Client* (path to *mcp.json*,
        already‑loaded dict, etc.).  Entries with `command`/`args` are launched
        as local subprocesses speaking MCP over stdio; pure `url` entries are
        treated as remote.
    allowed_tool_list : list[str] | None, optional
        If provided, only tools whose names are present in the list can be
        registered or executed via this sandbox.
    output_parsers : dict[str, OutputParser] | None, optional
        Optional map of tool‑name → custom parser functions.
    """

    # ---------------------------------------------------------------------
    def __init__(
        self,
        mcp_config: Union[str, Path, Dict[str, Any]],
        *,
        allowed_tool_list: Optional[List[str]] = None,
        output_parsers: Optional[Dict[str, OutputParser]] = None,
    ) -> None:
        # 0. Store allow‑list *before* initialising the base class.
        self.allowed_tool_list = allowed_tool_list.copy() if allowed_tool_list else None
        self.output_parsers = output_parsers.copy() if output_parsers else {}

        # 1. Initialise the base sandbox (no allow‑list in base).
        super().__init__()

        # 2. Spin up the background event‑loop.
        self._loop_thread = AsyncLoopThread()

        # 3. Create an *open* fastmcp.Client inside that loop.
        from fastmcp import Client  # local import to avoid hard dep for users not calling this

        async def _open_client(cfg):
            client_obj = Client(cfg)
            await client_obj.__aenter__()  # manually enter async context – keep alive
            return client_obj

        if isinstance(mcp_config, (str, Path)):
            cfg_payload = mcp_config  # pass through (fastmcp will parse file)
        else:
            cfg_payload = copy.deepcopy(mcp_config)

        self._client = self._loop_thread.submit(_open_client(cfg_payload)).result()

        # 4. Discover and register every tool with thin sync wrappers.
        self._register_all_tools()

        # 5. Shutdown guard.
        self._shutdown = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register_all_tools(self):
        """Query the client for its tool inventory and expose them through *self*."""

        async def _list_tools():
            return await self._client.list_tools()

        tool_objs = self._loop_thread.submit(_list_tools()).result()

        for t in tool_objs:
            if self.allowed_tool_list and t.name not in self.allowed_tool_list:
                continue  # skip – not in allow‑list

            super().register_tool(
                t.name,
                impl=self._make_sync_tool(t.name),
                description=t.description or "",
                parser=self.output_parsers.get(t.name),
                mcp_native=True,
                property_schema=t.inputSchema or {},
            )
    
    def _make_sync_tool(self, tool_name: str) -> Callable[..., Any]:
        """Return a *synchronous* wrapper around `client.call_tool` for one tool."""

        def _impl(*args, rollout_id: Optional[str] = None, **kwargs):
            # Accept either positional dict payload or keyword args.
            if args and isinstance(args[0], dict):
                payload = args[0]
            else:
                payload = kwargs  # tolerate kwargs instead of dict
            
            if rollout_id is not None:          # <- same injection
                payload["rollout_id"] = rollout_id
            
            async def _call():
                return await self._client.call_tool(tool_name, payload)

            fut = self._loop_thread.submit(_call())
            result = fut.result()

            # fastmcp returns a list of Content objects; grab `.text` if simple.
            if result and hasattr(result[0], "text"):
                return result[0].text

            return result

        _impl.__name__ = tool_name  # prettier repr
        return _impl

    def register_tool(
        self,
        name: str,
        impl: Callable[..., Any],
        *,
        description: str = "",
        parser: Optional[OutputParser] = None,
        mcp_native: bool = True,
    ) -> None:
        """Override to enforce the allow‑list at registration time."""
        if self.allowed_tool_list and name not in self.allowed_tool_list:
            raise ValueError(f"Tool '{name}' is not allowed: {self.allowed_tool_list}")
        super().register_tool(
            name,
            impl,
            description=description,
            parser=parser,
            mcp_native=mcp_native,
        )

    def shutdown(self):
        """Gracefully close MCP transports, terminate any child processes, and stop
        the background event loop.  Safe to call multiple times."""
        if self._shutdown:
            return

        async def _close_client():
            try:
                await self._client.__aexit__(None, None, None)
            finally:
                # Let fastmcp handle child process cleanup.
                pass

        # Run __aexit__ on the loop and wait.
        self._loop_thread.submit(_close_client()).result()

        # Stop the loop thread itself.
        self._loop_thread.stop()

        self._shutdown = True

    def __del__(self):
        with contextlib.suppress(Exception):
            self.shutdown()
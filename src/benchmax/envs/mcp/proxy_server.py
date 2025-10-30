import os
import sys
import shutil
import uuid
import yaml
import asyncio
import argparse
import psutil
import random

from typing import Any, Awaitable, Callable, Dict, Union, Optional, Tuple, List
from pathlib import Path
from functools import wraps

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.auth import AccessToken
from fastmcp import FastMCP, Client
from starlette.requests import Request
from starlette.responses import PlainTextResponse, FileResponse, JSONResponse, Response
from starlette.datastructures import UploadFile

from reward_fn import reward_functions  # type: ignore

RewardFunction = Callable[..., Union[float, Awaitable[float]]]
DEFAULT_API_SECRET = "dev_default_api_secret_please_change_me_32chars!"


# ---------------- Utility Functions ---------------- #
def setup_workspace(base_dir: Path) -> Path:
    """Create a unique workspace directory."""
    ws = (base_dir / uuid.uuid4().hex).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def load_config(config_path: Path, workspace: Path) -> Dict[str, Any]:
    """Load YAML config and inject workspace paths."""
    with open(config_path, "r") as f:
        content = f.read().replace(
            "${{ sync_workdir }}", str(Path(__file__).resolve().parent)
        )
    config: Dict[str, Any] = yaml.safe_load(content)
    if "mcpServers" in config:
        for server in config["mcpServers"].values():
            server["cwd"] = str(workspace)
    return config


# ---------------- Auth Decorator ---------------- #
def require_auth(
    func: Callable[..., Awaitable[Response]],
) -> Callable[..., Awaitable[Response]]:
    """Require JWT authentication using FastMCP's JWTVerifier."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Response:
        request: Optional[Request] = None
        self: Optional[ProxyServer] = None

        # Extract self and request
        if len(args) >= 2 and isinstance(args[0], ProxyServer):
            self = args[0]
            request = args[1]
        else:
            request = args[0] if args else kwargs.get("request")
            self = None

        if not request:
            return JSONResponse(
                {"error": "server_error", "detail": "Request object not found"},
                status_code=500,
            )

        if not self or not self.jwt_verifier:
            return JSONResponse(
                {"error": "server_error", "detail": "JWT verifier not initialized"},
                status_code=500,
            )

        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            return JSONResponse(
                {"error": "unauthorized", "detail": "Missing Authorization header"},
                status_code=401,
            )

        # Remove "Bearer " prefix if present
        token = auth_header.replace("Bearer ", "")

        # Use FastMCP's JWTVerifier to verify token
        access_token: Optional[AccessToken] = await self.jwt_verifier.verify_token(
            token
        )

        if not access_token:
            print("[WARN] JWT verification failed")
            return JSONResponse(
                {"error": "unauthorized", "detail": "Invalid or expired token"},
                status_code=401,
            )

        # Store AccessToken in request state for use in endpoint
        # AccessToken includes: token, client_id, scopes, expires_at, claims
        request.state.access_token = access_token

        return await func(*args, **kwargs)

    return wrapper


# ---------------- Memory Check Decorator ---------------- #
def with_memory_check(
    func: Callable[..., Awaitable[Response]],
) -> Callable[..., Awaitable[Response]]:
    """Check memory after each HTTP request."""

    @wraps(func)
    async def wrapper(
        self: "ProxyServer", request: Request, *args: Any, **kwargs: Any
    ) -> Response:
        response = await func(self, request, *args, **kwargs)
        await self.check_memory_usage()
        return response

    return wrapper


# ---------------- Memory Middleware ---------------- #
class MemoryMonitorMiddleware(Middleware):
    """
    Middleware that checks system memory after each MCP request or tool call.
    Can also be extended to log or reset the server if memory usage is too high.
    """

    def __init__(
        self,
        server: "ProxyServer",
        threshold: float = 0.9,
        reset_prob: float = 0.25,
    ) -> None:
        self.server = server
        self.threshold = threshold
        self.reset_prob = reset_prob

    async def on_message(
        self,
        context: MiddlewareContext,
        call_next: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        result = await call_next(context)
        await self.server.check_memory_usage()
        return result


# ---------------- Proxy Server ---------------- #
class ProxyServer:
    def __init__(
        self,
        base_dir: Union[str, Path],
        host: str = "0.0.0.0",
        port: int = 8080,
        enable_background_memory_monitor: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.port = port
        self.workspace: Optional[Path] = None
        self.client: Optional[Client] = None
        self.proxy: Optional[FastMCP] = None
        self.jwt_verifier: Optional[JWTVerifier] = None
        self.config_path = Path(__file__).parent / "mcp_config.yaml"
        self.enable_background_memory_monitor = enable_background_memory_monitor

    # ---------------- Memory ---------------- #
    async def check_memory_usage(
        self, threshold: float = 0.85, reset_prob: float = 1.0
    ) -> None:
        usage = psutil.virtual_memory().percent / 100.0
        if usage > threshold:
            if random.random() < reset_prob:
                print(
                    f"[WARN] Memory {usage * 100:.1f}% > {threshold * 100:.0f}%, triggering reset"
                )
                sys.stdout.flush()
                await self._server_reset()
            else:
                print(
                    f"[INFO] Memory high ({usage * 100:.1f}%), skipping reset this time"
                )
                sys.stdout.flush()

    async def monitor_memory(self, interval: int = 60) -> None:
        """Optional slow background monitor for idle memory leaks."""
        while True:
            await self.check_memory_usage()
            await asyncio.sleep(interval)

    # ---------------- Setup ---------------- #
    async def _setup(self) -> None:
        """Initialize workspace, MCP client, and proxy server."""
        self.workspace = setup_workspace(self.base_dir)
        config = load_config(self.config_path, self.workspace)

        self.client = Client(config)
        await self.client._connect()

        # Get API secret for JWT verification
        api_secret = os.getenv("API_SECRET", "")

        # Create single JWT verifier instance for all authentication
        self.jwt_verifier = JWTVerifier(
            public_key=api_secret,
            issuer="mcp-client",
            audience="mcp-proxy-server",
            algorithm="HS256",
        )

        # Create proxy with JWT authentication for /mcp endpoint
        # This uses the same jwt_verifier instance
        self.proxy = FastMCP.as_proxy(self.client, name="proxy", auth=self.jwt_verifier)

        # Register custom HTTP endpoints (these use our require_auth decorator
        # which internally uses the same jwt_verifier)
        self.proxy.custom_route("/health", methods=["GET"])(self._health)
        self.proxy.custom_route("/upload", methods=["POST"])(self._upload)
        self.proxy.custom_route("/download", methods=["GET"])(self._download)
        self.proxy.custom_route("/compute_reward", methods=["POST"])(
            self._compute_reward
        )
        self.proxy.custom_route("/reset", methods=["POST"])(self._reset)

        # Add memory monitoring middleware
        self.proxy.add_middleware(MemoryMonitorMiddleware(self))

        # Background memory monitor
        if self.enable_background_memory_monitor:
            asyncio.create_task(self.monitor_memory())

    # ---------------- Endpoints ---------------- #
    async def _health(self, request: Request) -> PlainTextResponse:
        return PlainTextResponse("OK")

    @require_auth
    @with_memory_check
    async def _upload(self, request: Request) -> PlainTextResponse:
        if not self.workspace:
            return PlainTextResponse("No workspace available", 500)

        # Access JWT claims from AccessToken
        access_token: AccessToken = request.state.access_token
        rollout_id = access_token.claims.get("rollout_id", "unknown")
        client_id = access_token.client_id

        print(f"[{rollout_id}] Processing upload request from client: {client_id}")

        form = await request.form()
        uploaded: List[str] = []

        for file in form.values():
            if isinstance(file, UploadFile) and file.filename:
                dest = self.workspace / file.filename
                with open(dest, "wb") as f:
                    f.write(await file.read())
                uploaded.append(file.filename)

        if not uploaded:
            return PlainTextResponse("No files uploaded", 400)

        return PlainTextResponse(f"Uploaded: {', '.join(uploaded)}")

    @require_auth
    @with_memory_check
    async def _download(self, request: Request) -> Response:
        if not self.workspace:
            return PlainTextResponse("No workspace", 500)

        # Access JWT claims from AccessToken
        access_token: AccessToken = request.state.access_token
        rollout_id = access_token.claims.get("rollout_id", "unknown")

        print(f"[{rollout_id}] Processing download request")

        file_path = request.query_params.get("file_path")
        if not file_path:
            return PlainTextResponse("file_path required", 400)

        full_path = self.workspace / file_path
        if not full_path.exists() or not full_path.is_file():
            return PlainTextResponse("File not found", 404)

        return FileResponse(str(full_path), filename=full_path.name)

    @require_auth
    @with_memory_check
    async def _compute_reward(self, request: Request) -> JSONResponse:
        """Compute reward scores using both sync and async reward functions passed as a dict."""
        # Access JWT claims from AccessToken
        access_token: AccessToken = request.state.access_token
        rollout_id = access_token.claims.get("rollout_id", "unknown")

        print(f"[{rollout_id}] Processing compute_reward request")

        try:
            data: Dict[str, Any] = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "invalid_request", "detail": "Invalid JSON payload"},
                status_code=400,
            )

        completion = data.get("completion")
        ground_truth = data.get("ground_truth")

        if completion is None or ground_truth is None:
            return JSONResponse(
                {
                    "error": "missing_fields",
                    "detail": "Both 'completion' and 'ground_truth' are required",
                },
                status_code=400,
            )

        kwargs: Dict[str, Any] = {
            "completion": completion,
            "ground_truth": ground_truth,
            "workspace": self.workspace,
            "mcp_client": self.client,
            **{
                k: v for k, v in data.items() if k not in ("completion", "ground_truth")
            },
        }

        async def _call_reward(name: str, func: RewardFunction) -> Tuple[str, float]:
            """Call reward function (sync or async) safely and return its result."""
            try:
                result = func(**kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                return (
                    (name, float(result))
                    if isinstance(result, (float, int))
                    else (name, float("nan"))
                )
            except Exception as e:
                print(f"[WARN] reward {name} failed: {e}")
                return (name, float("nan"))

        rf: Dict[str, RewardFunction] = reward_functions or {}

        try:
            tasks = [_call_reward(name, func) for name, func in rf.items()]
            results_list: List[Tuple[str, float]] = await asyncio.gather(*tasks)
            results: Dict[str, float] = dict(results_list)
            return JSONResponse(results)
        except Exception as e:
            print(f"Reward computation failed: {e}")
            return JSONResponse(
                {
                    "error": "internal_error",
                    "detail": f"Reward computation failed: {str(e)}",
                },
                status_code=500,
            )

    @require_auth
    async def _reset(self, request: Request) -> PlainTextResponse:
        # Access JWT claims from AccessToken
        access_token: AccessToken = request.state.access_token
        rollout_id = access_token.claims.get("rollout_id", "unknown")

        print(f"[{rollout_id}] Processing reset request")
        return await self._server_reset()

    # ---------------- Reset / Cleanup ---------------- #
    async def _server_reset(self) -> PlainTextResponse:
        async def do_reset() -> None:
            await asyncio.sleep(0.1)
            print("[INFO] Resetting server...")
            sys.stdout.flush()
            os.execv(sys.executable, [sys.executable] + sys.argv)

        if self.client:
            await self.client._disconnect()

        self.cleanup_workspace()
        asyncio.create_task(do_reset())
        return PlainTextResponse("Server reset scheduled")

    def cleanup_workspace(self) -> None:
        if self.workspace and self.workspace.exists():
            shutil.rmtree(self.workspace)

    # ---------------- Run ---------------- #
    async def start(self) -> None:
        await self._setup()
        if self.proxy:
            await self.proxy.run_async(transport="http", host=self.host, port=self.port)


# ---------------- Main ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastMCP Proxy Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--base-dir", type=str, default="workspace")
    parser.add_argument("--disable-background-memory-monitor", action="store_true")

    args = parser.parse_args()

    if "API_SECRET" not in os.environ or not os.environ["API_SECRET"]:
        os.environ["API_SECRET"] = DEFAULT_API_SECRET
        print(
            f"[INFO] No API_SECRET provided. Using default: {os.environ['API_SECRET']}"
        )
    else:
        api_secret = os.environ["API_SECRET"]
        print(f"[INFO] API_SECRET loaded (length: {len(api_secret)})")
        if len(api_secret) < 32:
            print("[WARN] API_SECRET should be at least 32 characters for security")

    server = ProxyServer(
        base_dir=args.base_dir,
        host=args.host,
        port=args.port,
        enable_background_memory_monitor=not args.disable_background_memory_monitor,
    )

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        server.cleanup_workspace()

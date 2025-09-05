import yaml
import uuid
import os
import asyncio
import shutil
from typing import Optional
from pathlib import Path
from functools import wraps
from fastmcp import FastMCP, Client
from starlette.requests import Request
from starlette.responses import PlainTextResponse, FileResponse
from starlette.datastructures import UploadFile


# Utility Functions
def setup_workspace(base_dir: Path = Path("workspace")) -> Path:
    """Create and return a new workspace directory using a UUID."""
    workspace = base_dir / uuid.uuid4().hex
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def prepare_config(config: dict, workspace: Path) -> dict:
    """Prepare config with workspace path for MCP servers."""
    config = config.copy()
    if "mcpServers" in config:
        for server in config["mcpServers"].values():
            server["cwd"] = str(workspace)
    return config


# Authentication Decorator
def require_auth(func):
    """Decorator to require Authorization header for endpoint access."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Handle both bound methods (self, request) and unbound functions (request)
        if len(args) == 2 and hasattr(args[0], '__class__'):
            # Bound method: (self, request)
            self, request = args
        else:
            # Unbound function: (request,)
            request = args[0]
            
        token = request.headers.get('Authorization')
        expected_token = os.getenv('API_TOKEN', 'default-secret-token')
        if token != expected_token:
            return PlainTextResponse("Unauthorized", status_code=401)
        return await func(*args, **kwargs)
    return wrapper


class ProxyServer:
    def __init__(self, base_dir: Path | str = "workspace"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_workspace = None
        self.local_servers = None
        self.proxy = None
        self.config_path = Path(__file__).parent / "mcp_config.yaml"

    async def _setup(self):
        """Initialize workspace, servers, and endpoints."""
        # Create new workspace
        self.current_workspace = setup_workspace(self.base_dir)
        
        # Load and prepare config
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        config = prepare_config(config, self.current_workspace)
        
        # Initialize MCP servers
        self.local_servers = Client(config)
        await self.local_servers._connect()

        # Create proxy with endpoints
        self.proxy = FastMCP.as_proxy(self.local_servers, name="proxy")
        
        # Register endpoints
        self.proxy.custom_route("/health", methods=["GET"])(self._handle_health_check)
        self.proxy.custom_route("/reset", methods=["POST"])(self._handle_reset)
        self.proxy.custom_route("/upload", methods=["POST"])(self._handle_upload)
        self.proxy.custom_route("/download", methods=["GET"])(self._handle_download)

    async def _reset_server(self, preserve_as: Optional[str] = None):
        """Reset server with new workspace, optionally preserving current one."""
        if not self.current_workspace:
            return

        if preserve_as:
            # Preserve workspace with given name
            preserved_dir = self.current_workspace.parent / preserve_as
            
            # Handle naming conflicts
            if preserved_dir.exists():
                import time
                timestamp = int(time.time())
                preserved_dir = self.current_workspace.parent / f"{preserve_as}_{timestamp}"
            
            try:
                self.current_workspace.rename(preserved_dir)
                print(f"Workspace preserved as: {preserved_dir}")
            except OSError as e:
                print(f"Warning: Could not preserve workspace as {preserve_as}: {e}")
                # Fall back to deletion
                shutil.rmtree(self.current_workspace, ignore_errors=True)
        else:
            # Delete current workspace
            shutil.rmtree(self.current_workspace, ignore_errors=True)

        # Clean up connections
        if self.local_servers:
            await self.local_servers._disconnect()
            self.local_servers = None
            self.proxy = None
        
        # Reinitialize with new workspace
        await self._setup()

    # Endpoint Handlers
    async def _handle_health_check(self, request: Request) -> PlainTextResponse:
        """Health check endpoint - no authentication required."""
        return PlainTextResponse("OK")

    @require_auth
    async def _handle_reset(self, request: Request) -> PlainTextResponse:
        """Reset workspace endpoint."""
        preserve_as = request.query_params.get('preserve_as')
        
        # Validate preserve_as if provided
        if preserve_as:
            if not preserve_as.replace('-', '').replace('_', '').replace('.', '').isalnum():
                return PlainTextResponse(
                    "Invalid preserve_as: must contain only alphanumeric characters, hyphens, underscores, and periods", 
                    status_code=400
                )
        
        await self._reset_server(preserve_as)
        
        if preserve_as:
            return PlainTextResponse(f"Server reset complete. Previous workspace preserved as: {preserve_as}")
        else:
            return PlainTextResponse("Server reset complete. Previous workspace deleted.")

    @require_auth
    async def _handle_upload(self, request: Request) -> PlainTextResponse:
        """Upload files to workspace."""
        if not self.current_workspace:
            return PlainTextResponse("No workspace available", status_code=500)
        
        # Get optional subdirectory path
        subpath = request.query_params.get('path', '')
        target_dir = self.current_workspace
        
        if subpath:
            # Ensure path is safe (no directory traversal)
            if '..' in subpath or subpath.startswith('/'):
                return PlainTextResponse("Invalid path: no directory traversal allowed", status_code=400)
            target_dir = self.current_workspace / subpath
            target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            form = await request.form()
            uploaded_files = []
            
            for field_name, file_data in form.items():
                if isinstance(file_data, UploadFile):
                    if file_data.filename:
                        file_path = target_dir / file_data.filename
                        
                        # Write file
                        with open(file_path, 'wb') as f:
                            content = await file_data.read()
                            f.write(content)
                        
                        uploaded_files.append(file_data.filename)
            
            if uploaded_files:
                return PlainTextResponse(f"Successfully uploaded: {', '.join(uploaded_files)}")
            else:
                return PlainTextResponse("No files received", status_code=400)
                
        except Exception as e:
            return PlainTextResponse(f"Upload failed: {str(e)}", status_code=500)

    @require_auth
    async def _handle_download(self, request: Request) -> FileResponse | PlainTextResponse:
        """Download files from workspace."""
        if not self.current_workspace:
            return PlainTextResponse("No workspace available", status_code=500)
        
        file_path = request.query_params.get('file_path')
        if not file_path:
            return PlainTextResponse("file_path parameter required", status_code=400)
        
        # Ensure path is safe (no directory traversal)
        if '..' in file_path or file_path.startswith('/'):
            return PlainTextResponse("Invalid file_path: no directory traversal allowed", status_code=400)
        
        full_path = self.current_workspace / file_path
        
        try:
            if not full_path.exists():
                return PlainTextResponse("File not found", status_code=404)
            
            if not full_path.is_file():
                return PlainTextResponse("Path is not a file", status_code=400)
            
            return FileResponse(
                path=str(full_path),
                filename=full_path.name
            )
            
        except Exception as e:
            return PlainTextResponse(f"Download failed: {str(e)}", status_code=500)

    async def start(self):
        """Start the proxy server."""
        try:
            await self._setup()
            if not self.proxy:
                raise RuntimeError("Proxy server failed to initialize")
            await self.proxy.run_async(transport="http", host="0.0.0.0", port=8080)
        except Exception as e:
            print(f"Error starting server: {e}")
            raise


def main():
    return ProxyServer()


if __name__ == "__main__":
    server = main()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
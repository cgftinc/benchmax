import asyncio
from typing import Optional
import aiohttp



async def check_health(address: str) -> bool:
    """
    Check if a server's /health endpoint is responding.
    
    Args:
        address: Server address in format "host:port"
    
    Returns:
        True if server is healthy, False otherwise
    """
    host, port = address.split(":")
    url = f"http://{host}:{port}/health"
    
    timeout_obj = aiohttp.ClientTimeout(total=2.0)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.get(url) as response:
                return response.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return False


async def wait_for_server_health(address: str, timeout: float = 60.0) -> bool:
    """
    Wait for a server to be healthy by polling its /health endpoint.
    
    Args:
        address: Server address in format "host:port"
        timeout: Maximum time to wait in seconds
    
    Returns:
        True if server becomes healthy, False if timeout
    """
    start_time = asyncio.get_event_loop().time()
    
    while (asyncio.get_event_loop().time() - start_time) < timeout:
        if await check_health(address):
            return True
        await asyncio.sleep(1.0)
    
    return False

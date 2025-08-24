"""Base host interface for remote container management."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

@dataclass
class RemoteContainerConfig:
    """Configuration for remote container deployment - host agnostic."""
    image_name: Optional[str]  # Docker image to use (e.g., "benchmax/excel-mcp:v1.0")
    startup_script: str = "/usr/local/bin/start_mcp_servers.sh"
    cleanup_script: Optional[str] = "/usr/local/bin/cleanup.sh"
    port_range: Tuple[int, int] = (8000, 8010)
    idle_timeout: Optional[int] = 3600  # Seconds, None for no timeout
    cpu: Optional[float] = None
    memory: Optional[int] = None  # MB
    env_vars: Optional[Dict[str, str]] = None

@dataclass
class RemoteContainer:
    """Represents a running container - host agnostic."""
    container_id: str  # Unique identifier for the container
    base_url: str  # URL to reach the container (e.g., "http://sandbox123.modal.com")
    workspace_path: str  # Path inside container (e.g., "/workspace/abc123")
    host_handle: Any  # Host-specific handle (modal.Sandbox, docker.Container, etc.)

class RemoteHost(ABC):
    """Abstract base class for remote container hosts.
    
    This interface defines the contract that all remote container hosts
    must implement to work with RemoteMCPEnv. hosts handle the low-level
    container lifecycle, command execution, and file operations.
    """
    
    @abstractmethod
    def create_container(self, config: RemoteContainerConfig, worker_id: str) -> RemoteContainer:
        """Create and start a new container.
        
        Args:
            config: Configuration for the container
            worker_id: Unique identifier for this worker instance
            
        Returns:
            RemoteContainer representing the running container
            
        Raises:
            RuntimeError: If container creation fails
        """
        pass
    
    @abstractmethod
    def execute_command(
        self, 
        container: RemoteContainer, 
        command: List[str],
        workdir: Optional[str] = None, 
        envvars: Optional[Dict[str, str]] = None
    ) -> Tuple[int, str, str]:
        """Execute a command in the container.
        
        Args:
            container: The container to execute the command in
            command: Command and arguments to execute
            workdir: Working directory for the command (defaults to container workspace)
            envvars: Environment variables to set for the command
            
        Returns:
            Tuple of (return_code, stdout, stderr)
            
        Raises:
            RuntimeError: If command execution fails at the infrastructure level
        """
        pass
    
    @abstractmethod
    def upload_file(self, container: RemoteContainer, local_path: Path, remote_path: str) -> None:
        """Upload a file to the container.
        
        Args:
            container: The target container
            local_path: Local file path to upload
            remote_path: Destination path inside the container
            
        Raises:
            FileNotFoundError: If local_path does not exist
            RuntimeError: If upload fails
        """
        pass
    
    @abstractmethod
    def download_file(self, container: RemoteContainer, remote_path: str, local_path: Path) -> None:
        """Download a file from the container.
        
        Args:
            container: The source container  
            remote_path: Path inside the container to download from
            local_path: Local destination path
            
        Raises:
            FileNotFoundError: If remote_path does not exist in container
            RuntimeError: If download fails
        """
        pass
    
    @abstractmethod
    def terminate_container(self, container: RemoteContainer) -> None:
        """Terminate and cleanup the container.
        
        Args:
            container: The container to terminate
            
        Note:
            This should be idempotent - safe to call multiple times.
            Should not raise exceptions if container is already terminated.
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the host."""
        return f"{self.__class__.__name__}()"
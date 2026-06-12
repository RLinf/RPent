"""Driver clients for the agent/tool process to driver process boundary.

The package name is kept as ``transport`` for compatibility, but the primary
abstraction is ``DriverClient``: command delivery plus driver artifact access.
"""
from pathlib import Path

from physical_agent.transport.base import DriverClient, TransportClient
from physical_agent.transport.file import FileDriverClient, FileTransportClient
from physical_agent.transport.socket_transport import (
    SocketDriverClient,
    SocketTransportClient,
)

_SOCKET_ENDPOINTS: dict[str, tuple[str, int]] = {}


def set_socket_endpoint(workdir: str | Path, host: str, port: int) -> None:
    """Record the socket endpoint discovered during driver startup."""
    _SOCKET_ENDPOINTS[str(Path(workdir).resolve())] = (host, int(port))


def create_driver_client(kind: str, workdir: str | Path) -> DriverClient:
    """Create a driver client for an initialized driver workdir."""
    wd = Path(workdir)
    if kind == "file":
        return FileDriverClient(wd)
    if kind == "socket":
        endpoint = _SOCKET_ENDPOINTS.get(str(wd.resolve()))
        if endpoint is None:
            raise RuntimeError(f"socket endpoint not registered for workdir: {wd}")
        host, port = endpoint
        return SocketDriverClient(host, port)
    raise ValueError(f"unknown transport kind: {kind}")


def create_transport_client(kind: str, workdir: str | Path) -> DriverClient:
    """Compatibility wrapper for the old transport factory name."""
    return create_driver_client(kind, workdir)


__all__ = [
    "DriverClient",
    "FileDriverClient",
    "FileTransportClient",
    "SocketDriverClient",
    "SocketTransportClient",
    "TransportClient",
    "create_driver_client",
    "create_transport_client",
    "set_socket_endpoint",
]

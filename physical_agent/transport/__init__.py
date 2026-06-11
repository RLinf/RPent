"""Transport clients for the agent/tool process to driver process boundary."""
from pathlib import Path

from physical_agent.transport.base import TransportClient
from physical_agent.transport.file import FileTransportClient
from physical_agent.transport.socket_transport import SocketTransportClient

_SOCKET_ENDPOINTS: dict[str, tuple[str, int]] = {}


def set_socket_endpoint(workdir: str | Path, host: str, port: int) -> None:
    """Record the socket endpoint discovered during driver startup."""
    _SOCKET_ENDPOINTS[str(Path(workdir).resolve())] = (host, int(port))


def create_transport_client(kind: str, workdir: str | Path) -> TransportClient:
    """Create a transport client for an initialized driver workdir."""
    wd = Path(workdir)
    if kind == "file":
        return FileTransportClient(wd)
    if kind == "socket":
        endpoint = _SOCKET_ENDPOINTS.get(str(wd.resolve()))
        if endpoint is None:
            raise RuntimeError(f"socket endpoint not registered for workdir: {wd}")
        host, port = endpoint
        return SocketTransportClient(host, port)
    raise ValueError(f"unknown transport kind: {kind}")


__all__ = [
    "FileTransportClient",
    "SocketTransportClient",
    "TransportClient",
    "create_transport_client",
    "set_socket_endpoint",
]

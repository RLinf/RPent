"""Driver clients for the agent/tool process to driver process boundary."""
from pathlib import Path

from physical_agent.driver_client.base import DriverClient
from physical_agent.driver_client.file import FileDriverClient
from physical_agent.driver_client.socket import SocketDriverClient

_SOCKET_ENDPOINTS: dict[str, tuple[str, int]] = {}


def set_socket_endpoint(workdir: str | Path, host: str, port: int) -> None:
    """Record the socket endpoint discovered during driver startup."""
    _SOCKET_ENDPOINTS[str(Path(workdir).resolve())] = (host, int(port))


def get_socket_endpoint(workdir: str | Path) -> tuple[str, int] | None:
    """Return the socket endpoint registered for a driver workdir."""
    return _SOCKET_ENDPOINTS.get(str(Path(workdir).resolve()))


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


__all__ = [
    "DriverClient",
    "FileDriverClient",
    "SocketDriverClient",
    "create_driver_client",
    "get_socket_endpoint",
    "set_socket_endpoint",
]

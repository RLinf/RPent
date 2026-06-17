"""Driver clients for the agent/tool process to driver process boundary."""
from pathlib import Path

from physical_agent.driver_client.base import DriverClient
from physical_agent.driver_client.proxies import RemoteEnvProxy
from physical_agent.driver_client.socket import SocketDriverClient

_SOCKET_ENDPOINTS: dict[str, tuple[str, int]] = {}


def set_socket_endpoint(output_dir: str | Path, host: str, port: int) -> None:
    """Record the socket endpoint discovered during driver startup."""
    _SOCKET_ENDPOINTS[str(Path(output_dir).resolve())] = (host, int(port))


def get_socket_endpoint(output_dir: str | Path) -> tuple[str, int] | None:
    """Return the socket endpoint registered for a driver output dir."""
    return _SOCKET_ENDPOINTS.get(str(Path(output_dir).resolve()))


def create_driver_client(output_dir: str | Path) -> DriverClient:
    """Create a driver client for an initialized driver output dir."""
    od = Path(output_dir)
    endpoint = _SOCKET_ENDPOINTS.get(str(od.resolve()))
    if endpoint is None:
        raise RuntimeError(f"socket endpoint not registered for output_dir: {od}")
    host, port = endpoint
    return SocketDriverClient(host, port)


__all__ = [
    "DriverClient",
    "RemoteEnvProxy",
    "SocketDriverClient",
    "create_driver_client",
    "get_socket_endpoint",
    "set_socket_endpoint",
]

"""Driver clients for the agent/tool process to driver process boundary."""
from pathlib import Path

from physical_agent.rpc_driver.base import RpcClient
from physical_agent.rpc_driver.socket import SocketRpcClient

_SOCKET_ENDPOINTS: dict[str, tuple[str, int]] = {}


def set_socket_endpoint(output_dir: str | Path, host: str, port: int) -> None:
    """Record the socket endpoint discovered during driver startup."""
    _SOCKET_ENDPOINTS[str(Path(output_dir).resolve())] = (host, int(port))


def get_socket_endpoint(output_dir: str | Path) -> tuple[str, int] | None:
    """Return the socket endpoint registered for a driver output dir."""
    return _SOCKET_ENDPOINTS.get(str(Path(output_dir).resolve()))


def create_rpc_client(output_dir: str | Path) -> RpcClient:
    """Create a driver client for an initialized driver output dir."""
    od = Path(output_dir)
    endpoint = _SOCKET_ENDPOINTS.get(str(od.resolve()))
    if endpoint is None:
        raise RuntimeError(f"socket endpoint not registered for output_dir: {od}")
    host, port = endpoint
    return SocketRpcClient(host, port)


__all__ = [
    "RpcClient",
    "SocketRpcClient",
    "create_rpc_client",
    "get_socket_endpoint",
    "set_socket_endpoint",
]

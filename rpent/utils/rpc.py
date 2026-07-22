"""RPC client protocol and endpoint registry for the agent/server boundary."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class RpcClient(Protocol):
    """Generic method-call RPC transport (agent → out-of-process server)."""

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Invoke a remote method and return its result."""

    def close(self) -> None:
        """Release any client-side transport resources."""


# ---------------------------------------------------------------------------
# Endpoint registry (transport-agnostic; entries carry a ``kind`` discriminator)
# ---------------------------------------------------------------------------

_ENDPOINTS: dict[str, dict[str, Any]] = {}


def set_socket_endpoint(output_dir: str | Path, host: str, port: int) -> None:
    """Register a socket-based RPC endpoint against an env_server output dir."""
    _ENDPOINTS[str(Path(output_dir).resolve())] = {
        "kind": "socket", "host": host, "port": int(port)
    }


def get_socket_endpoint(output_dir: str | Path) -> tuple[str, int] | None:
    """Return ``(host, port)`` if a socket endpoint is registered, else ``None``."""
    ep = _ENDPOINTS.get(str(Path(output_dir).resolve()))
    if ep is None or ep.get("kind") != "socket":
        return None
    return ep["host"], ep["port"]


def get_endpoint(output_dir: str | Path) -> dict[str, Any] | None:
    """Return the raw endpoint record for an output dir, or ``None``."""
    return _ENDPOINTS.get(str(Path(output_dir).resolve()))


def create_rpc_client(output_dir: str | Path) -> RpcClient:
    """Build an RPC client for whichever transport was registered at ``output_dir``.

    Dispatch is by the endpoint's ``kind`` field. Each transport is imported
    lazily inside the branch so this module stays dependency-free at load time
    and adding a new transport (e.g. ``http_rpc``) is a one-branch edit.
    """
    ep = _ENDPOINTS.get(str(Path(output_dir).resolve()))
    if ep is None:
        raise RuntimeError(f"no RPC endpoint registered for output_dir: {output_dir}")
    kind = ep.get("kind")
    if kind == "socket":
        from rpent.utils.socket_rpc import SocketRpcClient
        return SocketRpcClient(ep["host"], ep["port"])
    raise RuntimeError(f"unknown RPC endpoint kind: {kind!r}")


__all__ = [
    "RpcClient",
    "create_rpc_client",
    "get_endpoint",
    "get_socket_endpoint",
    "set_socket_endpoint",
]

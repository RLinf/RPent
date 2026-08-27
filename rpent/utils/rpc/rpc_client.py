"""RPC client protocol: error type, transport-agnostic client base, response
envelope validation.

Server-side counterparts live in :mod:`rpent.utils.rpc.rpc_facade` (the
``RpcFacade`` base and the ``make_error_response`` envelope builder).
"""
from __future__ import annotations

from typing import Any


class RpcError(RuntimeError):
    """Raised when a remote method call returns an error."""

    def __init__(self, method: str, message: str, *, traceback: str | None = None):
        super().__init__(f"{method}: {message}")
        self.method = method
        self.server_traceback = traceback


class RpcClient:
    """Base for transport-specific RPC clients."""

    def close(self) -> None:
        """Close the client connection."""
        pass

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Invoke a remote method and return its result. Override in subclasses."""
        raise NotImplementedError


def check_response(response: Any, method: str) -> Any:
    """Validate RPC response envelope; raise ``RpcError`` on failure, return result."""
    if not isinstance(response, dict):
        raise RpcError(method, f"bad response type: {type(response).__name__}")
    if not response.get("ok"):
        raise RpcError(
            method,
            str(response.get("error", "<no error message>")),
            traceback=response.get("traceback"),
        )
    return response.get("result")


__all__ = ["RpcClient", "RpcError", "check_response"]

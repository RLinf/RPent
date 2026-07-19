"""Utility helpers: config, logging, path resolution, templates."""

from rpent.utils.logging import get_logger, get_output_dir, init_output_dir
from rpent.utils.rpc import (
    RpcClient,
    create_rpc_client,
    get_endpoint,
    get_socket_endpoint,
    set_socket_endpoint,
)
from rpent.utils.socket_rpc import (
    RpcError,
    SocketRpcClient,
    SocketRpcServer,
)
from rpent.utils.templates import (
    default_variables,
    substitute,
    substitute_text,
)

__all__ = [
    "RpcClient",
    "RpcError",
    "SocketRpcClient",
    "SocketRpcServer",
    "create_rpc_client",
    "default_variables",
    "get_endpoint",
    "get_logger",
    "get_output_dir",
    "get_socket_endpoint",
    "init_output_dir",
    "set_socket_endpoint",
    "substitute",
    "substitute_text",
]

# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility helpers: config, logging, path resolution, templates."""

from rpent.utils.logging import get_logger, get_output_dir, init_output_dir
from rpent.utils.rpc import RpcClient, RpcError, make_rpc_client, parse_endpoint
from rpent.utils.rpc.socket_rpc import (
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
    "make_rpc_client",
    "parse_endpoint",
    "default_variables",
    "get_logger",
    "get_output_dir",
    "init_output_dir",
    "substitute",
    "substitute_text",
]

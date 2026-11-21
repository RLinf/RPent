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

"""Client utility functions: endpoint parsing, server discovery, and readiness polling."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon
    from rpent.utils.rpc.rpc_client import RpcClient

logger = get_logger("client_utils")
# Printed by RpcFacade once the server has bound its port.
_ANNOUNCEMENT = re.compile(r"^RPC server listening on (\S+)$", re.MULTILINE)


def parse_endpoint(endpoint: str) -> tuple[str, str, int]:
    """Parse ``[protocol://]host:port`` into ``(protocol, host, port)``.

    Protocol defaults to ``http`` when the prefix is omitted.
    """
    if "://" in endpoint:
        protocol, _, rest = endpoint.partition("://")
    else:
        protocol, rest = "http", endpoint
    host, _, port = rest.partition(":")
    if not host or not port:
        raise ValueError(f"endpoint must be [protocol://]host:port, got {endpoint!r}")
    return protocol, host, int(port)


def make_rpc_client(endpoint: str, *, enable_sessions: bool = False) -> "RpcClient":
    """Build an HTTP or socket client for ``endpoint``."""
    from rpent.utils.rpc.http_rpc import HttpRpcClient
    from rpent.utils.rpc.socket_rpc import SocketRpcClient

    protocol, host, port = parse_endpoint(endpoint)
    if protocol == "http":
        return HttpRpcClient(f"http://{host}:{port}", enable_sessions=enable_sessions)
    if protocol == "socket":
        return SocketRpcClient(host, port, enable_sessions=enable_sessions)
    raise ValueError(f"endpoint protocol must be http or socket, got {protocol!r}")


def wait_for_endpoint(daemon: "ProcessDaemon", *, timeout_s: float = 300.0) -> str:
    """Return the endpoint that ``daemon``'s RPC server announced in its log.

    Start the server with ``--port 0`` and a fresh log, so that it binds a free
    port itself; a port picked by the parent can be taken by another process
    before the server binds it.
    """
    deadline = time.monotonic() + timeout_s
    log = Path(daemon.log_path)
    while (match := _ANNOUNCEMENT.search(log.read_text())) is None:
        if daemon.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError(
                f"{daemon.name} did not announce its endpoint "
                f"(exit code {daemon.poll()}); see {log}"
            )
        time.sleep(0.5)
    return match[1]


def wait_for_ready(
    client: "RpcClient",
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.5,
    daemon: "ProcessDaemon | None" = None,
) -> None:
    """Poll ``client.call("healthz")`` until it succeeds or ``timeout_s`` elapses.

    If ``daemon`` is given and its subprocess exits before becoming ready,
    fail fast with ``RuntimeError`` (carrying the exit code) instead of
    blocking until ``timeout_s``.

    Once the server is reachable, if the client is session-aware (it owns a
    private ``_session_id``), this function registers the session with the
    server once. The sid stays private to the client; business code never
    sees it.
    """
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        if daemon is not None:
            rc = daemon.poll()
            if rc is not None:
                detail = last_err if last_err is not None else "no healthz attempt yet"
                raise RuntimeError(
                    f"{daemon.name} exited with code {rc} before becoming "
                    f"ready; check its log. last healthz error: {detail}"
                )
        try:
            client.call("healthz", timeout_s=1.0)
        except Exception as exc:
            last_err = exc
            time.sleep(poll_interval_s)
            continue
        sid = getattr(client, "_session_id", None)
        if sid is not None:
            try:
                client.call("session.register", timeout_s=30.0)
            except Exception as exc:
                raise RuntimeError(
                    f"server is reachable but session.register failed for "
                    f"sid={sid!r}: {exc}. The server process is now in a "
                    f"half-ready state; check its log."
                ) from exc
        return
    raise TimeoutError(
        f"server did not become ready within {timeout_s:.0f}s: {last_err}"
    )

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

"""Base class for subprocess RPC servers.

``RpcFacade`` owns the shutdown event, the ``healthz`` / ``shutdown`` RPC
methods, transport binding, parent-watch, and clean teardown. Subclasses
register business methods in ``_register_rpc`` (called from ``__init__``);
read-only methods listed in ``self._readonly_methods`` run under a shared
read lock, mutating methods acquire an exclusive write lock.

Client-side counterparts live in :mod:`rpent.utils.rpc.rpc_client`.

Usage::

    class MyFacade(RpcFacade):
        def __init__(self):
            super().__init__()
            self._rpc["hello"] = self.say_hello

        def say_hello(self):
            return "world"


    MyFacade().serve(transport="http", host="127.0.0.1", port=0)
"""

from __future__ import annotations

import threading
from typing import Any, Literal

from rpent.utils.logging import get_logger
from rpent.utils.rwlock import RWLock

logger = get_logger("rpc")


def make_error_response(exc: Exception) -> dict:
    """Build the error envelope for a caught exception."""
    import traceback as _tb

    return {"ok": False, "error": str(exc), "traceback": _tb.format_exc()}


class RpcFacade:
    """Base class for subprocess RPC servers.

    Subclasses register methods in ``self._rpc`` (typically in ``__init__``
    or a ``_register_rpc`` hook). Read-only methods listed in
    ``self._readonly_methods`` run under a shared read lock; mutating
    methods acquire an exclusive write lock.

    The base owns the shutdown event, the ``shutdown`` / ``healthz`` RPC
    methods, transport binding, parent-watch, and clean teardown.
    """

    def __init__(self) -> None:
        self._shutdown_event = threading.Event()
        self._dispatch_lock = RWLock()
        self._rpc: dict[str, Any] = {}
        self._readonly_methods: set[str] = set()

    def close(self) -> None:
        """Clean up resources. Override in subclasses that hold resources."""
        pass

    def _builtin_dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        """Handle framework methods (healthz, shutdown).

        Returns ``None`` for business methods so that :meth:`_dispatch`
        can fall through to RWLock-based routing.
        """
        if method == "healthz":
            return {"status": "ok"}
        if method == "shutdown":
            with self._dispatch_lock.write():
                self._shutdown_event.set()
            return {"ok": True}
        return None

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        """Business RPC dispatch using a registration dict.

        Subclasses register handlers in ``_register_rpc``. Read-only methods
        (registered in ``_readonly_methods``) run under a shared read lock;
        mutating methods acquire an exclusive write lock.
        """
        result = self._builtin_dispatch(method, args, kwargs)
        if result is not None:
            return result
        handler = self._rpc.get(method)
        if handler is None:
            raise ValueError(f"unknown RPC method: {method!r}")
        if method in self._readonly_methods:
            with self._dispatch_lock.read():
                return handler(*args, **kwargs)
        with self._dispatch_lock.write():
            return handler(*args, **kwargs)

    def serve(
        self,
        *,
        transport: Literal["socket", "http"],
        host: str,
        port: int,
        parent_watch: bool = False,
    ) -> None:
        """Bind, announce, watch-parent, serve-forever, shut down cleanly.

        When *parent_watch* is True, a background thread reads stdin (a pipe
        from :class:`ProcessDaemon`) and triggers shutdown when the pipe
        closes — i.e., when the parent process dies.
        """
        from rpent.utils.daemon import watch_parent_death
        from rpent.utils.rpc.http_rpc import HttpRpcServer
        from rpent.utils.rpc.socket_rpc import SocketRpcServer

        server_cls = HttpRpcServer if transport == "http" else SocketRpcServer
        server = server_cls((host, port), self._dispatch)
        bound_host, bound_port = server.server_address
        client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
        url = f"{transport}://{client_host}:{bound_port}"
        print(f"RPC server listening on {url}", flush=True)
        logger.info("RPC server listening on %s", url)

        if parent_watch:
            watch_parent_death(self._shutdown_event.set)
        try:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self._shutdown_event.wait()
        finally:
            server.shutdown()
            server.server_close()
            self.close()


__all__ = ["RpcFacade", "make_error_response"]

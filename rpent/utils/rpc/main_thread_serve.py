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

"""Optional single-thread dispatch capability for RPC facades.

Some simulation backends (MuJoCo EGL rendering, Isaac Sim) are not
thread-safe: every env operation must run on a single, fixed thread.
``MainThreadServeMixin`` provides a ``serve`` that overrides
:meth:`rpent.utils.rpc.rpc_facade.RpcFacade.serve` to run the transport
server on a daemon thread but execute every dispatch on the calling (main)
thread via a work queue. Mix it into a facade subclass only when the backend
needs this; otherwise inherit :class:`rpent.utils.rpc.rpc_facade.RpcFacade`
and use the plain ``serve``.
"""

from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Literal

from rpent.utils.daemon import watch_parent_death


class MainThreadServeMixin:
    """Mixin overriding ``serve`` so every dispatch runs on one thread.

    The transport server runs on a daemon thread; ``_dispatch_main_thread``
    (the transport-side proxy) enqueues each request on ``_main_thread_queue``
    and blocks until the consumer loop — running on the thread that called
    ``serve`` — has executed it via ``self._dispatch``. Subclasses mix this in
    ahead of :class:`RpcFacade` and inherit ``serve`` as-is; they do not need
    to wrap it.
    """

    def _dispatch_main_thread(
        self, method: str, args: tuple, kwargs: dict, *, session_id: str | None = None
    ) -> Any:
        """Transport-side proxy for :meth:`serve`.

        Called by the transport server's daemon thread on every RPC. Framework
        methods (``healthz`` / ``shutdown``) are answered through
        :meth:`RpcFacade._builtin_dispatch`; ``shutdown`` additionally pushes a
        ``None`` sentinel to wake the consumer loop. Everything else (business
        calls, session-aware or not) is enqueued with its ``session_id`` and
        executed on the main thread.
        """
        result = self._builtin_dispatch(method, args, kwargs)
        if result is not None:
            if method == "shutdown":
                self._main_thread_queue.put(None)  # sentinel: wake the consumer loop
            return result
        event = threading.Event()
        req: dict = {
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "session_id": session_id,
            "result": None,
            "error": None,
        }
        self._main_thread_queue.put((event, req))
        event.wait()
        if req["error"]:
            raise RuntimeError(req["error"])
        return req["result"]

    def serve(
        self,
        *,
        transport: Literal["socket", "http"],
        host: str,
        port: int,
        parent_watch: bool = False,
        session_sweep_s: float | None = None,
    ) -> None:
        """Serve RPC, dispatching every call on the calling (main) thread.

        Same contract as :meth:`RpcFacade.serve` — transport binding,
        ``healthz`` / ``shutdown``, parent-watch, and session support — but
        every dispatch runs on the thread that calls ``serve`` (normally the
        process main thread) via :meth:`_dispatch_main_thread` and
        ``self._main_thread_queue``. Use it for backends that must stay on a
        single thread (MuJoCo EGL rendering, Isaac Sim); subclasses mix this
        in and inherit ``serve`` without wrapping it.

        Exits when the shutdown event is set (``shutdown`` RPC or
        *parent_watch* parent death); both paths unblock the consumer loop.
        """
        self._main_thread_queue: "queue.Queue[tuple[threading.Event, dict] | None]" = (
            queue.Queue()
        )
        server = self._bind_and_announce(
            transport, host, port, self._dispatch_main_thread
        )
        if self._enable_sessions and (session_sweep_s is None or session_sweep_s <= 0):
            raise ValueError(
                "session_sweep_s is required (and > 0) when sessions "
                "are enabled; idle timeout is only enforced by the "
                f"sweep thread, got {session_sweep_s!r}"
            )
        if parent_watch:
            watch_parent_death(self._shutdown_event.set)
        if self._enable_sessions:
            threading.Thread(
                target=self._sweep_sessions,
                args=(session_sweep_s,),
                daemon=True,
                name="rpc-session-sweep",
            ).start()

        # Dispatch runs on THIS thread; poll the queue so shutdown via the
        # event (parent-watch) or the None sentinel (shutdown RPC) both
        # unblock the consumer.
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            while not self._shutdown_event.is_set():
                try:
                    item = self._main_thread_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                event, req = item
                try:
                    req["result"] = self._dispatch(
                        req["method"],
                        req["args"],
                        req["kwargs"],
                        session_id=req["session_id"],
                    )
                except Exception:
                    req["error"] = traceback.format_exc()
                event.set()
        finally:
            server.shutdown()
            server.server_close()
            self.close()


__all__ = ["MainThreadServeMixin"]

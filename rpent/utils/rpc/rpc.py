"""RPC client protocol and Facade base for subprocess RPC servers."""
from __future__ import annotations

import atexit
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Literal

from rpent.utils.logging import get_logger
from rpent.utils.rwlock import RWLock

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon

logger = get_logger("rpc")

# Fallback idle timeout for a registered session when the server does not
# configure one explicitly. Enforced only while the session sweep thread is
# running (see RpcFacade.serve); without a sweep, idle sessions are never
# dropped.
DEFAULT_SESSION_TIMEOUT_S = 3600.0


class RpcError(RuntimeError):
    """Raised when a remote method call returns an error."""

    def __init__(self, method: str, message: str, *, traceback: str | None = None):
        super().__init__(f"{method}: {message}")
        self.method = method
        self.server_traceback = traceback


class RpcClient:
    """Base for transport-specific RPC clients.

    Owns the session id (transport-private; business code never sees it)
    and the atexit close hook. Subclasses implement :meth:`call` for the
    transport-specific request path (payload construction, wire send, and
    response validation).
    """

    def __init__(self, *, enable_sessions: bool = False) -> None:
        self._session_id: str | None = (
            f"rpc_{uuid.uuid4().hex[:8]}" if enable_sessions else None
        )
        self._closed = False
        if self._session_id is not None:
            atexit.register(self.close)

    def close(self) -> None:
        """Notify the server to drop this client's session.

        Called automatically at exit via atexit. Failures are swallowed
        (the process is exiting anyway; the server's sweep thread is the
        fallback for crashed clients). Idempotent: a ``_closed`` flag guards
        against double-close when atexit fires after a manual ``close()``.

        Short timeout (1s) with a 0.5s healthz preflight so an already-dead
        server doesn't stall atexit by the full ``timeout_s``.
        """
        if self._closed:
            return
        self._closed = True
        try:
            # Preflight: if the server is unreachable, skip the session.close
            # RPC instead of blocking atexit for the full timeout.
            self.call("healthz", timeout_s=0.5)
        except Exception:
            return
        try:
            self.call("session.close", timeout_s=1.0)
        except Exception:
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


def make_error_response(exc: Exception) -> dict:
    """Build the error envelope for a caught exception."""
    import traceback as _tb
    return {"ok": False, "error": str(exc), "traceback": _tb.format_exc()}


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


def wait_for_ready(
    client: RpcClient,
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
        # Session-aware client: register its sid with the server now. The
        # sid is private to the client; business code never touches it.
        # healthz already succeeded, so a failure here is a session-register
        # specific problem (e.g. server-side ValueError) — wrap it so the
        # caller doesn't see a generic "did not become ready" timeout.
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


class RpcFacade:
    """Base class for subprocess RPC servers.

    Subclasses implement :meth:`_dispatch`; the base owns the shutdown
    event, the ``shutdown`` / ``healthz`` RPC methods, transport binding,
    parent-watch, and clean teardown.

    Usage::

        class MyFacade(RpcFacade):
            def _dispatch(self, method, args, kwargs, *, session_id=None):
                if method == "hello":
                    return "world"
                raise ValueError(f"unknown RPC method: {method!r}")

        MyFacade().serve(transport="http", host="127.0.0.1", port=0)
    """

    def __init__(
        self,
        *,
        enable_sessions: bool = False,
        session_timeout_s: float | None = DEFAULT_SESSION_TIMEOUT_S,
    ) -> None:
        self._enable_sessions = enable_sessions
        # Per-session idle timeout applied to every registered session. The
        # client does not carry one — the server owns the policy. Defaults to
        # one hour (:data:`DEFAULT_SESSION_TIMEOUT_S`); explicit ``None`` means
        # never expires. Only meaningful when sessions are enabled.
        self._session_timeout_s = (
            float(session_timeout_s)
            if session_timeout_s is not None
            else None
        )
        self._shutdown_event = threading.Event()
        # Per-session bookkeeping, sid -> (timeout_s | None, last_active).
        # Only used when sessions are enabled; all access (request dispatch +
        # the sweep thread) is guarded by _lock.
        self._session_lock = threading.Lock()
        self._sessions: dict[str, tuple[float | None, float]] = {}
        # simple impl of method register and dispatch
        self._dispatch_lock = RWLock()
        self._rpc: dict[str, Callable] = {}
        self._readonly_methods: set[str] = set()

    def _dispatch(
        self, method: str, args: tuple, kwargs: dict, *, session_id: str | None = None
    ) -> Any:
        """Business RPC dispatch. Override in subclasses.

        Subclasses MUST take ``self._session_lock`` around the dispatch body: the
        threading servers dispatch each request on its own thread, and most
        env/model servers touch a single subprocess worker or EGL/CUDA
        context that is not concurrency-safe. The lock also serializes
        dispatch against the sweep thread's ``_expire_session``.

        ``session_id`` is the caller's bound session (``None`` when
        unbound or sessions disabled); pass it through to business methods
        that need it. Do not handle ``shutdown``, ``healthz`` or
        ``session.*`` here — the base takes care of them.

        There is a simple impl to support dispatch with a rw lock.
        """
        if self._enable_sessions:
            return self._dispatch_session(
                method, args, kwargs, session_id=session_id,
            )
        else:
            return self._dispatch_nosession(
                method, args, kwargs,
            )

    def _dispatch_nosession(self, method: str, args: tuple, kwargs: dict) -> Any:
        handler = self._rpc.get(method)
        if handler is None:
            raise ValueError(f"unknown RPC method: {method!r}")
        if method in self._readonly_methods:
            with self._dispatch_lock.read():
                return handler(*args, **kwargs)
        with self._dispatch_lock.write():
            return handler(*args, **kwargs)

    def _dispatch_session(self, method: str, args: tuple, kwargs: dict, *,
                  session_id: str | None = None) -> Any:
        handler = self._rpc.get(method)
        if handler is None:
            raise ValueError(f"unknown RPC method: {method!r}")
        if method in self._readonly_methods:
            with self._dispatch_lock.read():
                return handler(*args, **kwargs, session_id=session_id)
        with self._dispatch_lock.write():
            return handler(*args, **kwargs, session_id=session_id)

    # ---- session lifecycle (RPC methods + hooks) -------------------------

    def register_session(self, session_id: str) -> dict:
        """Register a session.

        The idle timeout is the server's own ``session_timeout_s``
        (set at construction); the client does not carry one. Registering an
        already-live session refreshes its ``last_active``.

        Called automatically by :func:`wait_for_ready` on connect for
        session-aware clients; business code does not call this.
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"session_id must be a non-empty string, got {session_id!r}")
        with self._session_lock:
            self._sessions[session_id] = (
                self._session_timeout_s,
                time.monotonic(),
            )
        return {"ok": True, "session_id": session_id}

    def drop_session(self, session_id: str) -> dict:
        """Drop a session and fire its cleanup hook (no expiry check).

        Called by the ``session.close`` RPC (client-side atexit); unlike
        :meth:`_expire_session`, this removes the session unconditionally.
        """
        with self._session_lock:
            existed = self._sessions.pop(session_id, None) is not None
        if existed:
            try:
                self._on_session_drop(session_id)
            except Exception:
                # Don't let a failing cleanup hook (e.g. policy.reset) mask
                # the close ack. The session is already removed and nothing
                # retries this — the state leak is logged and we move on.
                logger.warning(
                    "session %s close: cleanup hook failed",
                    session_id,
                    exc_info=True,
                )
        return {"ok": True, "session_id": session_id}

    def _touch_session(self, session_id: str) -> None:
        """Refresh last_active for an active session; raise if unknown.

        Expiry is the sweep thread's job — this only checks the session
        is known and bumps its idle timer.
        """
        with self._session_lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                raise RpcError("session", f"session not found: {session_id}")
            tmo, _ = entry
            self._sessions[session_id] = (tmo, time.monotonic())

    def _on_session_drop(self, session_id: str) -> None:
        """Hook: policy-state cleanup when a session is dropped.

        Fired on both drop paths: the ``session.close`` RPC (client-side
        atexit) and idle-expiry by the sweep thread. See :meth:`drop_session`
        and :meth:`_expire_session`.
        """

    def _expire_session(self, session_id: str) -> None:
        """Drop a session whose idle timeout has elapsed and run its cleanup hook.

        Re-checks liveness under the lock: the sweep thread snapshots candidates,
        but a request may have refreshed ``last_active`` between the snapshot and
        this drop — removing it anyway would kill a live session and wrongly
        reset its policy state.
        """
        with self._session_lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return
            tmo, last = entry
            if tmo is None or (time.monotonic() - last) <= tmo:
                return  # not actually expired (refreshed since the snapshot)
            self._sessions.pop(session_id, None)
        self._on_session_drop(session_id)

    def _sweep_sessions(self, interval_s: float) -> None:
        """Background loop: drop idle-expired sessions every ``interval_s`` s."""
        while not self._shutdown_event.wait(interval_s):
            now = time.monotonic()
            with self._session_lock:
                expired = [
                    sid
                    for sid, (tmo, last) in self._sessions.items()
                    if tmo is not None and (now - last) > tmo
                ]
            for sid in expired:
                try:
                    self._expire_session(sid)
                except Exception:
                    # A failing cleanup hook (e.g. policy.reset) must not kill the
                    # whole sweep loop, or no further session ever gets cleaned up.
                    logger.warning(
                        "session sweep: drop %s failed", sid, exc_info=True
                    )

    def serve(
        self,
        *,
        transport: Literal["socket", "http"],
        host: str,
        port: int,
        parent_watch: bool = False,
        session_sweep_s: float | None = None,
    ) -> None:
        """Bind, announce, watch-parent, serve-forever, shut down cleanly.

        Session support (per-session state + idle-timeout validation) is
        governed by ``enable_sessions`` passed to :meth:`__init__` — servers
        that don't isolate per-client policy state leave it off.

        When *parent_watch* is True, a background thread reads stdin (a pipe
        from :class:`ProcessDaemon`) and triggers shutdown when the pipe
        closes — i.e., when the parent process dies.

        When sessions are enabled (``enable_sessions=True`` at construction),
        *session_sweep_s* MUST be a positive number — the idle timeout is
        only enforced by the sweep thread, so a non-positive value raises
        ``ValueError``. The sweep thread drops idle-expired sessions every
        that many seconds and fires :meth:`_on_session_drop`.
        """
        from rpent.utils.daemon import watch_parent_death
        from rpent.utils.rpc.http_rpc import HttpRpcServer
        from rpent.utils.rpc.socket_rpc import SocketRpcServer

        def dispatch(
            method: str, args: tuple, kwargs: dict, *, session_id: str | None = None
        ) -> Any:
            if method == "healthz":
                return {"status": "ok"}
            if method == "shutdown":
                with self._session_lock:
                    self._shutdown_event.set()
                return {"ok": True}
            if not self._enable_sessions:
                # _dispatch takes _lock internally — concurrent
                # ThreadingHTTPServer requests serialise there.
                return self._dispatch(method, args, kwargs)
            if session_id is None:
                # Sessions exist to isolate per-client policy state; running a
                # business call unbound would silently share that state across
                # clients.
                raise RpcError(
                    "session",
                    "this server requires a bound session: "
                    "call session.register first",
                )
            if method == "session.register":
                return self.register_session(session_id)
            if method == "session.close":
                return self.drop_session(session_id)
            # _touch_session and _dispatch each take _lock internally; the
            # brief gap between them cannot drop the session (only the sweep
            # thread drops idle ones, and it re-checks liveness under lock).
            self._touch_session(session_id)
            return self._dispatch(method, args, kwargs, session_id=session_id)

        server_cls = HttpRpcServer if transport == "http" else SocketRpcServer
        server = server_cls((host, port), dispatch)
        bound_host, bound_port = server.server_address
        client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
        url = f"{transport}://{client_host}:{bound_port}"
        print(f"RPC server listening on {url}", flush=True)
        logger.info("RPC server listening on %s", url)

        if parent_watch:
            watch_parent_death(self._shutdown_event.set)
        if self._enable_sessions:
            if session_sweep_s is None or session_sweep_s <= 0:
                raise ValueError(
                    "session_sweep_s is required (and > 0) when sessions "
                    "are enabled; idle timeout is only enforced by the "
                    f"sweep thread, got {session_sweep_s!r}"
                )
            threading.Thread(
                target=self._sweep_sessions,
                args=(session_sweep_s,),
                daemon=True,
                name="rpc-session-sweep",
            ).start()
        try:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self._shutdown_event.wait()
        finally:
            server.shutdown()
            server.server_close()


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


__all__ = [
    "RpcClient",
    "RpcError",
    "RpcFacade",
    "check_response",
    "make_error_response",
    "parse_endpoint",
    "wait_for_ready",
]

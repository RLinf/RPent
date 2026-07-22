"""HTTP-transport RPC for the env + model RPC boundary.

Uses HTTP POST with JSON payloads instead of pickle-framed TCP.
Numpy arrays cross the wire tagged as ``{"__ndarray__": [...], "dtype": ...}``
so the decode is explicit.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable

import numpy as np

from rpent.utils.socket_rpc import RpcError




DEFAULT_TIMEOUT_S = 30.0


def _from_json(obj: Any) -> Any:
    """Rehydrate ``{"__ndarray__": ..., "dtype": ...}`` back into ndarrays.

    Everything else is passed through unchanged.
    """
    if isinstance(obj, dict):
        if "__ndarray__" in obj and set(obj) <= {"__ndarray__", "dtype"}:
            return np.array(obj["__ndarray__"], dtype=obj.get("dtype"))
        return {k: _from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_json(v) for v in obj]
    return obj


class HttpRpcClient:
    """RPC client that talks to a driver server via HTTP POST.

    Parameters
    ----------
    base_url : str
        Server address, e.g. ``"http://127.0.0.1:8080"``.
    """

    def __init__(self, base_url: str) -> None:
        """Initialize with a base URL, e.g. ``"http://127.0.0.1:8080"``."""
        self._base_url = base_url.rstrip("/")

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Invoke a remote method via HTTP POST and return the result."""
        req_id = str(uuid.uuid4())
        payload = {
            "id": req_id,
            "method": method,
            "args": list(args),
            "kwargs": kwargs or {},
        }
        body = json.dumps(payload, cls=_NumpyEncoder).encode("utf-8")
        url = f"{self._base_url}/call"
        request_timeout = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            # HTTPError is an OSError subclass; catch first so we can parse
            # the ok=False body the server sent alongside the status code.
            raw = exc.read()
        except OSError as exc:
            raise RpcError(method, f"HTTP request failed: {exc}") from exc

        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RpcError(method, f"invalid JSON response: {exc}") from exc

        if not isinstance(response, dict):
            raise RpcError(method, f"bad response type: {type(response).__name__}")

        # Check ok before id: a server that failed to parse the request echoes
        # id=None, so id-mismatch would mask the real error.
        if not response.get("ok"):
            raise RpcError(
                method,
                str(response.get("error", "<no error message>")),
                traceback=response.get("traceback"),
            )

        if response.get("id") != req_id:
            raise RpcError(
                method, f"id mismatch ({response.get('id')!r} != {req_id!r})"
            )

        return _from_json(response.get("result"))

    def close(self) -> None:
        """Release any client-side transport resources (no-op for HTTP)."""
        return None


# ---------------------------------------------------------------------------
#  Server
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that tags numpy arrays and normalizes numpy scalars."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


class _HttpRpcHandler(BaseHTTPRequestHandler):
    """Handles POST /call with JSON-RPC body, dispatches to server.dispatch()."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:
        if self.path != "/call":
            self.send_response(404)
            self.end_headers()
            return

        req_id = None
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            request = json.loads(body)
            req_id = request.get("id") if isinstance(request, dict) else None
            method = request["method"]
            args = tuple(request.get("args", []))
            kwargs = dict(request.get("kwargs", {}))
            result = self.server.dispatch(method, args, kwargs)  # type: ignore[attr-defined]
            response: dict = {"id": req_id, "ok": True, "result": result}
        except Exception as exc:
            import traceback as _tb
            response = {
                "id": req_id,
                "ok": False,
                "error": str(exc),
                "traceback": _tb.format_exc(),
            }

        # Always 200; failures are described inside the body via ok=False.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(response, cls=_NumpyEncoder).encode("utf-8")
        )


class HttpRpcServer(ThreadingHTTPServer):
    """HTTP server that dispatches JSON-RPC calls.

    Same interface as ``SocketRpcServer`` — drop-in replacement at dispatch
    sites that use ``server_address``, ``serve_forever``, ``shutdown``, and
    ``server_close``.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        dispatch: Callable[[str, tuple, dict], Any],
    ) -> None:
        super().__init__(server_address, _HttpRpcHandler)
        self._dispatch = dispatch
        self._dispatch_lock = threading.Lock()

    def dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "healthz":
            return {"status": "ok"}
        with self._dispatch_lock:
            return self._dispatch(method, args, kwargs)

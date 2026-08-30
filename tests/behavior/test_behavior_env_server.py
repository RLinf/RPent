from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer

from robots.behavior.env_server import BehaviorMainThreadHttpRpcServer
from rpent.utils.rpc.http_rpc import HttpRpcClient


def test_behavior_env_rpc_dispatches_on_serving_thread() -> None:
    thread_ids: dict[str, int] = {}

    def dispatch(method: str, _args: tuple, _kwargs: dict) -> dict[str, int | str]:
        thread_ids["dispatch"] = threading.get_ident()
        return {"method": method, "thread_id": thread_ids["dispatch"]}

    server = BehaviorMainThreadHttpRpcServer(("127.0.0.1", 0), dispatch)
    ready = threading.Event()

    def serve() -> None:
        thread_ids["serve_forever"] = threading.get_ident()
        ready.set()
        server.serve_forever(poll_interval=0.01)

    server_thread = threading.Thread(target=serve)
    server_thread.start()
    try:
        assert ready.wait(timeout=2.0)
        response = HttpRpcClient(f"http://127.0.0.1:{server.server_address[1]}").call(
            "healthz"
        )

        assert response == {
            "method": "healthz",
            "thread_id": thread_ids["serve_forever"],
        }
        assert thread_ids["dispatch"] == thread_ids["serve_forever"]
        assert thread_ids["dispatch"] != threading.get_ident()
        assert not isinstance(server, ThreadingHTTPServer)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)

    assert not server_thread.is_alive()

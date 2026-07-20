from __future__ import annotations

import threading

from rpent.utils.socket_rpc import SocketRpcClient, SocketRpcServer


def test_priority_rpc_bypasses_serialized_command_lock() -> None:
    slow_started = threading.Event()
    release_slow = threading.Event()
    slow_result: list[object] = []

    def dispatch(method: str, _args: tuple, _kwargs: dict):
        if method == "slow":
            slow_started.set()
            if not release_slow.wait(timeout=2.0):
                raise TimeoutError("test did not release slow RPC")
            return "slow-finished"
        if method == "stop":
            return "stopped"
        raise ValueError(method)

    server = SocketRpcServer(("127.0.0.1", 0), dispatch, priority_methods={"stop"})
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    port = int(server.server_address[1])
    client = SocketRpcClient("127.0.0.1", port)

    def call_slow() -> None:
        slow_result.append(client.call("slow", timeout_s=2.0))

    slow_thread = threading.Thread(target=call_slow)
    slow_thread.start()
    assert slow_started.wait(timeout=1.0)

    try:
        assert client.call("stop", timeout_s=0.25) == "stopped"
        assert slow_thread.is_alive()
    finally:
        release_slow.set()
        slow_thread.join(timeout=1.0)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)

    assert slow_result == ["slow-finished"]

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

"""One SAM3 RPC endpoint backed by a bounded, round-robin worker pool."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from functools import partial
from pathlib import Path

from rpent.utils.rpc import RpcFacade, make_rpc_client, wait_for_ready


class Sam3Pool(RpcFacade):
    """Forward each complete image/query request to one idle SAM3 instance."""

    def __init__(
        self,
        clients: list,
        *,
        timeout_s: float = 100,
        max_pending: int = 256,
        per_backend_concurrency: int = 1,
        metrics_path: Path | None = None,
    ) -> None:
        super().__init__()
        if (
            not clients
            or timeout_s <= 0
            or max_pending < 1
            or per_backend_concurrency < 1
        ):
            raise ValueError(
                "SAM3 pool requires workers, a positive timeout and queue capacity"
            )
        self.clients = clients
        self.timeout_s, self.max_pending = timeout_s, max_pending
        self.metrics_path = metrics_path
        self.per_backend_concurrency = per_backend_concurrency
        self._condition = threading.Condition()
        self._pending = deque()
        self._inflight = [0] * len(clients)
        self._next = 0
        self._closed = False
        self._counts = [0] * len(clients)
        self._errors = [0] * len(clients)
        for method in ("sam3.segment", "sam3.segment_queries"):
            self._rpc[method] = partial(self.forward, method)
            self._readonly_methods.add(method)
        self._rpc["sam3.pool_stats"] = self.stats
        self._readonly_methods.add("sam3.pool_stats")

    def stats(self) -> dict:
        """Read queue counters without making diagnostic inference requests."""
        with self._condition:
            return {
                "busy": [count > 0 for count in self._inflight],
                "inflight": list(self._inflight),
                "per_backend_concurrency": self.per_backend_concurrency,
                "pending": len(self._pending),
                "requests": list(self._counts),
                "errors": list(self._errors),
            }

    def forward(self, method: str, *args, **kwargs):
        """Admit in arrival order and round-robin among currently idle workers."""
        started = time.monotonic()
        deadline = started + self.timeout_s
        ticket = object()
        with self._condition:
            if self._closed or len(self._pending) >= self.max_pending:
                raise RuntimeError("SAM3 pool closed or queue full")
            self._pending.append(ticket)
            try:
                while True:
                    if self._closed:
                        raise RuntimeError("SAM3 pool closed")
                    idle = [
                        (self._next + offset) % len(self.clients)
                        for offset in range(len(self.clients))
                        if self._inflight[(self._next + offset) % len(self.clients)]
                        < self.per_backend_concurrency
                    ]
                    if self._pending[0] is ticket and idle:
                        worker = idle[0]
                        self._next = (worker + 1) % len(self.clients)
                        self._inflight[worker] += 1
                        self._pending.popleft()
                        self._condition.notify_all()
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("SAM3 pool queue deadline exceeded")
                    self._condition.wait(remaining)
            except BaseException:
                self._pending.remove(ticket)
                self._condition.notify_all()
                raise
        dispatched = time.monotonic()
        error = None
        try:
            remaining = deadline - dispatched
            if remaining <= 0:
                raise TimeoutError("SAM3 pool deadline exceeded before dispatch")
            return self.clients[worker].call(
                method, args=args, kwargs=kwargs, timeout_s=remaining
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            finished = time.monotonic()
            with self._condition:
                self._inflight[worker] -= 1
                self._counts[worker] += 1
                self._errors[worker] += error is not None
                if self.metrics_path is not None:
                    self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.metrics_path.open("a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "time": time.time(),
                                    "worker": worker,
                                    "method": method,
                                    "queue_seconds": dispatched - started,
                                    "inference_seconds": finished - dispatched,
                                    "error": error,
                                }
                            )
                            + "\n"
                        )
                self._condition.notify_all()

    def _builtin_dispatch(self, method: str, args: tuple, kwargs: dict):
        if method == "shutdown":
            with self._condition:
                self._closed = True
                self._condition.notify_all()
        return super()._builtin_dispatch(method, args, kwargs)

    def close(self) -> None:
        """Wake queued requests and release this pool's client handles."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        for client in self.clients:
            client.close()


def main() -> None:
    """Connect already-running SAM3 instances and expose their shared endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--timeout-s", type=float, default=100)
    parser.add_argument("--max-pending", type=int, default=256)
    parser.add_argument("--per-backend-concurrency", type=int, default=1)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    clients = [make_rpc_client(endpoint) for endpoint in args.backends]
    for client in clients:
        wait_for_ready(client, timeout_s=30)
    pool = Sam3Pool(
        clients,
        timeout_s=args.timeout_s,
        max_pending=args.max_pending,
        per_backend_concurrency=args.per_backend_concurrency,
        metrics_path=args.metrics,
    )
    try:
        pool.serve(
            transport="http",
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        pool.close()


if __name__ == "__main__":
    main()

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

"""Route complete groups to real simulator slots over toolkit RPC."""

import argparse
import threading
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from rpent.utils.rpc import RpcFacade
from rpent.utils.rpc.http_rpc import EncodedRpcResponse, HttpRpcClient


@dataclass
class Attempt:
    """A backend slot stays owned until that backend acknowledges release."""

    backend: int
    group: str
    session: str | None = None


class ToolkitRouter(RpcFacade):
    """Reserve groups FIFO and route each episode to one native process owner."""

    def __init__(self, backends: list[tuple[str, int]], group_size: int):
        super().__init__()
        if (
            not backends
            or group_size < 1
            or any(slots < 1 for _, slots in backends)
            or len({url for url, _ in backends}) != len(backends)
        ):
            raise ValueError(
                "expected unique backends with positive slots and group size"
            )
        self.clients = [HttpRpcClient(url) for url, _ in backends]
        self.capacities = [slots for _, slots in backends]
        self.group_size = group_size
        self.capacity = sum(self.capacities) // group_size
        if self.capacity == 0:
            raise ValueError("toolkit capacity cannot fit a complete group")
        self.lock = threading.Lock()
        self.waiting: deque[str] = deque()
        self.admitted: set[str] = set()
        self.closed: set[str] = set()
        self.attempts: dict[str, Attempt] = {}
        self.sessions: dict[str, str] = {}
        self._rpc.update(
            {
                "backend.acquire_group": self.acquire_group,
                "backend.finish_group": self.finish_group,
                "backend.group_status": self.group_status,
                "backend.open": self.open,
                "backend.call": self.forward_call,
                "backend.renew": self.renew,
                "backend.release": self.release,
                "backend.cancel_owners": self.cancel_owners,
                "backend.status": self.status,
            }
        )
        self._readonly_methods.update(self._rpc)

    def acquire_group(self, group: str) -> str:
        """Reserve capacity for a full group before any of its members open."""
        with self.lock:
            if group in self.closed:
                return "closed"
            if group in self.admitted:
                return "admitted"
            if group not in self.waiting:
                self.waiting.append(group)
            if len(self.admitted) < self.capacity and self.waiting[0] == group:
                self.admitted.add(self.waiting.popleft())
                return "admitted"
            return "waiting"

    def finish_group(self, group: str) -> None:
        """Reject late opens and release all owned attempts before returning capacity."""
        with self.lock:
            self.closed.add(group)
            if group in self.waiting:
                self.waiting.remove(group)
            owners = [
                owner
                for owner, attempt in self.attempts.items()
                if attempt.group == group
            ]
        if owners:
            self.cancel_owners(owners)
        with self.lock:
            self.admitted.discard(group)

    def group_status(self, cohort_id: str | None = None) -> dict:
        """Report admission for the requested cohort and total configured capacity."""
        with self.lock:

            def matches(group: str) -> bool:
                return cohort_id is None or group.rpartition(":")[0] == cohort_id

            return {
                "capacity": self.capacity,
                "slots": sum(self.capacities),
                "group_size": self.group_size,
                "admitted": sum(matches(group) for group in self.admitted),
                "waiting": sum(matches(group) for group in self.waiting),
            }

    def open(self, request: dict, owner_id: str, group: str) -> dict:
        """Allocate a physical backend slot and initialize the native episode."""
        if request["gpu"] is not None:
            raise ValueError("routed episodes let the backend assign the GPU")
        with self.lock:
            if (
                group not in self.admitted
                or group in self.closed
                or owner_id in self.attempts
            ):
                raise ValueError("expected an admitted group and a new attempt owner")
            if (
                sum(attempt.group == group for attempt in self.attempts.values())
                >= self.group_size
            ):
                raise ValueError("group has already used its reserved slots")
            used = Counter(attempt.backend for attempt in self.attempts.values())
            backend = min(
                range(len(self.clients)), key=lambda i: used[i] / self.capacities[i]
            )
            attempt = self.attempts[owner_id] = Attempt(backend, group)
        try:
            reply = self.clients[backend].call(
                "backend.open",
                kwargs={"request": request, "owner_id": owner_id},
                timeout_s=1760,
            )
        except Exception:
            self.cancel_owners([owner_id])
            raise
        with self.lock:
            if group in self.closed or self.attempts.get(owner_id) is not attempt:
                raise RuntimeError("episode was cancelled during startup")
            attempt.session = reply["session"]
            self.sessions[attempt.session] = owner_id
        return reply

    def _forget(self, owner: str) -> None:
        attempt = self.attempts.pop(owner, None)
        if attempt is not None and attempt.session is not None:
            self.sessions.pop(attempt.session, None)

    def _session(self, session: str) -> tuple[str, HttpRpcClient]:
        with self.lock:
            owner = self.sessions[session]
            return owner, self.clients[self.attempts[owner].backend]

    def forward_call(
        self, session: str, method: str, arguments: dict
    ) -> EncodedRpcResponse:
        """Forward the original tool and artifact envelope without decoding it."""
        _, client = self._session(session)
        return client.forward(
            "backend.call", args=(session, method, arguments), timeout_s=605
        )

    def renew(self, session: str) -> dict:
        """Renew the native worker lease."""
        return self._session(session)[1].call(
            "backend.renew", args=(session,), timeout_s=12
        )

    def release(self, session: str) -> dict:
        """Release a native worker before making its slot reusable."""
        try:
            owner, client = self._session(session)
        except KeyError:
            return {"released": False}
        result = client.call("backend.release", args=(session,), timeout_s=125)
        with self.lock:
            self._forget(owner)
        return result

    def _owners_by_backend(self, owners: list[str]) -> dict[int, list[str]]:
        selected: dict[int, list[str]] = {}
        for owner in owners:
            if attempt := self.attempts.get(owner):
                selected.setdefault(attempt.backend, []).append(owner)
        return selected

    def cancel_owners(self, owner_ids: list[str]) -> dict:
        """Cancel exact owners, including initializations, on their assigned backends."""
        owners = list(dict.fromkeys(owner_ids))
        with self.lock:
            by_backend = self._owners_by_backend(owners)

        def cancel(backend: int, subset: list[str]) -> None:
            receipt = self.clients[backend].call(
                "backend.cancel_owners", args=(subset,), timeout_s=175
            )
            if not receipt["cleanup_complete"] or receipt["remaining"]:
                raise RuntimeError(f"backend cleanup failed: {receipt}")
            with self.lock:
                for owner in subset:
                    self._forget(owner)

        if by_backend:
            with ThreadPoolExecutor(max_workers=len(by_backend)) as pool:
                tuple(pool.map(cancel, by_backend, by_backend.values()))
        return {"owner_ids": owners, "remaining": {}, "cleanup_complete": True}

    def status(self, cohort_id: str | None = None) -> dict:
        """Read actual native readiness and reservations for the requested cohort."""
        with self.lock:
            by_backend = self._owners_by_backend(
                [
                    owner
                    for owner, attempt in self.attempts.items()
                    if cohort_id is None
                    or attempt.group.rpartition(":")[0] == cohort_id
                ]
            )

        def snapshot(index: int) -> dict:
            owners = by_backend.get(index)
            if cohort_id is not None and not owners:
                return {"ready": 0, "reserved": 0, "ready_per_gpu": {}, "per_gpu": {}}
            return self.clients[index].call(
                "backend.status",
                args=() if cohort_id is None else (owners,),
                timeout_s=6,
            )

        with ThreadPoolExecutor(max_workers=len(self.clients)) as pool:
            backends = list(pool.map(snapshot, range(len(self.clients))))
        return {
            "ready": sum(row["ready"] for row in backends),
            "reserved": sum(row["reserved"] for row in backends),
            "backends": backends,
        }

    def close(self) -> None:
        """Release this router's sessions while leaving borrowed services running."""
        with self.lock:
            self.closed.update(self.admitted)
            owners = list(self.attempts)
        try:
            self.cancel_owners(owners)
        finally:
            for client in self.clients:
                client.close()


def main() -> None:
    """Serve a simulator router with explicit backend capacity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True, help="URL=slots")
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    backends = [
        (url, int(slots))
        for url, slots in (item.rsplit("=", 1) for item in args.backend)
    ]
    ToolkitRouter(backends, args.group_size).serve(
        transport="http", host=args.host, port=args.port, parent_watch=args.parent_watch
    )


if __name__ == "__main__":
    main()

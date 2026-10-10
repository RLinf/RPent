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

"""Read-only liveness checks; never reconnect/reset a robot automatically."""

import threading

from rpent.dashboard.events import RuntimeStatusEvent
from rpent.utils.rpc import make_rpc_client


class RuntimeHealthMonitor:
    def __init__(self, state, endpoints):
        self._state = state
        self._clients = {name: make_rpc_client(url) for name, url in endpoints.items()}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def check(self):
        for name, client in self._clients.items():
            try:
                client.call("healthz", timeout_s=1)
                event = RuntimeStatusEvent(name, "ready")
            except Exception as exc:
                event = RuntimeStatusEvent(name, "failed", str(exc))
            self._state.emit(event)

    def _run(self):
        while not self._stop.is_set():
            self.check()
            self._stop.wait(2)

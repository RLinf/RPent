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

from rpent.dashboard import runtime_health


def test_runtime_health_detects_loss_and_recovery_without_observing(monkeypatch):
    calls = []
    events = []

    class Client:
        available = True

        def call(self, method, **kwargs):
            calls.append(method)
            assert kwargs["timeout_s"] == 1
            if not self.available:
                raise ConnectionError("env disconnected")
            return {"status": "ok"}

    class State:
        def emit(self, event):
            events.append(event)

    client = Client()
    monkeypatch.setattr(runtime_health, "make_rpc_client", lambda url: client)
    monitor = runtime_health.RuntimeHealthMonitor(State(), {"env": "http://test"})
    monitor.check()
    client.available = False
    monitor.check()
    client.available = True
    monitor.check()
    assert [event.status for event in events] == ["ready", "failed", "ready"]
    assert calls == ["healthz"] * 3

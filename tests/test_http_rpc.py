# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from rpent.utils.rpc.http_rpc import _build_opener


def test_loopback_rpc_bypasses_environment_proxy(monkeypatch) -> None:
    handlers = []

    def build_opener(*received):
        handlers.append(received)
        return object()

    monkeypatch.setattr("urllib.request.build_opener", build_opener)

    for hostname in ("127.0.0.1", "::1", "localhost"):
        _build_opener(hostname)

    assert all(call[0].proxies == {} for call in handlers)


def test_remote_rpc_keeps_environment_proxy(monkeypatch) -> None:
    handlers = []
    monkeypatch.setattr(
        "urllib.request.build_opener",
        lambda *received: handlers.append(received) or object(),
    )

    _build_opener("models.example.com")

    assert handlers == [()]

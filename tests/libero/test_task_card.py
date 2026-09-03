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

from types import SimpleNamespace

import pytest

from robots.libero.task_card.replay import execute, replay


class _Toolkit:
    def __init__(self, result: dict | None = None) -> None:
        self.result = result or {}
        self.state = SimpleNamespace(latest_step=0)

    def execute_tool(self, name: str, arguments: dict):
        return SimpleNamespace(result=self.result)

    def solved(self) -> bool:
        return False


def test_execute_rejects_tool_error() -> None:
    with pytest.raises(RuntimeError, match="move_to failed: unreachable"):
        execute(_Toolkit({"error": "unreachable"}), "move_to", {})


def test_replay_reuses_toolkit_opening_observation() -> None:
    toolkit = _Toolkit()
    toolkit.state.latest_step = 7

    result = replay(
        toolkit,
        molmo=SimpleNamespace(),
        card={"plan": [], "reference": {}, "source_of": {}},
    )

    assert result == {"done": False, "anchors": 0, "plan": 0}

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
"""Observable trace contract for the OneJev decision backend."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from rpent.planner.onejev import _DecisionTrace
from rpent.session import EnvState


def test_request_and_raw_response_are_saved_before_validation(tmp_path: Path) -> None:
    trace = _DecisionTrace(SimpleNamespace(state=EnvState(tmp_path)))
    trace.start(3)
    trace.record_request(
        {
            "model": "OneJev-9B",
            "state": {"task": "open the drawer"},
            "media": [{"type": "image", "data": "public RGB"}],
            "questions": {"next_action": {"type": "choice"}},
        }
    )
    state = json.loads((tmp_path / "onejev_state.json").read_text())
    question = json.loads((tmp_path / "onejev_question.json").read_text())
    decision_path = tmp_path / "onejev_decision.json"

    assert state == [
        {"turn": 3, "state": {"task": "open the drawer"}, "media_count": 1}
    ]
    assert question[0]["questions"]["next_action"]["type"] == "choice"
    assert json.loads(decision_path.read_text())[0]["status"] == "requested"

    trace.record_response({"http_status": 500, "output": {"error": "unavailable"}})
    decision = json.loads(decision_path.read_text())[0]
    assert decision["output"] == {"error": "unavailable"}
    assert decision["status"] == "responded"
    assert decision_path.read_text().startswith('[\n    {\n        "turn"')

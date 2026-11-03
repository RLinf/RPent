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

"""CLI to R2 toolkit to env server to RLinf's simulated controller, then finish."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e_tests.common import publish_check, run_scripted_policy_chain
from tests.e2e_tests.offline_planner_server import ScriptedToolCall


@pytest.mark.timeout(600)
def test_policy_chain(r2_mock_endpoint: str, tmp_path: Path) -> None:
    result = run_scripted_policy_chain(
        robot="r2",
        robot_argv=["--task-id", "0", "--env-endpoint", r2_mock_endpoint],
        output_dir=tmp_path / "run",
        action=ScriptedToolCall(
            "move_delta", {"arm": "left", "delta_xyz": [0.0, 0.0, 0.03]}
        ),
        action_count_field="ok",
        # Without a terminal the operator cannot answer, so finish is unverified.
        before_finish=(ScriptedToolCall("request_operator_verdict", {}),),
    )
    assert result["finish_status"] == "failure"
    publish_check("r2_policy_chain", result)

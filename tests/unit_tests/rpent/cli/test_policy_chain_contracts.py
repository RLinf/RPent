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

"""Offline coverage of GPU policy-chain artifact validation."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.tools import Toolkit
from tests.e2e_tests import common


@pytest.mark.parametrize("completion", ["accepted", "missing", "wrong_summary"])
def test_policy_chain_checks_accepted_finish_payload(tmp_path, monkeypatch, completion):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    toolkit = Toolkit(
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    planner = MagicMock(spec=common.OfflinePlannerServer)
    planner.base_url = "http://127.0.0.1:1/v1"
    planner.__enter__.return_value = planner

    def planner_server(script):
        if completion != "missing":
            arguments = dict(script[-1].arguments)
            if completion == "wrong_summary":
                arguments["summary"] = "unrelated completion"
            toolkit.execute_tool("finish", arguments)
        return planner

    def run_cli(*args, **kwargs):
        (output_dir / "transcript_test.json").write_text(
            json.dumps({"finish": toolkit.finish_result})
        )
        (output_dir / "states.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "command": {"action": "rldx_skill"},
                            "result": {"steps_applied": 1},
                        }
                    ]
                }
            )
        )
        return SimpleNamespace(returncode=0)

    (tmp_path / "rpent").touch()
    monkeypatch.setattr(common.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(common, "OfflinePlannerServer", planner_server)
    monkeypatch.setattr(common.subprocess, "run", run_cli)
    arguments = {
        "robot": "robocasa",
        "robot_argv": [],
        "output_dir": output_dir,
        "action": common.ScriptedToolCall("rldx_skill", {}),
        "action_count_field": "steps_applied",
    }
    if completion == "accepted":
        result = common.run_scripted_policy_chain(**arguments)
        assert result["finish_status"] == "stuck"
        assert result["simulator_action_count"] == 1
    else:
        with pytest.raises(RuntimeError, match="did not call finish"):
            common.run_scripted_policy_chain(**arguments)
    planner.assert_complete.assert_called_once()

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

import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.libero import robot_spec
from robots.libero.flash import replay


def _args(*extra):
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", default="flash")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--memory-dir", default=None)
    robot_spec._add_cli_args(parser, use_dashboard=False)
    return parser.parse_args(["--suite", "libero_object_swap", "--task", "3", *extra])


def test_locator_endpoint_and_selected_runtime(monkeypatch, tmp_path):
    args = _args("--locator-endpoint", "http://127.0.0.1:8115")
    robot_spec._parse_config(args)
    rpc = Mock()
    monkeypatch.setattr(robot_spec, "make_rpc_client", lambda endpoint: rpc)
    monkeypatch.setattr(robot_spec, "try_spawn_server", lambda *a: a[-1]())
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *a, post_fn: post_fn())
    daemons, kwargs = robot_spec._init_runtime(args, tmp_path, None, {"locator"})
    assert daemons == []  # Borrowed endpoints are not owned subprocesses.
    rpc.call.return_value = {"point_xy": [12, 34]}
    assert kwargs["locator_client"].locate(b"png", "cup") == (12, 34)
    assert rpc.call.call_args.kwargs["timeout_s"] == 180.0
    args.planner = "api"
    assert robot_spec._init_runtime(args, tmp_path, None, {"locator"}) == ([], {})


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--locator-endpoint", "http://localhost:1", "--planner", "api"],
    ],
)
def test_invalid_flash_locator_configuration(extra):
    with pytest.raises(ValueError, match="locator"):
        robot_spec._parse_config(_args(*extra))


@pytest.mark.parametrize("point_locator", ["point", "molmo"])
def test_point_plan_localization_and_motion(monkeypatch, tmp_path, point_locator):
    plan = [
        {
            "action": "move_to",
            "arguments": {"xyz": [0, 0, 0.72], "gripper": -1},
            "anchor": "cup",
            "anchor_distance": 0.1,
            "offset": [0.02, -0.03],
        },
        {"action": "set_gripper", "arguments": {"value": 1}},
        {
            "action": "move_to",
            "arguments": {"xyz": [0, 0, 0.72], "gripper": 1},
            "anchor": "cup",
            "anchor_distance": 0.1,
            "offset": [0.02, -0.03],
        },
        {"action": "release", "arguments": {}},
    ]
    (tmp_path / "object_swap_t3_plan.json").write_text(json.dumps({"plan": plan}))
    (tmp_path / "object_swap_t3_anchors.json").write_text(
        json.dumps(
            {
                "anchors": [
                    {"phrase": "cup", "median_xy": [0, 0], "locator": point_locator}
                ]
            }
        )
    )
    program = replay.load(tmp_path, "object_swap_t3")
    assert program["locator_of"] == {"cup": "point"}
    state = SimpleNamespace(latest_step=0)
    state.load_bytes = lambda name, step: name.encode()
    state.get = lambda step: SimpleNamespace(state={"robot0_eef_pos": [0.1, 0.2, 0.7]})
    locator = Mock(locate=Mock(return_value=(80, 40)))
    calls = []

    def execute(name, arguments):
        calls.append((name, arguments))
        state.latest_step += 1
        return SimpleNamespace(result={})

    def back_project(**kwargs):
        assert kwargs["resolution"] == "high"
        if kwargs["camera"] == "agentview":
            xyz = [0.1, 0.2, 0.5]
        elif kwargs["step"] == 1:
            xyz = [0.11, 0.21, 0.5]
        else:
            xyz = [0.14, 0.22, 0.5]
        return {"world_xyz": xyz}

    monkeypatch.setattr(replay.libero_tools, "back_project", back_project)
    toolkit = SimpleNamespace(state=state, execute_tool=execute, solved=lambda: False)
    result = replay.replay(toolkit, locator, program)
    assert result["anchors"] == 1
    queries = [call.args for call in locator.locate.call_args_list]
    assert queries == [
        (b"agentview_high.png", "cup"),
        (b"wrist_high.png", "the center of the cup directly below the gripper"),
        (b"wrist_high.png", "the body of the object held in the gripper"),
    ]
    moves = [args["xyz"] for name, args in calls if name == "move_to"]
    np.testing.assert_allclose(
        moves, [[0.1, 0.2, 0.72], [0.13, 0.18, 0.72], [0.1131, 0.1571, 0.72]]
    )


def test_missing_point_does_not_back_project(monkeypatch):
    state = SimpleNamespace(load_bytes=lambda *a, **kw: b"png")
    project = Mock()
    monkeypatch.setattr(replay.libero_tools, "back_project", project)
    locator = Mock(locate=Mock(return_value=None))
    assert replay.locate(locator, state, 0, "agentview", "cup") is None
    assert replay.held_body(locator, state, 0, "cup") is None
    project.assert_not_called()


@pytest.mark.parametrize("flag", ["--molmo-endpoint", "--locator-timeout"])
def test_removed_locator_flags_are_rejected(flag):
    with pytest.raises(SystemExit):
        _args(flag, "unused")


def test_unknown_plan_locator_is_rejected(tmp_path):
    (tmp_path / "unknown_plan.json").write_text('{"plan": []}')
    (tmp_path / "unknown_anchors.json").write_text(
        json.dumps(
            {"anchors": [{"phrase": "cup", "median_xy": [0, 0], "locator": "unknown"}]}
        )
    )
    with pytest.raises(ValueError, match="unsupported Flash locator"):
        replay.load(tmp_path, "unknown")

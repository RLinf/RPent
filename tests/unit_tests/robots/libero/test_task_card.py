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

import json
from types import SimpleNamespace

import pytest

from robots.libero.robot_spec import _parse_config
from robots.libero.task_card.replay import (
    cards,
    execute,
    load,
    locate,
    pick_succeeded,
    replay,
)
from rpent.planner.task_card import TaskCardPlanner
from rpent.robots.components.molmo_client import MolmoResult
from rpent.tools import ToolResult


def test_locate_uses_native_back_projection(make_toolkit):
    grounded = []

    def ground(image, query):
        grounded.append((image, query))
        return MolmoResult(found=True, point_xy=(4, 4))

    molmo = SimpleNamespace(ground=ground)
    toolkit, env, _ = make_toolkit(molmo_client=molmo)
    opening = toolkit.state.latest_step

    found = locate(toolkit.molmo_client, toolkit, opening, "agentview", "bowl")

    assert found is not None
    assert found["xy"] == pytest.approx([0.0, 0.0])
    assert found["z_top"] == pytest.approx(1.0)
    assert grounded[0][0].startswith(b"\x89PNG")
    assert grounded[0][1] == "bowl"
    assert toolkit.state.latest_step == opening
    assert env.actions == []


def test_task_card_planner_executes_native_toolkit(make_toolkit, tmp_path, monkeypatch):
    (tmp_path / "object_task_t0_plan.json").write_text(
        json.dumps({"plan": [{"action": "pi0_pick", "arguments": {"prompt": "bowl"}}]})
    )
    (tmp_path / "object_task_t0_anchors.json").write_text('{"anchors": []}')
    monkeypatch.setattr("robots.libero.task_card.replay.CARDS", tmp_path)
    toolkit, env, model = make_toolkit(molmo_client=SimpleNamespace())
    env.after_step = lambda: setattr(env, "terminated", True)

    result = TaskCardPlanner(recipe_tag="object_task_t0_s1", robot_name="libero").solve(
        system_prompt="", user_message="", toolkit=toolkit, max_turns=1
    )

    assert result.error is None
    assert result.finish_result["status"] == "success"
    assert result.stats["total_input_tokens"] == 0
    assert model.instructions == ["bowl"]
    assert env.reset_calls == 1
    assert toolkit.solved()


@pytest.mark.parametrize(
    ("gripper_open_thresh", "descent_thresh", "success"),
    [(0.003, 0.0, True), (0.05, 0.0, False), (0.003, 0.10, False)],
)
def test_native_pick_applies_task_card_thresholds(
    make_toolkit, monkeypatch, gripper_open_thresh, descent_thresh, success
):
    toolkit, env, _ = make_toolkit()

    def lift(ctx, prompt):
        env.pos[2] = 0.36
        obs = env.obs()
        obs["states"][-2:] = [0.02, -0.02]
        ctx.robot.set_obs(obs)

    monkeypatch.setattr("robots.libero.tools._vlm_chunk", lift)
    result = toolkit.execute_tool(
        "pi0_pick",
        {
            "prompt": "bowl",
            "max_chunks": 1,
            "gripper_open_thresh": gripper_open_thresh,
            "descent_thresh": descent_thresh,
        },
    )

    assert not result.is_error
    assert result.data["log"]["result"]["success"] is success


class _Toolkit:
    def __init__(self, result: dict | None = None) -> None:
        self.result = result or {}
        self.state = SimpleNamespace(latest_step=0)
        self.calls = []

    def execute_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return ToolResult(data=self.result, error=self.result.get("error"))

    def solved(self) -> bool:
        return False


def test_execute_rejects_tool_error() -> None:
    with pytest.raises(RuntimeError, match="move_to failed: unreachable"):
        execute(_Toolkit({"error": "unreachable"}), "move_to", {})


def test_pick_succeeded_uses_pi0_pick_success_contract() -> None:
    wrapped = {"log": {"result": {"success": True, "peak_lift_m": 0.05}}}

    assert pick_succeeded(wrapped) is True
    assert (
        pick_succeeded(
            {
                "log": {
                    "result": {
                        "success": False,
                        "peak_lift_m": 0.10,
                        "final_gripper_opening": 0.01,
                    }
                }
            }
        )
        is False
    )


def test_replay_reuses_toolkit_opening_observation() -> None:
    toolkit = _Toolkit()
    toolkit.state.latest_step = 7

    result = replay(
        toolkit,
        molmo=SimpleNamespace(),
        card={"plan": [], "reference": {}, "locator_of": {}},
    )

    assert result == {"done": False, "anchors": 0, "plan": 0}


def test_replay_passes_legacy_pick_thresholds_to_pi0_pick() -> None:
    toolkit = _Toolkit({"success": True})

    replay(
        toolkit,
        molmo=SimpleNamespace(),
        card={
            "plan": [
                {
                    "action": "pi0_pick",
                    "arguments": {
                        "prompt": "pick up the bowl",
                        "lift_thresh": 0.08,
                    },
                }
            ],
            "reference": {},
            "locator_of": {},
        },
    )

    assert toolkit.calls == [
        (
            "pi0_pick",
            {
                "prompt": "pick up the bowl",
                "lift_thresh": 0.04,
                "gripper_closed_thresh": 0.07,
                "gripper_open_thresh": 0.003,
                "descent_thresh": 0.0,
            },
        )
    ]


def test_pick_retry_reuses_relocated_move_arguments() -> None:
    class RetryToolkit(_Toolkit):
        def execute_tool(self, name: str, arguments: dict):
            self.calls.append((name, dict(arguments)))
            if name == "segment":
                result = {"world_xyz": [0.2, 0.1, 0.0]}
            elif name == "pi0_pick":
                result = {"success": False}
            else:
                result = {}
            return ToolResult(data=result)

    toolkit = RetryToolkit()
    replay(
        toolkit,
        molmo=SimpleNamespace(),
        card={
            "plan": [
                {
                    "action": "move_to",
                    "arguments": {"xyz": [0.0, 0.0, 0.7], "gripper": -1},
                    "anchor": "bowl",
                    "anchor_distance": 0.0,
                    "offset": [0.01, -0.02],
                },
                {"action": "pi0_pick", "arguments": {"prompt": "pick up the bowl"}},
            ],
            "reference": {"bowl": [0.0, 0.0]},
            "locator_of": {"bowl": "segment"},
        },
    )

    retried_moves = [args for name, args in toolkit.calls if name == "move_to"]
    assert len(retried_moves) == 3
    assert all(args["xyz"] == [0.21, 0.08, 0.7] for args in retried_moves)


def test_replay_stops_when_attached_anchor_is_not_located() -> None:
    toolkit = _Toolkit()
    notes = []

    result = replay(
        toolkit,
        molmo=SimpleNamespace(),
        card={
            "plan": [
                {
                    "action": "move_to",
                    "arguments": {"xyz": [0.3, 0.2, 0.7], "gripper": -1},
                    "anchor": "bowl",
                    "anchor_distance": 0.0,
                    "offset": [0.01, -0.02],
                },
                {"action": "release", "arguments": {}},
            ],
            "reference": {"bowl": [0.0, 0.0]},
            "locator_of": {"bowl": "segment"},
        },
        note=notes.append,
    )

    assert result["done"] is False
    assert [name for name, _ in toolkit.calls] == ["segment"]
    assert any("unavailable; stopping replay" in note for note in notes)


def test_replay_propagates_toolkit_exceptions() -> None:
    class FailingToolkit(_Toolkit):
        def execute_tool(self, name: str, arguments: dict):
            raise ConnectionError("RPC disconnected")

    with pytest.raises(ConnectionError, match="RPC disconnected"):
        replay(
            FailingToolkit(),
            molmo=SimpleNamespace(),
            card={
                "plan": [
                    {
                        "action": "move_to",
                        "arguments": {"xyz": [0.1, 0.1, 0.7], "gripper": -1},
                    }
                ],
                "reference": {},
                "locator_of": {},
            },
        )


def test_task_card_rejects_unsupported_suite() -> None:
    args = SimpleNamespace(
        suite="libero_goal_swap",
        task=0,
        planner="task_card",
        molmo_endpoint="http://127.0.0.1:8115",
    )

    with pytest.raises(ValueError, match="supported suites: libero_object_swap"):
        _parse_config(args)


def test_non_task_card_planner_rejects_molmo_endpoint() -> None:
    args = SimpleNamespace(
        suite="libero_object_swap",
        task=0,
        planner="api",
        molmo_endpoint="http://127.0.0.1:8115",
    )

    with pytest.raises(ValueError, match="requires --planner task_card"):
        _parse_config(args)


def test_missing_cards_explains_where_to_download(monkeypatch, tmp_path) -> None:
    sync_calls = []
    monkeypatch.setattr(
        "rpent.memory.MemoryManager.sync",
        lambda *args, **kwargs: sync_calls.append(kwargs),
    )

    with pytest.raises(FileNotFoundError, match="RLinf/RPent-memory") as error:
        cards(tmp_path / "task_card")

    assert "--cards" not in str(error.value)
    assert sync_calls == [
        {
            "remote_repo": "RLinf/RPent-memory",
            "allow_patterns": ("libero/task_card/**",),
        }
    ]


def test_cards_do_not_require_an_index(tmp_path) -> None:
    root = tmp_path / "task_card"
    root.mkdir(parents=True)
    (root / "object_swap_t0_plan.json").write_text('{"plan": []}')

    assert cards(root) == root


def test_load_reads_only_runtime_card_fields(tmp_path) -> None:
    (tmp_path / "object_swap_t0_plan.json").write_text('{"plan": []}')
    (tmp_path / "object_swap_t0_anchors.json").write_text(
        '{"anchors": [{"phrase": "bowl", "locator": "segment", '
        '"median_xy": [0.1, 0.2]}]}'
    )

    card = load(tmp_path, "object_swap_t0")

    assert card["plan"] == []
    assert card["reference"]["bowl"].tolist() == [0.1, 0.2]
    assert card["locator_of"] == {"bowl": "segment"}

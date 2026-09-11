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

from robots.libero.robot_spec import TASK_CARD_SUITES, _parse_config
from robots.libero.task_card.replay import cards, execute, load, pick_succeeded, replay


class _Toolkit:
    def __init__(self, result: dict | None = None) -> None:
        self.result = result or {}
        self.state = SimpleNamespace(latest_step=0)
        self.calls = []

    def execute_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return SimpleNamespace(result=self.result)

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
            return SimpleNamespace(result=result)

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


def test_task_card_supports_all_libero_pro_task_and_swap_suites() -> None:
    assert TASK_CARD_SUITES == {
        f"libero_{family}_{regime}"
        for family in ("spatial", "object", "goal", "10")
        for regime in ("task", "swap")
    }


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

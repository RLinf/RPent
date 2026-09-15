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

"""Offline tests of source provenance, arm frames and operator decisions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.franka.task_card import common
from robots.franka.task_card import replay as replay_module
from robots.franka.task_card.generate import generate
from rpent.session import EnvState


def robot_state(robot, xyz=(0.4, 0.0, 0.3)):
    pose = [*xyz, 0, 0, 0, 1]
    return (
        {"raw_base_state": {"tcp_pose": pose}}
        if robot == "franka"
        else {
            "coordinate_frame": "right_base",
            "left_arm": {"tcp_pose": pose},
            "right_arm": {"tcp_pose": pose},
        }
    )


def source(tmp_path, robot):
    state = EnvState(tmp_path)
    with state.record_step(state=robot_state(robot)):
        pass
    args = {"arm": "left"} if robot == "dual_franka" else {}
    with state.record_step(
        state=robot_state(robot, (0.5, 0, 0.3)),
        command={"action": "move_delta", "delta_xyz": [0.1, 0, 0], **args},
        result={"ok": True},
    ):
        pass
    return state


@pytest.mark.parametrize("robot", ["franka", "dual_franka"])
def test_generate_and_retarget_changed_scene(tmp_path, monkeypatch, robot):
    if robot == "dual_franka":
        monkeypatch.setattr(
            "robots.dual_franka.perception.transform_point_between_base_frames",
            lambda point, **kw: point,
        )
    source(tmp_path, robot)
    import importlib

    generator = importlib.import_module("robots.franka.task_card.generate")
    monkeypatch.setattr(generator, "fingerprint", lambda: {"test": "same"})
    monkeypatch.setattr(generator, "localize", lambda *a, **k: np.array([0.5, 0, 0.2]))
    card = generate(
        tmp_path,
        {
            "1": {
                "phrase": "cup",
                "camera": "base" if robot == "dual_franka" else "third_person",
            }
        },
        robot=robot,
        task=f"{robot}_t1",
        molmo=None,
        human_verdict="success",
    )
    assert card["plan"][0]["offset_xyz"] == pytest.approx([0, 0, 0.1])
    monkeypatch.setattr(
        replay_module, "localize", lambda *a, **k: np.array([0.45, 0.02, 0.2])
    )
    workspace = {
        "ee_pose_limit_min": [[0, -1, 0]] * 2 if robot == "dual_franka" else [0, -1, 0],
        "ee_pose_limit_max": [[1, 1, 1]] * 2 if robot == "dual_franka" else [1, 1, 1],
    }
    state = SimpleNamespace(get=lambda: SimpleNamespace(state=robot_state(robot)))
    args = replay_module.motion_arguments(
        card["plan"][0], robot=robot, state=state, molmo=None, workspace=workspace
    )
    assert args["delta_xyz"] == pytest.approx([0.05, 0.02, 0])
    if robot == "dual_franka":
        assert args["arm"] == "left"


def test_failed_source_cannot_be_silently_filtered(tmp_path):
    state = source(tmp_path, "franka")
    with state.record_step(
        state=robot_state("franka"),
        command={"action": "close_gripper"},
        result={"ok": False},
    ):
        pass
    with pytest.raises(ValueError, match="human-confirmed"):
        generate(
            tmp_path,
            {},
            robot="franka",
            task="franka_t1",
            molmo=None,
            human_verdict="failure",
        )
    # An all-gripper source isolates error handling from perception.
    data = json.loads((tmp_path / "states.json").read_text())
    data["steps"][1]["command"] = {"action": "open_gripper"}
    (tmp_path / "states.json").write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="successfully"):
        generate(
            tmp_path,
            {},
            robot="franka",
            task="franka_t1",
            molmo=None,
            human_verdict="success",
        )


def test_generator_requires_explicit_motion_intent(tmp_path):
    source(tmp_path, "franka")
    with pytest.raises(ValueError, match="step 1 requires"):
        generate(
            tmp_path,
            {},
            robot="franka",
            task="franka_t1",
            molmo=None,
            human_verdict="success",
        )


def test_recorded_state_does_not_overwrite_source(tmp_path):
    source(tmp_path, "franka")
    before = (tmp_path / "states.json").read_bytes()
    state = common.RecordedState(tmp_path)
    state.save("camera_meta.json", {"wrong": True}, step=0)
    assert (tmp_path / "states.json").read_bytes() == before


def card():
    return {
        "version": 1,
        "robot": "dual_franka",
        "task": "dual_franka_t1",
        "human_verdict": "success",
        "plan": [
            {"source_step": 1, "action": "open_gripper", "arguments": {"arm": "left"}},
            {"source_step": 2, "action": "open_gripper", "arguments": {"arm": "right"}},
            {
                "source_step": 3,
                "action": "vla_right_grasp",
                "arguments": {"prompt": "grasp cup", "max_chunks": 2},
            },
        ],
    }


class FakeToolkit:
    def __init__(self, root, fail=False):
        self.state = EnvState(root)
        self.calls = []
        self.fail = fail

    def raise_if_cancelled(self):
        pass

    def refresh_task_card_state(self):
        with self.state.record_step(state=robot_state("dual_franka")):
            pass

    def execute_tool(self, action, args):
        self.calls.append((action, args))
        raw = {"ok": not self.fail}
        with self.state.record_step(
            state=robot_state("dual_franka"),
            command={"action": action, **args},
            result=raw,
        ):
            pass
        # Return rendered output: failure must be caught in the raw record.
        return SimpleNamespace(result={"state": {}})


@pytest.mark.parametrize("verdict", ["success", "failure"])
def test_dual_order_and_human_authority(tmp_path, verdict):
    toolkit = FakeToolkit(tmp_path)
    answers = iter(["start", verdict])
    outcome = replay_module.replay(
        toolkit, card(), None, {}, human=lambda *a: next(answers)
    )
    assert [args.get("arm") for _, args in toolkit.calls] == ["left", "right", None]
    assert outcome["done"] == (verdict == "success")
    assert toolkit._task_card_solved == outcome["done"]
    saved = json.loads((tmp_path / "task_card_outcome.json").read_text())
    assert saved["human_verdict"] == verdict
    assert len(saved["events"]) == 5


def test_abort_never_calls_a_primitive(tmp_path):
    toolkit = FakeToolkit(tmp_path)
    result = replay_module.replay(toolkit, card(), None, {}, human=lambda *a: "abort")
    assert not toolkit.calls
    assert result["status"] == "aborted"


def test_one_arm_error_stops_remaining_steps_and_records_failure(tmp_path):
    toolkit = FakeToolkit(tmp_path, fail=True)
    with pytest.raises(RuntimeError):
        replay_module.replay(toolkit, card(), None, {}, human=lambda *a: "start")
    assert len(toolkit.calls) == 1
    assert not toolkit._task_card_solved
    assert (
        json.loads((tmp_path / "task_card_outcome.json").read_text())["status"]
        == "failure"
    )


def test_fixed_rotation_uses_quaternion_composition():
    current = Rotation.from_euler("xyz", [0.1, -0.1, 0.15])
    target = Rotation.from_euler("xyz", [-0.05, 0.05, 0.1])
    pose = robot_state("franka")
    pose["raw_base_state"]["tcp_pose"][3:] = current.as_quat().tolist()
    entry = {
        "action": "rotate_delta",
        "arguments": {"delta_rpy": [0, 0, 0]},
        "rotation_mode": "fixed",
        "target_quaternion": target.as_quat().tolist(),
    }
    args = replay_module.motion_arguments(
        entry,
        robot="franka",
        state=SimpleNamespace(get=lambda: SimpleNamespace(state=pose)),
        molmo=None,
        workspace={},
    )
    delta = Rotation.from_euler("xyz", args["delta_rpy"])
    assert (delta * current * target.inv()).magnitude() < 1e-8


def test_left_grounding_preserves_current_shared_frame(tmp_path, monkeypatch):
    from PIL import Image

    from robots.dual_franka import perception

    path = tmp_path / "base.png"
    Image.new("RGB", (20, 20)).save(path)
    state = SimpleNamespace(artifact_path=lambda *a, **k: path)
    molmo = SimpleNamespace(
        ground=lambda *a: SimpleNamespace(found=True, point_xy=[10, 10])
    )
    monkeypatch.setattr(
        perception,
        "back_project",
        lambda **k: {"selection_valid": True, "point_xyz": [0.5, 0.7, 0.2]},
    )
    calls = []

    def transform(point, **kwargs):
        calls.append(kwargs)
        return point - [0.01, 0.69, 0]

    monkeypatch.setattr(perception, "transform_point_between_base_frames", transform)
    result = common.localize(
        state, "dual_franka", {"camera": "base", "phrase": "cup"}, molmo, arm="left"
    )
    assert result == pytest.approx([0.5, 0.7, 0.2])
    assert calls == []


def test_reset_gate_precedes_runtime(monkeypatch):
    monkeypatch.setattr(replay_module, "ask_human", lambda *a: "abort")
    with pytest.raises(RuntimeError, match="before environment initialization"):
        replay_module.authorize_runtime(SimpleNamespace(planner="task_card"))


def test_invalid_projection_never_falls_back_to_recorded_delta(tmp_path, monkeypatch):
    toolkit = FakeToolkit(tmp_path)
    plan = card()
    plan["plan"] = [
        {
            "source_step": 1,
            "action": "move_delta",
            "arguments": {"arm": "right", "delta_xyz": [0.1, 0, 0]},
            "anchor": {"phrase": "cup", "camera": "base"},
            "offset_xyz": [0, 0, 0.1],
        }
    ]

    def missing(*args, **kwargs):
        raise ValueError("anchor not found")

    monkeypatch.setattr(replay_module, "localize", missing)
    with pytest.raises(ValueError, match="anchor not found"):
        replay_module.replay(toolkit, plan, None, {}, human=lambda *a: "start")
    assert toolkit.calls == []
    assert not json.loads((tmp_path / "task_card_outcome.json").read_text())["done"]


def test_relative_rotation_rejects_changed_start():
    entry = {
        "action": "rotate_delta",
        "arguments": {"delta_rpy": [0, 0, 0.1]},
        "rotation_mode": "relative",
        "reference_quaternion": Rotation.from_euler("z", 0.5).as_quat().tolist(),
    }
    with pytest.raises(ValueError, match="starting pose"):
        replay_module.motion_arguments(
            entry,
            robot="franka",
            state=SimpleNamespace(
                get=lambda: SimpleNamespace(state=robot_state("franka"))
            ),
            molmo=None,
            workspace={},
        )


def test_prepare_checks_identity_and_configuration_before_runtime(
    tmp_path, monkeypatch
):
    prepared = card()
    prepared["requirements"] = {"test": "source"}
    path = tmp_path / "card.json"
    path.write_text(json.dumps(prepared))
    args = SimpleNamespace(
        planner="task_card",
        task_card=str(path),
        task_id=1,
        molmo_endpoint="http://test:1",
        calibration_path="/tmp/calibration",
    )
    monkeypatch.setattr(replay_module, "fingerprint", lambda: {"test": "changed"})
    with pytest.raises(ValueError, match="calibration changed"):
        replay_module.prepare(args, "dual_franka")
    with pytest.raises(ValueError, match="robot/task"):
        replay_module.prepare(args, "franka")


def test_both_specs_use_existing_replay_hook():
    from robots.dual_franka.robot_spec import get_robot_spec as dual
    from robots.franka.robot_spec import get_robot_spec as single

    assert single().replay_card is dual().replay_card
    assert single().replay_card is not None

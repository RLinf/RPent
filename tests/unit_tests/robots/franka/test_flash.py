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

"""Offline coverage for both Franka Flash adapters."""

import argparse
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from robots.franka.flash import common
from robots.franka.flash.replay import (
    authorize_runtime,
    motion_arguments,
    prepare,
    replay,
)
from rpent.robots.components.molmo_client import MolmoResult


@pytest.fixture(params=["franka", "dual_franka"])
def scene(request, tmp_path):
    robot = request.param
    image = tmp_path / "frame.png"
    Image.new("RGB", (20, 20)).save(image)
    pose = [0.4, 0, 0.3, 0, 0, 0, 1]
    state_data = {
        "raw_base_state": {"tcp_pose": pose},
        "coordinate_frame": "right_base",
        "left_arm": {"tcp_pose": pose},
        "right_arm": {"tcp_pose": pose},
    }
    saved, actions = {}, []
    state = SimpleNamespace(
        get=lambda *a: SimpleNamespace(state=state_data),
        artifact_path=lambda *a, **kw: image,
        latest_step=0,
        latest_record=lambda: SimpleNamespace(result={"ok": True}),
        save=lambda name, value, **kw: saved.update({name: value}),
    )
    entry = {
        "source_step": 1,
        "action": "move_delta",
        "arguments": {"delta_xyz": [0.01, 0, 0]},
        "anchor": {
            "camera": "third_person" if robot == "franka" else "base",
            "phrase": "cup rim",
        },
        "offset_xyz": [0, 0, 0],
    }
    if robot == "dual_franka":
        entry["arguments"]["arm"] = "right"
    card = {
        "version": 1,
        "robot": robot,
        "task": f"{robot}_t0",
        "requirements": {},
        "human_verdict": "success",
        "plan": [entry],
    }
    limits = [-1, -1, -1], [1, 1, 1]
    workspace = dict(
        zip(
            ("ee_pose_limit_min", "ee_pose_limit_max"),
            limits if robot == "franka" else ([limits[0]] * 2, [limits[1]] * 2),
        )
    )
    toolkit = SimpleNamespace(
        state=state,
        raise_if_cancelled=lambda: None,
        refresh_flash_state=lambda: None,
        execute_tool=lambda name, args: (
            actions.append((name, args)) or SimpleNamespace(result={"ok": True})
        ),
    )
    grounder = SimpleNamespace(ground=lambda *a: MolmoResult(True, (10, 10), (20, 20)))
    return SimpleNamespace(
        robot=robot,
        card=card,
        state=state,
        workspace=workspace,
        toolkit=toolkit,
        grounder=grounder,
        saved=saved,
        actions=actions,
    )


def projection(monkeypatch, scene, point=None, error=None):
    result = {"point_base": point, "point_xyz": point, "selection_valid": True}
    if error:
        result["error"] = error
    module = importlib.import_module(f"robots.{scene.robot}.perception")
    monkeypatch.setattr(module, "back_project", lambda **kw: result)


def test_live_point_changes_motion_and_operator_confirms_success(scene, monkeypatch):
    projection(monkeypatch, scene, [0.45, 0, 0.3])
    answers = iter(["start", "success"])
    outcome = replay(
        scene.toolkit,
        scene.card,
        scene.grounder,
        scene.workspace,
        human=lambda *a: next(answers),
    )
    assert outcome["done"] and scene.toolkit._flash_solved
    np.testing.assert_allclose(scene.actions[0][1]["delta_xyz"], [0.05, 0, 0])
    assert scene.saved["flash_outcome.json"]["human_verdict"] == "success"


@pytest.mark.parametrize(
    "failure", ["missing", "depth", "workspace", "distance", "bounds"]
)
def test_invalid_grounding_never_moves(scene, monkeypatch, failure):
    projection(
        monkeypatch,
        scene,
        [2, 0, 0.3]
        if failure == "workspace"
        else [0.7, 0, 0.3]
        if failure == "distance"
        else [0.45, 0, 0.3],
        error="no valid depth near pixel" if failure == "depth" else None,
    )
    if failure == "missing":
        scene.grounder.ground = lambda *a: MolmoResult(False, None)
    if failure == "bounds":
        scene.grounder.ground = lambda *a: MolmoResult(True, (100, 100))
    with pytest.raises(ValueError):
        replay(
            scene.toolkit,
            scene.card,
            scene.grounder,
            scene.workspace,
            human=lambda *a: "start",
        )
    assert scene.actions == []
    assert scene.saved["flash_outcome.json"]["status"] == "failure"


def test_operator_abort_does_not_execute(scene):
    outcome = replay(
        scene.toolkit,
        scene.card,
        scene.grounder,
        scene.workspace,
        human=lambda *a: "abort",
    )
    assert not outcome["done"] and not scene.actions


def test_failed_primitive_stops_before_next_action(scene, monkeypatch):
    projection(monkeypatch, scene, [0.45, 0, 0.3])
    scene.toolkit.execute_tool = lambda *a: SimpleNamespace(
        result={"error": "controller fault"}
    )
    with pytest.raises(RuntimeError, match="controller fault"):
        replay(
            scene.toolkit,
            scene.card,
            scene.grounder,
            scene.workspace,
            human=lambda *a: "start",
        )
    assert not scene.toolkit._flash_solved


def test_card_prepare_checks_fingerprint_before_runtime(scene, monkeypatch, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(scene.card))
    module = importlib.import_module("robots.franka.flash.replay")
    monkeypatch.setattr(module, "fingerprint", lambda: {"robot_config": "changed"})
    args = argparse.Namespace(
        planner="flash", flash_plan=path, molmo_endpoint="http://localhost:1", task_id=0
    )
    with pytest.raises(ValueError, match="configuration or calibration changed"):
        prepare(args, scene.robot)


def test_hook_registered_for_both_robots(scene):
    spec = importlib.import_module(f"robots.{scene.robot}.robot_spec").get_robot_spec()
    assert callable(spec.run_flash)
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, False)
    args = parser.parse_args(
        ["--flash-plan", "plan.json", "--molmo-endpoint", "http://localhost:1"]
    )
    assert args.flash_plan == "plan.json"


def test_runtime_confirmation_refusal(monkeypatch):
    module = importlib.import_module("robots.franka.flash.replay")
    monkeypatch.setattr(module, "ask_human", lambda *a: "abort")
    with pytest.raises(RuntimeError, match="aborted"):
        authorize_runtime(argparse.Namespace(planner="flash"))


def test_left_arm_workspace_uses_left_base(scene, monkeypatch):
    if scene.robot != "dual_franka":
        pytest.skip("left-base conversion only applies to dual Franka")
    projection(monkeypatch, scene, [0.45, 0, 0.3])
    scene.card["plan"][0]["arguments"]["arm"] = "left"
    module = importlib.import_module("robots.dual_franka.perception")
    monkeypatch.setattr(
        module,
        "transform_point_between_base_frames",
        lambda *a, **kw: np.array([2, 0, 0]),
    )
    with pytest.raises(ValueError, match="workspace"):
        motion_arguments(
            scene.card["plan"][0],
            robot=scene.robot,
            state=scene.state,
            molmo=scene.grounder,
            workspace=scene.workspace,
        )


def test_fingerprint_includes_calibration_content(tmp_path, monkeypatch):
    config, calibration = tmp_path / "robot.yaml", tmp_path / "camera.yaml"
    config.write_text("robot")
    calibration.write_text("first")
    monkeypatch.setattr(common, "get_robot_config_path", lambda: config)
    monkeypatch.setattr(
        common,
        "get_perception_calibration_mapping",
        lambda: {"external": str(calibration)},
    )
    first = common.fingerprint()
    calibration.write_text("second")
    assert first != common.fingerprint()


def test_generate_from_reviewed_recording(scene, tmp_path, monkeypatch):
    from robots.franka.flash.generate import generate

    module = importlib.import_module("robots.franka.flash.generate")
    monkeypatch.setattr(module, "fingerprint", lambda: {})
    monkeypatch.setattr(module, "localize", lambda *a, **kw: np.array([0.4, 0, 0.3]))
    state = scene.state.get().state
    steps = [
        {"step_idx": 0, "state": state},
        {
            "step_idx": 1,
            "state": state,
            "command": {"action": "move_delta", **scene.card["plan"][0]["arguments"]},
            "result": {"ok": True},
        },
    ]
    (tmp_path / "recording_fingerprint.json").write_text("{}")
    (tmp_path / "states.json").write_text(json.dumps({"steps": steps}))
    card = generate(
        tmp_path,
        {"1": {**scene.card["plan"][0]["anchor"], "intent": "approach rim"}},
        robot=scene.robot,
        task=scene.card["task"],
        molmo=scene.grounder,
        human_verdict="success",
    )
    np.testing.assert_allclose(card["plan"][0]["offset_xyz"], [0, 0, 0])
    with pytest.raises(ValueError, match="human-confirmed"):
        generate(
            tmp_path,
            {},
            robot=scene.robot,
            task=scene.card["task"],
            molmo=scene.grounder,
            human_verdict="failure",
        )


@pytest.mark.parametrize("failure_kind", ["depth", "rpc", "missing"])
def test_agent_fallback_is_shared_by_both_robots(scene, monkeypatch, failure_kind):
    from robots.franka.flash.grounding import localize_with_fallback

    calls = []
    projection(monkeypatch, scene, [0.45, 0, 0.3])
    module = importlib.import_module(f"robots.{scene.robot}.perception")
    monkeypatch.setattr(
        module,
        "back_project",
        lambda **kw: (
            {"error": "no valid depth near pixel"}
            if kw["col"] == 1
            else {
                "point_base": [0.45, 0, 0.3],
                "point_xyz": [0.45, 0, 0.3],
                "selection_valid": True,
            }
        ),
    )

    def molmo(*args):
        calls.append("molmo")
        if len(calls) == 1:
            if failure_kind == "rpc":
                raise ConnectionError("Molmo unavailable")
            if failure_kind == "missing":
                return MolmoResult(False, None)
            return MolmoResult(True, (1, 1))
        return MolmoResult(True, (10, 10))

    def agent(image, prompt):
        calls.append("agent")
        assert "cup rim" in prompt
        assert "previous selection failed" in prompt
        return MolmoResult(True, (10, 10))

    scene.grounder.ground = molmo
    fallback = SimpleNamespace(ground=agent)
    scene.toolkit.refresh_flash_state = lambda: calls.append("refresh")
    for _ in range(2):
        point = localize_with_fallback(
            scene.toolkit,
            scene.robot,
            scene.card["plan"][0]["anchor"],
            scene.grounder,
            arm="right" if scene.robot == "dual_franka" else None,
            agent=fallback,
        )
        np.testing.assert_allclose(point, [0.45, 0, 0.3])
    assert calls == ["molmo", "refresh", "agent", "molmo"]


def test_agent_failure_never_moves(scene, monkeypatch):
    projection(monkeypatch, scene, error="no valid depth near pixel")
    calls = []
    fallback = SimpleNamespace(
        ground=lambda *a: calls.append("agent") or MolmoResult(True, (10, 10))
    )
    with pytest.raises(ValueError, match="depth"):
        replay(
            scene.toolkit,
            scene.card,
            scene.grounder,
            scene.workspace,
            human=lambda *a: "start",
            grounding_agent=fallback,
        )
    assert calls == ["agent"] and scene.actions == []
    assert [e["provider"] for e in scene.saved["flash_grounding.json"]["attempts"]] == [
        "molmo",
        "agent",
    ]


def test_agent_success_reaches_motion(scene, monkeypatch):
    projection(monkeypatch, scene, [0.45, 0, 0.3])
    scene.grounder.ground = lambda *a: MolmoResult(False, None)
    fallback = SimpleNamespace(ground=lambda *a: MolmoResult(True, (10, 10)))
    answers = iter(["start", "success"])
    outcome = replay(
        scene.toolkit,
        scene.card,
        scene.grounder,
        scene.workspace,
        human=lambda *a: next(answers),
        grounding_agent=fallback,
    )
    assert outcome["done"] and len(scene.actions) == 1


def test_cancellation_does_not_trigger_agent(scene):
    from rpent.tools.toolkit import ToolCancelled

    def cancelled(*args):
        raise ToolCancelled("operator cancelled")

    scene.grounder.ground = cancelled
    calls = []
    fallback = SimpleNamespace(ground=lambda *a: calls.append("agent"))
    with pytest.raises(ToolCancelled):
        replay(
            scene.toolkit,
            scene.card,
            scene.grounder,
            scene.workspace,
            human=lambda *a: "start",
            grounding_agent=fallback,
        )
    assert not calls and not scene.actions


@pytest.mark.parametrize("failure", ["once", "always", "configuration"])
def test_dual_projection_exception_fallback(scene, monkeypatch, failure):
    if scene.robot != "dual_franka":
        pytest.skip("dual-Franka depth exception contract")
    from robots.dual_franka import perception

    calls = []

    def project(**kwargs):
        calls.append("project")
        if failure == "configuration":
            raise ValueError("missing calibration")
        if failure == "always" or calls.count("project") == 1:
            perception._median_depth(np.zeros((20, 20)), 10, 10, radius=2)
        return {"point_xyz": [0.45, 0, 0.3], "selection_valid": True}

    monkeypatch.setattr(perception, "back_project", project)
    fallback = SimpleNamespace(
        ground=lambda *a: calls.append("agent") or MolmoResult(True, (10, 10))
    )
    answers = iter(["start", "success"])
    kwargs = {"human": lambda *a: next(answers), "grounding_agent": fallback}
    if failure == "once":
        assert replay(
            scene.toolkit, scene.card, scene.grounder, scene.workspace, **kwargs
        )["done"]
        assert "agent" in calls and len(scene.actions) == 1
    else:
        with pytest.raises(ValueError, match="depth|calibration"):
            replay(scene.toolkit, scene.card, scene.grounder, scene.workspace, **kwargs)
        assert not scene.actions
        assert ("agent" in calls) == (failure == "always")


@pytest.mark.parametrize("provenance", [None, {"robot_config": "old"}])
def test_generation_rejects_missing_or_changed_calibration_before_grounding(
    scene, tmp_path, monkeypatch, provenance
):
    from robots.franka.flash.generate import generate

    module = importlib.import_module("robots.franka.flash.generate")
    monkeypatch.setattr(module, "fingerprint", lambda: {"robot_config": "new"})
    if provenance is not None:
        (tmp_path / "recording_fingerprint.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="provenance|changed since recording"):
        generate(
            tmp_path,
            {},
            robot=scene.robot,
            task=scene.card["task"],
            molmo=scene.grounder,
            human_verdict="success",
        )


@pytest.mark.parametrize("second_reset", [False, True])
def test_generation_filters_workflow_records_but_rejects_multiple_attempts(
    scene, tmp_path, monkeypatch, second_reset
):
    from robots.franka.flash.generate import generate

    module = importlib.import_module("robots.franka.flash.generate")
    monkeypatch.setattr(module, "fingerprint", lambda: {})
    reset = {"scene_reset_confirmed": True, "robot_reset": {"ok": True}}
    actions = [
        ("request_scene_reset", reset),
        ("open_gripper", {"ok": True}),
        ("observe_for_verdict", {}),
        ("request_operator_verdict", {"status": "success"}),
    ]
    if second_reset:
        actions.append(("request_scene_reset", reset))
    state = scene.state.get().state
    steps = [{"step_idx": 0, "state": state}]
    for index, (action, result) in enumerate(actions, 1):
        command = {"action": action}
        if action == "open_gripper" and scene.robot == "dual_franka":
            command["arm"] = "right"
        steps.append(
            {"step_idx": index, "state": state, "command": command, "result": result}
        )
    (tmp_path / "recording_fingerprint.json").write_text("{}")
    (tmp_path / "states.json").write_text(json.dumps({"steps": steps}))
    kwargs = {
        "robot": scene.robot,
        "task": scene.card["task"],
        "molmo": scene.grounder,
        "human_verdict": "success",
    }
    if second_reset:
        with pytest.raises(ValueError, match="multiple attempts"):
            generate(tmp_path, {}, **kwargs)
    else:
        card = generate(tmp_path, {}, **kwargs)
        assert [entry["action"] for entry in card["plan"]] == ["open_gripper"]
        assert card["plan"][0]["source_step"] == 2

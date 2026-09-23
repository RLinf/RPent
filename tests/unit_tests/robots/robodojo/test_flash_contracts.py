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

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo import env_server, robot_spec
from robots.robodojo.access import public_observation
from robots.robodojo.env_client import RoboDojoEnvClient
from robots.robodojo.flash.generate import generate_plan
from robots.robodojo.flash.grounding import ANCHOR_METHOD, mask_geometry
from robots.robodojo.flash.plan import validate_plan
from robots.robodojo.flash.replay import replay, run_flash
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.flash import FlashPlanner


def trace():
    return [
        {
            "action": "segment",
            "arguments": {"text_prompt": "bottle"},
            "result": {
                "found": True,
                "box_px": [0, 0, 2, 2],
                **mask_geometry([[0, 0], [0, 1]]),
            },
        },
        {
            "action": "back_project",
            "arguments": {"row": 1, "col": 1},
            "result": {"world_xyz": [0.1, 0.2, 0.7]},
        },
        {
            "action": "move_to",
            "arguments": {"xyz": [0.12, 0.23, 0.8], "arm": "left"},
            "result": {"reached": True, "reward": 999},
        },
    ]


class FakeToolkit:
    eval_fair = True

    def __init__(self):
        self.calls = []
        self.pick_results = iter([True])
        self.fine = [0.2, 0.3, 0.8]

    def flash_observation(self):
        return {
            "state": {"left_ee_pose": [0.2, 0.3, 0.8], "left_ee_joint_state": [0.2]}
        }

    def execute_tool(self, name, args):
        self.calls.append((name, args))
        if name == "segment":
            result = {"found": True, **mask_geometry([[0, 0], [0, 1]])}
        elif name == "back_project":
            result = {
                "world_xyz": [0.2, 0.3, 0.8]
                if args["camera"] == "cam_head"
                else self.fine
            }
        elif name == "move_to":
            result = {"log": {"result": {"reached": True}}}
        elif name == "pi0_pick":
            result = {"log": {"result": {"success": next(self.pick_results)}}}
        else:
            raise AssertionError(name)
        return SimpleNamespace(result=result)


def test_generate_and_replay_translates_waypoints_without_feedback():
    plan = generate_plan(trace(), "pick")
    assert plan["actions"][0]["offset"] == [0.02, 0.03, 0.1]
    assert "reward" not in json.dumps(plan)
    assert "xyz" not in json.dumps(plan)
    toolkit = FakeToolkit()
    assert replay(toolkit, plan, lambda _: None)["done"]
    np.testing.assert_allclose(toolkit.calls[-1][1]["xyz"], [0.22, 0.33, 0.9])


def test_mask_centroid_not_box_center_and_tampering_is_rejected():
    recorded = trace()
    recorded[0]["result"].update(mask_geometry([[1, 1, 0], [1, 0, 0], [0, 0, 1]]))
    recorded[1]["arguments"] = {"row": 0, "col": 0}
    plan = validate_plan(generate_plan(recorded, "pick"))
    assert plan["version"] == 2
    assert plan["anchors"]["bottle"]["method"] == ANCHOR_METHOD
    recorded[0]["result"]["centroid_rc"] = [1, 1]
    with pytest.raises(ValueError, match="centroid does not match"):
        generate_plan(recorded, "pick")
    recorded[0]["result"]["centroid_rc"] = [0, 0]
    recorded[1]["arguments"] = {"row": 1, "col": 1}
    with pytest.raises(ValueError, match="mask centroid"):
        generate_plan(recorded, "pick")


def test_legacy_anchor_plan_is_not_silently_reinterpreted():
    plan = generate_plan(trace(), "pick")
    plan["version"] = 1
    with pytest.raises(ValueError, match="Unsupported Flash plan"):
        validate_plan(plan)


@pytest.mark.parametrize("fine,passes", [([0.21, 0.3, 0.8], True), ([2, 3, 4], False)])
def test_wrist_refinement_cannot_overwrite_inconsistent_anchor(fine, passes):
    plan = generate_plan(trace(), "pick")
    plan["anchors"]["bottle"]["refine_camera"] = "cam_left_wrist"
    toolkit = FakeToolkit()
    toolkit.fine = fine
    if passes:
        replay(toolkit, plan, lambda _: None)
        np.testing.assert_allclose(toolkit.calls[-1][1]["xyz"], [0.23, 0.33, 0.9])
    else:
        with pytest.raises(RuntimeError, match="disagreement"):
            replay(toolkit, plan, lambda _: None)
        assert not any(name == "move_to" for name, _ in toolkit.calls)


@pytest.mark.parametrize(
    "results,passes", [([False, True], True), ([False] * 3, False)]
)
def test_grasp_retry_is_bounded_and_relocalizes(results, passes):
    plan = generate_plan(trace(), "pick")
    plan["actions"].append(
        {
            "action": "pi0_pick",
            "arguments": {"prompt": "pick bottle"},
            "anchor": "bottle",
            "offset": None,
        }
    )
    toolkit = FakeToolkit()
    toolkit.pick_results = iter(results)
    if passes:
        replay(toolkit, plan, lambda _: None)
    else:
        with pytest.raises(RuntimeError, match="not held"):
            replay(toolkit, plan, lambda _: None)
    assert sum(name == "pi0_pick" for name, _ in toolkit.calls) == len(results)
    assert sum(name == "move_to" for name, _ in toolkit.calls) == len(results)


def test_plan_rejects_unanchored_moves_and_privileged_actions():
    with pytest.raises(ValueError, match="fresh"):
        generate_plan(trace()[2:], "pick")
    plan = generate_plan(trace(), "pick")
    plan["actions"][0]["action"] = "get_reward_details"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_plan(plan)


def test_flash_planner_offline_memory_smoke(tmp_path):
    plan = generate_plan(trace(), "pick")
    root = tmp_path / "robodojo"
    (root / "flash").mkdir(parents=True)
    (root / "flash/pick_plan.json").write_text(json.dumps(plan))
    toolkit = FakeToolkit()
    toolkit.memory = MemoryManager(root)
    toolkit._task_name = "pick"
    result = FlashPlanner(recipe_tag="pick_l2", robot_name="robodojo").solve(
        system_prompt="unused", user_message="unused", toolkit=toolkit, max_turns=0
    )
    assert result.error is None
    assert result.stats["total_input_tokens"] == 0
    assert result.stats["turns_used"] == 0
    np.testing.assert_allclose(toolkit.calls[-1][1]["xyz"], [0.22, 0.33, 0.9])
    with pytest.raises(ValueError, match="identity"):
        run_flash(toolkit, "other_l2", lambda _: None)


def test_eval_facade_never_queries_privileged_feedback(
    monkeypatch, agent_module, make_agent
):
    raw = {
        "reward": 999,
        "instruction": "pick",
        "state": {"object_world_xyz": [9, 9, 9], "left_ee_pose": [0, 0, 1]},
        "vision": {"cam_head": {"color": np.zeros((2, 2, 3)), "object_id": 77}},
    }
    env = SimpleNamespace(take_action_cnt=[0], step_lim=2, apply_target=lambda *a: None)

    def apply(action, action_type, *, eval_fair):
        assert eval_fair
        env.take_action_cnt[0] += 1

    agent = make_agent(
        env,
        meta={"mode": "eval-fair"},
        slot=SimpleNamespace(env=env, apply_action=apply),
    )
    facade = env_server.RoboDojoEnvFacade(agent)
    monkeypatch.setattr(agent_module, "_obs_dict", lambda *a: copy.deepcopy(raw))
    monkeypatch.setattr(
        agent_module, "_reward_details", lambda *a: pytest.fail("reward read")
    )
    agent.bottle_mon.check = lambda *a: pytest.fail("truth monitor read")
    for method in (
        "env.get_reward_details",
        "env.get_safety_status",
        "env.is_success",
        "env.reset",
    ):
        assert method not in facade._rpc
    obs, reward, done, info = facade.step({})
    assert reward == 0 and done is False
    assert info == {"status": {"step": 1, "step_limit": 2}}
    assert obs == public_observation(obs)
    assert "object_world_xyz" not in obs["state"]
    assert "object_id" not in obs["vision"]["cam_head"]
    facade.step({})
    with pytest.raises(RuntimeError, match="budget"):
        facade.step({})


def test_dev_diagnostics_remain_callable_through_client_rpc(
    monkeypatch, agent_module, make_agent
):
    from robots.robodojo import tools

    agent = make_agent(
        SimpleNamespace(is_success=lambda **kw: True), meta={"mode": "dev"}
    )
    facade = env_server.RoboDojoEnvFacade(agent)
    monkeypatch.setattr(agent_module, "_reward_details", lambda *a: {"score": 100})
    monkeypatch.setattr(agent.bottle_mon, "status", lambda: {"rolling": []})
    facade._rpc["env.reset"] = lambda: {}

    def call(method, *, args=(), kwargs=None, **options):
        return facade._rpc[method](*args, **(kwargs or {}))

    client = RoboDojoEnvClient(
        SimpleNamespace(call=call), expected_meta={"mode": "dev"}
    )
    primitives = SimpleNamespace(env=client)
    assert client.get_reward_details() == {"score": 100}
    assert client.get_safety_status() == {"rolling": []}
    assert tools.get_reward_details(primitives, None) == {"score": 100}
    assert tools.get_safety_status(primitives, None) == {"rolling": []}
    toolkit = object.__new__(RoboDojoToolkit)
    toolkit.eval_fair = False
    toolkit._primitives = primitives
    assert toolkit.solved() is True


def test_agent_close_flushes_video_before_runtime(make_agent):
    events = []
    agent = make_agent(
        SimpleNamespace(),
        recorder=SimpleNamespace(close=lambda: events.append("writer")),
    )
    agent.venv.close = lambda clear_cache: events.append(("runtime", clear_cache))
    facade = env_server.RoboDojoEnvFacade(agent)
    facade.request_close()
    assert facade._shutdown_event.is_set()
    assert events == []
    facade.close()
    facade.close()
    assert events == ["writer", ("runtime", True)]


def test_eval_client_rejects_dev_service_and_does_not_reset():
    calls = []

    def call(method, **kwargs):
        calls.append(method)
        return {"mode": "eval-fair"}

    RoboDojoEnvClient(SimpleNamespace(call=call), expected_meta={"mode": "eval-fair"})
    assert calls == ["env.get_env_meta"]
    with pytest.raises(AssertionError, match="mismatch"):
        RoboDojoEnvClient(
            SimpleNamespace(call=lambda *a, **kw: {}),
            expected_meta={"mode": "eval-fair"},
        )


@pytest.mark.parametrize("eval_fair", [False, True])
def test_mode_tools_state_and_prompt_gate(monkeypatch, tmp_path, eval_fair):
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    env = SimpleNamespace(
        eval_fair=eval_fair,
        get_obs=lambda: {"vision": {}, "state": {"object_world_xyz": [9, 9, 9]}},
        get_status=lambda: {"step": 0, "step_limit": 10, "success": True, "score": 100},
        get_task_language=lambda: "pick",
    )
    toolkit = RoboDojoToolkit(
        primitives_kwargs={"env": env, "task": "pick"},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        eval_fair=eval_fair,
    )
    names = {s["name"] for s in toolkit.get_tools_spec()}
    state = toolkit.execute_tool("view_env_state", {}).result
    assert "terminated" not in state
    assert "object_world_xyz" not in state["state"]["state"]
    assert not toolkit._state.latest_record().terminated
    for name in ("get_reward_details", "get_safety_status"):
        assert name not in names
        assert "unknown tool" in toolkit.execute_tool(name, {}).result["error"]
    bundle = robot_spec.get_robot_spec().prompts
    variables = {
        "mode": "eval-fair" if eval_fair else "dev",
        "task": "put_bottles_into_dustbin",
        "layout": 0,
        "output_dir": str(tmp_path),
        "env_cfg_type": "arx_x5",
        "action_type": "joint",
        "task_summary": "scene",
    }
    prompt = bundle.render("system", variables=variables) + bundle.render(
        "user", variables=variables
    )
    for name in ("get_reward_details", "get_safety_status", "privileged"):
        assert name not in prompt
    assert ("read_text_file" in names) is not eval_fair
    if eval_fair:
        assert "get_safety_status" not in names
        assert (
            "unknown tool"
            in toolkit.execute_tool("get_reward_details", {}).result["error"]
        )
        bundle = robot_spec.get_robot_spec().prompts
        variables = {"mode": "eval-fair", "task": "pick", "layout": 0}
        prompt = bundle.render("system", variables=variables) + bundle.render(
            "user", variables=variables
        )
        for word in ("reward", "score", "predicate", "world_xyz"):
            assert word not in prompt
        assert not toolkit.solved()


def test_dev_recording_exports_actual_tool_results(monkeypatch, tmp_path):
    from robots.robodojo.flash.generate import trace_from_states
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    obs = {
        "vision": {
            "cam_head": {
                "color": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth": np.ones((2, 2)),
                "intrinsic_matrix": np.eye(3),
                "extrinsic_matrix": np.eye(4),
            }
        },
        "state": {"left_ee_pose": [1, 1, -1], "right_ee_pose": [0, 0, 1]},
    }
    env = SimpleNamespace(
        get_obs=lambda: obs,
        get_status=lambda: {"step": 0, "step_limit": 10},
        get_task_language=lambda: "pick",
    )
    toolkit = RoboDojoToolkit(
        primitives_kwargs={
            "env": env,
            "sam3_client": SimpleNamespace(
                segment=lambda *a, **kw: SimpleNamespace(
                    found=True, score=0.9, box=[0, 0, 2, 2], mask=[[0, 0], [0, 1]]
                )
            ),
        },
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
    )
    toolkit.execute_tool("segment", {"text_prompt": "bottle"})
    toolkit.execute_tool("back_project", {"row": 1, "col": 1})
    toolkit.execute_tool("move_to", {"xyz": [1, 1, -1], "arm": "left"})
    manifest = json.loads((tmp_path / "states.json").read_text())
    recorded = trace_from_states(manifest)
    assert not (tmp_path / "flash_trace.json").exists()
    assert len(manifest["steps"]) == 2  # Initial observation and one motion.
    state = toolkit.execute_tool("view_env_state", {}).result
    assert state["log"]["command"] == {
        "action": "move_to",
        "xyz": [1, 1, -1],
        "arm": "left",
    }
    assert state["task_language"] == "pick"
    assert state["state"]["eef"]["left"] == [1, 1, -1]
    assert "cam_head.png" in state["artifacts"]
    assert state["_image_bytes"]
    plan = generate_plan(recorded, "pick")
    assert plan["actions"][0]["offset"] == [0, 0, 0]
    manifest["steps"].append(
        {"command": {"action": "finish"}, "result": {"done": True}}
    )
    assert generate_plan(trace_from_states(manifest), "pick") == plan
    toolkit.execute_tool("move_to", {"xyz": [1, 1, -1], "arm": "left"})
    manifest = json.loads((tmp_path / "states.json").read_text())
    assert "perception" not in manifest["steps"][-1]["extras"]
    with pytest.raises(ValueError, match="fresh measured"):
        generate_plan(trace_from_states(manifest), "pick")


def test_flash_config_selects_mode_without_task_summary(monkeypatch, tmp_path):
    import argparse

    from robots.robodojo import tasks

    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser, False)
    args = parser.parse_args(["--task", "pick"])
    args.planner = "flash"
    args.output_dir = tmp_path
    monkeypatch.setattr(
        tasks, "task_summary", lambda *a: pytest.fail("dev context read")
    )
    config = robot_spec._parse_config(args)
    assert config.prompt_vars["mode"] == "eval-fair"
    assert config.prompt_vars["task_summary"] == {}

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

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from robots.robodojo.env_client import RoboDojoEnvClient
from robots.robodojo.xpolicylab.bridge import Bridge, PublicPrimitives


def make_primitives(bridge):
    return PublicPrimitives(
        env=RoboDojoEnvClient(bridge, expected_meta={"mode": "xpolicylab"}),
        check_cancelled=bridge.check,
    )


def observation():
    return {
        "instruction": "Test public feedback",
        "vision": {
            c: {"color": np.zeros((2, 2, 3), dtype=np.uint8)}
            for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        },
        "state": {
            **{f"{a}_arm_joint_state": [0.0] * 6 for a in ("left", "right")},
            **{f"{a}_ee_joint_state": [1.0] for a in ("left", "right")},
            **{
                f"{a}_ee_pose": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
                for a in ("left", "right")
            },
        },
    }


def feedback(bridge, action, step):
    obs = deepcopy(bridge.obs)
    for arm in ("left", "right"):
        obs["state"][f"{arm}_ee_joint_state"] = action[f"{arm}_ee_joint_state"]
        if f"{arm}_arm_joint_state" in action:
            # Simulate imperfect tracking, so commanded and measured differ.
            obs["state"][f"{arm}_arm_joint_state"] = (
                np.asarray(action[f"{arm}_arm_joint_state"]) * 0.8
            ).tolist()
    bridge.observe(obs, step, 100, False)


def test_gripper_target_is_not_reported_as_measured_opening():
    b = Bridge()
    b.observe(observation(), 0, 100, False)
    p = make_primitives(b)
    with ThreadPoolExecutor(1) as pool:
        try:
            f = pool.submit(p.set_gripper, arm="right", val=0.0, steps=3)
            targets = []
            for i in range(3):
                action = b.actions.get(timeout=3)
                targets.append(action["right_ee_joint_state"][0])
                feedback(b, action, i + 1)
            result = f.result(timeout=3).data
            assert result["completed"]
            assert result["executed_steps"] == p.env._history_step == 3
            assert result["gripper_target"] == 0
            assert "gripper_val" not in result
            assert np.allclose(targets, [2 / 3, 1 / 3, 0])
        finally:
            b.stop()


def test_hold_commanded_joints_then_relatch_actual_after_ee_action():
    b = Bridge()
    b.observe(observation(), 0, 100, False)
    p = make_primitives(b)
    with ThreadPoolExecutor(1) as pool:
        try:
            f = pool.submit(p.env.step_joint_update, arm="left", arm_qpos=[0.1] * 6)
            a = b.actions.get(timeout=3)
            feedback(b, a, 1)
            f.result(timeout=3)
            f = pool.submit(p.set_gripper, arm="right", val=0.0, steps=1)
            a = b.actions.get(timeout=3)
            assert np.allclose(a["left_arm_joint_state"], 0.1)
            feedback(b, a, 2)
            f.result(timeout=3)
            ee = {
                f"{arm}_{field}": b.obs["state"][f"{arm}_{field}"]
                for arm in ("left", "right")
                for field in ("ee_pose", "ee_joint_state")
            }
            f = pool.submit(p.env.step, ee)
            a = b.actions.get(timeout=3)
            feedback(b, a, 3)
            f.result(timeout=3)
            f = pool.submit(p.set_gripper, arm="right", val=1.0, steps=1)
            a = b.actions.get(timeout=3)
            assert np.allclose(a["left_arm_joint_state"], 0.08)
            feedback(b, a, 4)
            f.result(timeout=3)
            assert p.env._history_step == 4
        finally:
            b.stop()


def test_unacknowledged_action_does_not_enter_history():
    b = Bridge()
    b.observe(observation(), 0, 100, False)
    p = make_primitives(b)
    with ThreadPoolExecutor(1) as pool:
        f = pool.submit(p.set_gripper, arm="left", val=0.0, steps=1)
        b.actions.get(timeout=3)
        b.stop()
        result = f.result(timeout=3).data
    assert not result["completed"]
    assert result["executed_steps"] == 0
    assert p.env._history_step == 0
    with pytest.raises(RuntimeError):
        p.env.policy_observation()


def observed(step=0, layout=0):
    obs = observation()
    obs["episode_status"] = {"step": step, "step_limit": 100, "ended": False}
    obs["rpent_episode"] = {"task": "stack_blocks", "seed": 1, "layout": layout}
    obs["extra_info"] = {"detailed_instruction": "Official public task details."}
    return obs


@pytest.mark.parametrize("with_details", [False, True])
def test_two_episodes_clear_history_targets_and_use_actual_layout(
    tmp_path, monkeypatch, with_details
):
    from robots.robodojo.xpolicylab import model as module
    from rpent.planner.base import PlannerResult

    calls = []

    class Planner:
        def solve(self, **kwargs):
            for key in ("system_prompt", "user_message"):
                assert "Official public task details." not in kwargs[key]
                assert "Public scoring rules." not in kwargs[key]
            toolkit = kwargs["toolkit"]
            env = toolkit._primitives.env
            assert env._history_step == env._inference_index == 0
            assert toolkit._primitives.env._client.held is None
            assert "reset" not in toolkit._tools
            assert "not measured opening" in toolkit._tools["set_gripper"].description
            view = toolkit.execute_tool("view_env_state", {"step": -1})
            assert len(view.images) == 3
            visible = json.dumps(view.data)
            assert "Test public feedback" in visible
            assert "Official public task details." not in visible
            assert "Public scoring rules." not in visible
            result = toolkit.execute_tool(
                "set_gripper", {"arm": "right", "val": 0, "steps": 2}
            )
            assert not result.is_error, result
            assert result.data["log"]["result"]["gripper_target"] == 0
            assert "gripper_val" not in result.data["log"]["result"]
            env.policy_observation()
            assert env._inference_index == 1
            calls.append(env)
            return PlannerResult(
                finish_result={"status": "partial"}, messages=[], stats={}
            )

    monkeypatch.setattr(module, "build_planner", lambda *a, **kw: Planner())
    model = module.Model(
        {
            "env_cfg_type": "arx_x5",
            "output_dir": str(tmp_path),
            "memory_dir": str(tmp_path / "memory"),
            "am_endpoint": "socket://127.0.0.1:1",
        }
    )
    for layout in (0, 1):
        model.reset()
        initial = observed(layout=layout)
        if with_details:
            initial["extra_info"]["scoring"] = ["Public scoring rules."]
        else:
            initial.pop("extra_info")
        model.update_obs(initial)
        try:
            while model.worker.is_alive():
                actions = model.get_action()
                for action in actions:
                    obs = deepcopy(model.bridge.obs)
                    if not action.get("_rpent_capture"):
                        obs["episode_status"]["step"] += 1
                        obs["state"]["right_ee_joint_state"] = action[
                            "right_ee_joint_state"
                        ]
                    model.update_obs(obs)
            model.worker.join(5)
            assert model.report["status"] == "completed", model.report
            assert model.report["native_steps"] == model.report["history_steps"] == 2
            assert model.report["inference_count"] == 1
            assert model.report["episode"]["layout"] == layout
            assert json.loads((model.output / "task-instruction.json").read_text()) == {
                "instruction": "Test public feedback",
                "source": "observation.instruction",
            }
        finally:
            model.abort()
    assert calls[0] is not calls[1]
    assert len(list((tmp_path / "episodes").glob("*/report.json"))) == 2


@pytest.mark.parametrize("failure", [False, True])
def test_official_finish_score_and_infrastructure_separation(tmp_path, failure):
    from robots.robodojo.xpolicylab.deploy import eval_one_episode

    class Reward:
        def get_score(self):
            return [25.0]

    class Env:
        task_name = "stack_blocks"
        eval_seed = 1
        env_seeds = [7]
        save_dir = tmp_path
        take_action_cnt = [0]
        step_lim = 100
        success = [True]
        reward_manager = Reward()
        closed = False

        def is_episode_end(self):
            return not self.success[0]

        def get_obs(self):
            return observed()

        def get_score(self):
            pass

        def close(self):
            self.closed = True

    class Client:
        closed = False

        def call(self, func_name, **kw):
            if func_name == "get_action":
                return []
            if func_name == "episode_report":
                return {
                    "status": "failed" if failure else "completed",
                    "timeout_s": 4800,
                    "error": "connection lost" if failure else None,
                }
            if func_name == "update_obs":
                assert kw["obs"]["rpent_episode"] == {
                    "task": "stack_blocks",
                    "seed": 1,
                    "layout": 7,
                }

        def close(self):
            self.closed = True

    env, client = Env(), Client()
    output = tmp_path / "rpent/layout_7"
    if failure:
        with pytest.raises(SystemExit) as exc:
            eval_one_episode(env, client)
        assert exc.value.code == 2
        assert env.closed and client.closed
        assert not (output / "native-result.json").exists()
        assert (output / "infrastructure-error.json").exists()
    else:
        eval_one_episode(env, client)
        result = json.loads((output / "native-result.json").read_text())
        assert result["score"] == 25 and result["success"] is False
        assert result["layout"] == 7


def test_public_status_never_exposes_partial_score():
    b = Bridge()
    obs = observed()
    obs["episode_status"].update(success=True, score=75)
    b.observe(obs, 0, 100, True)
    assert b.status == {"step": 0, "step_limit": 100, "success": True}


def test_terminal_status_waits_for_cleanup(tmp_path, monkeypatch):
    from threading import Event

    from robots.robodojo.xpolicylab import model as module
    from rpent.planner.base import PlannerResult

    closing, finish_cleanup = Event(), Event()

    class Toolkit:
        def __init__(self, **kwargs):
            pass

        def close(self):
            closing.set()
            assert finish_cleanup.wait(5)

    class Planner:
        def solve(self, **kwargs):
            return PlannerResult(
                finish_result={"status": "partial"}, messages=[], stats={}
            )

    monkeypatch.setattr(module, "ExternalToolkit", Toolkit)
    monkeypatch.setattr(module, "build_planner", lambda *a, **kw: Planner())
    model = module.Model(
        {
            "env_cfg_type": "arx_x5",
            "output_dir": str(tmp_path),
            "memory_dir": str(tmp_path / "memory"),
            "am_endpoint": "socket://127.0.0.1:1",
        }
    )
    model.reset()
    model.update_obs(observed())
    try:
        assert closing.wait(5)
        assert model.episode_report()["status"] == "running"
    finally:
        finish_cleanup.set()
        model.worker.join(5)
    assert model.episode_report()["status"] == "completed"
    assert (tmp_path / "episodes/stack_blocks_s1_l0/report.json").is_file()


def test_official_task_counts_and_worker_arguments(tmp_path, monkeypatch):
    from robots.robodojo.xpolicylab import batch
    from robots.robodojo.xpolicylab.__main__ import build_parser

    cfg = tmp_path / "task/RoboDojo/config/_task.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "common:\n  eval_nums: 50\ntasks:\n  stack_blocks:\n    eval_nums: 25\n"
    )
    args = build_parser().parse_args(
        [
            "--runtime-root",
            str(tmp_path),
            "--dimensions",
            "generalization",
            "memory",
            "--am-endpoint",
            "socket://vpp2:9100",
            "--output-dir",
            str(tmp_path / "out"),
            "--worker",
            "0:2",
            "--worker",
            "1:3",
            "--model",
            "test-model",
        ]
    )
    monkeypatch.setattr(
        batch.subprocess,
        "check_output",
        lambda *a, **k: json.dumps(
            {
                "tasks": [
                    {
                        "name": "stack_blocks",
                        "dimension": "generalization",
                        "runnable": True,
                    },
                    {"name": "swap_T", "dimension": "memory", "runnable": True},
                ]
            }
        ),
    )
    assert [t["episodes"] for t in batch.task_plan(args)] == [25, 50]
    args.episodes = 30
    assert [t["episodes"] for t in batch.task_plan(args)] == [25, 30]
    args.tasks = ["swap_T"]
    assert [t["task"] for t in batch.task_plan(args)] == ["swap_T"]
    args.tasks = ["unknown"]
    with pytest.raises(ValueError, match="outside selected dimensions"):
        batch.task_plan(args)
    args.tasks = None
    args.episodes = None
    args.layouts = "2,7,9"
    assert [t["layouts"] for t in batch.task_plan(args)] == [[2, 7, 9]] * 2
    command = batch.task_command(args, "swap_T", tmp_path / "out", "1:3")
    single = build_parser().parse_args(command[3:])
    assert (single.task, single.seed, single.model) == ("swap_T", 1, "test-model")
    assert (single.env_cuda_device, single.policy_cuda_device) == ("1", "3")
    assert single.layouts == "2,7,9"
    assert single.am_endpoint == args.am_endpoint
    assert single.reasoning_effort == args.reasoning_effort


def test_batch_resume_does_not_rerun_failed_or_completed_tasks(tmp_path, monkeypatch):
    from robots.robodojo.xpolicylab import batch
    from robots.robodojo.xpolicylab.__main__ import build_parser

    args = build_parser().parse_args(
        [
            "--runtime-root",
            str(tmp_path),
            "--dimensions",
            "memory",
            "--am-endpoint",
            "socket://vpp2:9100",
            "--output-dir",
            str(tmp_path / "out"),
            "--memory-dir",
            str(tmp_path),
            "--worker",
            "0:0",
            "--worker",
            "1:1",
            "--dry-run",
        ]
    )
    (tmp_path / "XPolicyLab/policy").mkdir(parents=True)
    tasks = [
        {"task": t, "dimension": "memory", "episodes": 1, "layouts": [0]}
        for t in ("a", "b", "c")
    ]
    monkeypatch.setattr(batch, "task_plan", lambda _: tasks)
    monkeypatch.setattr(batch, "fingerprint", lambda _: "fixed-inputs")
    batch.run_batch(args)
    started = []

    class Worker:
        def __init__(self, name, command, **kwargs):
            self.task = command[command.index("--task") + 1]
            self.output = Path(command[command.index("--output-dir") + 1])
            self.subprocess_env = {}

        def start(self):
            started.append(self.task)
            self.output.mkdir()
            result = tmp_path / "results" / self.task
            result.mkdir(parents=True)
            (self.output / "manifest.json").write_text(
                json.dumps({"result_dir": str(result)})
            )

        def poll(self):
            return 1 if self.task == "b" else 0

        def stop(self, **kwargs):
            pass

    monkeypatch.setattr(batch, "ProcessDaemon", Worker)
    monkeypatch.setattr(batch.time, "sleep", lambda _: None)
    monkeypatch.setattr(batch.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(batch, "verify_results", lambda *a, **k: None)
    args.dry_run, args.resume = False, True
    with pytest.raises(RuntimeError, match="Batch incomplete"):
        batch.run_batch(args)
    assert started == ["a", "b", "c"]
    with pytest.raises(RuntimeError, match="Batch incomplete"):
        batch.run_batch(args)
    assert started == ["a", "b", "c"]
    monkeypatch.setattr(batch, "fingerprint", lambda _: "changed-inputs")
    with pytest.raises(ValueError, match="inputs changed"):
        batch.run_batch(args)


def test_layout_selection_preserves_official_ids_and_resume_exclusions(tmp_path):
    from types import SimpleNamespace

    from robots.robodojo.xpolicylab.__main__ import selected_layouts
    from robots.robodojo.xpolicylab.evaluator import select_seed_queue

    cfg = tmp_path / "task/RoboDojo/config/_task.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "common:\n  eval_nums: 50\ntasks:\n  stack_blocks:\n    eval_nums: 25\n"
    )
    assert selected_layouts(tmp_path, "stack_blocks", "all") == list(range(25))
    assert selected_layouts(tmp_path, "swap_T", "all") == list(range(50))
    assert selected_layouts(tmp_path, "stack_blocks", "7,2-3") == [2, 3, 7]
    for invalid in ("", "-1", "2,2", "3-2", "0-25", "x", "0-999999999"):
        with pytest.raises(ValueError):
            selected_layouts(tmp_path, "stack_blocks", invalid)
    mapping = {i: {"scene_layout": f"scene_{i}.json"} for i in range(10)}
    manager = SimpleNamespace(seed_info=mapping, seed_list=list(range(10)))
    select_seed_queue(manager, [2, 7])
    assert manager.seed_list == [2, 7]
    assert manager.ed_idx == 2 and manager.idx == 0
    assert manager.seed_info is mapping
    assert manager.seed_info[7]["scene_layout"] == "scene_7.json"
    # The official resume filter has already excluded completed layout 2.
    manager.seed_list = [i for i in range(10) if i != 2]
    select_seed_queue(manager, [2, 7])
    assert manager.seed_list == [7]
    with pytest.raises(ValueError, match="unavailable"):
        select_seed_queue(manager, [10])

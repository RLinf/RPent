# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Real Toolkit/Planner dispatch, with deterministic control and model feedback."""

import json
from types import SimpleNamespace

import pytest
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.api_loop import ApiAgentLoop
from rpent.utils import logging
from tests.unit_tests.robots.robodojo.fakes import EpisodePolicy
from tests.unit_tests.robots.robodojo.fakes import FakeEnv as EpisodeEnv


class AgentEnv(EpisodeEnv):
    def get_motion_state(self):
        return {
            **super().get_motion_state(),
            **{
                f"{arm}_eef_pose": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
                for arm in ("left", "right")
            },
        }

    def plan_arm_path(self, arm, target):
        return {"status": "Failure", "position": []}

    def is_success(self):
        return self.frame == self.done_at


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    instances = []

    def build(limit=20, done_at=None, vla_agent=True, **kwargs):
        env, policy = AgentEnv(limit, done_at), EpisodePolicy()
        kit = RoboDojoToolkit(
            primitives_kwargs=dict(
                env=env,
                vla_client=policy,
                vla_backend="awomo05",
                vla_agent=vla_agent,
                **kwargs,
            ),
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            state_output_dir=tmp_path,
        )
        instances.append(kit)
        return env, policy, kit

    yield build
    for kit in instances:
        kit.close()


def report(result):
    return result.data.get("log", {}).get("result", result.data)


def test_actual_planner_dispatch_mixed_control_and_cleanup(setup, tmp_path):
    env, policy, kit = setup()
    script = iter(
        [
            ("view_env_state", {}),
            ("run_policy", {"max_steps": 3}),
            ("run_policy", {"max_steps": 2}),
            ("set_gripper", {"arm": "left", "val": 0.5, "steps": 1}),
            ("hold", {"steps": 2}),
            ("run_policy", {"max_steps": 2}),
            (
                "finish",
                {"status": "uncertain", "summary": "No native verdict inferred"},
            ),
        ]
    )
    feedback = []
    execute = kit.execute_tool

    def record(name, arguments):
        result = execute(name, arguments)
        feedback.append((name, result))
        return result

    kit.execute_tool = record

    async def stream(messages, info):
        name, args = next(script)
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(args))}

    result = ApiAgentLoop(
        model=FunctionModel(stream_function=stream),
        dashboard_events=NullDashboardEventSink(),
        no_images=True,
    ).solve(
        system_prompt="Solve through the registered tools.",
        user_message="stack the blocks",
        toolkit=kit,
        max_turns=7,
    )
    assert result.error is None
    assert result.finish_result["agent_assessment"] == "uncertain"
    assert result.finish_result["success"] is None
    assert not kit.solved()
    assert env.frame == 10
    assert [e for e in policy.events if e[0] == "infer"] == [("infer", 0), ("infer", 8)]
    assert [e[1] for e in policy.events if e[0] == "observe"] == list(range(11))
    assert policy.events.count(("reset",)) == 1
    assert policy.events.count(("end",)) == 1
    kit.close()
    assert policy.events.count(("end",)) == 1
    controls = [
        r for name, r in feedback if name in {"run_policy", "hold", "set_gripper"}
    ]
    assert [r.data["execution"]["confirmed_actions"] for r in controls] == [
        3,
        2,
        1,
        2,
        2,
    ]
    assert all(not r.is_error and len(r.images) == 3 for r in controls)
    assert not {"write_text_file", "reset", "chunk_step"} & kit._tools.keys()


def test_fixed_planner_uses_registered_tool_until_native_episode_end(setup):
    native, client, toolkit = setup(limit=400, done_at=11, vla_agent=False)
    script = iter(
        [
            ("view_env_state", {}),
            ("run_policy", {"max_steps": 3}),
            ("run_policy", {"max_steps": 5}),
            ("run_policy", {"max_steps": 32}),
            (
                "finish",
                {
                    "status": "stuck",
                    "summary": "native episode ended; success is not inferred",
                },
            ),
        ]
    )

    async def stream(messages, info):
        name, arguments = next(script)
        if name == "finish":
            assert native.frame == 11
            assert toolkit._primitives.env.last_run["episode_ended"]
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(arguments))}

    try:
        for name in ("pi0_pick", "move_to", "set_gripper", "place_in_bin", "stabilize"):
            assert name not in toolkit._tools
        invalid = toolkit.execute_tool(
            "run_policy", {"max_steps": 3, "prompt": "pick another object"}
        )
        assert invalid.is_error
        assert native.frame == 0
        result = ApiAgentLoop(
            model=FunctionModel(stream_function=stream),
            dashboard_events=NullDashboardEventSink(),
            no_images=True,
        ).solve(
            system_prompt="Execute the original instruction until the environment ends.",
            user_message="stack the blocks",
            toolkit=toolkit,
            max_turns=5,
        )
        assert result.error is None
        assert result.finish_result["_finish"]
        records = toolkit._state.records()
        reports = [
            r.result
            for r in records
            if r.command and r.command["action"] == "run_policy"
        ]
        assert [r["executed_steps"] for r in reports] == [3, 5, 3]
        assert [len(r["inferences"]) for r in reports] == [1, 0, 1]
        assert client.events.count(("reset",)) == 1
    finally:
        toolkit.close()


def test_cancel_between_calls_discards_pending_actions_without_model_reset(setup):
    native, client, toolkit = setup(vla_agent=False)
    try:
        first = toolkit.execute_tool("run_policy", {"max_steps": 3})
        assert not first.is_error
        assert first.data["log"]["result"]["pending_actions"] == 5
        assert first.data["log"]["result"]["inference_count"] == 1
        assert "inferences" not in first.data["log"]["result"]
        toolkit.cancel_active_and_wait()
        toolkit.resume_calls()
        assert not toolkit._primitives.env._pending
        assert not toolkit.execute_tool("run_policy", {"max_steps": 1}).is_error
        assert native.frame == 4
        assert client.events.count(("reset",)) == 1
        assert toolkit._primitives.env._pending
    finally:
        toolkit.close()


def test_reads_and_zero_execution_failures_preserve_actions(setup):
    env, policy, kit = setup()
    assert (
        report(kit.execute_tool("run_policy", {"max_steps": 3}))["pending_actions"] == 5
    )
    before = list(policy.events)
    for name, args in [
        ("view_env_state", {}),
        ("render", {}),
        ("move_delta", {"arm": "right"}),
        ("move_to", {"arm": "left", "xyz": [0.0, 0.0, 1.0]}),
        ("hold", {"steps": -1}),
        ("run_policy", {"max_steps": 1, "instruction": "changed"}),
    ]:
        kit.execute_tool(name, {**args})
        assert env.frame == 3
        assert policy.events == before
    result = report(kit.execute_tool("run_policy", {"max_steps": 2}))
    assert result["pending_actions"] == 3
    assert len([e for e in policy.events if e[0] == "infer"]) == 1


@pytest.mark.parametrize(
    "tool_name, arguments, method",
    [
        ("hold", {"steps": 1}, "env.hold_step"),
        (
            "set_gripper",
            {"arm": "left", "val": 0.5, "steps": 1},
            "env.step_joint_update",
        ),
    ],
)
def test_server_confirmed_zero_action_rejection_preserves_chunk(
    setup, tool_name, arguments, method
):
    env, policy, kit = setup()
    assert not kit.execute_tool("run_policy", {"max_steps": 3}).is_error
    before = list(policy.events)
    rpc = env._client.call

    def reject(name, **kwargs):
        if name == method:
            return {
                "step_before": 3,
                "step_after": 3,
                "executed_actions": 0,
                "terminated": False,
                "truncated": False,
                "episode_status": env.get_status(),
                "rejection": "target rejected",
            }
        return rpc(name, **kwargs)

    env._client.call = reject
    rejected = kit.execute_tool(tool_name, arguments)
    assert rejected.is_error
    assert rejected.data["execution"]["confirmed_actions"] == 0
    assert rejected.data["execution"]["synchronization_trusted"]
    assert policy.events == before
    assert env.frame == 3
    assert len(kit._primitives.env._pending) == 5
    result = kit.execute_tool("run_policy", {"max_steps": 2})
    assert not result.is_error
    assert report(result)["pending_actions"] == 3
    assert [e for e in policy.events if e[0] == "infer"] == [("infer", 0)]


@pytest.mark.parametrize("done_at", [None, 6])
def test_shared_native_budget_and_termination(setup, done_at):
    env, policy, kit = setup(limit=8, done_at=done_at)
    kit.execute_tool("run_policy", {"max_steps": 3})
    result = report(kit.execute_tool("hold", {"steps": 10}))
    assert env.frame == (done_at or 8)
    assert result["executed_steps"] == env.frame - 3
    assert result["stop_reason"] == ("episode_end" if done_at else "budget_exhausted")
    assert result["episode_status"]["step"] == env.frame
    kit.execute_tool("run_policy", {"max_steps": 3})
    assert env.frame == (done_at or 8)
    assert [e[1] for e in policy.events if e[0] == "observe"] == list(range(env.frame))


@pytest.mark.parametrize("failure", ["rpc", "feedback", "missing_obs", "counter"])
def test_uncertainty_blocks_later_actions_without_retry(setup, monkeypatch, failure):
    env, policy, kit = setup()
    first = kit.execute_tool("run_policy", {"max_steps": 2})
    assert first.data["execution"]["confirmed_actions"] == 2
    if failure == "rpc":

        def lost():
            raise TimeoutError("result uncertain")

        env.after_step = lost
    elif failure == "feedback":

        def lost(obs, step):
            raise RuntimeError("model feedback unavailable")

        policy.update_obs = lost
    elif failure == "missing_obs":
        rpc = env._client.call

        def missing_observation(method, **kwargs):
            result = rpc(method, **kwargs)
            if method == "env.hold_step":
                result.pop("observation")
            return result

        monkeypatch.setattr(env._client, "call", missing_observation)
    else:
        original = env._native_step

        def wrong(*args, **kwargs):
            obs, reward, done, info = original(*args, **kwargs)
            info["step_after"] += 1
            return obs, reward, done, info

        env._native_step = wrong
    result = kit.execute_tool("hold", {"steps": 1})
    assert env.frame == 3
    confirmed = failure in {"feedback", "missing_obs"}
    assert result.data["execution"]["confirmed_actions"] == int(confirmed)
    assert result.data["execution"]["confirmed_step"] == 2 + int(confirmed)
    assert not result.data["execution"]["synchronization_trusted"]
    receipt = report(result)
    assert receipt["executed_actions"] == int(confirmed)
    assert receipt["episode_status"]["step"] == 2 + int(confirmed)
    assert receipt["execution_uncertain"] is (not confirmed)
    infer_count = policy.events.count(("infer", 0))
    kit.execute_tool("run_policy", {"max_steps": 1})
    kit.execute_tool("hold", {"steps": 1})
    assert env.frame == 3
    assert policy.events.count(("infer", 0)) == infer_count
    assert policy.events.count(("reset",)) == 1


def test_partial_cancel_discards_actions_and_closes(setup):
    env, policy, kit = setup()
    kit.execute_tool("run_policy", {"max_steps": 2})

    def cancel():
        if env.frame == 4:
            kit._scheduler.cancel()

    env.after_step = cancel
    result = report(kit.execute_tool("hold", {"steps": 8}))
    assert env.frame == 4
    assert result["executed_steps"] == 2
    assert result["interrupted"]
    kit.close()
    assert policy.events.count(("end",)) == 1


def test_primitive_can_precede_policy_without_resetting_history(setup):
    env, policy, kit = setup()
    assert not kit.execute_tool("hold", {"steps": 2}).is_error
    assert env.frame == 2
    assert not [e for e in policy.events if e[0] == "infer"]
    assert not kit.execute_tool("run_policy", {"max_steps": 1}).is_error
    assert [e for e in policy.events if e[0] == "infer"] == [("infer", 2)]
    assert policy.events.count(("reset",)) == 1


def test_geometry_queries_require_the_selected_views_saved_data(setup, monkeypatch):
    original = AgentEnv.get_obs

    def head_only(self):
        obs = original(self)
        for name in ("cam_left_wrist", "cam_right_wrist"):
            obs["vision"][name].pop("depth")
        return obs

    monkeypatch.setattr(AgentEnv, "get_obs", head_only)
    env, policy, kit = setup(agent_tools=["geometry"])
    before = list(policy.events)
    assert not kit.execute_tool("back_project", {"row": 0, "col": 0}).is_error
    result = kit.execute_tool(
        "back_project", {"row": 0, "col": 0, "camera": "cam_left_wrist"}
    )
    assert result.data["error"]["code"] == "world_xyz_not_found"
    assert "left_wrist_world_xyz.npy" not in kit.state.get().artifacts
    assert policy.events == before and env.frame == 0


@pytest.mark.parametrize(
    "groups",
    [
        None,
        ["policy"],
        ["motion"],
        ["geometry"],
        ["sam3"],
        ["policy", "sam3"],
        ["policy", "motion", "geometry", "sam3"],
    ],
)
def test_tool_groups_match_observations_dependencies_and_prompts(
    setup, monkeypatch, groups
):
    from robots.robodojo.prompt_bundle import system_prompt, user_prompt
    from rpent.prompt.utils import format_prompt

    original = AgentEnv.get_obs
    geometry = groups is not None and "geometry" in groups

    def observation(self):
        obs = original(self)
        if not geometry:
            for camera in obs["vision"].values():
                camera.pop("depth")
                # Disabled geometry must not try to interpret calibration.
                camera["intrinsic_matrix"] = [[float("nan")]]
        return obs

    monkeypatch.setattr(AgentEnv, "get_obs", observation)
    env, policy, kit = setup(
        vla_agent=groups is not None, agent_tools=groups or [], sam3_client=object()
    )
    variables = {
        "vla_backend": "awomo05",
        "vla_agent": groups is not None,
        "agent_tools": groups,
        "memory_dir": str(kit.memory.root),
        "task": "put_bottles_into_dustbin",
        "seed": 0,
        "layout": 0,
        "env_cfg_type": "test",
        "output_dir": "output",
        "task_summary": "Visible scene",
        "recipe_tag": "result",
    }
    prompt = format_prompt(system_prompt(variables), variables=variables)
    prompt += format_prompt(user_prompt(variables), variables=variables)
    assert "{{" not in prompt
    for obsolete in (
        "set_stage_plan",
        "review_execution",
        "decision_id",
        "agent_budget",
        "vpp2_act",
        "pi0_pick",
        "place_in_bin",
        "stabilize",
    ):
        assert obsolete not in prompt
    if kit.vla_agent:
        assert str(kit.memory.root) in prompt
        assert "read_text_file" in prompt and "list_dir" in prompt
    groups = ["policy"] if groups is None else groups
    motion = {"move_to", "move_delta", "hold", "rotate_wrist", "set_gripper", "release"}
    expected = {"view_env_state", "finish"}
    if kit.vla_agent:
        expected |= {"render", "read_text_file", "list_dir"}
    else:
        expected |= {"read_text_file", "list_dir", "write_text_file"}
    expected |= {"run_policy"} if "policy" in groups else set()
    expected |= motion if "motion" in groups else set()
    expected |= (
        {"back_project", "sample_world_xyz", "query_world_map"} if geometry else set()
    )
    expected |= {"segment"} if "sam3" in groups else set()
    assert set(kit._tools) == expected
    for group, name in [
        ("policy", "run_policy"),
        ("motion", "move_delta"),
        ("geometry", "sample_world_xyz"),
        ("sam3", "segment("),
    ]:
        assert (name in prompt) == (group in groups)
    assert all(
        ("world_xyz" in fields) == geometry
        for fields in kit.state.get().state["artifacts"].values()
    )
    assert len(kit.execute_tool("view_env_state", {}).images) == 3
    assert env.frame == 0 and policy.events == [("reset",), ("observe", 0)]
    # Service validation precedes model reset or initial observation capture.
    with pytest.raises(ValueError, match="SAM3"):
        setup(agent_tools=["sam3"])


@pytest.mark.parametrize("native_success", [False, True])
def test_finish_keeps_agent_assessment_separate_from_native_verdict(
    setup, monkeypatch, native_success
):
    env, policy, kit = setup()
    status = env.get_status()
    monkeypatch.setattr(
        env, "get_status", lambda: {**status, "success": native_success}
    )
    result = kit.execute_tool(
        "finish", {"status": "success", "summary": "Observed outcome"}
    )
    assert result.data["_finish"]
    assert result.data["agent_assessment"] == "success"
    assert result.data["success"] is native_success
    assert result.data["status"] == ("success" if native_success else "failure")
    assert policy.events.count(("end",)) == 1


@pytest.mark.parametrize("profile", ["local", "hf"])
def test_mixed_control_uses_standard_memory_preparation(monkeypatch, tmp_path, profile):
    from types import SimpleNamespace

    from robots.robodojo.robot_spec import _validate_args, get_robot_spec
    from rpent.memory import loading
    from tests.unit_tests.robots.robodojo.fakes import args

    parsed = args(
        "--vla-backend", "awomo05", "--vla-endpoint", "http://bridge", "--vla-agent"
    )
    parsed.planner, parsed.model = "api", "openai:test"
    parsed.memory_profile = profile
    parsed.memory_repo = "example/recipes" if profile == "hf" else None
    parsed.memory_dir = str(tmp_path) if profile == "local" else None
    _validate_args(parsed)
    config = SimpleNamespace(prompt_vars={"memory_dir": parsed.memory_dir})
    prepared = []

    def prepare(repo, robot):
        prepared.append((repo, robot))
        return tmp_path

    monkeypatch.setattr(loading, "prepare_explicit_source", prepare)
    loading.prepare_run_memory(parsed, get_robot_spec(), config)
    assert config.prompt_vars["memory_dir"] == str(tmp_path)
    assert prepared == ([("example/recipes", "robodojo")] if profile == "hf" else [])


def test_mixed_control_reads_memory_without_exposing_other_files(setup, tmp_path):
    env, policy, kit = setup()
    root = kit.memory.root
    root.mkdir(parents=True, exist_ok=True)
    note = root / "MEMORY.md"
    note.write_text("Use fresh observations to adapt prior experience.")
    external = tmp_path / "hidden.txt"
    external.write_text("private scene data")
    (root / "escape.md").symlink_to(external)
    before = list(policy.events)
    assert "MEMORY.md" in kit.execute_tool("list_dir", {}).data["files"]
    assert (
        kit.execute_tool("read_text_file", {"path": str(note)}).data["content"]
        == note.read_text()
    )
    for path in (external, root / "escape.md"):
        assert kit.execute_tool("read_text_file", {"path": str(path)}).is_error
    assert kit.execute_tool(
        "write_text_file", {"path": str(note), "content": "changed"}
    ).is_error
    assert policy.events == before and env.frame == 0
    assert note.read_text() == "Use fresh observations to adapt prior experience."


def test_new_toolkit_can_reuse_runtime_after_previous_session_closes(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    native, policy = AgentEnv(), EpisodePolicy()
    previous = None
    for i in range(2):
        kit = RoboDojoToolkit(
            primitives_kwargs={
                "env": native,
                "vla_client": policy,
                "vla_backend": "awomo05",
                "vla_agent": True,
            },
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            state_output_dir=tmp_path / str(i),
        )
        if previous is not None:
            previous.close()
            previous.resume_calls()
            assert policy.events.count(("end",)) == 1
        assert not kit.execute_tool("run_policy", {"max_steps": 1}).is_error
        kit.close()
        assert native._execution is None
        assert not native.closed
        # The next environment task starts explicitly, preserving runtime ownership.
        native.reset()
        previous = kit
    assert policy.events.count(("end",)) == 2


@pytest.mark.parametrize("mixed", [False, True])
def test_full_inferences_saved_once_and_summary_does_not_mutate_them(
    setup, monkeypatch, mixed
):
    from copy import deepcopy

    env, policy, kit = setup(vla_agent=mixed)
    reads, publishes = [], []
    observe = env.get_obs

    def capture():
        reads.append(env.frame)
        return observe()

    monkeypatch.setattr(env, "get_obs", capture)
    monkeypatch.setattr(
        kit,
        "_dashboard_events",
        SimpleNamespace(emit=lambda event: publishes.append(event.record.step_idx)),
    )
    result = kit.execute_tool("run_policy", {"max_steps": 2})
    assert not result.is_error
    assert reads == [1, 2, 2]  # two action receipts, one standard capture
    assert publishes == [1]
    assert len(result.images) == 3
    full = deepcopy(kit._primitives.env.last_run)
    stored = kit.state.get()
    saved_text = (kit.state._output_dir / "states.json").read_text()
    manifest = json.loads(saved_text)
    assert stored.result == manifest["steps"][-1]["result"] == full
    assert full["inferences"][0]["info"]["upstream"]["latency_ms"] == 123.5
    assert report(result)["inference_count"] == 1
    assert "inferences" not in report(result) and "inferences" not in result.data
    assert result.data["step"] == stored.step_idx == 1
    assert result.data["state"]["control"]["step"] == 2
    assert (
        report(result)["observation_step"]
        == result.data["execution"]["confirmed_step"]
        == 2
    )
    before = list(policy.events)
    viewed = kit.execute_tool("view_env_state", {"step": 1})
    assert report(viewed) == report(result)
    assert viewed.images == result.images
    assert reads == [1, 2, 2] and policy.events == before
    viewed.data["log"]["result"]["pending_actions"] = -1
    assert kit.state.get().result == full == kit._primitives.env.last_run
    assert (kit.state._output_dir / "states.json").read_text() == saved_text


@pytest.mark.parametrize(
    "mixed,failure",
    [
        (mixed, failure)
        for mixed in (False, True)
        for failure in ("capture", "save", "load", "publish", "counter", "rgb")
    ]
    + [(True, "calibration")],
)
def test_post_action_observation_failure_retains_progress_and_blocks_control(
    setup, monkeypatch, mixed, failure
):
    env, policy, kit = setup(vla_agent=mixed)
    before_records = len(kit.state.records())

    def fail(*args, **kwargs):
        raise OSError("observation delivery failed")

    if failure == "capture":
        monkeypatch.setattr(kit, "get_env_state", fail)
    elif failure == "save":
        monkeypatch.setattr(kit.state, "save", lambda *args, **kwargs: None)
    elif failure == "load":
        monkeypatch.setattr(kit.state, "load_bytes", fail)
    elif failure == "publish":
        monkeypatch.setattr(kit, "_dashboard_events", SimpleNamespace(emit=fail))
    else:
        observe = env.get_obs

        def advance_during_capture():
            obs = observe()
            if env.frame == 2 and env.get_execution_state()["confirmed_step"] == 2:
                if failure == "counter":
                    env.frame += 1
                elif failure == "rgb":
                    obs["vision"]["cam_head"].pop("color")
                else:
                    obs["vision"]["cam_head"]["intrinsic_matrix"][0, 0] = 0
            return obs

        monkeypatch.setattr(env, "get_obs", advance_during_capture)
    result = kit.execute_tool("run_policy", {"max_steps": 2})
    assert result.is_error
    assert result.data["execution"]["confirmed_actions"] == 2
    assert result.data["execution"]["confirmed_step"] == 2
    assert not result.data["execution"]["synchronization_trusted"]
    assert "inferences" not in result.data
    assert kit._primitives.env.last_run["executed_steps"] == 2
    assert "state_capture_error" not in kit._primitives.env.last_run
    assert len(kit._primitives.env.last_run["inferences"]) == 1
    assert len(kit.state.records()) == before_records + int(
        failure in {"load", "publish"}
    )
    if failure in {"load", "publish"}:
        assert kit.state.get().result == kit._primitives.env.last_run
    if failure == "publish":
        assert result.data["step"] == kit.state.latest_step
        assert len(result.data["artifacts"]) >= 3
        assert report(result)["executed_steps"] == 2
        assert report(result)["inference_count"] == 1
    before = list(policy.events)
    frame = env.frame
    kit.resume_calls()
    assert kit.execute_tool("run_policy", {"max_steps": 1}).is_error
    assert env.frame == frame and policy.events == before


@pytest.mark.parametrize("publish_failure", [False, True])
def test_queued_controls_keep_per_call_progress_and_delivery_guard(
    setup, monkeypatch, publish_failure
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    env, policy, kit = setup()
    entered, proceed = Event(), Event()

    def first_action():
        if env.frame == 1:
            entered.set()
            assert proceed.wait(3)

    env.after_step = first_action
    if publish_failure:

        def fail(event):
            raise OSError("publication failed")

        monkeypatch.setattr(kit, "_dashboard_events", SimpleNamespace(emit=fail))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(kit.execute_tool, "run_policy", {"max_steps": 1})
        assert entered.wait(3)
        second = pool.submit(kit.execute_tool, "run_policy", {"max_steps": 1})
        try:
            with kit._scheduler.condition:
                assert kit._scheduler.condition.wait_for(
                    lambda: len(kit._scheduler.pending) == 1, timeout=3
                )
        finally:
            proceed.set()
        a, b = first.result(timeout=3), second.result(timeout=3)
    assert a.data["execution"]["confirmed_actions"] == 1
    assert b.data["execution"]["confirmed_actions"] == (0 if publish_failure else 1)
    assert b.data["execution"]["step_before"] == 1
    assert env.frame == (1 if publish_failure else 2)
    assert a.is_error is b.is_error is publish_failure
    assert len([e for e in policy.events if e[0] == "infer"]) == 1


def test_default_prompt_preserves_vpp2_capabilities():
    from robots.robodojo.prompt_bundle import system_prompt
    from rpent.prompt.utils import format_prompt

    prompt = format_prompt(
        system_prompt(), variables={"output_dir": "output", "recipe_tag": "result"}
    )
    for tool in ("vpp2_act", "move_delta", "segment", "back_project"):
        assert tool in prompt
    assert "run_policy" not in prompt

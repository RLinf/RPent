# Copyright 2026 The RPent Authors.
"""RoboDojo toolkit registration, observations and policy result delivery."""

import json

import pytest


def report(result):
    return result.data.get("log", {}).get("result", result.data)


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_backend_alone_exposes_policy_primitives_geometry_and_memory(
    policy_toolkit, backend
):
    from robots.robodojo.prompt_bundle import system_prompt, user_prompt
    from rpent.prompt.utils import format_prompt

    env, policy, kit = policy_toolkit(backend=backend)
    policy_tool = "awomo05_act" if backend == "awomo05" else "vpp2_act"
    variables = {
        "vla_backend": backend,
        "task": "stack_blocks",
        "seed": 0,
        "layout": 0,
        "env_cfg_type": "arx_x5",
        "output_dir": str(kit.state._output_dir),
        "task_summary": "Visible blocks",
        "recipe_tag": "stack_blocks",
        "memory_dir": str(kit.memory.root),
    }
    prompt = format_prompt(system_prompt(variables), variables=variables)
    prompt += format_prompt(user_prompt(variables), variables=variables)
    expected = {
        "view_env_state",
        "render",
        "finish",
        policy_tool,
        "move_to",
        "rotate_wrist",
        "set_gripper",
        "release",
        "back_project",
        "segment",
        "sample_world_xyz",
        "query_world_map",
        "read_text_file",
        "list_dir",
    }
    assert expected <= set(kit._tools)
    other_policy = "vpp2_act" if backend == "awomo05" else "awomo05_act"
    assert other_policy not in kit._tools
    assert other_policy not in prompt
    for name in (policy_tool, "move_to", "sample_world_xyz"):
        assert name in prompt
    assert "{{" not in prompt
    assert all(
        "world_xyz" in fields for fields in kit.state.get().state["artifacts"].values()
    )
    assert len(kit.execute_tool("view_env_state", {}).images) == 3
    assert env.frame == 0
    assert policy.events == (
        [("reset",), ("observe", 0)] if backend == "awomo05" else []
    )
    assert "segment" in prompt


@pytest.mark.parametrize("backend,horizon", [("vpp2", 24), ("awomo05", 8)])
@pytest.mark.parametrize("missing", ["capture", "saved_file"])
def test_partial_rgb_delivery_is_independent_of_policy_history(
    policy_toolkit, monkeypatch, backend, horizon, missing
):
    env, policy, kit = policy_toolkit(limit=100, backend=backend)
    before = list(policy.events)
    if missing == "capture":
        observe = env.get_obs

        def partial_observation():
            obs = observe()
            obs["vision"]["cam_head"].pop("color")
            return obs

        with monkeypatch.context() as patch:
            patch.setattr(env, "get_obs", partial_observation)
            result = kit.execute_tool("render", {})
        assert "cam_head.png" not in kit.state.get().artifacts
    else:
        kit.state.artifact_path("cam_head.png").unlink()
        result = kit.execute_tool("view_env_state", {})
    assert not result.is_error
    assert len(result.images) == 2
    assert result.data["step"] == kit.state.latest_step
    assert result.data["state"]["cameras"]["cam_left_wrist"]["shape"] == [2, 2, 3]
    assert env.frame == 0 and policy.events == before
    arguments = {"instruction": "stack blocks"} if backend == "vpp2" else {}
    executed = kit.execute_tool(f"{backend}_act", arguments)
    assert not executed.is_error
    assert report(executed)["executed_steps"] == env.frame == horizon


@pytest.mark.parametrize("backend,horizon", [("vpp2", 24), ("awomo05", 8)])
@pytest.mark.parametrize("stop", ["budget", "cancel", "uncertain", "reject"])
def test_policy_backends_share_partial_progress_and_error_feedback(
    policy_toolkit, monkeypatch, backend, horizon, stop
):
    from rpent.tools.toolkit import ToolCancelled

    env, policy, kit = policy_toolkit(
        limit=3 if stop == "budget" else 100, backend=backend
    )
    action = env._native_step

    def execute(*args, **kwargs):
        if stop == "reject" and env.frame == 3:
            return (
                env.get_obs(),
                0.0,
                False,
                {
                    "status": env.get_status(),
                    "step_before": 3,
                    "step_after": 3,
                    "executed_actions": 0,
                    "terminated": False,
                    "truncated": False,
                    "rejection": "target rejected",
                },
            )
        result = action(*args, **kwargs)
        if stop == "uncertain" and env.frame == 4:
            raise TimeoutError("reply lost after action")
        return result

    def cancelled():
        if stop == "cancel" and env.frame == 3:
            raise ToolCancelled("cancelled after three actions")

    monkeypatch.setattr(env, "_native_step", execute)
    monkeypatch.setattr(kit._primitives, "_check_cancelled", cancelled)
    name = f"{backend}_act"
    arguments = {
        "chunks": 2,
        **({"instruction": "stack blocks"} if backend == "vpp2" else {}),
    }
    result = kit.execute_tool(name, arguments)
    receipt = report(result)
    assert receipt["requested_chunks"] == 2
    assert receipt["executed_chunks"] == 0
    assert receipt["requested_steps"] == 2 * horizon
    assert receipt["executed_steps"] == receipt["executed_actions"] == 3
    assert not receipt["completed"]
    assert receipt["episode_status"] == {
        "step": 3,
        "step_limit": env.limit,
        "remaining_steps": env.limit - 3,
    }
    assert (
        receipt["stop_reason"]
        == {
            "budget": "budget_exhausted",
            "cancel": "cancelled",
            "uncertain": "execution_unknown",
            "reject": "rejected",
        }[stop]
    )
    assert result.is_error is (stop != "budget")
    assert "execution" not in result.data and "diagnostics" not in receipt
    if stop == "cancel":
        assert receipt["code"] == "tool_cancelled" and receipt["interrupted"]
    if stop == "uncertain":
        assert receipt["execution_uncertain"]
        assert env.frame == 4
        kit.resume_calls()
        assert kit.execute_tool(name, arguments).is_error
        assert env.frame == 4
    assert len([event for event in policy.events if event[0] == "infer"]) == 1


@pytest.mark.parametrize("backend,horizon", [("vpp2", 24), ("awomo05", 8)])
@pytest.mark.parametrize("completed_chunks", [0, 1])
def test_status_failure_before_chunk_preserves_confirmed_progress(
    policy_toolkit, monkeypatch, backend, horizon, completed_chunks
):
    env, policy, kit = policy_toolkit(limit=100, backend=backend)
    method = "infer" if backend == "vpp2" else "get_action"
    infer = getattr(policy, method)
    get_status = env.get_status

    def fail_once():
        monkeypatch.setattr(env, "get_status", get_status)
        raise TimeoutError("status unavailable before chunk")

    def infer_then_fail(*args):
        actions = infer(*args)
        if env.frame == completed_chunks * horizon:
            monkeypatch.setattr(env, "get_status", fail_once)
        return actions

    monkeypatch.setattr(policy, method, infer_then_fail)
    result = kit.execute_tool(
        f"{backend}_act",
        {
            "chunks": 3,
            **({"instruction": "stack blocks"} if backend == "vpp2" else {}),
        },
    )
    receipt = report(result)
    assert result.is_error
    assert not receipt["success"] and not receipt["completed"]
    assert receipt["stop_reason"] == "runtime_failure"
    assert receipt["error"] == "status unavailable before chunk"
    assert receipt["requested_chunks"] == 3
    assert receipt["requested_steps"] == receipt["requested_actions"] == 3 * horizon
    assert receipt["executed_chunks"] == completed_chunks
    assert (
        receipt["executed_steps"]
        == receipt["executed_actions"]
        == env.frame
        == completed_chunks * horizon
    )
    assert receipt["episode_status"]["step"] == env.frame
    assert kit.state.get().result["executed_steps"] == env.frame
    assert [event for event in policy.events if event[0] == "infer"] == [
        ("infer", i * horizon) for i in range(completed_chunks + 1)
    ]


@pytest.mark.parametrize("backend,horizon", [("vpp2", 24), ("awomo05", 8)])
def test_capture_failure_does_not_invalidate_either_model_history(
    policy_toolkit, monkeypatch, backend, horizon
):
    env, policy, kit = policy_toolkit(limit=100, backend=backend)
    name = f"{backend}_act"
    arguments = {"instruction": "stack blocks"} if backend == "vpp2" else {}

    def fail(*args, **kwargs):
        raise OSError("presentation unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(kit, "get_env_state", fail)
        result = kit.execute_tool(name, arguments)
        assert result.is_error and report(result)["executed_steps"] == horizon
    assert not kit.execute_tool(
        "set_gripper", {"arm": "left", "val": 0.5, "steps": 1}
    ).is_error
    assert not kit.execute_tool(name, arguments).is_error
    assert env.frame == 2 * horizon + 1
    assert [e for e in policy.events if e[0] == "infer"] == [
        ("infer", 0),
        ("infer", horizon + 1),
    ]
    if backend == "awomo05":
        assert policy.events.count(("reset",)) == 1
        assert [e[1] for e in policy.events if e[0] == "observe"] == list(
            range(env.frame + 1)
        )
    else:
        assert env._history_step == env.frame


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_environment_only_toolkit_keeps_motion_and_observation_tools(
    policy_toolkit, backend
):
    env, policy, kit = policy_toolkit(backend=backend, vla_client=None)
    assert not {"vpp2_act", "awomo05_act"} & kit._tools.keys()
    assert env._execution.policy_observer is None
    assert not kit.execute_tool("view_env_state", {}).is_error
    result = kit.execute_tool("set_gripper", {"arm": "left", "val": 0.5, "steps": 1})
    assert not result.is_error
    assert env.frame == 1
    with pytest.raises(RuntimeError, match="not connected"):
        getattr(kit._primitives, f"{backend}_act")(
            **({"instruction": "stack blocks"} if backend == "vpp2" else {})
        )
    kit.close()
    assert policy.events == []


def test_default_prompt_preserves_vpp2_capabilities():
    from robots.robodojo.prompt_bundle import system_prompt
    from rpent.prompt.utils import format_prompt

    prompt = format_prompt(
        system_prompt(),
        variables={
            "output_dir": "output",
            "recipe_tag": "result",
            "memory_dir": "memory",
        },
    )
    for tool in ("vpp2_act", "move_to", "segment", "back_project"):
        assert tool in prompt
    assert "awomo05_act" not in prompt


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
@pytest.mark.parametrize("native_success", [False, True])
def test_finish_uses_native_verdict(
    policy_toolkit, monkeypatch, backend, native_success
):
    env, policy, kit = policy_toolkit(backend)
    status = env.get_status()
    monkeypatch.setattr(
        env, "get_status", lambda: {**status, "success": native_success}
    )
    result = kit.execute_tool(
        "finish", {"status": "success", "summary": "Observed outcome"}
    )
    assert result.data["_finish"] and result.data["requested_success"]
    assert result.data["success"] is native_success
    assert result.data["status"] == ("success" if native_success else "failure")
    kit.close()
    assert not env.closed
    if backend == "awomo05":
        assert policy.events.count(("end",)) == 1


@pytest.mark.parametrize("backend,horizon", [("vpp2", 24), ("awomo05", 8)])
def test_policy_result_is_saved_once_and_read_without_mutation(
    policy_toolkit, backend, horizon
):
    from copy import deepcopy

    env, policy, kit = policy_toolkit(backend)
    arguments = {"instruction": "stack blocks"} if backend == "vpp2" else {}
    result = kit.execute_tool(f"{backend}_act", arguments)
    assert not result.is_error and len(result.images) == 3
    receipt = deepcopy(report(result))
    assert receipt["executed_steps"] == horizon
    assert kit.state.latest_step == 1
    manifest_path = kit.state._output_dir / "states.json"
    saved = manifest_path.read_text()
    assert json.loads(saved)["steps"][-1]["result"] == kit.state.get().result == receipt
    before = list(policy.events)
    viewed = kit.execute_tool("view_env_state", {"step": 1})
    assert report(viewed) == receipt and viewed.images == result.images
    report(viewed)["executed_chunks"] = -1
    assert kit.state.get().result == receipt
    assert manifest_path.read_text() == saved
    assert env.frame == horizon and policy.events == before


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_invalid_policy_budget_does_not_execute(policy_toolkit, backend):
    env, policy, kit = policy_toolkit(backend)
    before = list(policy.events)
    for chunks in (0, -1, 1.5):
        arguments = {
            "chunks": chunks,
            **({"instruction": "stack blocks"} if backend == "vpp2" else {}),
        }
        assert kit.execute_tool(f"{backend}_act", arguments).is_error
        assert env.frame == 0 and policy.events == before


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_runtime_reused_after_toolkit_closes(policy_toolkit, tmp_path, backend):
    from robots.robodojo.toolkit import RoboDojoToolkit
    from rpent.dashboard.events import NullDashboardEventSink

    env, policy, first = policy_toolkit(backend)
    first.close()
    first.close()
    env.reset()
    second = RoboDojoToolkit(
        primitives_kwargs={"env": env, "vla_client": policy, "vla_backend": backend},
        dashboard_events=NullDashboardEventSink(),
        memory=first.memory,
        state_output_dir=tmp_path / "second",
    )
    try:
        result = second.execute_tool(
            f"{backend}_act",
            {"instruction": "stack blocks"} if backend == "vpp2" else {},
        )
        assert not result.is_error and report(result)["executed_chunks"] == 1
    finally:
        second.close()
    assert not env.closed
    if backend == "awomo05":
        assert policy.events.count(("end",)) == 2


@pytest.mark.parametrize("backend", ["vpp2", "awomo05"])
def test_toolkit_startup_failure_releases_execution(
    policy_toolkit, monkeypatch, backend
):
    from robots.robodojo.toolkit import RoboDojoToolkit

    closed = []

    def fail(kit, record):
        closed.append(kit._primitives)
        raise ValueError("initial publication failed")

    monkeypatch.setattr(RoboDojoToolkit, "_publish_step", fail)
    with pytest.raises(ValueError, match="initial publication failed"):
        policy_toolkit(backend)
    primitive = closed[0]
    assert not primitive.env.closed
    assert primitive.env._execution.policy_observer is None
    if backend == "awomo05":
        assert primitive.vla_client.events.count(("end",)) == 1

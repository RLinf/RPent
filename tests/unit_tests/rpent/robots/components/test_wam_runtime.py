# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Runtime selection and native execution through registered WAM backends."""

from unittest.mock import Mock

import numpy as np
import pytest

from robots.libero.control import LIBERO_OSC
from robots.robotwin.control import ROBOTWIN_QPOS
from rpent.cli.main import _build_argparser
from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots import get_robot_spec
from rpent.robots.components import wam_runtime


def _args(spec, argv):
    parser = _build_argparser()
    spec.add_cli_args(parser, False)
    return parser.parse_args(["--seed", "0", *argv])


@pytest.mark.parametrize(
    "robot,identity,controller",
    [
        ("libero", ["--suite", "libero_spatial", "--task", "0"], LIBERO_OSC),
        ("robotwin", ["--task-name", "beat_block_hammer"], ROBOTWIN_QPOS),
    ],
)
def test_fast_wam_runtime_and_prompts_use_platform_contract(
    tmp_path, monkeypatch, robot, identity, controller
):
    spec = get_robot_spec(robot)
    args = _args(
        spec,
        [
            *identity,
            "--wam-backend",
            "fast-wam",
            "--wam-endpoint",
            "http://localhost:8117",
        ],
    )
    config = spec.parse_config(args)
    assert config.prompt_vars["memory_enabled"] is False
    assert args.memory_profile == "local"
    for name in ("system", "user"):
        prompt = spec.prompts.render(
            name, variables={**config.prompt_vars, "output_dir": str(tmp_path)}
        )
        assert "wam_act" in prompt
        assert all(
            word not in prompt
            for word in ("cosmos_act", "lingbot_act", "read_text_file", "16 actions")
        )
    components = {
        entry["name"] for entry in spec.resolve_dashboard(args)["runtime_components"]
    }
    assert "wam" in components and "vla" not in components
    caps = {"backend": "fast_wam", "control": controller}
    rpc = Mock(
        call=Mock(
            side_effect=lambda method, **kwargs: (
                caps if method == "wam.capabilities" else {"ok": True}
            )
        )
    )
    monkeypatch.setattr(wam_runtime, "make_rpc_client", lambda *a, **kw: rpc)
    monkeypatch.setattr("rpent.robots.runtime.wait_for_ready", lambda *a, **kw: None)
    owned, runtime = spec.init_runtime(
        args, tmp_path, NullDashboardEventSink(), {"wam"}
    )
    assert owned == []
    assert runtime["model"].wam.get_capabilities()["control"] == controller
    assert all(call.args[0] != "wam.reset" for call in rpc.call.call_args_list)
    runtime["model"].close()
    rpc.close.assert_called_once()


def test_owned_fast_wam_launch_and_incompatible_pair_fail_early(tmp_path, monkeypatch):
    spec = get_robot_spec("robotwin")
    for name in ("python", "checkpoint", "worker.yaml", "stats.json"):
        (tmp_path / name).touch()
    args = _args(
        spec,
        [
            "--task-name",
            "beat_block_hammer",
            "--wam-backend",
            "fast-wam",
            "--wam-python",
            str(tmp_path / "python"),
            "--wam-checkpoint",
            str(tmp_path / "checkpoint"),
            "--wam-config",
            str(tmp_path / "worker.yaml"),
            "--wam-dataset-stats",
            str(tmp_path / "stats.json"),
        ],
    )
    factory = Mock()
    monkeypatch.setattr(wam_runtime, "ProcessDaemon", factory)
    selected = wam_runtime.select_wam(args, "robotwin")
    _, rpc = selected.start_service(args, tmp_path)
    command = factory.call_args.kwargs["cmd"]
    assert command[2] == "rpent.robots.components.fast_wam.server"
    assert command[command.index("--platform") + 1] == "robotwin"
    assert command[command.index("--config") + 1] == str(tmp_path / "worker.yaml")
    rpc.close()
    args.wam_predict_future = True
    with pytest.raises(ValueError, match="cosmos-policy worker options"):
        wam_runtime.select_wam(args, "robotwin")
    args.wam_predict_future = False
    args.wam_config = args.wam_dataset_stats = None
    args.wam_backend = "cosmos-policy"
    with pytest.raises(ValueError, match="no adapter for robotwin"):
        wam_runtime.select_wam(args, "robotwin")


@pytest.mark.parametrize("action_type,dim", [("qpos", 14), ("ee", 16)])
def test_robotwin_wam_executes_negotiated_controller_and_refreshes_observation(
    action_type, dim
):
    from robots.robotwin.primitives import RoboTwinPrimitives

    status = {"eval_success": False, "step_lim": 20, "take_action_cnt": 0}
    env = Mock(
        terminated=False,
        truncated=False,
        last_info={
            "episode_status": status,
            "robot_state": {"qpos_target14": np.zeros(14)},
        },
    )
    env.get_task_language.return_value = "native task"
    env.render_camera.return_value = np.zeros((8, 8, 3), np.uint8)

    def step(actions, **kwargs):
        assert kwargs["action_type"] == action_type
        status["take_action_cnt"] += len(actions)
        env.last_info = {
            "episode_status": status,
            "robot_state": {"qpos_target14": np.full(14, status["take_action_cnt"])},
        }
        return {}, 0, False, False, {**env.last_info, "executed_actions": len(actions)}

    env.chunk_step.side_effect = step
    model = Mock(action_type=action_type)
    model.predict.return_value = np.zeros((3, dim))
    primitive = RoboTwinPrimitives(
        env=env, model=model, seed=0, check_cancelled=lambda: None, policy_kind="wam"
    )
    assert primitive.wam_act("subtask").data["executed_steps"] == 3
    assert primitive.wam_act().data["executed_steps"] == 3
    model.infer.assert_not_called()
    calls = model.predict.call_args_list
    assert [call.args[0]["task_language"] for call in calls] == [
        "subtask",
        "native task",
    ]
    np.testing.assert_array_equal(
        calls[1].args[0]["robot_state"]["qpos_target14"], np.full(14, 3)
    )
    env.reset.return_value = ({}, {})
    for episode in range(2):
        primitive.reset()
        assert model.reset.call_count == episode + 1

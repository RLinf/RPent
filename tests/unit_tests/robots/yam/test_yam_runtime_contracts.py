# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from robots.yam.cameras import YamRgbdCameraRig, YamRgbdFrame
from robots.yam.contracts import YAM_CAMERA_NAMES
from robots.yam.env_server import YamEnvFacade
from robots.yam.geometry import YamCalibration


def test_connect_observes_without_reset_or_motion(client, env):
    assert env._runtime.events == ["connect", "hold"]
    assert not env._runtime.commands
    status = client.last_info["episode_status"]
    assert not status["ready_for_motion"] and not status["eval_success"]
    with pytest.raises(RuntimeError, match="ready receipt"):
        client.reset()
    assert env._episode_id == status["episode_id"]


@pytest.mark.parametrize(
    "arm,offset,other", [("left", 0, slice(7, 14)), ("right", 7, slice(0, 7))]
)
def test_single_arm_preserves_other_command_target(primitives, env, arm, offset, other):
    command = env._previous_command.copy()
    env._runtime.qpos[other] -= 0.01
    result = primitives.set_gripper(arm=arm, val=1.0, steps=2)
    assert result["executed_steps"] == 2
    for q in env._runtime.commands:
        np.testing.assert_array_equal(q[other], command[other])
        np.testing.assert_array_equal(
            q[offset : offset + 6], command[offset : offset + 6]
        )
    assert env._runtime.commands[-1][offset + 6] == 1.0


@pytest.mark.parametrize(
    "arm,offset,other", [("left", 0, slice(7, 14)), ("right", 7, slice(0, 7))]
)
@pytest.mark.parametrize("tool", ["move_to", "rotate_wrist", "set_gripper", "release"])
@pytest.mark.parametrize("compact", [True, False])
def test_primitive_refreshes_after_operator_command(
    primitives, env, monkeypatch, arm, offset, other, tool, compact
):
    # An operator command changes accepted targets after this client's last RGBD
    # observation, as reset_pose did during the real diagnostic acceptance.
    command = env._previous_command.copy()
    command[:6] += 0.08
    command[7:13] += 0.08
    command[[6, 13]] = 0.8
    env.control_step(command, expected_episode_id=env._episode_id)
    env._runtime.commands.clear()
    env._runtime.qpos[other] -= 0.01  # target must not be replaced by feedback
    primitives.env.execution_capabilities["compact_control"] = compact
    measured = env._runtime.qpos.copy()
    planned = []

    def plan(side, target):
        planned.append(np.asarray(target))
        path = np.repeat(measured[None, offset : offset + 6], 2, axis=0)
        path[-1, 0] += 0.03
        return {"status": "Success", "position": path}

    monkeypatch.setattr(primitives.env, "plan_arm_path", plan)
    if tool == "move_to":
        result = primitives.move_to(
            arm=arm, xyz=[measured[offset] + 0.03, 0, 0], substeps=2
        )
    elif tool == "rotate_wrist":
        result = primitives.rotate_wrist(arm=arm, delta_yaw_deg=5, substeps=2)
        assert planned[0][0] == pytest.approx(measured[offset])
    else:
        result = getattr(primitives, tool)(arm=arm, val=1.0, steps=2)
        assert env._runtime.commands[0][offset + 6] == pytest.approx(0.9)
    assert result["executed_steps"] == 2
    for q in env._runtime.commands:
        np.testing.assert_array_equal(q[other], command[other])
        if tool in {"move_to", "rotate_wrist"}:
            assert q[offset + 6] == command[offset + 6]
        else:
            np.testing.assert_array_equal(
                q[offset : offset + 6], command[offset : offset + 6]
            )


@pytest.mark.parametrize(
    "fault",
    [
        "shape",
        "nan",
        "gripper",
        "hard_limit",
        "collision",
        "stale_episode",
        "not_ready",
    ],
)
def test_rejection_precedes_dispatch(ready_client, env, monkeypatch, fault):
    target = env._runtime.qpos.copy()
    expected = env._episode_id
    if fault == "shape":
        target = np.zeros(13)
    elif fault == "nan":
        target[0] = np.nan
    elif fault == "gripper":
        target[6] = 1.1
    elif fault == "hard_limit":
        target[0] = 10
    elif fault == "collision":
        monkeypatch.setattr(
            env.geometry,
            "check_qpos_transition",
            lambda *a: {"ok": False, "reason": "collision_guard"},
        )
    elif fault == "stale_episode":
        expected = "previous"
    else:
        env._operator_ready_receipt = None
    with pytest.raises((ValueError, RuntimeError)):
        ready_client.step(target, expected_episode_id=expected)
    assert not env._runtime.commands


@pytest.mark.parametrize("fault", ["stop", "runtime", "camera"])
def test_chunk_fault_never_dispatches_next_step(ready_client, env, monkeypatch, fault):
    original = env._runtime.command

    def command(target):
        result = original(target)
        if fault == "stop":
            env.request_stop()
        elif fault == "runtime":
            raise RuntimeError("CAN feedback lost")
        else:
            monkeypatch.setattr(
                env._cameras,
                "snapshot",
                lambda **kwargs: (_ for _ in ()).throw(RuntimeError("capture failed")),
            )
        return result

    monkeypatch.setattr(env._runtime, "command", command)
    actions = np.repeat(env._runtime.qpos[None], 3, axis=0)
    if fault == "stop":
        result = ready_client.chunk_step(actions)
        assert result[-1]["executed_actions"] == 1
    else:
        with pytest.raises(RuntimeError):
            ready_client.chunk_step(actions)
    assert env._stop_requested.is_set() and len(env._runtime.commands) == 1
    assert env._runtime.events[-1] == "hold"


def test_stop_is_idempotent_and_does_not_recapture_sag(ready_client, env):
    ready_client.request_stop()
    holds = env._runtime.events.count("hold")
    env._runtime.qpos[3] -= 0.025
    ready_client.request_stop()
    ready_client.read_control_state()
    ready_client.chunk_step(env._runtime.qpos[None])
    assert env._runtime.events.count("hold") == holds
    assert not env._runtime.commands


def test_shutdown_waits_for_home_and_allows_retry(client, env, monkeypatch):
    env.config["park_on_close"] = {
        "enabled": True,
        "left_qpos": env._runtime.qpos[:7].tolist(),
        "right_qpos": env._runtime.qpos[7:].tolist(),
        "duration_s": 1,
        "max_joint_delta": 0.02,
        "tolerance": 0.04,
        "timeout_s": 2,
    }
    facade = YamEnvFacade(env)

    def failed(*args, **kwargs):
        raise RuntimeError("home not reached")

    monkeypatch.setattr(env._runtime, "move_to", failed, raising=False)
    with pytest.raises(RuntimeError, match="home not reached"):
        facade._dispatch("shutdown", (), {})
    assert env.is_started() and "close" not in env._runtime.events
    assert not facade._shutdown_event.is_set()
    monkeypatch.setattr(
        env._runtime, "move_to", lambda *a, **k: env._runtime.events.append("home")
    )
    facade._dispatch("shutdown", (), {})
    assert env._runtime.events[-2:] == ["home", "close"]
    assert facade._shutdown_event.is_set()


@pytest.mark.parametrize("fault", [None, "thread", "stale", "missing", "old"])
def test_snapshot_requires_fresh_complete_frames(monkeypatch, fault):
    rig = YamRgbdCameraRig(
        {"cameras": {name: {"serial": name} for name in YAM_CAMERA_NAMES}},
        calibration=YamCalibration({}, None, {}),
    )
    monkeypatch.setattr(rig, "open", lambda: None)
    clock = iter([100.0, 100.1, 101.0])
    monkeypatch.setattr("robots.yam.cameras.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("robots.yam.cameras.time.sleep", lambda _: None)
    for name in YAM_CAMERA_NAMES:
        rig._last_frames[name] = YamRgbdFrame(
            np.ones((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=np.float32),
            {
                "timestamps": {
                    "host_before_monotonic_s": 100.0,
                    "host_after_monotonic_s": 100.0,
                }
            },
        )
    if fault == "thread":
        rig._thread_errors["right"] = RuntimeError("capture failed")
    elif fault == "stale":
        rig._last_frames["right"].camera_meta["timestamps"][
            "host_after_monotonic_s"
        ] = 90.0
    elif fault == "missing":
        del rig._last_frames["right"]
    elif fault == "old":
        rig._last_frames["right"].camera_meta["timestamps"][
            "host_before_monotonic_s"
        ] = 99.0
    if fault is not None:
        match = {
            "thread": "capture thread error: right",
            "stale": "stale: right",
            "missing": "no cached frames yet:.*right",
            "old": "requested host time: right",
        }[fault]
        with pytest.raises(RuntimeError, match=match):
            rig.snapshot(not_before_monotonic_s=100.0)
        return
    snapshot = rig.snapshot(not_before_monotonic_s=100.0)
    assert set(snapshot["views"]) == set(YAM_CAMERA_NAMES)
    for name, frame in snapshot["views"].items():
        rig._last_frames[name].rgb.fill(0)
        rig._last_frames[name].depth.fill(0)
        assert frame.rgb.all() and frame.depth.all()


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("feedback", "feedback_error"),
        ("episode", "episode_changed"),
        ("budget", "budget_exhausted"),
        ("cancel", "cancelled"),
    ],
)
def test_servo_fault_stops_without_correction(
    primitives, env, clock, monkeypatch, fault, reason
):
    from robots.yam.servo import JointServoConfig, run_joint_servo

    episode = env._episode_id
    target = primitives.env.last_info["robot_state"]["left_eef_pose"]

    def fail():
        raise RuntimeError("injected fault")

    if fault == "feedback":
        monkeypatch.setattr(primitives.env, "read_control_state", fail)
    elif fault == "episode":
        env._episode_id = "next"
    elif fault == "budget":
        env._take_action_cnt = env.step_lim
    result = run_joint_servo(
        env=primitives.env,
        apply_updates=primitives.apply_qpos_updates,
        check_cancelled=fail if fault == "cancel" else lambda: None,
        arm="left",
        nominal=env._previous_command[:6],
        target_pose=target,
        episode_id=episode,
        config=JointServoConfig.from_config({"enabled": True}),
    )
    assert not result["recoverable"] and result["stop_reason"] == reason
    assert env._stop_requested.is_set() and not env._runtime.commands


@pytest.mark.parametrize("substeps", [1, 7])
def test_move_to_never_drops_safety_waypoints(primitives, env, monkeypatch, substeps):
    start = env._previous_command.copy()
    path = np.repeat(start[None, :6], 3, axis=0)
    path[:, 0] += [0.01, 0.02, 0.03]
    monkeypatch.setattr(
        env, "plan_arm_path", lambda *a, **k: {"status": "Success", "position": path}
    )
    result = primitives.move_to(
        arm="left", xyz=[path[-1, 0], 0, 0], quat=[1, 0, 0, 0], substeps=substeps
    )
    sent = np.array(env._runtime.commands)
    assert len(sent) == max(substeps, len(path)) == result["executed_steps"]
    cursor = 0
    for waypoint in path:
        matches = np.flatnonzero(
            np.all(np.isclose(sent[cursor:, :6], waypoint), axis=1)
        )
        assert len(matches)
        cursor += int(matches[0]) + 1
    np.testing.assert_array_equal(
        sent[:, 7:], np.repeat(start[None, 7:], len(sent), axis=0)
    )

# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Check the public dual-arm API rather than a YAM-only spelling of it."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robotwin.primitives import RoboTwinPrimitives
from robots.yam.primitives import YamPrimitives
from robots.yam.tasks import classify_episode
from robots.yam.tools import TOOLS_SPEC
from robots.yam.vla_server import YamVLAFacade


def test_dual_arm_signatures_and_schemas_match():
    specs = {s["name"]: s["input_schema"] for s in TOOLS_SPEC}
    for name in ("move_to", "rotate_wrist", "set_gripper", "release"):
        yam = inspect.signature(getattr(YamPrimitives, name)).parameters
        reference = inspect.signature(getattr(RoboTwinPrimitives, name)).parameters
        reference = {
            k: v for k, v in reference.items() if not k.startswith("_") and k != "self"
        }
        public = {k: v for k, v in yam.items() if k not in {"self", "xyz_bounds"}}
        assert public == reference
        assert set(specs[name]["properties"]) == set(yam) - {"self"}
        for key, param in public.items():
            schema = specs[name]["properties"][key]
            if "default" in schema:
                assert schema["default"] == param.default
    assert "status" in specs
    assert not ({"exploration_status", "open", "close"} & set(specs))


def test_status_reads_receipt_updates_without_observe():
    native = {
        "episode_id": "current",
        "take_action_cnt": 1,
        "step_lim": 5,
        "stop_requested": False,
        "ready_for_motion": True,
        "eval_success": False,
    }
    env = SimpleNamespace(
        last_info={"episode_status": {"episode_id": "old"}},
        read_control_state=lambda: ({}, {"episode_status": native}),
    )
    p = YamPrimitives(env=env, check_cancelled=lambda: None)
    assert p.status()["episode_id"] == "current"
    assert p.status()["can_continue"]
    native["terminal_event"] = "abort"
    assert p.status()["reason"] == "abort"
    assert not p.status()["can_continue"]
    assert classify_episode(native)["eval_success"] is False


@pytest.mark.parametrize("length", [1, 20, 30])
def test_pi05_views_and_execution_length(primitives, env, model, length):
    result = primitives.pi05_act(chunks=2, use_length=length)
    assert result["completed"] and result["executed_steps"] == 2 * length
    assert len(env._runtime.commands) == 2 * length
    obs = model.calls[0]
    assert obs["states"].shape == (1, 14)
    assert obs["main_images"].shape == (1, 2, 3, 3)
    assert obs["extra_view_images"].shape == (1, 2, 2, 3, 3)
    assert obs["main_images"].min() == 1
    assert obs["extra_view_images"][0, 0].min() == 2
    assert obs["extra_view_images"][0, 1].min() == 3
    assert obs["task_descriptions"] == ["place the cube"]


@pytest.mark.parametrize("length", [0, 31, True])
def test_invalid_vla_length_never_predicts(primitives, env, model, length):
    with pytest.raises(ValueError):
        primitives.pi05_act(use_length=length)
    assert not model.calls and not env._runtime.commands


def test_vla_facade_validation_and_rlinf_gripper_saturation(primitives):
    observation = primitives._build_policy_observation()
    raw = np.linspace(-0.2, 1.2, 30 * 14, dtype=np.float32).reshape(1, 30, 14)
    calls = []

    def predict(obs, mode):
        calls.append(mode)
        return raw.copy(), {}

    facade = YamVLAFacade(model=SimpleNamespace(predict_action_batch=predict))
    with pytest.raises(ValueError):
        facade.predict({**observation, "states": np.zeros((1, 13))})
    assert calls == []
    result = facade.predict(observation)
    assert result.shape == (1, 30, 14) and calls == ["eval"]
    joints = [i for i in range(14) if i not in (6, 13)]
    np.testing.assert_array_equal(result[..., joints], raw[..., joints])
    np.testing.assert_array_equal(
        result[..., [6, 13]], np.clip(raw[..., [6, 13]], 0, 1)
    )
    raw[:] = np.nan
    with pytest.raises(ValueError):
        facade.predict(observation)


def test_world_z_rotation_and_forwarded_arguments(primitives, monkeypatch):
    from scipy.spatial.transform import Rotation

    primitives.env.last_info["robot_state"]["left_eef_pose"] = [
        0.2,
        0.1,
        0.3,
        0.5,
        0.5,
        0.5,
        0.5,
    ]
    seen = {}
    monkeypatch.setattr(
        primitives, "move_to", lambda **kwargs: seen.update(kwargs) or {"success": True}
    )
    primitives.rotate_wrist(arm="left", delta_yaw_deg=37, gripper=0.4, substeps=9)
    expected = Rotation.from_euler("z", 37, degrees=True) * Rotation.from_quat(
        [0.5] * 4, scalar_first=True
    )
    np.testing.assert_allclose(seen["quat"], expected.as_quat(scalar_first=True))
    assert (seen["xyz"], seen["gripper"], seen["substeps"]) == ([0.2, 0.1, 0.3], 0.4, 9)

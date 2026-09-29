# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.tools import LIBERO_ACTION_SCHEMA, TOOLS_SPEC, LiberoPrimitives
from rpent.robots.components.action_model_protocol import (
    ActionModelCapabilities,
    ActionModelPrediction,
    ActionModelProtocolError,
)


def _obs(value: int = 0) -> dict:
    return {
        "main_images": np.full((8, 8, 3), value, np.uint8),
        "wrist_images": np.full((8, 8, 3), value + 1, np.uint8),
        "states": np.arange(8, dtype=np.float32),
        "task_descriptions": "put the bowl on the plate",
    }


class _Env:
    return_all_frames = False

    def __init__(self) -> None:
        self.terminated = False
        self.truncated = False
        self.calls = []
        self.raw_value = 10

    def reset(self):
        return _obs(), {}

    def raw_obs(self):
        primary = np.full((8, 8, 3), self.raw_value, np.uint8)
        return {
            "agentview_image": primary,
            "robot0_eye_in_hand_image": primary + 1,
            "robot0_gripper_qpos": np.array([0.03, -0.03], np.float32),
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3], np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], np.float32),
        }

    def chunk_step(self, actions, *, return_all_frames=None):
        self.calls.append(np.array(actions, copy=True))
        observations = [_obs(index + 1) for index in range(len(actions))]
        self.raw_value += len(actions)
        obs = observations if return_all_frames else observations[-1]
        count = len(actions)
        return obs, np.zeros(count), np.zeros(count), np.zeros(count), {}


class _Wam:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = actions
        self.requests = []
        self.capabilities = ActionModelCapabilities(
            backend="cosmos_policy",
            checkpoint="predict2-2b-libero",
            supported_embodiments=("libero_7d",),
            action_dim=7,
            camera_roles=("primary", "wrist"),
            action_schema=LIBERO_ACTION_SCHEMA,
        )

    def get_capabilities(self):
        return self.capabilities

    def predict(self, request):
        self.requests.append(request)
        return ActionModelPrediction(
            actions=self.actions,
            value=0.4,
            metadata={
                "backend": "cosmos_policy",
                "checkpoint": "predict2-2b-libero",
                "embodiment": "libero_7d",
            },
        )


def _primitives(env: _Env, wam: _Wam) -> LiberoPrimitives:
    primitives = LiberoPrimitives(
        env=env,
        model=SimpleNamespace(),
        sam3_client=SimpleNamespace(),
        check_cancelled=lambda: None,
        wam_model=wam,
    )
    primitives.reset()
    return primitives


def test_wam_act_builds_unified_observation_and_executes_bounded_chunk() -> None:
    env = _Env()
    wam = _Wam(np.ones((20, 7), np.float32))
    result = _primitives(env, wam).wam_act(max_actions_per_chunk=4)

    assert result == {
        "executed_steps": 4,
        "backend": "cosmos_policy",
        "checkpoint": "predict2-2b-libero",
        "value": pytest.approx(0.4),
        "done": False,
    }
    assert env.calls[0].shape == (4, 7)
    request = wam.requests[0]
    assert request["embodiment"] == "libero_7d"
    assert request["instruction"] == "put the bowl on the plate"
    assert request["images"]["primary"].shape == (8, 8, 3)
    np.testing.assert_allclose(
        request["proprio"],
        [0.03, -0.03, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
    )
    assert np.all(request["images"]["primary"] == 10)


def test_wam_act_repredicts_from_latest_observation() -> None:
    env = _Env()
    wam = _Wam(np.ones((1, 7), np.float32))
    result = _primitives(env, wam).wam_act(max_chunks=2)
    assert result["executed_steps"] == 2
    assert len(wam.requests) == 2
    assert np.all(wam.requests[1]["images"]["primary"] == 11)


def test_wam_act_rejects_invalid_actions_before_execution() -> None:
    env = _Env()
    wam = _Wam(np.ones((2, 6), np.float32))
    with pytest.raises(ActionModelProtocolError, match=r"\[T, 7\]"):
        _primitives(env, wam).wam_act()
    assert env.calls == []


@pytest.mark.parametrize("column", [0, 3])
def test_wam_act_rejects_out_of_range_actions_before_execution(column: int) -> None:
    env = _Env()
    actions = np.zeros((1, 7), np.float32)
    actions[0, column] = 1.01
    wam = _Wam(actions)

    with pytest.raises(ActionModelProtocolError, match="outside LIBERO bounds"):
        _primitives(env, wam).wam_act()
    assert env.calls == []


def test_wam_act_clips_gripper_to_libero_bounds_before_execution() -> None:
    env = _Env()
    actions = np.zeros((2, 7), np.float32)
    actions[:, 6] = [-1.005, 1.003]
    wam = _Wam(actions)

    result = _primitives(env, wam).wam_act()

    assert result["executed_steps"] == 2
    np.testing.assert_array_equal(env.calls[0][:, 6], [-1.0, 1.0])
    np.testing.assert_array_equal(env.calls[0][:, :6], np.zeros((2, 6)))


def test_wam_act_honors_cancellation_after_inference_before_execution() -> None:
    env = _Env()
    wam = _Wam(np.ones((2, 7), np.float32))
    checks = 0

    def check_cancelled() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("cancelled during inference")

    primitives = LiberoPrimitives(
        env=env,
        model=SimpleNamespace(),
        sam3_client=SimpleNamespace(),
        check_cancelled=check_cancelled,
        wam_model=wam,
    )
    primitives.reset()

    with pytest.raises(RuntimeError, match="cancelled during inference"):
        primitives.wam_act()
    assert env.calls == []


def test_wam_act_tool_does_not_accept_agent_authored_instruction() -> None:
    spec = next(spec for spec in TOOLS_SPEC if spec["name"] == "wam_act")

    assert "instruction" not in spec["input_schema"]["properties"]
    assert spec["input_schema"].get("required", []) == []

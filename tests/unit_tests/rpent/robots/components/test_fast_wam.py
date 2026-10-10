# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Check native Fast-WAM tensor contracts without downloading model weights."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.libero.observation import physical_observation
from robots.libero.wam_client import LiberoWAMClient, libero_request
from robots.robotwin.control import ROBOTWIN_EEF
from robots.robotwin.observation import physical_observation as robotwin_observation
from robots.robotwin.vla_client import LingBotVLAClient
from robots.robotwin.wam_client import RoboTwinWAMClient
from rpent.robots.components.cosmos_policy.adapter import make_adapter as cosmos_adapter
from rpent.robots.components.fast_wam import adapter as fast_adapter
from rpent.robots.components.fast_wam.adapter.encode import encode_libero
from rpent.robots.components.fast_wam.server import FastWAMFacade
from rpent.robots.components.wam_facade_base import WAMAdapterSpec
from rpent.robots.components.wam_rpc_protocol import (
    WAMPrediction,
    normalize_wam_request,
)

torch = pytest.importorskip("torch")


def _processor(cameras):
    normalizer = Mock()
    normalizer.forward.side_effect = lambda batch: {
        "state": {"default": batch["state"]["default"] + 10}
    }
    normalizer.normalizers = {
        "action": {"default": SimpleNamespace(backward=lambda value: value * 2)}
    }
    return SimpleNamespace(
        num_output_cameras=cameras,
        shape_meta={
            "images": [{"shape": (3, 256, 256)}] * cameras,
            "state": [{"key": "default"}],
            "action": [{"key": "default"}],
        },
        action_state_transform=lambda batch: batch,
        normalizer=normalizer,
    )


def _facade(platform="libero", **overrides):
    dim = 7 if platform == "libero" else 14
    actions = torch.zeros((4, dim))
    actions[:, -1] = torch.tensor([0, 0.25, 0.5, 0.75])
    model = SimpleNamespace(
        device="cpu",
        torch_dtype=torch.float32,
        infer_action=Mock(return_value={"action": actions}),
    )
    options = {
        "checkpoint": "test",
        "platform": platform,
        "action_horizon": 4,
        "execute_steps": 3,
        "video_size": (256, 512) if platform == "libero" else (384, 320),
        "concat": "horizontal" if platform == "libero" else "robotwin",
        "prompt_template": "Task: {task}",
        **overrides,
    }
    return FastWAMFacade(model, _processor(2 if platform == "libero" else 3), **options)


def _raw_obs():
    image = np.zeros((256, 256, 3), np.uint8)
    image[0, 0] = [1, 2, 3]
    image[-1, -1] = [4, 5, 6]
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 10,
        "robot0_eef_pos": np.array([1, 2, 3]),
        "robot0_eef_quat": np.array([0, 0, np.sqrt(0.5), -np.sqrt(0.5)]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
        "task_descriptions": "pick the bowl",
    }


def test_same_physical_observation_drives_cosmos_and_fast_wam():
    raw = _raw_obs()
    raw["robot0_joint_pos"] = np.arange(7)
    raw["robot0_joint_vel"] = np.ones(7)
    physical = physical_observation(raw)
    np.testing.assert_array_equal(physical["state"]["joint_positions"], np.arange(7))
    np.testing.assert_array_equal(physical["state"]["joint_velocities"], np.ones(7))
    assert "protocol_version" not in physical and "action_space" not in physical
    request = normalize_wam_request(libero_request(raw))
    _, (encode, _) = cosmos_adapter("libero", "test")
    cosmos = encode(request)["observation"]
    np.testing.assert_allclose(
        cosmos["proprio"],
        np.r_[
            raw["robot0_gripper_qpos"], raw["robot0_eef_pos"], raw["robot0_eef_quat"]
        ],
    )
    np.testing.assert_array_equal(cosmos["primary_image"], raw["agentview_image"][::-1])

    facade = _facade()
    rpc = Mock(call=lambda method, args=(), **kw: facade._dispatch(method, args, {}))
    result = LiberoWAMClient(rpc, expected_backend="fast_wam").predict_result(raw)
    assert result.actions.shape == (3, 7)
    np.testing.assert_allclose(result.actions[:, -1], [1, 0, -1])
    native = facade._model.infer_action.call_args.kwargs
    expected = np.concatenate(
        (
            raw["agentview_image"][::-1, ::-1],
            raw["robot0_eye_in_hand_image"][::-1, ::-1],
        ),
        axis=1,
    )
    np.testing.assert_allclose(
        native["input_image"][0].permute(1, 2, 0).numpy(),
        expected.astype(np.float32) * (2 / 255) - 1,
    )
    np.testing.assert_allclose(
        native["proprio"].numpy()[0],
        np.array([1, 2, 3, 0, 0, 1.5 * np.pi, 0.04, -0.04]) + 10,
        atol=1e-6,
    )
    assert native["prompt"] == "Task: pick the bowl" and native["action_horizon"] == 4
    np.testing.assert_array_equal(raw["agentview_image"][0, 0], [1, 2, 3])


def test_robotwin_pair_keeps_joint_actions_and_three_camera_layout():
    facade = _facade("robotwin")
    observation = {
        "task_language": "move",
        "views": {
            role: {"rgb": np.full((8, 8, 3), value, np.uint8)}
            for role, value in (("head", 10), ("left_wrist", 20), ("right_wrist", 30))
        },
        "robot_state": {
            "qpos_target14": np.arange(14),
            "arm_qpos_real12": np.arange(12) + 100,
            "left_eef_pose": np.array([1, 2, 3, 1, 0, 0, 0]),
            "right_eef_pose": np.array([4, 5, 6, 1, 0, 0, 0]),
            "left_gripper": 0.25,
            "right_gripper": 0.75,
        },
    }
    physical = robotwin_observation(observation)
    np.testing.assert_array_equal(
        physical["state"]["joint_positions"], np.arange(12) + 100
    )
    np.testing.assert_array_equal(physical["state"]["joint_targets"], np.arange(14))
    vla = LingBotVLAClient("localhost", 9999)
    native_vla = Mock(return_value={"action": np.zeros((3, 16))})
    # Exercise the existing VLA transport boundary with the same physical input.
    vla._client = Mock(call=native_vla)
    vla.infer(observation)
    payload = native_vla.call_args.kwargs["args"][0]
    np.testing.assert_array_equal(
        payload["observation.state"][:7], observation["robot_state"]["left_eef_pose"]
    )
    rpc = Mock(call=lambda method, args=(), **kw: facade._dispatch(method, args, {}))
    client = RoboTwinWAMClient(rpc, expected_backend="fast_wam")
    result = client.predict(observation)
    assert result.shape == (3, 14) and client.action_type == "qpos"
    np.testing.assert_allclose(result[:, -1], [0, 0.5, 1])
    native = facade._model.infer_action.call_args.kwargs
    image = native["input_image"].numpy()[0]
    assert image.shape == (3, 384, 320)
    for point, value in (((0, 0), 10), ((300, 0), 20), ((300, 200), 30)):
        np.testing.assert_allclose(image[:, point[0], point[1]], value * 2 / 255 - 1)
    np.testing.assert_array_equal(native["proprio"].numpy()[0], np.arange(14) + 10)
    assert native["prompt"] == "Task: move"
    client.reset()
    client.close()
    rpc.close.assert_called_once()


def test_registered_pair_owns_capabilities_and_negotiates_another_robotwin_controller(
    monkeypatch,
):
    def capabilities(*, processor, binarize_gripper, **common):
        return ROBOTWIN_EEF.capabilities(
            camera_roles=("head",), state_schema={"left_eef_pose": 7}, **common
        )

    encode = Mock(
        side_effect=lambda request, **kw: {"pose": request["state"]["left_eef_pose"]}
    )
    decode = Mock(side_effect=lambda result, **kw: WAMPrediction(np.zeros((3, 16))))
    monkeypatch.setitem(
        fast_adapter.ADAPTERS,
        "robotwin_eef",
        WAMAdapterSpec(encode, decode, capabilities),
    )
    facade = _facade("robotwin_eef")
    client = RoboTwinWAMClient(
        Mock(call=lambda method, args=(), **kw: facade._dispatch(method, args, {}))
    )
    observation = {
        "views": {"head": {"rgb": np.zeros((8, 8, 3), np.uint8)}},
        "robot_state": {"left_eef_pose": np.arange(7)},
        "task_language": "move",
    }
    assert client.predict(observation).shape == (3, 16)
    assert client.action_type == "ee"
    encode.assert_called_once()
    decode.assert_called_once()


def test_robotwin_client_rejects_a_libero_controller_before_inference():
    facade = _facade("libero")
    rpc = Mock(call=lambda method, args=(), **kw: facade._dispatch(method, args, {}))
    with pytest.raises(ValueError):
        RoboTwinWAMClient(rpc).predict({})
    facade._model.infer_action.assert_not_called()


@pytest.mark.parametrize(
    "bad_actions",
    [
        torch.zeros(4, 8),
        torch.zeros(2, 4, 7),
        torch.full((4, 7), float("nan")),
        torch.cat((torch.zeros(4, 6), torch.full((4, 1), float("inf"))), dim=1),
    ],
)
def test_fast_wam_rejects_invalid_native_output(bad_actions):
    facade = _facade(binarize_gripper=True)
    facade._model.infer_action.return_value = {"action": bad_actions}
    with pytest.raises(ValueError):
        facade.predict(libero_request(_raw_obs()))


def test_platform_and_camera_mismatch_fail_before_inference():
    with pytest.raises(ValueError, match="unsupported"):
        _facade("robocasa")
    facade = _facade(video_size=(128, 128))
    with pytest.raises(ValueError, match="video_size"):
        facade.predict(libero_request(_raw_obs()))
    facade._model.infer_action.assert_not_called()


@pytest.mark.parametrize(
    "cameras,concat,size", [(1, "horizontal", (256, 256)), (2, "vertical", (512, 256))]
)
def test_libero_camera_profiles_and_identity_orientation(cameras, concat, size):
    raw = _raw_obs()
    raw["robot0_eef_quat"] = np.array([0, 0, 0, 1])
    encoded = encode_libero(
        normalize_wam_request(libero_request(raw)),
        processor=_processor(cameras),
        device="cpu",
        dtype=torch.float32,
        prompt_template="{task}",
        video_size=size,
        concat=concat,
    )
    assert tuple(encoded["input_image"].shape) == (1, 3, *size)
    np.testing.assert_array_equal(encoded["proprio"].numpy()[0, 3:6], [10, 10, 10])

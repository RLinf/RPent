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

"""Check physical-to-model transforms and native action restoration on CPU."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.libero.observation import physical_observation
from robots.libero.wam_client import LiberoWAMClient, libero_request
from robots.robotwin.observation import physical_observation as robotwin_observation
from robots.robotwin.wam_client import RoboTwinWAMClient
from rpent.robots.components.cosmos_policy.adapter import make_adapter as cosmos_adapter
from rpent.robots.components.fast_wam.server import FastWAMFacade, load_libero_policy

torch = pytest.importorskip("torch")


def test_libero_loader_preserves_worker_options_in_rlinf_config(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    factory = Mock()
    monkeypatch.setitem(
        sys.modules,
        "rlinf.models.embodiment.fastwam",
        SimpleNamespace(get_model=factory),
    )
    cfg = OmegaConf.create(
        {
            "action_horizon": 32,
            "execute_steps": 10,
            "concat": "horizontal",
            "binarize_gripper": True,
            "inference_options": {
                "seed": 1,
                "num_inference_steps": 5,
                "compile_action_infer": False,
            },
        }
    )
    path = tmp_path / "worker.yaml"
    load_libero_policy(str(path), cfg, "weights.pt", "stats.json", torch.bfloat16)
    resolved = factory.call_args.args[0]
    assert (resolved.model_path, resolved.dataset_stats_path) == (
        "weights.pt",
        "stats.json",
    )
    assert (resolved.action_horizon, resolved.num_action_chunks) == (32, 10)
    assert (resolved.seed, resolved.num_inference_steps, resolved.binarize_gripper) == (
        1,
        5,
        True,
    )
    assert resolved.fastwam.config_dir == str(tmp_path)
    assert resolved.fastwam.config_name == path.name
    composed = OmegaConf.merge(
        {"processor": {"num_output_cameras": 2}},
        OmegaConf.from_dotlist(list(resolved.fastwam.overrides)),
    )
    assert composed.data.train.processor.num_output_cameras == 2
    assert composed.data.train.num_frames == 33
    assert composed.data.train.concat_multi_camera == "horizontal"
    assert factory.call_args.kwargs == {"torch_dtype": torch.bfloat16}
    cfg.inference_options.compile_action_infer = True
    with pytest.raises(ValueError, match="compile_action_infer"):
        load_libero_policy(str(path), cfg, "weights.pt", "stats.json", torch.bfloat16)


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
        predict_action_batch=Mock(
            return_value=(np.full((1, 3, 7), 0.25, np.float32), {})
        ),
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
    assert "action_space" not in physical
    request = libero_request(raw)
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
    result = LiberoWAMClient(rpc).predict_result(raw)
    assert result["actions"].shape == (3, 7)
    np.testing.assert_array_equal(result["actions"], np.full((3, 7), 0.25))
    native = facade._model.predict_action_batch.call_args.args[0]
    np.testing.assert_array_equal(
        native["main_images"][0], raw["agentview_image"][::-1, ::-1]
    )
    np.testing.assert_array_equal(
        native["wrist_images"][0], raw["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    np.testing.assert_allclose(
        native["states"][0],
        np.array([1, 2, 3, 0, 0, 1.5 * np.pi, 0.04, -0.04]),
        atol=1e-6,
    )
    assert native["task_descriptions"] == ["pick the bowl"]
    facade._model.infer_action.assert_not_called()
    facade._model.predict_action_batch.assert_called_once_with(native, mode="eval")
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
    rpc = Mock(call=lambda method, args=(), **kw: facade._dispatch(method, args, {}))
    client = RoboTwinWAMClient(rpc)
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


def test_robotwin_client_rejects_a_libero_controller_before_inference():
    facade = _facade("libero")
    rpc = Mock(call=lambda method, args=(), **kw: facade._dispatch(method, args, {}))
    with pytest.raises(ValueError):
        RoboTwinWAMClient(rpc).predict({})
    facade._model.infer_action.assert_not_called()
    facade._model.predict_action_batch.assert_not_called()


@pytest.mark.parametrize(
    "bad_actions",
    [
        torch.zeros(4, 15),
        torch.zeros(2, 4, 14),
        torch.full((4, 14), float("nan")),
        torch.cat((torch.zeros(4, 13), torch.full((4, 1), float("inf"))), dim=1),
    ],
)
def test_robotwin_rejects_invalid_native_output_before_prefix_selection(bad_actions):
    from rpent.robots.components.fast_wam.adapter.decode import decode_robotwin

    with pytest.raises(ValueError):
        decode_robotwin(
            {"action": bad_actions}, processor=_processor(3), horizon=4, execute_steps=3
        )

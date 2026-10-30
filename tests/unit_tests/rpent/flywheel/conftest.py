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
from copy import deepcopy

import numpy as np
import pytest

from rpent.flywheel.reader import iter_trajectories
from rpent.session import EnvState, MissingInfo, Payload


def _array(value):
    value = np.asarray(value)
    return {
        "$type": "ndarray",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "value": value.tolist(),
    }


@pytest.fixture
def trajectory(tmp_path, request):
    vla_steps = getattr(request, "param", 6)
    state = EnvState(tmp_path)
    state.start_task(
        instruction="Put the bowl on the stove",
        config=Payload(
            "rpent.config",
            {
                "arguments": {
                    "robot_name": "libero",
                    "suite": "libero_goal",
                    "task": 1,
                    "seed": 0,
                }
            },
        ),
    )
    planner = state.register_planner(backend="api")
    tools = [
        state.register_tool_call(
            planner_id=planner, name=name, arguments=Payload("json", {})
        )
        for name in ("move_to", "pi0_pick")
    ]
    image = state.save_trajectory_artifact(np.zeros((256, 256, 3), dtype=np.uint8))
    observations = [
        state.register_observation(
            payload=Payload(
                "libero.observation",
                {
                    "main_images": image,
                    "wrist_images": image,
                    "states": np.full(8, index, dtype=np.float32),
                    "task_descriptions": "Put the bowl on the stove",
                },
                completeness="complete",
                inline_array_paths=("/states",),
            )
        )
        for index in range(vla_steps + 3)
    ]
    values = np.arange((vla_steps + 2) * 7, dtype=np.float32).reshape(-1, 7) / 100
    actions = [
        state.register_action(
            tool_call_id=tools[1] if 1 <= index <= vla_steps else tools[0],
            source="vla" if 1 <= index <= vla_steps else "tool_computed",
            action=Payload(
                "libero.action",
                value,
                completeness="complete",
                inline_array_paths=("",),
            ),
            before_observation_id=observations[index],
            after_observation_id=observations[index + 1],
            result=Payload(
                "libero.step_result",
                {"terminated": index == vla_steps + 1, "truncated": False},
                completeness="complete",
            ),
            missing={"executed_at": MissingInfo("not_exposed", "No execution clock")},
        )
        for index, value in enumerate(values)
    ]
    for start in range(1, vla_steps + 1, 3):
        stop = min(start + 3, vla_steps + 1)
        state.register_prediction(
            tool_call_id=tools[1],
            instruction="pick up the bowl",
            action_ids=actions[start:stop],
        )
    state.end_task(
        status="completed",
        stop_reason="finish",
        result=Payload("libero.result", {"environment_success": True}),
    )
    root = state.save_planner_trajectory()
    return next(iter_trajectories(root))


@pytest.fixture(params=["robodojo", "robotwin"])
def bimanual_trajectory(trajectory, request):
    robot = request.param
    root = trajectory["root"]
    arguments = trajectory["task"]["config"]["data"]["arguments"]
    arguments.update(robot_name=robot, layout=0)

    images = []
    for index in range(3):
        image = np.full((16, 24, 3), index + 1, dtype=np.uint8)
        image[0, 0] = [10 + index, 20 + index, 30 + index]
        np.save(root / f"camera-{index}.npy", image)
        images.append(
            {
                "$type": "artifact_ref",
                "kind": "ndarray",
                "dtype": "uint8",
                "shape": list(image.shape),
                "path": f"camera-{index}.npy",
            }
        )

    observations = {}
    count = len(trajectory["actions"])
    for index in range(count + 1):
        if robot == "robodojo":
            values = np.arange(14, dtype=np.float32) + index * 100
            fields = {
                "vision": {
                    camera: {"color": image}
                    for camera, image in zip(
                        ("cam_head", "cam_left_wrist", "cam_right_wrist"), images
                    )
                },
                "state": {
                    "right_ee_joint_state": [float(values[13])],
                    "left_ee_joint_state": [float(values[12])],
                    "right_arm_joint_state": _array(values[6:12]),
                    "left_arm_joint_state": _array(values[:6]),
                },
            }
        else:
            values = np.arange(16, dtype=np.float64) + index * 100
            fields = {
                "observation": {
                    camera: {"rgb": image}
                    for camera, image in zip(
                        ("head_camera", "left_camera", "right_camera"), images
                    )
                },
                "robot_state": {
                    "right_gripper": float(values[15]),
                    "left_gripper": float(values[7]),
                    "right_eef_pose": _array(values[8:15]),
                    "left_eef_pose": _array(values[:7]),
                    "qpos_target14": _array(np.full(14, -999.0)),
                    "left_tcp_pose": _array(np.full(7, -999.0)),
                    "right_tcp_pose": _array(np.full(7, -999.0)),
                },
            }
        observation_id = f"O{index + 1}"
        observations[observation_id] = {
            "observation_id": observation_id,
            "payload": {
                "format": f"{robot}.observation",
                "data": deepcopy(fields),
                "completeness": "complete",
                "missing": {},
            },
            "missing": {},
        }

    for index, action in enumerate(trajectory["actions"]):
        dim = 14 if robot == "robodojo" else 16
        values = np.arange(dim, dtype=np.float64) / 10 + index
        if robot == "robodojo":
            payload = {
                "right_ee_joint_state": _array(values[13:14]),
                "left_ee_joint_state": _array(values[12:13]),
                "right_arm_joint_state": _array(values[6:12]),
                "left_arm_joint_state": _array(values[:6]),
            }
            format = "robodojo.action.joint"
            result = {"done": index == count - 1}
        else:
            payload = _array(values)
            format = "robotwin.action.ee"
            result = {"terminated": index == count - 1, "truncated": False}
        action["action"] = {
            "format": format,
            "data": payload,
            "completeness": "complete",
            "missing": {},
        }
        action["result"] = {"format": f"{robot}.step_result", "data": result}
        for side in ("before", "after"):
            action[f"{side}_observation"] = observations[
                action[f"{side}_observation_id"]
            ]
    document = json.loads((root / "task.json").read_text())
    document["task"] = trajectory["task"]
    (root / "task.json").write_text(json.dumps(document))
    (root / "actions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trajectory["actions"])
    )
    return trajectory


@pytest.fixture
def robocasa_trajectory(trajectory):
    root = trajectory["root"]
    trajectory["task"]["config"]["data"]["arguments"]["robot_name"] = "robocasa"

    observations = {}
    for index in range(len(trajectory["actions"]) + 1):
        images = {}
        for camera_index, camera in enumerate(
            ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
        ):
            image = np.full((64, 64, 3), index * 4, dtype=np.uint8)
            image[..., camera_index] = 180
            image[:16, :16] = [220, 40, 80]
            path = f"{camera}-{index}.npy"
            np.save(root / path, image)
            images[camera] = {
                "$type": "artifact_ref",
                "kind": "ndarray",
                "dtype": "uint8",
                "shape": list(image.shape),
                "path": path,
            }
        values = np.arange(16, dtype=np.float64) + index * 100
        fields = {"images": images}
        start = 0
        for name, size in (
            ("robot0_base_to_eef_pos", 3),
            ("robot0_base_to_eef_quat", 4),
            ("robot0_gripper_qpos", 2),
            ("robot0_base_pos", 3),
            ("robot0_base_quat", 4),
        ):
            fields[name] = _array(values[start : start + size])
            start += size
        observation_id = f"O{index + 1}"
        observations[observation_id] = {
            "observation_id": observation_id,
            "payload": {
                "format": "robocasa.observation",
                "data": fields,
                "completeness": "complete",
                "missing": {},
            },
            "missing": {},
        }
    for index, action in enumerate(trajectory["actions"]):
        values = np.arange(12, dtype=np.float64) / 100 + index / 20
        values[[6, 11]] = [(-1) ** index, (-1) ** (index + 1)]
        action["action"] = {
            "format": "robocasa.action",
            "data": _array(values),
            "completeness": "complete",
            "missing": {},
        }
        action["result"] = {
            "format": "robocasa.step_result",
            "data": {"reward": 0.0, "done": False, "info": {}},
        }
        for side in ("before", "after"):
            action[f"{side}_observation"] = observations[
                action[f"{side}_observation_id"]
            ]
    document = json.loads((root / "task.json").read_text())
    document["task"] = trajectory["task"]
    (root / "task.json").write_text(json.dumps(document))
    (root / "actions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trajectory["actions"])
    )
    return trajectory

# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hardware-independent YAM wire and policy contracts."""

from dataclasses import dataclass

import numpy as np

YAM_CAMERA_NAMES = ("top", "left", "right")
YAM_STATUS_KEYS = (
    "eval_success",
    "take_action_cnt",
    "step_lim",
    "actual_seed",
    "episode_id",
)


@dataclass(frozen=True)
class YamModelSpec:
    policy_name: str = "pi05_yam_joint"
    camera_order: tuple[str, ...] = YAM_CAMERA_NAMES
    state_layout: str = "qpos14"
    action_layout: str = "qpos14"
    action_horizon: int = 30
    use_length: int = 30
    control_hz: int = 30


MODEL_SPEC = YamModelSpec()


def env_runtime_contract(
    *,
    task_name: str,
    seed: int = 0,
    max_episode_steps: int = 1000,
    joint_servo: dict | None = None,
) -> dict:
    contract = {
        "runtime": "yam_real_env",
        "task_name": task_name,
        "seed": int(seed),
        "seed_mode": "label_only",
        "action_layouts": ["qpos14"],
        "execution": {
            "reset": True,
            "step": True,
            "chunk_step": True,
            "action_layouts": ["qpos14"],
            "chunk_step_all_frames": True,
            "step_limit": int(max_episode_steps),
            "control_hz": 30,
        },
        "extensions": {
            "render_camera": {
                "camera_names": list(YAM_CAMERA_NAMES),
                "metric_depth": True,
            },
            "get_camera_meta": True,
            "get_task_language": True,
            "plan_arm_path": True,
            "observe": True,
            "request_stop": True,
        },
    }
    if joint_servo is not None:
        from robots.yam.servo import JointServoConfig

        contract["execution"]["joint_servo"] = JointServoConfig.from_config(
            joint_servo
        ).as_dict()
    return contract


def vla_runtime_contract() -> dict:
    """Return the identity required from a YAM Pi0.5 VLA server."""
    return {
        "runtime": "yam_vla",
        "policy_name": MODEL_SPEC.policy_name,
        "camera_order": list(MODEL_SPEC.camera_order),
        "state_layout": MODEL_SPEC.state_layout,
        "action_layout": MODEL_SPEC.action_layout,
        "action_dim": 14,
        "action_horizon": MODEL_SPEC.action_horizon,
        "use_length": MODEL_SPEC.use_length,
    }


def validate_actions(actions, *, action_type="qpos") -> np.ndarray:
    if action_type != "qpos":
        raise ValueError(
            "YAM accepts only absolute qpos14 actions (action_type='qpos')"
        )
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 14:
        raise ValueError(f"YAM actions must be [N,14], N >= 1; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("YAM actions must be finite")
    if np.any((array[:, [6, 13]] < 0) | (array[:, [6, 13]] > 1)):
        raise ValueError("YAM grippers must be in [0,1], 0=closed, 1=open")
    return array

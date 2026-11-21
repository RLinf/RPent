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

#: Identity a YAM Pi0.5 VLA server must report back to the control station.
#: The capability-side copies in ``rpent/components/models/platforms.py`` state what
#: each model server expects; keep them equal.
YAM_VLA_CONTRACT = {
    "runtime": "yam_vla",
    "policy_name": "pi05_yam_joint",
    "camera_order": ["top", "left", "right"],
    "state_layout": "qpos14",
    "action_layout": "qpos14",
    "action_dim": 14,
    "action_horizon": 30,
    "use_length": 5,
}

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
    """Values the control station and the policy share.

    Everything here that a model server also receives comes from
    ``YAM_VLA_CONTRACT``, so the wire contract and the local defaults cannot
    drift. ``control_hz`` is local: the policy contract does not carry a
    frequency.
    """

    policy_name: str = YAM_VLA_CONTRACT["policy_name"]
    camera_order: tuple[str, ...] = tuple(YAM_VLA_CONTRACT["camera_order"])
    state_layout: str = YAM_VLA_CONTRACT["state_layout"]
    action_layout: str = YAM_VLA_CONTRACT["action_layout"]
    action_horizon: int = YAM_VLA_CONTRACT["action_horizon"]
    use_length: int = YAM_VLA_CONTRACT["use_length"]
    control_hz: int = 30


MODEL_SPEC = YamModelSpec()


def env_runtime_contract(
    *,
    task_id: int,
    seed: int = 0,
    max_episode_steps: int = 1000,
    joint_servo: dict | None = None,
) -> dict:
    contract = {
        "runtime": "yam_real_env",
        "task_id": task_id,
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
            "control_hz": MODEL_SPEC.control_hz,
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
    return dict(YAM_VLA_CONTRACT)


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

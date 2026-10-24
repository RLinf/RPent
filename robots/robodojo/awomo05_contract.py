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

"""Awomo05 RoboDojo wire contract, also usable in the standalone model environment."""

import numpy as np

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_FIELDS = {
    "left_arm_joint_state": 6,
    "left_ee_joint_state": 1,
    "right_arm_joint_state": 6,
    "right_ee_joint_state": 1,
}
POLICY_METHODS = ("reset", "update_obs", "get_action")
SESSION_METHODS = (
    "vla.reset_session",
    "vla.update_obs",
    "vla.get_action",
    "vla.end_episode",
)


def validate_model_meta(meta: dict, *, bridge: bool = False) -> dict:
    """Reject incompatible endpoints before any episode reset or inference."""
    required = {
        "policy_name": "Awomo05",
        "env_cfg_type": "arx_x5",
        "action_type": "joint",
        "action_dim": 14,
        "action_layout": "native_joint_dict",
        "camera_order": list(CAMERAS),
        "input_color_order": "bgr",
        "control_hz": 25,
        "history_enabled": True,
    }
    if not isinstance(meta, dict):
        raise ValueError("Awomo05 endpoint did not return model metadata")
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise ValueError(
                f"Awomo05 {key}: expected {expected!r}, got {meta.get(key)!r}"
            )
    horizon, replan = meta.get("action_horizon"), meta.get("replan_steps")
    if (
        type(horizon) is not int
        or type(replan) is not int
        or not 1 <= replan <= horizon
    ):
        raise ValueError("Awomo05 requires 1 <= replan_steps <= action_horizon")
    methods = meta.get("methods", [])
    if not isinstance(methods, list) or not set(POLICY_METHODS).issubset(methods):
        raise ValueError(
            "Awomo05 endpoint lacks reset/update_obs/get_action capabilities"
        )
    rpc_methods = meta.get("rpc_methods", [])
    if bridge and "action_info" in meta and type(meta["action_info"]) is not bool:
        raise ValueError("Awomo05 action_info capability must be a boolean")
    if bridge and (
        meta.get("stateful") is not True
        or not isinstance(rpc_methods, list)
        or not set(SESSION_METHODS).issubset(rpc_methods)
    ):
        raise ValueError("Awomo05 requires an exclusive stateful RPent bridge")
    return meta


def validate_native_state(state: dict) -> None:
    """Validate named fields without applying Pi0.5's different vector ordering."""
    for key, size in STATE_FIELDS.items():
        value = np.asarray(state.get(key))
        if (
            value.shape != (size,)
            or not np.issubdtype(value.dtype, np.number)
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Awomo05 {key} must contain {size} finite values")

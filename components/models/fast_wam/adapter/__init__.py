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

"""Fast-WAM's supported platforms and the contract each one declares."""

from typing import Any

from components.models.fast_wam import USES_SESSIONS
from rpent.components.models.contracts import WAMModelMeta
from rpent.components.models.platforms import (
    LIBERO_OSC,
    ROBOTWIN_CAMERAS,
    ROBOTWIN_QPOS,
)


def _require_execution_window(action_horizon: Any, execute_steps: Any) -> None:
    if (
        type(action_horizon) is not int
        or type(execute_steps) is not int
        or not 0 < execute_steps <= action_horizon
    ):
        raise ValueError(
            "Fast-WAM requires 0 < execute_steps <= action_horizon (integers)"
        )


def libero_model_meta(*, checkpoint: str, model: Any) -> WAMModelMeta:
    """Read RLinf's policy configuration through the loaded model."""
    processor = model.processor
    if processor.num_output_cameras not in (1, 2):
        raise ValueError("Fast-WAM LIBERO supports one or two cameras")
    action_horizon = model.policy_cfg.action_horizon
    execute_steps = model.policy_cfg.num_action_chunks
    _require_execution_window(action_horizon, execute_steps)
    return WAMModelMeta(
        backend="fast_wam",
        checkpoint=checkpoint,
        control=LIBERO_OSC,
        camera_roles=("primary", "wrist")[: processor.num_output_cameras],
        state_schema={"eef_position": 3, "eef_quaternion_xyzw": 4, "gripper_qpos": 2},
        chunk_size=execute_steps,
        uses_sessions=USES_SESSIONS,
        returns_future_observation=False,
        returns_value=False,
        metadata={"prediction_horizon": action_horizon},
    )


def robotwin_model_meta(
    *, checkpoint: str, processor: Any, config: Any
) -> WAMModelMeta:
    """Require the official three-camera qpos14 layout and a valid prefix."""
    if (
        tuple(config.video_size) != (384, 320)
        or config.concat != "robotwin"
        or processor.num_output_cameras != 3
    ):
        raise ValueError(
            "RoboTwin requires three cameras in the [384,320] robotwin layout"
        )
    if config.get("binarize_gripper", False):
        raise ValueError("RoboTwin qpos does not use LIBERO gripper binarization")
    state_meta = processor.shape_meta["state"]
    action_meta = processor.shape_meta["action"]
    if len(state_meta) != 1 or len(action_meta) != 1:
        raise ValueError("Fast-WAM requires one merged state key and action key")
    _require_execution_window(config.action_horizon, config.execute_steps)
    return WAMModelMeta(
        backend="fast_wam",
        checkpoint=checkpoint,
        control=ROBOTWIN_QPOS,
        camera_roles=ROBOTWIN_CAMERAS,
        state_schema={"joint_targets": 14},
        chunk_size=config.execute_steps,
        uses_sessions=USES_SESSIONS,
        returns_future_observation=False,
        returns_value=False,
        metadata={"prediction_horizon": config.action_horizon},
    )


#: Platform -> the checkpoint contract this model serves on it.
ADAPTERS = {"libero": libero_model_meta, "robotwin": robotwin_model_meta}

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

"""Select matched Fast-WAM platform encoders, decoders and controller contracts."""

from functools import partial
from typing import Any

from robots.libero.control import LIBERO_OSC
from robots.robotwin.control import ROBOTWIN_QPOS
from robots.robotwin.observation import ROBOTWIN_CAMERAS
from rpent.robots.components.am_contracts import WAMModelMeta
from rpent.robots.components.am_facade_base import (
    WAMAdapter,
    WAMAdapterSpec,
)
from rpent.robots.components.fast_wam import USES_SESSIONS
from rpent.robots.components.fast_wam.adapter.decode import (
    decode_libero,
    decode_robotwin,
)
from rpent.robots.components.fast_wam.adapter.encode import (
    encode_libero,
    encode_robotwin,
)


def _libero_model_meta(*, processor: Any, **common: Any) -> WAMModelMeta:
    if processor.num_output_cameras not in (1, 2):
        raise ValueError("Fast-WAM LIBERO supports one or two cameras")
    return WAMModelMeta(
        control=LIBERO_OSC,
        camera_roles=("primary", "wrist")[: processor.num_output_cameras],
        state_schema={"eef_position": 3, "eef_quaternion_xyzw": 4, "gripper_qpos": 2},
        **common,
    )


def _robotwin_model_meta(*, processor: Any, **common: Any) -> WAMModelMeta:
    return WAMModelMeta(
        control=ROBOTWIN_QPOS,
        camera_roles=ROBOTWIN_CAMERAS,
        state_schema={"joint_targets": 14},
        **common,
    )


ADAPTERS = {
    "libero": WAMAdapterSpec(encode_libero, decode_libero, _libero_model_meta),
    "robotwin": WAMAdapterSpec(encode_robotwin, decode_robotwin, _robotwin_model_meta),
}


def make_adapter(
    platform: str,
    checkpoint: str,
    *,
    model: Any,
    processor: Any = None,
    config: Any = None,
    prompt_template: str = "{task}",
) -> tuple[WAMModelMeta, WAMAdapter]:
    """Bind checkpoint preprocessing and a supported execution prefix."""
    if platform not in ADAPTERS:
        raise ValueError(f"unsupported Fast-WAM platform: {platform!r}")
    spec = ADAPTERS[platform]
    if platform == "libero":
        processor = model.processor
        action_horizon = model.policy_cfg.action_horizon
        execute_steps = model.policy_cfg.num_action_chunks
        adapter = (partial(spec.encode, processor=processor), spec.decode)
    else:
        action_horizon = config.action_horizon
        execute_steps = config.execute_steps
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
        adapter = (
            partial(
                spec.encode,
                processor=processor,
                state_key=state_meta[0]["key"],
                device=model.device,
                dtype=model.torch_dtype,
                prompt_template=prompt_template,
            ),
            partial(
                spec.decode,
                normalizer=processor.normalizer.normalizers["action"][
                    action_meta[0]["key"]
                ],
                horizon=action_horizon,
                execute_steps=execute_steps,
            ),
        )
    if (
        type(action_horizon) is not int
        or type(execute_steps) is not int
        or not 0 < execute_steps <= action_horizon
    ):
        raise ValueError(
            "Fast-WAM requires 0 < execute_steps <= action_horizon (integers)"
        )
    model_meta = spec.model_meta(
        processor=processor,
        backend="fast_wam",
        checkpoint=checkpoint,
        chunk_size=execute_steps,
        uses_sessions=USES_SESSIONS,
        returns_future_observation=False,
        returns_value=False,
        metadata={"prediction_horizon": action_horizon},
    )
    return model_meta, adapter

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

"""Cosmos Policy's supported platforms and the contract each one declares."""

from components.models.cosmos_policy import USES_SESSIONS
from rpent.components.models.contracts import WAMModelMeta
from rpent.components.models.platforms import LIBERO_OSC

STATE_SCHEMA = {"gripper_qpos": 2, "eef_position": 3, "eef_quaternion_xyzw": 4}


def libero_model_meta(*, checkpoint: str, predict_future: bool) -> WAMModelMeta:
    """Declare the official two-camera OSC contract and its optional outputs."""
    return WAMModelMeta(
        backend="cosmos_policy",
        checkpoint=checkpoint,
        control=LIBERO_OSC,
        camera_roles=("primary", "wrist"),
        state_schema=STATE_SCHEMA,
        chunk_size=16,
        uses_sessions=USES_SESSIONS,
        returns_future_observation=predict_future,
        returns_value=predict_future,
        metadata={},
    )


#: Platform -> the checkpoint contract this model serves on it.
ADAPTERS = {"libero": libero_model_meta}

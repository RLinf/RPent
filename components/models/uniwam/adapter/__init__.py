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

"""UniWAM's supported platforms and the contract each one declares."""

from components.models.uniwam import (
    ACTION_CHUNK_SIZE,
    HISTORY_LENGTH,
    USES_SESSIONS,
)
from rpent.components.models.contracts import WAMModelMeta
from rpent.components.models.platforms import ROBOTWIN_QPOS


def robotwin_model_meta(*, checkpoint: str) -> WAMModelMeta:
    """Declare the fixed history-flow checkpoint contract."""
    return WAMModelMeta(
        backend="uniwam",
        checkpoint=checkpoint,
        control=ROBOTWIN_QPOS,
        camera_roles=("head", "left_wrist", "right_wrist"),
        state_schema={"joint_targets": 14},
        joint_target_history_shape=(HISTORY_LENGTH, 14),
        chunk_size=ACTION_CHUNK_SIZE,
        uses_sessions=USES_SESSIONS,
        prediction_timeout_s=600.0,
    )


#: Platform -> the checkpoint contract this model serves on it.
ADAPTERS = {"robotwin": robotwin_model_meta}

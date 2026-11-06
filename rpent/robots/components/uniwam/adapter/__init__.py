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

"""Paired UniWAM RoboTwin transforms and control metadata."""

from functools import partial

from robots.robotwin.control import ROBOTWIN_QPOS
from rpent.robots.components.am_contracts import WAMModelMeta
from rpent.robots.components.am_facade_base import WAMAdapter, WAMAdapterSpec
from rpent.robots.components.uniwam import ACTION_CHUNK_SIZE, USES_SESSIONS
from rpent.robots.components.uniwam.adapter.decode import decode_robotwin
from rpent.robots.components.uniwam.adapter.encode import encode_robotwin

ADAPTERS = {
    "robotwin": WAMAdapterSpec(
        encode_robotwin,
        decode_robotwin,
        partial(
            WAMModelMeta,
            control=ROBOTWIN_QPOS,
            camera_roles=("head", "left_wrist", "right_wrist"),
            state_schema={"joint_targets": 14},
        ),
    ),
}


def make_adapter(platform: str, checkpoint: str) -> tuple[WAMModelMeta, WAMAdapter]:
    """Declare the fixed history-flow checkpoint contract."""
    spec = ADAPTERS[platform]
    return spec.model_meta(
        backend="uniwam",
        checkpoint=checkpoint,
        chunk_size=ACTION_CHUNK_SIZE,
        uses_sessions=USES_SESSIONS,
        prediction_timeout_s=600.0,
    ), (spec.encode, spec.decode)

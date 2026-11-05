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

"""Pair Cosmos3's RLinf transforms with the LIBERO control contract."""

from functools import partial
from typing import Any

from robots.libero.control import LIBERO_OSC
from rpent.robots.components.am_contracts import WAMModelMeta
from rpent.robots.components.am_facade_base import WAMAdapter, WAMAdapterSpec
from rpent.robots.components.cosmos3 import USES_SESSIONS
from rpent.robots.components.cosmos3.adapter.decode import decode_libero
from rpent.robots.components.cosmos3.adapter.encode import encode_libero

ADAPTERS = {
    "libero": WAMAdapterSpec(
        encode_libero,
        decode_libero,
        partial(
            WAMModelMeta,
            control=LIBERO_OSC,
            camera_roles=("primary", "wrist"),
            state_schema={},
        ),
    ),
}


def make_adapter(
    platform: str, checkpoint: str, *, policy: Any
) -> tuple[WAMModelMeta, WAMAdapter]:
    """Bind RLinf's existing adapter to the selected platform."""
    spec = ADAPTERS[platform]
    meta = spec.model_meta(
        backend="cosmos3",
        checkpoint=checkpoint,
        chunk_size=int(policy.model_cfg.num_action_chunks),
        uses_sessions=USES_SESSIONS,
        returns_future_observation=False,
        returns_value=False,
    )
    return meta, (
        partial(spec.encode, policy=policy),
        partial(spec.decode, policy=policy),
    )

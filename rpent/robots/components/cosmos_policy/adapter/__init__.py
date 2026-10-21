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

"""Matched Cosmos Policy platform encoders and decoders."""

from functools import partial

from robots.libero.control import LIBERO_OSC
from rpent.robots.components.am_contracts import WAMModelMeta
from rpent.robots.components.am_facade_base import (
    WAMAdapter,
    WAMAdapterSpec,
)
from rpent.robots.components.cosmos_policy import USES_SESSIONS
from rpent.robots.components.cosmos_policy.adapter.decode import decode_libero
from rpent.robots.components.cosmos_policy.adapter.encode import (
    STATE_SCHEMA,
    encode_libero,
)

ACTION_SCHEMA = LIBERO_OSC["action_schema"]
ADAPTERS = {
    "libero": WAMAdapterSpec(
        encode_libero,
        decode_libero,
        partial(
            WAMModelMeta,
            control=LIBERO_OSC,
            camera_roles=("primary", "wrist"),
            state_schema=STATE_SCHEMA,
        ),
    ),
}


def make_adapter(
    platform: str, checkpoint: str, *, predict_future: bool = False
) -> tuple[WAMModelMeta, WAMAdapter]:
    """Select a supported checkpoint contract and its paired transforms."""
    if platform not in ADAPTERS:
        raise ValueError(f"unsupported Cosmos Policy platform: {platform!r}")
    spec = ADAPTERS[platform]
    model_meta = spec.model_meta(
        backend="cosmos_policy",
        checkpoint=checkpoint,
        chunk_size=16,
        uses_sessions=USES_SESSIONS,
        returns_future_observation=predict_future,
        returns_value=predict_future,
        metadata={},
    )
    return model_meta, (
        spec.encode,
        partial(spec.decode, predict_future=predict_future),
    )

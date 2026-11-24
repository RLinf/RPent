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

"""Cosmos3's supported platforms and the contract each one declares."""

from typing import Any

from components.models.cosmos3 import USES_SESSIONS
from rpent.components.models.contracts import WAMModelMeta
from rpent.components.models.platforms import LIBERO_OSC


def libero_model_meta(*, checkpoint: str, policy: Any) -> WAMModelMeta:
    """Take the executable chunk length from RLinf's already-validated config."""
    return WAMModelMeta(
        backend="cosmos3",
        checkpoint=checkpoint,
        control=LIBERO_OSC,
        camera_roles=("primary", "wrist"),
        state_schema={},
        chunk_size=int(policy.model_cfg.num_action_chunks),
        uses_sessions=USES_SESSIONS,
        returns_future_observation=False,
        returns_value=False,
    )


#: Platform -> the checkpoint contract this model serves on it.
ADAPTERS = {"libero": libero_model_meta}

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

"""Delegate Cosmos3 action normalization and rotation conversion to RLinf."""

from typing import Any

from rpent.robots.components.am_contracts import WAMPrediction


def decode_libero(result: tuple[dict, dict], *, policy: Any) -> WAMPrediction:
    """Unbatch native actions after RLinf restores LIBERO's control convention."""
    from rlinf.envs.action_utils import prepare_actions_for_libero

    response, state = result
    actions, _ = policy.parse_response(response, state)
    if actions.ndim != 3 or actions.shape[0] != 1:
        raise ValueError("Cosmos3 must return one batched action chunk")
    actions = prepare_actions_for_libero(actions[0].cpu().numpy(), model_type="cosmos3")
    return WAMPrediction(actions=actions)

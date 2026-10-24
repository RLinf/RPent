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

"""RoboDojo Awomo05 endpoint contract; the official adapter owns preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np

from robots.robodojo.awomo05_contract import (
    CAMERAS,
    STATE_FIELDS,
    validate_model_meta,
    validate_native_state,
)
from rpent.robots.components.xpolicylab_vla_client import XPolicyLabVLAClient


class Awomo05Client(XPolicyLabVLAClient):
    """Validated stateful client; no image transforms or action cache."""

    def __init__(self, client) -> None:
        super().__init__(client)
        try:
            meta = client.call("vla.get_model_meta")
        except Exception as exc:
            raise RuntimeError(
                "Awomo05 endpoint must expose vla.get_model_meta; use the RoboDojo Awomo05 bridge"
            ) from exc
        self.model_meta = validate_model_meta(meta, bridge=True)
        self.last_action_info: dict[str, Any] = {}

    def update_obs(self, obs: dict[str, Any], step: int) -> None:
        validate_native_state(obs.get("state", {}))
        for camera in CAMERAS:
            image = np.asarray(obs.get("vision", {}).get(camera, {}).get("color"))
            if (
                image.ndim != 3
                or image.shape[-1] != 3
                or min(image.shape[:2]) == 0
                or image.dtype != np.uint8
            ):
                raise ValueError(f"Awomo05 {camera} requires HWC uint8 RGB")
        if (
            not isinstance(obs.get("instruction"), str)
            or not obs["instruction"].strip()
        ):
            raise ValueError("Awomo05 requires the official task instruction")
        # Agent geometry may include depth/calibration. Keep the learned
        # model's inputs at the official RGB + joint-state + language boundary.
        payload = {
            "instruction": obs["instruction"],
            "state": {key: obs["state"][key] for key in STATE_FIELDS},
            "vision": {
                name: {"color": obs["vision"][name]["color"]} for name in CAMERAS
            },
        }
        super().update_obs(payload, step)

    def get_action(self) -> list[dict[str, Any]]:
        self.last_action_info = {}
        if self.model_meta.get("action_info"):
            result = super().get_action(include_info=True)
            if (
                not isinstance(result, dict)
                or "actions" not in result
                or not isinstance(result.get("info"), dict)
            ):
                raise ValueError("Awomo05 returned an invalid action info envelope")
            actions = result["actions"]
            self.last_action_info = result["info"]
        else:
            actions = super().get_action()
        if (
            not isinstance(actions, (list, tuple))
            or not 1 <= len(actions) <= self.model_meta["replan_steps"]
        ):
            raise ValueError("Awomo05 returned an invalid action chunk")
        for action in actions:
            if not isinstance(action, dict):
                raise ValueError("Awomo05 requires native joint action dictionaries")
            if set(action) != set(STATE_FIELDS):
                raise ValueError("Awomo05 returned unexpected action fields")
            validate_native_state(action)
        return actions

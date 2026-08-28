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

"""Pi0.5 VLA client for LIBERO."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.robots.components.vla_client_base import BaseVLAClient


class Pi05VLAClient(BaseVLAClient):
    """Adapt LIBERO observations to the common VLA RPC protocol."""

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **_kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Encode one LIBERO observation and return a Pi0.5 action chunk."""
        main_image = np.asarray(env_obs["main_images"])
        if main_image.ndim != 3:
            raise ValueError(
                f"main_images expected shape [H,W,3]; got {main_image.shape}"
            )
        if main_image.dtype != np.uint8:
            main_image = main_image.astype(np.uint8)
        observation: dict[str, Any] = {
            "main_images": main_image[None],
            "wrist_images": None,
            "extra_view_images": None,
        }
        for source_key in ("wrist_images", "extra_view_images"):
            view = env_obs.get(source_key)
            if view is None:
                continue
            array = np.asarray(view)
            if array.size > 0 and array.ndim == 3:
                if array.dtype != np.uint8:
                    array = array.astype(np.uint8)
                observation[source_key] = array[None]

        states = np.asarray(env_obs["states"], dtype=np.float32)
        if states.ndim != 1:
            raise ValueError(
                f"states must be single-env shape [state_dim]; got {states.shape}"
            )

        observation["states"] = states[None]
        observation["task_descriptions"] = [str(env_obs.get("task_descriptions") or "")]
        response = super().predict(observation, options={"mode": mode})
        actions = np.asarray(response, dtype=np.float32)[0]
        return actions, {}

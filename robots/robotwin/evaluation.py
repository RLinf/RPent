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

"""Bind RoboTwin's native action stream to one evaluation trial."""

from __future__ import annotations

import argparse
import time
from typing import Any

from rpent.evaluation.runtime import EpisodeTrial
from rpent.robots.robot_spec import RunConfig


class RoboTwinTrial(EpisodeTrial):
    """Read native success and action budgets from the RoboTwin executor."""

    native_source = "RoboTwin.TASK_ENV.eval_success"

    def __init__(
        self, args: argparse.Namespace, config: RunConfig, *, direct: bool = False
    ) -> None:
        super().__init__(
            args,
            config,
            robot="robotwin",
            task=args.task_name,
            scene=args.task_config,
            direct=direct,
        )

    def reset_evidence(self, runtime: dict[str, Any]) -> tuple[str, bool | None]:
        info = self.env.last_reset_info
        return info["instruction"], info["episode_status"].get("eval_success")

    def is_exhausted(self) -> bool:
        status = self.env.last_info.get("episode_status", {})
        limit = status.get("step_lim")
        return limit is not None and status.get("take_action_cnt", 0) >= limit

    def observe(self, info: dict[str, Any]) -> None:
        """Use the age of first native success to bound it on the runner clock."""
        received = time.monotonic()
        status = info.get("episode_status", {})
        value = status.get("eval_success")
        age = info.get("native_success_age_s")
        if isinstance(age, (int, float)) and age >= 0:
            received -= age
            value = True
        limit = status.get("step_lim")
        self.native_update(
            value,
            source=self.native_source,
            observed_at=received,
            exhausted=limit is not None and status.get("take_action_cnt", 0) >= limit,
            sealed=info.get("sealed") is True,
        )

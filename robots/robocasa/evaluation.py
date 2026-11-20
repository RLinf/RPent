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


"""Bind RoboCasa native success and executor quiescence to one measured run."""

from __future__ import annotations

import argparse
import time
from typing import Any

from rpent.evaluation.runtime import EpisodeTrial
from rpent.robots.robot_spec import RunConfig
from rpent.utils.rpc import RpcError


class RoboCasaTrial(EpisodeTrial):
    """Measure the episode admitted to the RoboCasa policy."""

    native_source = "RoboCasa.Kitchen._check_success"

    def __init__(
        self, args: argparse.Namespace, config: RunConfig, *, direct: bool = False
    ):
        super().__init__(
            args,
            config,
            robot="robocasa",
            task=args.task_name,
            scene=args.split,
            direct=direct,
            extra_budget={"actions_per_chunk": args.actions_per_chunk}
            if direct
            else None,
        )
        self.max_steps = getattr(args, "max_episode_steps", None)
        self.instruction: str
        self.exhausted = False

    def reset_evidence(self, runtime: dict[str, Any]) -> tuple[str, bool]:
        # The client already reset on connect. Move the policy's normal second
        # reset here to bind the same observation, language and episode ID.
        vla = runtime.get("vla_client")
        if vla is not None:
            checkpoint = vla.get_modality_config().get("checkpoint")
            if isinstance(checkpoint, str) and checkpoint:
                self.trial.set_checkpoint(checkpoint)
        info = self.env.reset_trial(self.max_steps)
        self.instruction = info["instruction"]
        return self.instruction, info["success"]

    def start(self, runtime: dict[str, Any]) -> dict[str, Any]:
        try:
            measured = super().start(runtime)
        except RpcError as exc:
            if exc.method != "env.reset_trial" or "unknown RPC method:" not in str(exc):
                raise
            runtime = self.legacy_runtime(runtime)
            if self.max_steps is not None:
                raise ValueError(
                    "Update the RoboCasa environment worker to enforce --max-episode-steps"
                ) from exc
            return runtime
        return {**measured, "initial_observation_ready": True}

    def is_exhausted(self) -> bool:
        return self.exhausted

    def observe(self, info: dict[str, Any]) -> None:
        """Use elapsed server time to preserve first-success timing at seal."""
        self.exhausted = self.exhausted or info.get("exhausted", False)
        age = info.get("native_success_age_s")
        observed_at = time.monotonic() - age if age is not None else None
        self.native_update(
            info["success"],
            source=self.native_source,
            observed_at=observed_at,
            exhausted=info.get("exhausted", False),
            sealed=info["sealed"],
        )

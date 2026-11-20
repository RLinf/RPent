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

"""Native LIBERO evaluation lifecycle for standard and perturbed suites."""

from __future__ import annotations

import time

from rpent.evaluation.runtime import EpisodeTrial
from rpent.utils.rpc import RpcError


class LiberoTrial(EpisodeTrial):
    """Measure the native check_success predicate and step termination."""

    native_source = "LIBERO.check_success / LiberoEnv.step.terminated"

    def __init__(self, args, config, *, direct=False):
        super().__init__(
            args,
            config,
            robot="libero",
            environment=f"libero-{args.libero_type}",
            task=str(args.task),
            scene=args.suite,
            direct=direct,
        )

    def reset_evidence(self, runtime):
        model = runtime.get("model")
        if hasattr(model, "status"):
            try:
                info = model.status()
            except (RpcError, OSError):
                # Earlier/third-party workers may not implement the optional status RPC.
                info = {}
            if isinstance(info, dict) and isinstance(info.get("checkpoint"), str):
                self.trial.set_checkpoint(info["checkpoint"])
        instruction = self.env.last_obs["task_descriptions"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("LIBERO reset did not return a task instruction")
        return instruction, self.env.begin_trial()

    def start(self, runtime):
        if not runtime["env"].last_reset_info.get("episode_id"):
            return self.legacy_runtime(runtime)
        runtime = super().start(runtime)
        for key in ("sam3_client", "molmo_client", "geometry_grounder"):
            client = runtime.get(key)
            if client is not None:
                runtime[key] = client.for_trial(self.trial.tokens)
        return runtime

    def is_exhausted(self):
        return self.env.truncated

    def observe(self, info):
        age = info.get("native_success_age_s")
        observed_at = time.monotonic() - age if age is not None else None
        self.native_update(
            info["success"],
            source=self.native_source,
            observed_at=observed_at,
            exhausted=info["truncated"],
            sealed=info["sealed"],
        )

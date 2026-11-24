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

"""Evaluate a direct VLA or WAM on one native LIBERO episode."""

from __future__ import annotations

import argparse

from robots.libero.evaluation import LiberoTrial
from robots.libero.robot_spec import get_robot_spec
from robots.libero.tools import LiberoPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("libero_direct_eval")


def main(argv=None) -> int:
    spec = get_robot_spec()
    parser = argparse.ArgumentParser(description=__doc__)
    spec.add_cli_args(parser, False)
    parser.add_argument("--output-dir")
    parser.set_defaults(
        direct_eval=True, memory_profile="local", memory_dir=None, explore=False
    )
    args = parser.parse_args(argv)
    config = spec.parse_config(args)
    output = init_output_dir(config.output_dir)
    trial = LiberoTrial(args, config, direct=True)
    policy_kind = config.task_desc["policy_kind"]
    daemons, error = [], None
    try:
        daemons, runtime = spec.init_runtime(
            args, output, NullDashboardEventSink(), {"env", policy_kind}
        )
        policy = LiberoPrimitives(
            **trial.start(runtime),
            policy_kind=policy_kind,
            sam3_client=None,
            check_cancelled=trial.remaining,
        )
        env = policy.env
        policy.set_obs(env.last_obs)
        if policy_kind == "wam":
            policy.model.reset()
        while not (env.terminated or env.truncated):
            if policy_kind == "wam":
                policy.wam_act(max_chunks=1)
            else:
                policy._vlm_chunk(env.last_obs["task_descriptions"])
    except Exception as exc:
        error = exc
        logger.exception("Direct LIBERO evaluation stopped")
    finally:
        try:
            trial.finalize(error=error is not None, startup_error=error)
        finally:
            for daemon in reversed(daemons):
                daemon.stop()
    logger.info("trial metrics: %s", trial.trial.path)
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())

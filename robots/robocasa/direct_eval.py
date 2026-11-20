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


"""Evaluate RLDX-1 directly in a native RoboCasa365 episode."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from robots.robocasa.evaluation import RoboCasaTrial
from robots.robocasa.rldx_skill import RLDXSkill
from robots.robocasa.robot_spec import get_robot_spec
from rpent.dashboard.events import NullDashboardEventSink
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("robocasa_direct_eval")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local or remote RLDX action model against a bounded native episode."""
    spec = get_robot_spec()
    parser = argparse.ArgumentParser(description=__doc__)
    spec.add_cli_args(parser, False)
    parser.add_argument("--output-dir")
    parser.add_argument("--actions-per-chunk", type=int, default=8)
    parser.set_defaults(max_episode_steps=560)
    args = parser.parse_args(argv)
    if args.actions_per_chunk <= 0:
        parser.error("--actions-per-chunk must be positive")
    config = spec.parse_config(args)
    output = init_output_dir(config.output_dir)
    trial = RoboCasaTrial(args, config, direct=True)
    daemons = []
    error = None
    try:
        daemons, runtime = spec.init_runtime(
            args, output, NullDashboardEventSink(), None
        )
        measured = trial.start(runtime)
        env = measured["env_client"]
        policy = RLDXSkill(
            env,
            measured["vla_client"],
            check_cancelled=trial.remaining,
            token_ledger=measured["token_ledger"],
        )
        while not env.check_success() and not trial.is_exhausted():
            result = policy.run(
                trial.instruction, max_chunks=1, n_action_steps=args.actions_per_chunk
            )
            if result["steps_applied"] == 0:
                raise RuntimeError("RLDX prediction advanced no native actions")
    except Exception as exc:
        error = exc
        logger.error("Direct RoboCasa evaluation failed: %s", exc)
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

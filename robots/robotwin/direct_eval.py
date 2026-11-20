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

"""Evaluate the RoboTwin action model directly, one native episode at a time."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from robots.robotwin.evaluation import RoboTwinTrial
from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.robot_spec import get_robot_spec
from rpent.dashboard.events import NullDashboardEventSink
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("robotwin_direct_eval")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one direct VLA or WAM on its native instruction and write metrics."""
    spec = get_robot_spec()
    parser = argparse.ArgumentParser(description=__doc__)
    spec.add_cli_args(parser, False)
    parser.add_argument("--output-dir")
    parser.set_defaults(memory_profile="local", memory_dir=None, explore=False)
    args = parser.parse_args(argv)
    config = spec.parse_config(args)
    output = init_output_dir(config.output_dir)
    trial = RoboTwinTrial(args, config, direct=True)
    devices = []
    error = None
    try:
        devices, runtime = spec.init_runtime(
            args, output, NullDashboardEventSink(), None
        )
        kwargs = trial.start(runtime)
        policy = RoboTwinPrimitives(**kwargs, check_cancelled=trial.remaining)
        trial.trial.record_usage(
            request_id="no-planner",
            model="none",
            module="planner",
            stage="online",
            granularity="session",
            input_tokens=0,
            output_tokens=0,
        )
        if policy.policy_kind == "wam":
            policy.model.reset()
        while True:
            status = policy.status()
            if status.get("eval_success") is True or (
                status.get("step_lim") is not None
                and status["take_action_cnt"] >= status["step_lim"]
            ):
                break
            trial.remaining()
            result = (
                policy.wam_act(max_chunks=1)
                if policy.policy_kind == "wam"
                else policy.lingbot_act(chunks=1)
            )
            if not result.data["executed_steps"]:
                raise RuntimeError("action-model prediction advanced no native actions")
    except TimeoutError as exc:
        error = exc
        logger.warning("%s", exc)
    except Exception as exc:
        error = exc
        logger.error("Direct RoboTwin evaluation failed: %s", exc)
    finally:
        try:
            trial.finalize(error=error is not None, startup_error=error)
        finally:
            for device in reversed(devices):
                device.stop()
    logger.info("trial metrics: %s", trial.trial.path)
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())

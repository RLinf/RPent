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

"""Minimal import/runtime-contract selfcheck for the BEHAVIOR robot plugin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def run_import_selfcheck() -> dict[str, Any]:
    from robots.behavior.robot_spec import get_robot_spec
    from robots.behavior.schemas import BEHAVIOR_TOOL_NAMES
    from robots.behavior.task_specs import get_task_spec

    spec = get_robot_spec()
    parser = argparse.ArgumentParser(prog="behavior-selfcheck")
    spec.add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(
        [
            "--task-name",
            "turning_on_radio",
            "--public-seed",
            "1",
        ]
    )
    args.output_dir = str(Path("/tmp/rpent_behavior_selfcheck"))
    config = spec.parse_config(args)
    return {
        "robot": spec.name,
        "robot_spec_fields": sorted(spec.__dataclass_fields__),
        "task_name": config.prompt_vars["task_name"],
        "activity_instance_id": config.prompt_vars["activity_instance_id"],
        "behavior_mode": config.prompt_vars["behavior_mode"],
        "memory_profile": config.prompt_vars["memory_profile"],
        "memory_dir": config.prompt_vars["memory_dir"],
        "tool_count_without_finish": len(BEHAVIOR_TOOL_NAMES),
        "radio_task_language": get_task_spec("turning_on_radio").task_language,
    }


def main() -> None:
    print(json.dumps(run_import_selfcheck(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["run_import_selfcheck"]

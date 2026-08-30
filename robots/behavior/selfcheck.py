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
        "behavior_episode_memory": config.prompt_vars["behavior_episode_memory"],
        "tool_count_without_finish": len(BEHAVIOR_TOOL_NAMES),
        "radio_task_language": get_task_spec("turning_on_radio").task_language,
    }


def main() -> None:
    print(json.dumps(run_import_selfcheck(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["run_import_selfcheck"]

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

"""Strict, versioned frozen-plan data contract."""

import math
import re

from jsonschema import validate

from robots.robodojo.flash.grounding import ANCHOR_METHOD
from robots.robodojo.tools import TOOLS_SPEC

ACTIONS = {"move_to", "set_gripper", "pi0_pick"}


def task_name(value: str) -> str:
    """Validate a task-derived memory filename."""
    if not re.fullmatch(r"[a-zA-Z0-9_]+", value):
        raise ValueError("Invalid Flash task name")
    return value


def vector(value) -> list[float]:
    """Require a finite three-dimensional vector."""
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    ):
        raise ValueError("Expected finite xyz vector")
    return [float(v) for v in value]


def validate_plan(plan: dict) -> dict:
    """Reject unrecognized fields/actions before executing any motion."""
    if set(plan) != {"version", "task", "anchors", "actions"} or plan["version"] != 2:
        raise ValueError("Unsupported Flash plan")
    task_name(plan["task"])
    anchors = plan["anchors"]
    if not isinstance(anchors, dict) or not anchors:
        raise ValueError("Flash plan requires anchors")
    for anchor in anchors.values():
        if (
            set(anchor) != {"query", "method", "refine_camera", "min_score"}
            or anchor["method"] != ANCHOR_METHOD
            or not isinstance(anchor["query"], str)
            or not anchor["query"].strip()
        ):
            raise ValueError("Invalid symbolic anchor")
        if (
            not isinstance(anchor["min_score"], (int, float))
            or not 0 <= anchor["min_score"] <= 1
        ):
            raise ValueError("Invalid segmentation threshold")
        if anchor["refine_camera"] not in (None, "cam_left_wrist", "cam_right_wrist"):
            raise ValueError("Invalid refinement camera")
    if not isinstance(plan["actions"], list) or not 1 <= len(plan["actions"]) <= 200:
        raise ValueError("Plan must contain 1..200 actions")
    specs = {s["name"]: s["input_schema"] for s in TOOLS_SPEC}
    for entry in plan["actions"]:
        if (
            set(entry) != {"action", "arguments", "anchor", "offset"}
            or entry["action"] not in ACTIONS
        ):
            raise ValueError("Unsupported Flash action")
        if entry["anchor"] not in anchors:
            raise ValueError("Unknown anchor")
        args = dict(entry["arguments"])
        if entry["action"] == "move_to":
            if "xyz" in args:
                raise ValueError("Frozen moves must use relative offsets")
            args["xyz"] = vector(entry["offset"])
        elif entry["offset"] is not None:
            raise ValueError("Only move_to has an offset")
        validate(args, {**specs[entry["action"]], "additionalProperties": False})
        if "gripper" in args and args["gripper"] not in (-1, 0, 1):
            raise ValueError("Invalid gripper command")
        for key, value in args.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Non-finite argument")
            if key in {"max_steps", "max_chunks"} and not 1 <= value <= 200:
                raise ValueError("Invalid action budget")
    return plan

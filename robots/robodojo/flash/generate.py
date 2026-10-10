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

"""Export a dev perception/action trace, without copying diagnostic feedback."""

import argparse
import json
from pathlib import Path

from robots.robodojo.flash.grounding import ANCHOR_METHOD, centroid_pixel
from robots.robodojo.flash.plan import ACTIONS, task_name, validate_plan, vector


def generate_plan(
    trace: list[dict], task: str, *, refine_camera: str | None = None
) -> dict:
    """Associate head-camera mask-centroid depth with command offsets.

    Require a new segment/depth pair after motion; ambiguous or unsupported
    motion traces fail closed rather than inventing an anchor.
    """
    anchors, actions = {}, []
    segment = None
    measured = None
    for entry in trace:
        name, args, result = entry["action"], entry["arguments"], entry["result"]
        if result.get("error"):
            raise ValueError("Cannot freeze a failed tool call")
        if name == "segment":
            segment = (
                (args, result)
                if result.get("found") and args.get("camera", "cam_head") == "cam_head"
                else None
            )
            measured = None
        elif name == "back_project":
            if segment is None or args.get("camera", "cam_head") != "cam_head":
                raise ValueError("Depth must follow head-camera segmentation")
            query = segment[0]["text_prompt"]
            row, col = centroid_pixel(segment[1])
            if (args["row"], args["col"]) != (row, col):
                raise ValueError("Depth pixel must be the segmentation mask centroid")
            measured = (query, vector(result["world_xyz"]))
            anchor = {
                "query": query,
                "method": ANCHOR_METHOD,
                "refine_camera": refine_camera,
                "min_score": segment[0].get("min_score", 0.2),
            }
            if query in anchors and anchors[query] != anchor:
                raise ValueError(
                    "A symbolic anchor must retain its segmentation threshold"
                )
            anchors[query] = anchor
        elif name in ACTIONS:
            if measured is None:
                raise ValueError(
                    "Each action requires a fresh measured symbolic anchor"
                )
            query, xyz = measured
            arguments = dict(args)
            offset = None
            if name == "move_to":
                target = vector(arguments.pop("xyz"))
                offset = [round(a - b, 8) for a, b in zip(target, xyz)]
            actions.append(
                {
                    "action": name,
                    "arguments": arguments,
                    "anchor": query,
                    "offset": offset,
                }
            )
            segment = measured = None
        else:
            raise ValueError(f"Unsupported recorded action: {name}")
    return validate_plan(
        {"version": 2, "task": task_name(task), "anchors": anchors, "actions": actions}
    )


def trace_from_states(manifest: dict) -> list[dict]:
    """Read action records and their preceding perception from EnvState."""
    trace = []
    for record in manifest["steps"]:
        command = record.get("command")
        if not command:
            continue
        trace.extend(record.get("extras", {}).get("perception", []))
        name = command["action"]
        if name not in {*ACTIONS, "place_in_bin", "stabilize"}:
            continue
        trace.append(
            {
                "action": name,
                "arguments": {
                    key: value for key, value in command.items() if key != "action"
                },
                "result": record["result"],
            }
        )
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--refine-camera", choices=["cam_left_wrist", "cam_right_wrist"]
    )
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    if isinstance(trace, dict):
        trace = trace_from_states(trace)
    plan = generate_plan(trace, args.task, refine_camera=args.refine_camera)
    args.destination.mkdir(parents=True, exist_ok=True)
    with (args.destination / f"{plan['task']}_plan.json").open("x") as output:
        json.dump(plan, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()

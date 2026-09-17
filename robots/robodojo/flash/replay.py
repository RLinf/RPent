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

"""Deterministic replay with head/wrist RGB-D grounding and bounded retries."""

import json
import math
from collections.abc import Callable
from pathlib import Path

from robots.robodojo.flash.plan import task_name, validate_plan, vector

PICK_ATTEMPTS = 3
REFINE_ACCEPT = 0.05
MAX_HELD_DISTANCE = 0.12
MAX_OFFSET = 0.5


def execute(toolkit, name: str, arguments: dict) -> dict:
    """Call the registered tool and stop on errors or exhausted episode budget."""
    raw = toolkit.execute_tool(name, arguments).result
    if not isinstance(raw, dict) or raw.get("error"):
        raise RuntimeError(f"Flash tool failed: {name}")
    if raw.get("truncated"):
        raise RuntimeError("Episode step budget exhausted")
    result = raw.get("log", {}).get("result", raw)
    if result.get("error") or result.get("terminated"):
        raise RuntimeError(f"Flash action stopped: {name}")
    if name == "move_to" and result.get("reached") is not True:
        raise RuntimeError("Waypoint not reached")
    return result


def locate(toolkit, query: str, camera: str, min_score: float = 0.2) -> list[float]:
    """Ground a symbolic SAM3 query through the same box-center depth contract."""
    result = execute(
        toolkit,
        "segment",
        {"text_prompt": query, "camera": camera, "min_score": min_score},
    )
    if result.get("found") is not True:
        raise RuntimeError("Anchor not visible")
    box = result["box_px"]
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise RuntimeError("Invalid segmentation box")
    result = execute(
        toolkit,
        "back_project",
        {
            "row": int((box[1] + box[3]) / 2),
            "col": int((box[0] + box[2]) / 2),
            "camera": camera,
        },
    )
    return vector(result["world_xyz"])


def held(toolkit, anchor: dict, pick_result: dict) -> bool:
    """Require both the unchanged pick heuristic and visual proximity to an EEF."""
    if pick_result.get("success") is not True:
        return False
    obs = toolkit.flash_observation()
    for arm in ("left", "right"):
        state = obs["state"]
        pose = state.get(f"{arm}_ee_pose")
        grip = state.get(f"{arm}_ee_joint_state")
        if pose is None or grip is None or float(grip[0]) >= 0.55:
            continue
        try:
            point = locate(
                toolkit, anchor["query"], f"cam_{arm}_wrist", anchor["min_score"]
            )
        except RuntimeError:
            continue
        if math.dist(point, vector(list(pose[:3]))) <= MAX_HELD_DISTANCE:
            return True
    return False


def replay(toolkit, plan: dict, note: Callable[[str], None]) -> dict:
    """Execute a validated frozen plan without consulting task verdicts."""
    if not toolkit.eval_fair:
        raise ValueError("Flash replay requires eval-fair mode")
    validate_plan(plan)
    for entry in plan["actions"]:
        if (
            entry["offset"] is not None
            and math.dist(entry["offset"], [0, 0, 0]) > MAX_OFFSET
        ):
            raise ValueError("Waypoint offset exceeds Flash safety bound")
    toolkit.flash_observation()
    coarse = {
        key: locate(toolkit, anchor["query"], "cam_head", anchor["min_score"])
        for key, anchor in plan["anchors"].items()
    }
    count = 0
    previous_move = None

    def move(entry, *, refresh=False):
        anchor = plan["anchors"][entry["anchor"]]
        point = coarse[entry["anchor"]]
        if refresh:
            toolkit.flash_observation()
            point = locate(toolkit, anchor["query"], "cam_head", anchor["min_score"])
        if camera := anchor["refine_camera"]:
            toolkit.flash_observation()
            fine = locate(toolkit, anchor["query"], camera, anchor["min_score"])
            if math.dist(fine, point) > REFINE_ACCEPT:
                raise RuntimeError("Wrist/head anchor disagreement")
            point = fine
        args = {
            **entry["arguments"],
            "xyz": [a + b for a, b in zip(point, entry["offset"])],
        }
        execute(toolkit, "move_to", args)

    for entry in plan["actions"]:
        name = entry["action"]
        note(f"Flash action {count + 1}: {name}")
        if name == "move_to":
            move(entry)
            previous_move = entry
        elif name == "pi0_pick":
            anchor = plan["anchors"][entry["anchor"]]
            for attempt in range(PICK_ATTEMPTS):
                result = execute(toolkit, name, entry["arguments"])
                if held(toolkit, anchor, result):
                    break
                if (
                    attempt == PICK_ATTEMPTS - 1
                    or previous_move is None
                    or previous_move["anchor"] != entry["anchor"]
                ):
                    raise RuntimeError("Grasp not held; replay stopped")
                move(previous_move, refresh=True)
            previous_move = None
        else:
            execute(toolkit, name, entry["arguments"])
            previous_move = None
        count += 1
    return {
        "done": True,
        "program": plan["task"],
        "plan": count,
        "anchors": len(coarse),
    }


def load_plan(root: Path, task: str) -> dict:
    """Read only a validated task plan within the selected memory directory."""
    task = task_name(task)
    path = root / "flash" / f"{task}_plan.json"
    if path.resolve().parent != (root / "flash").resolve():
        raise ValueError("Flash plan must remain within the selected memory")
    plan = validate_plan(json.loads(path.read_text()))
    if plan["task"] != task:
        raise ValueError("Flash task identity mismatch")
    return plan


def run_flash(toolkit, cell_tag: str, note: Callable[[str], None]) -> dict:
    """Load the task plan from the selected robot memory and replay it once."""
    task = task_name(toolkit._task_name)
    plan = load_plan(toolkit.memory.root, task)
    if not (cell_tag.startswith(f"{task}_l") or cell_tag == f"{task}_random"):
        raise ValueError("Flash task identity mismatch")
    return replay(toolkit, plan, note)

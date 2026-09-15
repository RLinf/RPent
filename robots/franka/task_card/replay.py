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

"""Supervised visual replay using the existing TaskCardPlanner hook."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robots.franka.task_card.common import (
    VLA_ACTIONS,
    checked_result,
    fingerprint,
    localize,
    tcp_pose,
    validate_card,
    vector,
)


def ask_human(question, choices):
    """Wait for an explicit operator answer; EOF never grants permission."""
    if not sys.stdin.isatty():
        raise RuntimeError("Franka task cards require an operator terminal")
    while True:
        answer = input(f"{question} [{' / '.join(choices)}]: ").strip().lower()
        if answer in choices:
            return answer


def add_cli_args(parser):
    parser.add_argument("--task-card", help="Generated Franka task-card JSON")
    parser.add_argument("--molmo-endpoint", help="Live Molmo grounding endpoint")


def prepare(args, robot):
    if getattr(args, "planner", None) != "task_card":
        return None
    if getattr(args, "dashboard", False) or getattr(args, "interactive", False):
        raise ValueError(
            "Franka task-card confirmation uses its own terminal; omit --dashboard/--interactive"
        )
    if not getattr(args, "task_card", None) or not getattr(
        args, "molmo_endpoint", None
    ):
        raise ValueError("Franka task_card requires --task-card and --molmo-endpoint")
    from robots.franka.runtime_config import set_calibration_path

    set_calibration_path(args.calibration_path)
    card = validate_card(json.loads(Path(args.task_card).read_text()))
    if card["robot"] != robot or card["task"] != f"{robot}_t{args.task_id}":
        raise ValueError("card robot/task does not match requested robot/task")
    if card["requirements"] != fingerprint():
        raise ValueError(
            "card robot configuration or calibration changed; regenerate/review the card"
        )
    return {"card": card, "endpoint": args.molmo_endpoint}


def authorize_runtime(args):
    """Called before creating an EnvClient, whose constructor resets hardware."""
    if getattr(args, "planner", None) == "task_card":
        if getattr(args, "dashboard", False) or getattr(args, "interactive", False):
            raise ValueError(
                "Franka task cards require the dedicated operator terminal"
            )
        options = getattr(args, "_task_card_options", None)
        if options and any(e["action"] in VLA_ACTIONS for e in options["card"]["plan"]):
            if options["card"]["robot"] == "franka" and not args.vla_endpoint:
                raise ValueError("single-Franka VLA task card requires --vla-endpoint")
        result = ask_human(
            "Restore the scene and clear BOTH arm workspaces. Allow environment initialization/reset?",
            ("ready", "abort"),
        )
        if result != "ready":
            raise RuntimeError("operator aborted before environment initialization")


def motion_arguments(entry, *, robot, state, molmo, workspace):
    args = dict(entry["arguments"])
    action = entry["action"]
    if action not in {"move_delta", "rotate_delta"}:
        return args
    pose = tcp_pose(state.get().state, robot, args.get("arm"))
    if action == "move_delta":
        point = localize(state, robot, entry["anchor"], molmo, arm=args.get("arm"))
        target = point + vector(entry["offset_xyz"])
        lower, upper = workspace["ee_pose_limit_min"], workspace["ee_pose_limit_max"]
        if robot == "dual_franka":
            index = 0 if args["arm"] == "left" else 1
            lower, upper = lower[index], upper[index]
        workspace_target = target
        if robot == "dual_franka" and args["arm"] == "left":
            from robots.dual_franka.perception import (
                transform_point_between_base_frames,
            )

            workspace_target = transform_point_between_base_frames(
                target, source="right_base", target="left_base"
            )
        if np.any(workspace_target < vector(lower[:3])) or np.any(
            workspace_target > vector(upper[:3])
        ):
            raise ValueError("grounded target is outside the configured arm workspace")
        delta = target - pose[:3]
        if np.linalg.norm(delta) > 0.20:
            raise ValueError(
                "grounded translation exceeds 0.20m; use intermediate recorded steps"
            )
        args["delta_xyz"] = delta.tolist()
    else:
        current = Rotation.from_quat(pose[3:])
        if entry["rotation_mode"] == "relative":
            reference = Rotation.from_quat(entry["reference_quaternion"])
            if (current * reference.inv()).magnitude() > 0.15:
                raise ValueError(
                    "relative rotation starting pose differs by more than 0.15rad"
                )
        else:
            target = Rotation.from_quat(entry["target_quaternion"])
            args["delta_rpy"] = (target * current.inv()).as_euler("xyz").tolist()
        if Rotation.from_euler("xyz", vector(args["delta_rpy"])).magnitude() > 0.35:
            raise ValueError(
                "rotation exceeds 0.35rad; use intermediate recorded steps"
            )
    return args


def replay(toolkit, card, molmo, workspace, *, human=ask_human, note=lambda _: None):
    validate_card(card)
    events = []
    outcome = {
        "card": card["task"],
        "done": False,
        "plan": 0,
        "anchors": 0,
        "events": events,
        "human_verdict": None,
    }
    toolkit._task_card_solved = False
    if any(e["action"] in VLA_ACTIONS for e in card["plan"]):
        primitives = getattr(toolkit, "_primitives", None)
        if primitives is not None and primitives.model is None:
            raise ValueError("task card requires a connected VLA model")
    try:
        decision = human(
            "Environment initialized. Clear both workspaces and start card execution?",
            ("start", "abort"),
        )
        events.append(
            {
                "kind": "start",
                "answer": decision,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if decision != "start":
            outcome["status"] = "aborted"
            return outcome
        for entry in card["plan"]:
            toolkit.raise_if_cancelled()
            # Refresh observations at each boundary, including after operator wait.
            toolkit.refresh_task_card_state()
            args = motion_arguments(
                entry,
                robot=card["robot"],
                state=toolkit.state,
                molmo=molmo,
                workspace=workspace,
            )
            note(f"source step {entry['source_step']}: {entry['action']} {args}")
            event = {
                "source_step": entry["source_step"],
                "before_step": toolkit.state.latest_step,
                "action": entry["action"],
                "arguments": args,
            }
            events.append(event)
            result = toolkit.execute_tool(entry["action"], args).result
            event["replay_step"] = toolkit.state.latest_step
            # Franka's tool output is a rendered state; inspect the raw result too.
            checked_result(result)
            checked_result(toolkit.state.latest_record().result)
            outcome["plan"] += 1
            outcome["anchors"] += int(entry["action"] == "move_delta")
        verdict = human(
            "Card actions finished. Did the physical task succeed?",
            ("success", "failure"),
        )
        events.append(
            {
                "kind": "outcome",
                "answer": verdict,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        outcome.update(human_verdict=verdict, done=verdict == "success", status=verdict)
        toolkit._task_card_solved = outcome["done"]
        return outcome
    except BaseException as exc:
        outcome.update(status="failure", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        toolkit.state.save("task_card_outcome.json", outcome, step=None)
        toolkit.state.save(
            "task_card_recipe.jsonl",
            [
                {"action": event["action"], **event["arguments"]}
                for event in events
                if "action" in event
            ],
            step=None,
        )


def replay_card(toolkit, cell_tag, note):
    from robots.franka.runtime_config import get_robot_config_path, load_mapping
    from rpent.robots.components.molmo_client import MolmoClient
    from rpent.utils.rpc import make_rpc_client

    options = toolkit.task_card_options
    if not options or options["card"]["task"] != cell_tag:
        raise ValueError("missing or mismatched task card")
    if options["card"]["requirements"] != fingerprint():
        raise ValueError("configuration changed after card preparation")
    rpc = make_rpc_client(options["endpoint"])
    try:
        return replay(
            toolkit,
            options["card"],
            MolmoClient(rpc),
            load_mapping(get_robot_config_path())["workspace"],
            note=note,
        )
    finally:
        rpc.close()

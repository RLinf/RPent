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

"""Select continuous ranges of executed VLA actions."""

from typing import Any

from rpent.flywheel.flywheel import (
    read_actions,
    successful_prefix,
    validate_observation,
)
from rpent.robots import get_robot_spec


def select_vla_segments(trajectory: dict[str, Any]) -> dict[str, list[dict]]:
    """Return continuous ranges of executed VLA actions and exclusion reasons."""
    task = trajectory["task"]
    robot = task["config"]["data"]["arguments"].get("robot_name")
    if not isinstance(robot, str) or not robot.isidentifier():
        raise ValueError("Task config must identify a robot_name")
    config = get_robot_spec(robot).flywheel
    if config is None:
        raise ValueError(f"Trajectory data selection is not supported for {robot}")

    actions = trajectory["actions"]
    result = task.get("result") or {}
    stop = 0
    if (result.get("data") or {}).get("environment_success") is True and not any(
        key in result.get("missing", {}) for key in ("", "/environment_success")
    ):
        stop = successful_prefix(actions, config)
    segments = []
    rejected = []
    if not stop:
        return {
            "segments": [],
            "rejected": [
                {
                    "task_id": task["task_id"],
                    "reason": "no_confirmed_environment_success",
                }
            ],
        }

    positions = {}
    for index, action in enumerate(actions):
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise ValueError(f"Action at index {index} has no action_id")
        if action_id in positions:
            raise ValueError(f"Duplicate action_id: {action_id}")
        positions[action_id] = index

    references = {}
    seen_predictions = set()
    for prediction in trajectory["predictions"]:
        prediction_id = prediction.get("prediction_id")
        if (
            not isinstance(prediction_id, str)
            or not prediction_id
            or prediction_id in seen_predictions
        ):
            raise ValueError(f"Missing or duplicate prediction_id: {prediction_id}")
        seen_predictions.add(prediction_id)
        try:
            action_ids = prediction["action_ids"]
            if not isinstance(action_ids, list) or any(
                not isinstance(item, str) for item in action_ids
            ):
                raise ValueError("Prediction action_ids must be a list of IDs")
            if not action_ids or not any(
                positions.get(item, stop) < stop for item in action_ids
            ):
                continue
            if len(set(action_ids)) != len(action_ids):
                raise ValueError("Prediction repeats an action_id")
            instruction = prediction.get("instruction")
            if (
                not isinstance(instruction, str)
                or not instruction.strip()
                or not prediction.get("tool_call_id")
            ):
                raise ValueError("Missing local instruction or tool call")
            if any(
                key in prediction.get("missing", {})
                for key in ("action_ids", "instruction")
            ):
                raise ValueError("Missing prediction data")
            known_positions = [
                positions[item] for item in action_ids if item in positions
            ]
            if known_positions != sorted(known_positions):
                raise ValueError("Prediction action order differs from execution order")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            rejected.append(
                {"prediction_id": prediction_id, "reason": f"invalid_prediction: {exc}"}
            )
            continue
        for action_id in action_ids:
            if action_id not in positions:
                rejected.append(
                    {
                        "prediction_id": prediction_id,
                        "action_id": action_id,
                        "reason": "missing_action_record",
                    }
                )
                continue
            references.setdefault(action_id, []).append(prediction)

    segment = previous = None
    for index, action in enumerate(actions):
        action_id = action["action_id"]
        claims = references.get(action_id, [])
        if not claims:
            segment = previous = None
            continue
        reason = None
        if index >= stop:
            reason = "after_environment_success"
        elif len(claims) > 1:
            reason = "multiple_predictions_claim_action"
        else:
            prediction = claims[0]
            try:
                if action.get("source") != "vla":
                    raise ValueError("Prediction references a non-VLA action")
                if action.get("tool_call_id") != prediction["tool_call_id"]:
                    raise ValueError(
                        "Action and prediction belong to different tool calls"
                    )
                if any(
                    key in action.get("missing", {})
                    for key in (
                        "action",
                        "before_observation_id",
                        "after_observation_id",
                    )
                ):
                    raise ValueError("Missing action or observation reference")
                for side in ("before", "after"):
                    observation = action.get(f"{side}_observation")
                    observation_id = action.get(f"{side}_observation_id")
                    if (
                        not observation_id
                        or not isinstance(observation, dict)
                        or observation.get("observation_id") != observation_id
                    ):
                        raise ValueError(f"Missing or inconsistent {side} observation")
                    if side == "before":
                        validate_observation(trajectory["root"], observation, config)
                read_actions(trajectory["root"], action["action"], config)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                reason = f"invalid_action: {exc}"

        if reason is not None:
            rejected.extend(
                {
                    "prediction_id": prediction["prediction_id"],
                    "action_id": action_id,
                    "reason": reason,
                }
                for prediction in claims
            )
            segment = previous = None
            continue

        consecutive = (
            segment is not None
            and segment["stop_index"] == index
            and segment["instruction"] == prediction["instruction"]
            and previous["action"]["format"] == action["action"]["format"]
            and previous["after_observation"] == action["before_observation"]
        )
        if not consecutive:
            segment = {
                "start_index": index,
                "stop_index": index,
                "instruction": prediction["instruction"],
            }
            segments.append(segment)
        segment["stop_index"] = index + 1
        previous = action

    return {"segments": segments, "rejected": rejected}

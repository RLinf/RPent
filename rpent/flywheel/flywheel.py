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

"""Convert native trajectories using robot field declarations."""

from pathlib import Path
from typing import Any

import numpy as np

from rpent.flywheel.reader import read_array, validate_image


def successful_prefix(actions: list[dict], config: dict[str, Any]) -> int:
    """Return the prefix through native success, or all actions for task-only success."""
    success = config["success"]
    if success is None:
        return len(actions)
    field = success["field"]
    truncated = success.get("truncated")
    for index, action in enumerate(actions):
        result = action.get("result") or {}
        if result.get("format") != config["result_format"]:
            continue
        outcome = result.get("data") or {}
        if outcome.get(field) is True and not any(
            key in result.get("missing", {}) for key in ("", f"/{field}")
        ):
            return index + 1
        if truncated and outcome.get(truncated) is True:
            return 0
    return 0


def _read_field(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Read a field unless it or an ancestor is marked missing."""
    missing = payload.get("missing", {})
    pointer = ""
    if pointer in missing:
        raise ValueError(f"Missing {payload['format']} data")
    value = payload.get("data")
    for key in path:
        pointer += "/" + key.replace("~", "~0").replace("/", "~1")
        if pointer in missing:
            raise ValueError(f"Missing {payload['format']} field {pointer}")
        try:
            value = value[key]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing {payload['format']} field {pointer}") from exc
    return value


def _read_vector(
    root: Path, payload: dict[str, Any], layout: dict[str, Any]
) -> np.ndarray:
    """Validate and concatenate numeric fields in their declared order."""
    parts = []
    for field in layout["fields"]:
        value = _read_field(payload, field["path"])
        if isinstance(value, dict):
            part = read_array(root, value)
        elif "dtype_from" in field:
            part = np.asarray(value, dtype=parts[field["dtype_from"]].dtype)
        elif "raw_dtype" in layout:
            part = np.asarray(value, dtype=layout["raw_dtype"])
        else:
            raise ValueError(
                f"Expected an encoded ndarray for {payload['format']} "
                f"field {field['path']}"
            )
        if part.shape != field["shape"] or not np.isfinite(part).all():
            raise ValueError(
                f"Invalid {payload['format']} field {field['path']}: "
                f"expected shape {field['shape']} with finite values, got {part.shape}"
            )
        parts.append(part.reshape(-1))
    return np.concatenate(parts)


def read_actions(
    root: Path, payload: dict[str, Any], config: dict[str, Any]
) -> np.ndarray:
    """Read one executed action and apply its declared training value maps."""
    payload = payload or {}
    layout = config["action"]
    if (
        payload.get("format") != layout["format"]
        or payload.get("completeness") != "complete"
        or payload.get("missing")
    ):
        raise ValueError(f"Missing or unsupported {layout['format']} action payload")
    values = _read_vector(root, payload, layout)
    for index, mapping in layout.get("value_maps", {}).items():
        if values[index] not in mapping:
            raise ValueError(
                f"Invalid {layout['format']} action[{index}]: "
                f"expected one of {tuple(mapping)}, got {values[index]}"
            )
        values[index] = mapping[values[index]]
    return values


def validate_observation(
    root: Path, observation: dict[str, Any], config: dict[str, Any]
) -> np.ndarray:
    """Check required observation fields and return the state, leaving images unloaded."""
    payload = observation.get("payload") or {}
    if payload.get("format") != config["observation_format"]:
        raise ValueError(f"Expected a {config['observation_format']} observation")
    if "payload" in observation.get("missing", {}):
        raise ValueError("Missing observation payload")
    state = _read_vector(root, payload, config["state"])
    for name, field in config["images"].items():
        validate_image(
            _read_field(payload, field["path"]),
            name,
            expected_shape=field.get("shape"),
        )
    return state


def read_frame(
    root: Path,
    action_record: dict[str, Any],
    instruction: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Read one executed action and its before observation as a training frame."""
    observation = action_record["before_observation"]
    state = validate_observation(root, observation, config)
    actions = read_actions(root, action_record["action"], config)
    return {
        **{
            name: read_array(root, _read_field(observation["payload"], field["path"]))
            for name, field in config["images"].items()
        },
        "observation.state": state,
        "action": actions,
        "prompt": instruction,
    }

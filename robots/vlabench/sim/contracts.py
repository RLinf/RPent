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

"""VLABench environment identity, cameras and episode budget."""

from __future__ import annotations

from typing import Any

import numpy as np

#: Camera roles the VLABench EnvServer exposes, in fixed order.
VLABENCH_CAMERA_NAMES = ("agentview", "left", "right", "wrist")

#: Camera role -> camera name inside the VLABench MuJoCo task.
VLABENCH_NATIVE_CAMERAS = {
    "agentview": "forward",
    "left": "left",
    "right": "right",
    "wrist": "franka/Franka_wrist_cam",
}

#: Official evaluation track used when ``--track`` is not given.
DEFAULT_TRACK = "track_1_in_distribution"

#: Native control steps one run may use unless ``--max-episode-steps``
#: overrides it. This is a custom run budget, not the official VLABench one;
#: the server still reports the official limit for provenance.
DEFAULT_EPISODE_STEPS = 10_000

#: The only robot this integration drives.
VLABENCH_ROBOT = "franka"


def env_identity(
    *, task_name: str, track: str, episode_index: int, seed: int
) -> dict[str, Any]:
    """Return the launch identity a VLABench EnvServer must report."""
    return {
        "benchmark": "vlabench",
        "robot": VLABENCH_ROBOT,
        "task_name": task_name,
        "track": track,
        "episode_index": int(episode_index),
        "seed": int(seed),
    }


def check_env_meta(
    meta: dict[str, Any],
    *,
    task_name: str,
    track: str,
    episode_index: int,
    seed: int,
) -> dict[str, Any]:
    """Check the identity fields of a server's metadata and return a copy.

    The server also reports provenance it alone can compute, such as the track
    file hash and the VLABench commit, so only the launch identity is compared.

    Raises:
        ValueError: If the server was launched for a different task or episode.
    """
    expected = env_identity(
        task_name=task_name, track=track, episode_index=episode_index, seed=seed
    )
    mismatched = {
        key: (value, meta.get(key))
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatched:
        raise ValueError(
            f"VLABench server identity mismatch (expected, actual): {mismatched}"
        )
    return dict(meta)


#: Reasons a motion receipt may report for having stopped.
MOTION_STOP_REASONS = frozenset(
    {
        "reached",
        "max_steps",
        "ik_failed",
        "completed",
        "cancelled",
        "native_success",
        "budget_exhausted",
        "environment_terminal",
    }
)


def validate_motion_receipt(
    result: Any, method: str, arguments: dict[str, Any], motion_id: str
) -> None:
    """Check a motion receipt for internal consistency after the motion ran.

    A receipt that is incomplete or contradicts itself leaves the execution
    state unknown, so the caller must not retry the motion. Raises
    ``RuntimeError`` if the receipt is malformed or inconsistent.

    Args:
        result: The receipt returned by ``env.execute_pose`` or
            ``env.execute_gripper``.
        method: The RPC method that produced it.
        arguments: The keyword arguments the motion was dispatched with.
        motion_id: The token the motion was dispatched under.
    """

    def require(condition: bool, detail: str) -> None:
        if not condition:
            raise RuntimeError("invalid VLABench motion receipt: " + detail)

    def number(value: Any) -> bool:
        return type(value) in (int, float) and bool(np.isfinite(value))

    require(isinstance(result, dict), "expected object")
    for key in ("ok", "terminated", "truncated", "interrupted"):
        require(type(result.get(key)) is bool, key)
    require(result.get("motion_id") == motion_id, "motion identity")
    requested = arguments.get("max_steps", arguments.get("steps"))
    used = result.get("steps_used")
    require(type(used) is int and 0 <= used <= requested, "steps_used")
    require(
        type(result.get("requested_steps")) is int
        and result["requested_steps"] == requested,
        "requested_steps",
    )
    budget = result.get("episode_status")
    require(isinstance(budget, dict), "episode_status")
    for key in ("step", "step_limit", "remaining_steps"):
        require(type(budget.get(key)) is int and budget[key] >= 0, key)
    require(
        used <= budget["step"] <= budget["step_limit"]
        and budget["remaining_steps"] == budget["step_limit"] - budget["step"],
        "inconsistent episode counters",
    )
    require(not (result["terminated"] and result["truncated"]), "terminal flags")
    reason = result.get("stop_reason")
    require(reason in MOTION_STOP_REASONS, "stop_reason")
    require(
        not result["interrupted"] or (not result["ok"] and reason == "cancelled"),
        "interrupted motion reported success",
    )
    require((reason == "cancelled") == result["interrupted"], "cancellation flags")
    require((reason == "native_success") == result["terminated"], "success termination")
    require(
        (reason in {"budget_exhausted", "environment_terminal"}) == result["truncated"],
        "truncation flags",
    )
    if reason == "budget_exhausted":
        require(budget["remaining_steps"] == 0, "exhausted budget")
    if method == "env.execute_pose":
        require(type(result.get("ik_failed")) is bool, "ik_failed")
        for key in ("final_dist_m", "final_orientation_error_rad"):
            require(number(result.get(key)) and result[key] >= 0, key)
        position = result.get("final_eef_pos")
        require(
            isinstance(position, (list, tuple))
            and len(position) == 3
            and all(number(v) for v in position),
            "final_eef_pos",
        )
        require(
            not result["ok"]
            or (
                not result["ik_failed"]
                and result["final_dist_m"] <= arguments["tol"]
                and result["final_orientation_error_rad"] <= arguments["ori_tol"]
            ),
            "pose success disagrees with measured error",
        )
    else:
        require(
            type(result.get("commands_completed")) is bool
            and result["commands_completed"] == (used == requested)
            and result["ok"] == result["commands_completed"],
            "commands_completed",
        )

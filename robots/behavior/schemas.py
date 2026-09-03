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

"""Validated BEHAVIOR/R1Pro observation, action, and public tool contracts."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from robots.behavior.task_specs import BehaviorTaskSpec, get_task_spec

ACTION_DIM = 23
DEFAULT_ACTION_CHUNK = 32
CAMERA_KEYS = ("main", "left_wrist", "right_wrist")
PHYSICAL_CAMERAS = ("head", "left_wrist", "right_wrist")
HEAD_VIEW_PRESETS = (
    "center",
    "up",
    "down",
    "left",
    "right",
    "down_left",
    "down_right",
)
FRAME_REVIEW_ASSESSMENTS = (
    "target_bearing_surface_confirmed",
    "opposite_surface_confirmed",
    "side_or_indeterminate",
)

PUBLIC_TOOL_CONTRACTS: dict[int, tuple[str, ...]] = {
    1: (
        "pi0_nav_pick",
        "observe",
        "pixel_to_world",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
        "save_robot_state_checkpoint",
    ),
    2: (
        "pi0_nav_pick",
        "observe",
        "pixel_to_world",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
        "save_robot_state_checkpoint",
        "navigate_to",
    ),
    3: (
        "pi0_nav_pick",
        "observe",
        "pixel_to_world",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
        "save_robot_state_checkpoint",
        "navigate_to",
        "move_both_to",
    ),
    4: (
        "pi0_nav_pick",
        "observe",
        "pixel_to_world",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
        "save_robot_state_checkpoint",
        "navigate_to",
        "move_both_to",
        "get_prepared_motion_status",
    ),
    5: (
        "pi0_nav_pick",
        "observe",
        "pixel_to_world",
        "navigate_to",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
    ),
}
CURRENT_PUBLIC_TOOL_CONTRACT_VERSION = 5
BEHAVIOR_TOOL_NAMES = PUBLIC_TOOL_CONTRACTS[CURRENT_PUBLIC_TOOL_CONTRACT_VERSION]
PUBLIC_PRIMITIVE_ENTRYPOINTS = {
    "pi0_nav_pick": "BehaviorPrimitives.pi0_nav_pick",
    "observe": "BehaviorPrimitives.observe",
    "pixel_to_world": "BehaviorPrimitives.pixel_to_world",
    "navigate_to": "BehaviorPrimitives.navigate_to",
    "move_to": "BehaviorPrimitives.move_to",
    "rotate_wrist": "BehaviorPrimitives.rotate_wrist",
    "close": "BehaviorPrimitives.close",
    "open": "BehaviorPrimitives.open",
    "press": "BehaviorPrimitives.press",
}
if tuple(PUBLIC_TOOL_CONTRACTS) != (1, 2, 3, 4, 5):
    raise ValueError("BEHAVIOR public tool contract versions must be contiguous")
if tuple(PUBLIC_PRIMITIVE_ENTRYPOINTS) != BEHAVIOR_TOOL_NAMES:
    raise ValueError("BEHAVIOR primitive entrypoints must match the public contract")
if len(BEHAVIOR_TOOL_NAMES) != 9 or len(set(BEHAVIOR_TOOL_NAMES)) != 9:
    raise ValueError("BEHAVIOR toolkit must expose 9 unique public primitives")

POLICY_STATE_SEGMENTS: dict[str, slice] = {
    "base": slice(0, 3),
    "trunk": slice(3, 7),
    "left_arm": slice(7, 14),
    "right_arm": slice(14, 21),
    "left_gripper": slice(21, 22),
    "right_gripper": slice(22, 23),
}
ENV_ACTION_SEGMENTS: dict[str, slice] = {
    "base": slice(0, 3),
    "trunk": slice(3, 7),
    "left_arm": slice(7, 14),
    "left_gripper": slice(14, 15),
    "right_arm": slice(15, 22),
    "right_gripper": slice(22, 23),
}
RAW_PROPRIO_SEGMENTS: dict[str, slice] = {
    "left_arm": slice(158, 165),
    "left_gripper": slice(193, 195),
    "right_arm": slice(197, 204),
    "right_gripper": slice(232, 234),
    "trunk": slice(236, 240),
    "base": slice(253, 256),
}


def _validate_segments(name: str, segments: Mapping[str, slice]) -> None:
    covered: list[int] = []
    for segment, indices in segments.items():
        if (
            indices.start is None
            or indices.stop is None
            or indices.step not in (None, 1)
        ):
            raise ValueError(f"{name}.{segment} must be a contiguous slice")
        covered.extend(range(indices.start, indices.stop))
    if covered != list(range(ACTION_DIM)):
        raise ValueError(
            f"{name} must cover 0..{ACTION_DIM - 1} exactly, got {covered}"
        )


_validate_segments("POLICY_STATE_SEGMENTS", POLICY_STATE_SEGMENTS)
_validate_segments("ENV_ACTION_SEGMENTS", ENV_ACTION_SEGMENTS)
if POLICY_STATE_SEGMENTS == ENV_ACTION_SEGMENTS:
    raise ValueError("policy state and env action layouts must remain distinct")


def segment_ranges(segments: Mapping[str, slice]) -> dict[str, list[int]]:
    return {name: [part.start, part.stop] for name, part in segments.items()}


def validate_policy_state(state: Any) -> np.ndarray:
    array = np.asarray(state, dtype=np.float32)
    if array.shape != (ACTION_DIM,):
        raise ValueError(
            f"compact policy state must be [{ACTION_DIM}], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("compact policy state contains NaN or infinity")
    return array


def extract_policy_state(raw_proprio: Any) -> np.ndarray:
    raw = np.asarray(raw_proprio, dtype=np.float32)
    if raw.ndim != 1 or raw.shape[0] < RAW_PROPRIO_SEGMENTS["base"].stop:
        raise ValueError(
            "raw R1Pro proprio must be a vector with at least "
            f"{RAW_PROPRIO_SEGMENTS['base'].stop} values, got {raw.shape}"
        )
    compact = np.concatenate(
        [
            raw[RAW_PROPRIO_SEGMENTS["base"]],
            raw[RAW_PROPRIO_SEGMENTS["trunk"]],
            raw[RAW_PROPRIO_SEGMENTS["left_arm"]],
            raw[RAW_PROPRIO_SEGMENTS["right_arm"]],
            np.asarray([raw[RAW_PROPRIO_SEGMENTS["left_gripper"]].sum()]),
            np.asarray([raw[RAW_PROPRIO_SEGMENTS["right_gripper"]].sum()]),
        ]
    )
    return validate_policy_state(compact)


def validate_action_chunk(
    actions: Any, *, max_horizon: int | None = None
) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM or array.shape[0] < 1:
        raise ValueError(
            f"BEHAVIOR actions must be [T,{ACTION_DIM}], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("BEHAVIOR actions contain NaN or infinity")
    if max_horizon is not None and array.shape[0] > int(max_horizon):
        raise ValueError(
            f"BEHAVIOR action horizon {array.shape[0]} exceeds {int(max_horizon)}"
        )
    return array


ENV_WIRE_SCHEMA: dict[str, Any] = {
    "name": "behavior_env_rpc",
    "version": 1,
    "observation": {
        "main_images": "uint8[H,W,3]",
        "wrist_images": "uint8[2,H,W,3]",
        "states": "float[raw_proprio_dim]",
        "task_descriptions": "str",
    },
    "action": {
        "shape": f"float[T,{ACTION_DIM}]",
        "segments": segment_ranges(ENV_ACTION_SEGMENTS),
    },
    "official_success_path": ["info", "done", "success"],
}

VLA_WIRE_SCHEMA: dict[str, Any] = {
    "name": "pi05_vla_rpc_behavior",
    "version": 2,
    "request": {
        "method": "vla.predict",
        "main_images": "uint8[1,H,W,3]",
        "wrist_images": "uint8[1,2,H,W,3]",
        "states": "float[1,raw_proprio_dim]",
        "task_descriptions": "list[str]",
        "extra_view_images": "None",
        "compact_state_segments": segment_ranges(POLICY_STATE_SEGMENTS),
        "mode": "eval",
    },
    "response": {
        "actions": f"float[1,T,{ACTION_DIM}]",
        "env_action_segments": segment_ranges(ENV_ACTION_SEGMENTS),
    },
}


def _planner_spec(
    name: str,
    description: str,
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
    one_of: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
    if one_of is not None:
        schema["oneOf"] = one_of
    return {"name": name, "description": description, "input_schema": schema}


_CAMERA_ROLE_SCHEMA = {
    "type": "string",
    "enum": ["head", "left_wrist", "right_wrist"],
}
_HAND_SCHEMA = {"type": "string", "enum": ["left", "right"]}
_VISUAL_HAND_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "camera": {"type": "string", "enum": ["head", "left_wrist", "right_wrist"]},
        "frame_id": {"type": "string", "minLength": 1},
        "selected_hand": {"type": "string", "enum": ["left", "right"]},
        "assessment": {"type": "string", "const": "selected_hand_visually_confirmed"},
    },
    "required": ["camera", "frame_id", "selected_hand", "assessment"],
    "additionalProperties": False,
}
_RELEASE_VISUAL_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "camera": {"type": "string", "enum": ["head", "left_wrist", "right_wrist"]},
        "frame_id": {"type": "string", "minLength": 1},
        "selected_hand": {"type": "string", "enum": ["left", "right"]},
        "assessment": {
            "type": "string",
            "const": "attached_object_fully_inside_receptacle_opening",
        },
    },
    "required": ["camera", "frame_id", "selected_hand", "assessment"],
    "additionalProperties": False,
}
_DELTA_XYZ_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
}

PI0_NAV_PICK_SPEC = _planner_spec(
    "pi0_nav_pick",
    (
        "Invoke a Pi0.5 planner tool supporting navigation, grasping, and "
        "pressing. chunks is a positive requested work bound with no fixed "
        "maximum; execution is bounded only by the remaining episode steps, raw "
        "official success, termination, truncation, or infrastructure failure. "
        "task_success reports only raw info.done.success."
    ),
    {
        "instruction": {
            "type": "string",
            "minLength": 1,
            "description": "Exact VLA task language for this invocation.",
        },
        "chunks": {
            "type": "integer",
            "minimum": 1,
            "description": "Positive [32,23] action chunk request count.",
        },
    },
    required=["instruction", "chunks"],
)

OBSERVE_SPEC = _planner_spec(
    "observe",
    "Capture or review a public BEHAVIOR RGB-D observation without advancing physics.",
    {
        "camera": _CAMERA_ROLE_SCHEMA,
        "head_view": {"type": "string", "enum": list(HEAD_VIEW_PRESETS)},
        "paired_hand": _HAND_SCHEMA,
        "frame_review": {
            "type": "object",
            "properties": {
                "frame_id": {"type": "string", "minLength": 1},
                "assessment": {
                    "type": "string",
                    "enum": list(FRAME_REVIEW_ASSESSMENTS),
                },
            },
            "required": ["frame_id", "assessment"],
            "additionalProperties": False,
        },
        "depth_probe": {
            "type": "object",
            "properties": {
                "frame_id": {"type": "string", "minLength": 1},
                "u": {"type": "integer"},
                "v": {"type": "integer"},
                "depth_window_px": {"type": "integer", "minimum": 1, "maximum": 31},
                "assessment": {
                    "type": "string",
                    "const": "target_point_visually_confirmed",
                },
            },
            "required": ["frame_id", "u", "v", "depth_window_px", "assessment"],
            "additionalProperties": False,
        },
    },
    required=["camera"],
)
OBSERVE_SPEC["input_schema"]["allOf"] = [
    {"not": {"required": ["frame_review", "depth_probe"]}},
    {
        "if": {"required": ["head_view"]},
        "then": {
            "properties": {"camera": {"const": "head"}},
            "not": {
                "anyOf": [
                    {"required": ["frame_review"]},
                    {"required": ["depth_probe"]},
                ]
            },
        },
    },
    {
        "if": {"required": ["paired_hand"]},
        "then": {
            "properties": {"camera": {"const": "head"}},
            "not": {
                "anyOf": [
                    {"required": ["frame_review"]},
                    {"required": ["depth_probe"]},
                ]
            },
        },
    },
]

PIXEL_TO_WORLD_SPEC = _planner_spec(
    "pixel_to_world",
    "Back-project one pixel from a fresh public RGB-D frame.",
    {
        "camera": _CAMERA_ROLE_SCHEMA,
        "frame_id": {"type": "string", "minLength": 1},
        "u": {"type": "integer"},
        "v": {"type": "integer"},
        "depth_window_px": {
            "type": "integer",
            "default": 7,
            "minimum": 1,
            "maximum": 31,
        },
        "target_fact": {"type": "string", "const": "soda_can_floor_outside_receptacle"},
    },
    required=["camera", "frame_id", "u", "v"],
)

_MOVE_TARGET_SCHEMA = {
    "type": "object",
    "properties": {
        "projection_id": {"type": "string", "minLength": 1},
        "standoff_m": {"type": "number", "minimum": 0.0},
        "delta_xyz": _DELTA_XYZ_SCHEMA,
        "frame": {"type": "string", "enum": ["world", "eef"]},
    },
    "oneOf": [
        {
            "required": ["projection_id"],
            "not": {
                "anyOf": [
                    {"required": ["delta_xyz"]},
                    {"required": ["frame"]},
                ]
            },
        },
        {
            "required": ["delta_xyz", "frame"],
            "not": {
                "anyOf": [
                    {"required": ["projection_id"]},
                    {"required": ["standoff_m"]},
                ]
            },
        },
    ],
    "additionalProperties": False,
}

_MOVE_TO_HAND_SCHEMA = {"type": "string", "enum": ["left", "right", "both"]}
_MOVE_TO_BOTH_TARGETS_SCHEMA = {
    "type": "object",
    "properties": {
        "left": {
            "type": "object",
            "properties": {
                "delta_xyz": _DELTA_XYZ_SCHEMA,
                "frame": {"type": "string", "enum": ["world", "eef"]},
            },
            "required": ["delta_xyz", "frame"],
            "additionalProperties": False,
        },
        "right": {
            "type": "object",
            "properties": {
                "delta_xyz": _DELTA_XYZ_SCHEMA,
                "frame": {"type": "string", "enum": ["world", "eef"]},
            },
            "required": ["delta_xyz", "frame"],
            "additionalProperties": False,
        },
    },
    "required": ["left", "right"],
    "additionalProperties": False,
}
_MOVE_TO_BOTH_VISUAL_HAND_CHECKS_SCHEMA = {
    "type": "object",
    "properties": {
        "left": _VISUAL_HAND_CHECK_SCHEMA,
        "right": _VISUAL_HAND_CHECK_SCHEMA,
    },
    "required": ["left", "right"],
    "additionalProperties": False,
}

MOVE_TO_SPEC = _planner_spec(
    "move_to",
    "Move the selected BEHAVIOR hand, or coordinate both hands when hand is both.",
    {
        "hand": _MOVE_TO_HAND_SCHEMA,
        "visual_hand_check": _VISUAL_HAND_CHECK_SCHEMA,
        "target": _MOVE_TARGET_SCHEMA,
        "targets": _MOVE_TO_BOTH_TARGETS_SCHEMA,
        "visual_hand_checks": _MOVE_TO_BOTH_VISUAL_HAND_CHECKS_SCHEMA,
        "support_motion_phase": {
            "type": "string",
            "enum": ["carry_can", "transit_next_can"],
        },
    },
    one_of=[
        {
            "properties": {"hand": {"enum": ["left", "right"]}},
            "required": ["hand", "target"],
            "not": {
                "anyOf": [
                    {"required": ["targets"]},
                    {"required": ["visual_hand_checks"]},
                ]
            },
        },
        {
            "properties": {"hand": {"const": "both"}},
            "required": ["hand", "targets", "visual_hand_checks"],
            "not": {
                "anyOf": [
                    {"required": ["target"]},
                    {"required": ["visual_hand_check"]},
                    {"required": ["support_motion_phase"]},
                ]
            },
        },
    ],
)

ROTATE_WRIST_SPEC = _planner_spec(
    "rotate_wrist",
    "Rotate one selected BEHAVIOR wrist.",
    {
        "hand": _HAND_SCHEMA,
        "visual_hand_check": _VISUAL_HAND_CHECK_SCHEMA,
        "angle_deg": {"type": "number"},
        "direction": {"type": "string", "enum": ["clockwise", "counterclockwise"]},
    },
    required=["hand", "visual_hand_check", "angle_deg"],
)

CLOSE_SPEC = _planner_spec(
    "close",
    "Close one selected BEHAVIOR gripper.",
    {
        "hand": _HAND_SCHEMA,
        "visual_hand_check": _VISUAL_HAND_CHECK_SCHEMA,
    },
    required=["hand", "visual_hand_check"],
)

OPEN_SPEC = _planner_spec(
    "open",
    "Open one selected BEHAVIOR gripper.",
    {
        "hand": _HAND_SCHEMA,
        "visual_hand_check": _VISUAL_HAND_CHECK_SCHEMA,
        "release_visual_check": _RELEASE_VISUAL_CHECK_SCHEMA,
    },
    required=["hand", "visual_hand_check"],
)

PRESS_SPEC = _planner_spec(
    "press",
    "Press with one selected BEHAVIOR hand.",
    {
        "hand": _HAND_SCHEMA,
        "visual_hand_check": _VISUAL_HAND_CHECK_SCHEMA,
        "duration_s": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 10.0},
    },
    required=["hand", "visual_hand_check"],
)

_NAVIGATION_VISUAL_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "camera": {"type": "string", "const": "head"},
        "frame_id": {"type": "string", "minLength": 1},
        "assessment": {
            "type": "string",
            "const": "navigation_target_visually_confirmed",
        },
    },
    "required": ["camera", "frame_id", "assessment"],
    "additionalProperties": False,
}
_RELATIVE_NAVIGATION_MOTION_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["translation", "rotation"]},
        "direction": {
            "type": "string",
            "enum": ["forward", "backward", "left", "right"],
        },
        "distance_m": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.5},
        "angle_deg": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 180.0},
    },
    "additionalProperties": False,
}
NAVIGATE_TO_SPEC = _planner_spec(
    "navigate_to",
    "Request a base navigation goal from a projection or explicit relative motion.",
    {
        "projection_id": {"type": "string", "minLength": 1},
        "navigation_visual_check": _NAVIGATION_VISUAL_CHECK_SCHEMA,
        "relative_motion": _RELATIVE_NAVIGATION_MOTION_SCHEMA,
        "standoff_m": {
            "type": "number",
            "default": 0.85,
            "minimum": 0.45,
            "maximum": 1.5,
        },
        "plan_only": {"type": "boolean"},
        "prepared_plan_id": {"type": "string", "minLength": 1},
    },
    one_of=[
        {
            "required": ["projection_id", "navigation_visual_check"],
            "not": {"required": ["relative_motion"]},
        },
        {
            "required": ["relative_motion"],
            "not": {
                "anyOf": [
                    {"required": ["projection_id"]},
                    {"required": ["navigation_visual_check"]},
                    {"required": ["standoff_m"]},
                ]
            },
        },
    ],
)


def _non_bool_number(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return value


def validate_observe_request(
    *,
    camera: Any,
    head_view: Any = None,
    paired_hand: Any = None,
    frame_review: Any = None,
    depth_probe: Any = None,
) -> dict[str, Any]:
    if not isinstance(camera, str) or camera not in PHYSICAL_CAMERAS:
        raise ValueError("camera must be head, left_wrist, or right_wrist")
    if frame_review is not None and depth_probe is not None:
        raise ValueError("frame_review and depth_probe are mutually exclusive")
    if head_view is not None:
        if not isinstance(head_view, str) or head_view not in HEAD_VIEW_PRESETS:
            raise ValueError("head_view must be a supported preset")
        if camera != "head":
            raise ValueError("head_view is available only for camera='head'")
        if frame_review is not None or depth_probe is not None:
            raise ValueError("head_view cannot be combined with review/probe")
    if paired_hand is not None:
        if paired_hand not in {"left", "right"}:
            raise ValueError("paired_hand must be 'left' or 'right'")
        if camera != "head":
            raise ValueError("paired_hand is available only for camera='head'")
        if frame_review is not None or depth_probe is not None:
            raise ValueError("paired_hand cannot be combined with review/probe")
    request: dict[str, Any] = {"camera": camera}
    if head_view is not None:
        request["head_view"] = head_view
    if paired_hand is not None:
        request["paired_hand"] = paired_hand
    if frame_review is not None:
        request["frame_review"] = frame_review
    if depth_probe is not None:
        request["depth_probe"] = depth_probe
    return request


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def validate_relative_navigation_motion(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("relative_motion must be an object")
    motion = dict(value)
    kind = motion.get("kind")
    direction = motion.get("direction")
    if kind == "translation":
        expected = {"kind", "direction", "distance_m"}
        if direction not in {"forward", "backward"}:
            raise ValueError("translation direction must be forward or backward")
        amount_name = "distance_m"
        maximum = 1.5
    elif kind == "rotation":
        expected = {"kind", "direction", "angle_deg"}
        if direction not in {"left", "right"}:
            raise ValueError("rotation direction must be left or right")
        amount_name = "angle_deg"
        maximum = 180.0
    else:
        raise ValueError("relative_motion.kind must be translation or rotation")
    if set(motion) != expected:
        raise ValueError(f"relative_motion.{kind} requires exactly {sorted(expected)}")
    amount = _non_bool_number(
        motion[amount_name], field=f"relative_motion.{amount_name}"
    )
    if amount <= 0.0 or amount > maximum:
        raise ValueError(f"relative_motion.{amount_name} must be within (0,{maximum}]")
    return {"kind": str(kind), "direction": str(direction), amount_name: amount}


def validate_visibility_recovery_check(value: Any, *, hand: Any) -> dict[str, str]:
    if hand not in {"left", "right"}:
        raise ValueError("hand must be 'left' or 'right'")
    required = {"view_pair_id", "selected_hand", "assessment"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(
            "visibility_recovery_check requires exactly view_pair_id, selected_hand, "
            "and assessment"
        )
    if value["selected_hand"] != hand:
        raise ValueError("hand must equal visibility_recovery_check.selected_hand")
    if value["assessment"] != "global_target_and_matching_wrist_observed":
        raise ValueError(
            "visibility_recovery_check.assessment must be "
            "global_target_and_matching_wrist_observed"
        )
    return {
        "view_pair_id": _identifier(value["view_pair_id"], name="view_pair_id"),
        "selected_hand": str(hand),
        "assessment": "global_target_and_matching_wrist_observed",
    }


def validate_move_both_targets(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"left", "right"}:
        raise ValueError("targets requires exactly left and right")
    normalized: dict[str, dict[str, Any]] = {}
    for hand in ("left", "right"):
        target = value[hand]
        if not isinstance(target, Mapping) or set(target) != {"delta_xyz", "frame"}:
            raise ValueError(f"targets.{hand} requires exactly delta_xyz and frame")
        frame = target["frame"]
        if frame not in {"world", "eef"}:
            raise ValueError(f"targets.{hand}.frame must be world or eef")
        delta = target["delta_xyz"]
        delta_items = delta.tolist() if isinstance(delta, np.ndarray) else delta
        if not isinstance(delta_items, (list, tuple)) or len(delta_items) != 3:
            raise ValueError(f"targets.{hand}.delta_xyz must contain three numbers")
        values = np.asarray(delta_items, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"targets.{hand}.delta_xyz must contain finite numbers")
        normalized[hand] = {"delta_xyz": values.tolist(), "frame": str(frame)}
    return normalized


def validate_move_both_visual_hand_checks(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != {"left", "right"}:
        raise ValueError("visual_hand_checks requires exactly left and right")
    normalized: dict[str, dict[str, str]] = {}
    required = {"camera", "frame_id", "selected_hand", "assessment"}
    for hand in ("left", "right"):
        check = value[hand]
        if not isinstance(check, Mapping) or set(check) != required:
            raise ValueError(f"visual_hand_checks.{hand} has invalid keys")
        if check["camera"] not in {"head", f"{hand}_wrist"}:
            raise ValueError(f"visual_hand_checks.{hand}.camera is invalid")
        if check["selected_hand"] != hand:
            raise ValueError(f"visual_hand_checks.{hand}.selected_hand must be {hand}")
        if check["assessment"] != "selected_hand_visually_confirmed":
            raise ValueError(f"visual_hand_checks.{hand}.assessment is invalid")
        normalized[hand] = {
            "camera": str(check["camera"]),
            "frame_id": _identifier(check["frame_id"], name="frame_id"),
            "selected_hand": hand,
            "assessment": "selected_hand_visually_confirmed",
        }
    return normalized


def behavior_tool_specs_for_task(
    task: str | BehaviorTaskSpec,
) -> tuple[dict[str, Any], ...]:
    task_spec = get_task_spec(task) if isinstance(task, str) else task
    specs = {
        "pi0_nav_pick": copy.deepcopy(PI0_NAV_PICK_SPEC),
        "observe": copy.deepcopy(OBSERVE_SPEC),
        "pixel_to_world": copy.deepcopy(PIXEL_TO_WORLD_SPEC),
        "navigate_to": copy.deepcopy(NAVIGATE_TO_SPEC),
        "move_to": copy.deepcopy(MOVE_TO_SPEC),
        "rotate_wrist": copy.deepcopy(ROTATE_WRIST_SPEC),
        "close": copy.deepcopy(CLOSE_SPEC),
        "open": copy.deepcopy(OPEN_SPEC),
        "press": copy.deepcopy(PRESS_SPEC),
    }
    if task_spec.release_visual_policy is None:
        specs["open"]["input_schema"]["properties"].pop("release_visual_check", None)
    return tuple(specs[name] for name in BEHAVIOR_TOOL_NAMES)


__all__ = [
    "ACTION_DIM",
    "BEHAVIOR_TOOL_NAMES",
    "CAMERA_KEYS",
    "CLOSE_SPEC",
    "CURRENT_PUBLIC_TOOL_CONTRACT_VERSION",
    "DEFAULT_ACTION_CHUNK",
    "ENV_ACTION_SEGMENTS",
    "ENV_WIRE_SCHEMA",
    "FRAME_REVIEW_ASSESSMENTS",
    "HEAD_VIEW_PRESETS",
    "MOVE_TO_SPEC",
    "NAVIGATE_TO_SPEC",
    "OBSERVE_SPEC",
    "OPEN_SPEC",
    "PI0_NAV_PICK_SPEC",
    "PHYSICAL_CAMERAS",
    "PIXEL_TO_WORLD_SPEC",
    "POLICY_STATE_SEGMENTS",
    "PRESS_SPEC",
    "PUBLIC_PRIMITIVE_ENTRYPOINTS",
    "PUBLIC_TOOL_CONTRACTS",
    "ROTATE_WRIST_SPEC",
    "VLA_WIRE_SCHEMA",
    "behavior_tool_specs_for_task",
    "extract_policy_state",
    "segment_ranges",
    "validate_action_chunk",
    "validate_move_both_targets",
    "validate_move_both_visual_hand_checks",
    "validate_observe_request",
    "validate_policy_state",
    "validate_relative_navigation_motion",
    "validate_visibility_recovery_check",
]

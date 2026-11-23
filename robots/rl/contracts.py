# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Publish and validate the same complete RL input contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from rpent.tools import Tool

EPISODE_CONTRACT_VERSION = 2
CAMERAS = {
    "libero": ("agentview", "wrist"),
    "robocasa": ("agentview", "agentview_right", "wrist"),
    "robotwin": ("head", "left_wrist", "right_wrist"),
    "robodojo": ("cam_head", "cam_left_wrist", "cam_right_wrist"),
}


def _present(name, kind):
    return {"required": [name], "properties": {name: {"type": kind}}}


def complete_schema_alternatives(schema):
    """Keep object alternatives self-contained for required-tool grammar backends.

    Some decoders shallow-merge a branch's properties over the parent map. A
    type-only override then loses array items and length constraints. Expand
    each alternative without changing which argument objects are valid.
    """

    def normalize_nullable(node):
        if isinstance(node, list):
            for child in node:
                normalize_nullable(child)
        elif isinstance(node, dict):
            for child in list(node.values()):
                normalize_nullable(child)
            choices = node.get("anyOf")
            if isinstance(choices, list) and len(choices) == 2 and "type" not in node:
                values = [x for x in choices if x.get("type") != "null"]
                nulls = [x for x in choices if x.get("type") == "null"]
                if len(values) == len(nulls) == 1 and isinstance(
                    values[0].get("type"), str
                ):
                    fields = {k: v for k, v in node.items() if k != "anyOf"}
                    node.clear()
                    node.update(values[0])
                    node.update(fields)
                    node["type"] = [values[0]["type"], "null"]

    def expand(node):
        if isinstance(node, list):
            for child in node:
                expand(child)
        elif isinstance(node, dict):
            for child in list(node.values()):
                expand(child)
            properties = node.get("properties")
            if not isinstance(properties, dict):
                return
            for key in ("anyOf", "oneOf"):
                for branch in node.get(key, []):
                    if not isinstance(branch, dict) or "$ref" in branch:
                        continue
                    overrides = branch.get("properties", {})
                    combined = deepcopy(properties)
                    for name, value in overrides.items():
                        combined[name] = {**combined.get(name, {}), **value}
                    branch["properties"] = combined
                    branch["type"] = node.get("type", "object")
                    branch["required"] = list(
                        dict.fromkeys(
                            [*node.get("required", []), *branch.get("required", [])]
                        )
                    )
                    if "additionalProperties" in node:
                        branch["additionalProperties"] = node["additionalProperties"]

    normalize_nullable(schema)
    expand(schema)
    return schema


def strict_tool(declaration: Tool, *, cameras=()):
    """Keep the public call shape while expressing mutually exclusive queries."""
    schema = deepcopy(declaration.input_schema)

    def forbid_extras(value):
        if isinstance(value, dict):
            if value.get("type") == "object" and "properties" in value:
                value["additionalProperties"] = False
            for child in value.values():
                forbid_extras(child)
        elif isinstance(value, list):
            for child in value:
                forbid_extras(child)

    forbid_extras(schema)
    schema["additionalProperties"] = False
    properties = schema.get("properties", {})

    def nonblank_strings(value):
        if isinstance(value, dict):
            if value.get("type") == "string" and value.get("minLength", 0) >= 1:
                value["pattern"] = r"^[\s\S]*\S[\s\S]*$"
            for child in value.values():
                nonblank_strings(child)
        elif isinstance(value, list):
            for child in value:
                nonblank_strings(child)

    nonblank_strings(schema)
    for name in ("prompt", "mask_id"):
        if name in properties:
            properties[name]["minLength"] = 1
            properties[name]["pattern"] = r"^[\s\S]*\S[\s\S]*$"
    for name, count in (("point", 2), ("pixel", 2), ("box", 4)):
        if name in properties:
            properties[name].update(
                minItems=count,
                maxItems=count,
                items={
                    "type": "integer",
                    "minimum": 0,
                },
            )
    if (
        cameras
        and "camera" in properties
        and declaration.name not in {"move_to", "set_gripper"}
    ):
        properties["camera"]["enum"] = [
            *cameras,
            *(["all"] if declaration.name == "view_env" else []),
            *([None] if declaration.name in {"segment", "back_project"} else []),
        ]
    if declaration.name == "segment":
        text, point = _present("prompt", "string"), _present("point", "array")
        text["properties"]["point"] = {"type": "null"}
        point["properties"]["prompt"] = {"type": "null"}
        point["required"].append("step")
        point["properties"]["step"] = {"type": "integer"}
        schema["oneOf"] = [text, point]
    elif declaration.name == "back_project":
        branches = []
        for name, kind in (("pixel", "array"), ("box", "array"), ("mask_id", "string")):
            branch = _present(name, kind)
            for other in ("pixel", "box", "mask_id"):
                if other != name:
                    branch["properties"][other] = {"type": "null"}
            if name == "mask_id":
                branch["properties"]["camera"] = {"type": "null"}
                branch["properties"]["step"] = {"type": "null"}
            else:
                branch["required"].append("step")
                branch["properties"]["step"] = {"type": "integer"}
            if name == "pixel":
                branch["properties"]["z_range_m"] = {"type": "null"}
            branches.append(branch)
        schema["oneOf"] = branches
    return replace(declaration, _input_schema=complete_schema_alternatives(schema))


def validate_relations(name: str, arguments: dict) -> None:
    """Check value-dependent constraints that JSON Schema cannot express."""
    box = arguments.get("box")
    if box is not None and (box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError("box must have row_min < row_max and col_min < col_max")
    z = arguments.get("z_range_m", arguments.get("z_range_cm"))
    if z is not None and z[0] > z[1]:
        raise ValueError("z_range_m must be [low,high] with low <= high")


def validate_options(
    environment,
    workflow,
    action_mode,
    max_actions,
    reasoning,
    images,
    *,
    world_cord="base",
):
    """Reject unsupported non-default options before allocating a simulator."""
    if action_mode != "batch" or not 1 <= max_actions <= 3:
        raise ValueError("episode requires batch actions with a limit of 1..3")
    if reasoning < -1 or images < -1:
        raise ValueError("history limits must be -1 or a nonnegative integer")
    if workflow != "episode":
        raise ValueError("only episode is supported")


def validate_image_size(environment: str, world_cord: str, image_size: int) -> None:
    """Reject unsupported VLM image sizes before starting a simulator."""
    from robots.rl.images import IMAGE_SIZES

    if image_size not in IMAGE_SIZES:
        raise ValueError(f"unsupported image size {image_size}")
    if image_size != 1024 and not (
        environment == "libero"
        or (environment == "robocasa" and world_cord == "general")
    ):
        raise ValueError(
            "--image_size currently supports LIBERO and RoboCasa general only"
        )


def validate_perception_options(workflow, mode="single", max_calls=8, concurrency=8):
    """Keep graph perception opt-in and confined to episode workflows."""
    if workflow != "episode" or mode != "single":
        raise ValueError("episode exposes only view_env, segment and back_project")
    if max_calls != 8 or concurrency != 8:
        raise ValueError("batch perception options are no longer supported")


def argument_hint(name: str, environment: str = "libero") -> str:
    """Short actionable model feedback, independently of validator formatting."""
    if environment == "robotwin" and name in {"move_to", "set_gripper"}:
        return "Follow the declared native action fields and orientation convention; include the selected arm where required."
    if environment == "robocasa" and name in {"move_to", "set_gripper"}:
        return (
            "Follow the declared move_to/set_gripper fields and orientation convention."
        )
    return {
        "segment": "Supply exactly one non-empty prompt or point [row,col]; follow the tool's camera/step requirements.",
        "back_project": "Supply exactly one pixel, box or mask_id; pixel/box must match the displayed camera and frame.",
        "move_to": "Supply xyz, orientation_rpy, or both as three finite numbers.",
        "move_base": "Supply bounded local velocities vx/vy/yaw_rate and duration_s within the declared limits.",
        "move_lift": "Supply delta_m within +/-0.25 and duration_s within 0.1..5.",
        "observe": "Use calls with unique id/op; view_env has camera only; segment/back_project need an issued source or @view.camera/@segment. Fix the whole list before retrying.",
        "execute_actions": "Supply a non-empty actions list within K; each item must be move_to with a target or set_gripper with opening 0..1.",
    }.get(name, "Follow the declared required fields and value ranges.")


def validate_coordinates(environment: str, world_cord: str) -> None:
    """Reject unsupported frames before starting a simulator."""
    if world_cord not in {"base", "general"}:
        raise ValueError("world_cord must be base or general")
    if world_cord == "general" and environment not in {
        "libero",
        "robotwin",
        "robocasa",
        "robodojo",
    }:
        raise ValueError("general coordinates require a supported simulator")


def coordinate_definition(declaration, world_cord: str):
    """Describe the selected coordinates without changing the tool wire schema."""
    if world_cord == "base":
        return declaration
    import re
    from dataclasses import replace

    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, str):
            return re.sub(
                r"\bworld(?:-frame)?\b",
                "general frame",
                value.replace(
                    " In episode_image only the final two-camera observation is returned.",
                    "",
                )
                .replace("grip_site", "native grasp TCP")
                .replace("grip-site", "native grasp TCP"),
            )
        return value

    return replace(
        declaration,
        description=convert(declaration.description),
        _input_schema=convert(declaration.input_schema),
    )

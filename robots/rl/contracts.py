# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Publish and validate the same complete RL input contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from rpent.tools import Tool


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


def strict_tool(
    declaration: Tool, *, image_size: int, latest: bool = False, cameras=()
):
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
                    "maximum": image_size if name == "box" else image_size - 1,
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
            *([None] if declaration.name == "back_project" else []),
        ]
    if declaration.name == "segment":
        text, point = _present("prompt", "string"), _present("point", "array")
        text["properties"]["point"] = {"type": "null"}
        point["properties"]["prompt"] = {"type": "null"}
        if not latest:
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
                if not latest:
                    branch["properties"]["step"] = {"type": "null"}
            elif not latest:
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
    z = arguments.get("z_range_m")
    if z is not None and z[0] > z[1]:
        raise ValueError("z_range_m must be [low,high] with low <= high")


def validate_options(
    environment, workflow, action_mode, max_actions, reasoning, images
):
    """Reject unsupported non-default options before allocating a simulator."""
    if action_mode not in {"single", "batch"} or max_actions < 1:
        raise ValueError(
            "action-mode must be single/batch and max-actions-per-call >= 1"
        )
    if reasoning < -1 or images < -1:
        raise ValueError("history limits must be -1 or a nonnegative integer")
    if (environment != "libero" or workflow not in {"episode", "episode_image"}) and (
        action_mode != "single" or reasoning != -1 or images != -1 or max_actions != 3
    ):
        raise ValueError(
            "action/history options currently support LIBERO episode and episode_image only"
        )


def validate_perception_options(workflow, mode="single", max_calls=8, concurrency=8):
    """Keep graph perception opt-in and confined to episode workflows."""
    if mode not in {"single", "batch"}:
        raise ValueError("perception-mode must be single or batch")
    if max_calls < 1 or concurrency < 1:
        raise ValueError("max-observe-calls and observe-concurrency must be positive")
    if workflow != "episode" and (
        mode != "single" or max_calls != 8 or concurrency != 8
    ):
        raise ValueError("observe options require --workflow episode")
    if mode == "single" and (max_calls != 8 or concurrency != 8):
        raise ValueError("observe limits require --perception-mode batch")


def argument_hint(name: str, environment: str = "libero") -> str:
    """Short actionable model feedback, independently of validator formatting."""
    if environment != "libero" and name in {"move_to", "set_gripper"}:
        return "Follow the declared native action fields and orientation convention; include the selected arm where required."
    return {
        "segment": "Supply exactly one non-empty prompt or point [row,col]; follow the tool's camera/step requirements.",
        "back_project": "Supply exactly one pixel, box or mask_id; pixel/box must match the displayed camera and frame.",
        "move_to": "Supply xyz, orientation_rpy, or both as three finite numbers.",
        "observe": "Use calls with unique id/op; view_env has camera only; segment/back_project need an issued source or @view.camera/@segment. Fix the whole list before retrying.",
        "execute_actions": "Supply a non-empty actions list within K; each item must be move_to with a target or set_gripper with opening 0..1.",
    }.get(name, "Follow the declared required fields and value ranges.")

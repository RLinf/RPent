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

"""Centimetre model interface, tool schema normalization, and input validation."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import replace
from typing import Any

from pydantic import ConfigDict, create_model

from rpent.tools import Tool

EPISODE_CONTRACT_VERSION = 3
CAMERAS: dict[str, tuple[str, ...]] = {
    "libero": ("agentview", "wrist"),
    "robocasa": ("agentview", "agentview_right", "wrist"),
    "robotwin": ("head", "left_wrist", "right_wrist"),
    "robodojo": ("cam_head", "cam_left_wrist", "cam_right_wrist"),
}

_LENGTH_FIELDS = frozenset({"xyz", "world_xyz", "tcp_xyz", "base_xyz", "vx", "vy"})
_LINEAR_BOUNDS = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
)


def _scale(value: Any, to_model: bool) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_scale(item, to_model) for item in value]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("A linear quantity must contain finite numbers")
    return value * 100.0 if to_model else value / 100.0


class Centimeters:
    """Translate model-facing centimetres and native metres."""

    factor = 100.0

    def field(self, name: str, *, to_model: bool = True) -> str:
        old, new = ("_m", "_cm") if to_model else ("_cm", "_m")
        return name[: -len(old)] + new if name.endswith(old) else name

    def payload(self, value: Any, *, to_model: bool = True) -> Any:
        """Convert structured feedback/arguments without touching image bytes."""
        if isinstance(value, list):
            return [self.payload(item, to_model=to_model) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            native_key = key if to_model else self.field(key, to_model=False)
            output_key = self.field(key, to_model=to_model)
            if native_key in _LENGTH_FIELDS or native_key.endswith("_m"):
                result[output_key] = _scale(item, to_model)
            elif key == "commanded_local_velocity":
                result[key] = [
                    _scale(item[0], to_model),
                    _scale(item[1], to_model),
                    item[2],
                ]
            elif key == "units" and isinstance(item, dict):
                result[key] = {
                    **item,
                    "position": "centimetres" if to_model else "metres",
                }
            else:
                result[output_key] = self.payload(item, to_model=to_model)
        return result

    def description(self, text: str) -> str:
        text = re.sub(r"\b([a-zA-Z_][a-zA-Z_0-9]*)_m\b", r"\1_cm", text)
        text = re.sub(r"\s*Example:.*", "", text, flags=re.S)
        text = re.sub(
            r"(?<![\w.])([+-]?(?:\d+\.?\d*|\.\d+))\s*(?:metres?|meters?|m)(?=/s|\b)",
            lambda match: f"{float(match[1]) * 100:g} cm",
            text,
        )
        text = re.sub(r"\b(metres?|meters?)\b", r"centi\1", text)
        return re.sub(r"\bm/s\b", "cm/s", text)

    def schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Rename fields and scale numeric bounds at their actual property nodes."""

        def walk(node: Any, linear: bool = False) -> Any:
            if isinstance(node, list):
                return [walk(item, linear) for item in node]
            if not isinstance(node, dict):
                return node
            result = {}
            for key, value in node.items():
                if key == "properties":
                    result[key] = {
                        self.field(name): walk(
                            child, name in _LENGTH_FIELDS or name.endswith("_m")
                        )
                        for name, child in value.items()
                    }
                elif key == "required":
                    result[key] = [self.field(name) for name in value]
                elif key == "description":
                    result[key] = self.description(value)
                elif linear and key in _LINEAR_BOUNDS:
                    result[key] = value * self.factor
                else:
                    result[key] = walk(value, linear)
            return result

        return walk(schema)


CENTIMETERS = Centimeters()


def model_tool(declaration: Tool) -> Tool:
    """Publish centimetres; convert arguments at the simulator boundary."""
    units = CENTIMETERS
    schema = units.schema(declaration.input_schema)

    def external_field(name: str):
        if name in schema.get("required", []):
            return Any, ...
        native_name = units.field(name, to_model=False)
        field = declaration.args_schema.model_fields[native_name]
        default = field.get_default(call_default_factory=True)
        return Any, units.payload({native_name: default})[name]

    external = create_model(
        declaration.args_schema.__name__ + "ModelInterface",
        __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
        **{name: external_field(name) for name in schema["properties"]},
    )

    def execute(**arguments):
        native = units.payload(arguments, to_model=False)
        parameters = declaration.args_schema.model_validate(native, strict=True)
        return declaration(
            **{
                name: getattr(parameters, name)
                for name in type(parameters).model_fields
            }
        )

    return replace(
        declaration,
        args_schema=external,
        _input_schema=schema,
        description=units.description(declaration.description),
        handler=execute,
        _unbound_method=False,
    )


def _present(name: str, kind: str) -> dict[str, Any]:
    return {"required": [name], "properties": {name: {"type": kind}}}


def complete_schema_alternatives(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize nullable anyOf and expand object alternative branches."""

    def normalize_nullable(node: Any) -> None:
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

    def expand(node: Any) -> None:
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


def strict_tool(declaration: Tool, *, cameras: tuple[str, ...] = ()) -> Tool:
    """Publish mutually exclusive query schemas with camera enums and strict objects."""
    schema = deepcopy(declaration.input_schema)

    def normalize(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" and "properties" in value:
                value["additionalProperties"] = False
            if value.get("type") == "string" and value.get("minLength", 0) >= 1:
                value["pattern"] = r"^[\s\S]*\S[\s\S]*$"
            for child in value.values():
                normalize(child)
        elif isinstance(value, list):
            for child in value:
                normalize(child)

    normalize(schema)
    schema["additionalProperties"] = False
    properties = schema.get("properties", {})
    for name in ("prompt", "mask_id"):
        if name in properties:
            properties[name]["minLength"] = 1
            properties[name]["pattern"] = r"^[\s\S]*\S[\s\S]*$"
    for name, count in (("point", 2), ("pixel", 2), ("box", 4)):
        if name in properties:
            properties[name].update(
                minItems=count,
                maxItems=count,
                items={"type": "integer", "minimum": 0},
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


def coordinate_definition(declaration: Tool) -> Tool:
    """Rewrite world/grip_site phrasing to the shared general frame."""

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, str):
            return re.sub(
                r"\bworld(?:-frame)?\b",
                "general frame",
                value.replace("grip_site", "native grasp TCP").replace(
                    "grip-site", "native grasp TCP"
                ),
            )
        return value

    return replace(
        declaration,
        description=convert(declaration.description),
        _input_schema=convert(declaration.input_schema),
    )


def validate_relations(name: str, arguments: dict[str, Any]) -> None:
    """Check value-dependent constraints that JSON Schema cannot express."""
    box = arguments.get("box")
    if box is not None and (box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError("box must have row_min < row_max and col_min < col_max")
    z = arguments.get("z_range_m", arguments.get("z_range_cm"))
    if z is not None and z[0] > z[1]:
        raise ValueError("z_range_m must be [low,high] with low <= high")


def argument_hint(name: str, environment: str = "libero") -> str:
    """Short actionable model feedback on invalid tool arguments."""
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
        "execute_actions": "Supply a non-empty actions list within K; each item must be move_to with a target or set_gripper with opening 0..1.",
    }.get(name, "Follow the declared required fields and value ranges.")

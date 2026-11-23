# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Convert the model interface once; physics, calibration and raw depth stay SI."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any

from pydantic import ConfigDict, create_model

from rpent.tools import Tool

# These public fields predate explicit unit suffixes. Pixel dimensions named
# size/height/width, angles, time and normalized jaw opening are not lengths.
_LENGTH_FIELDS = frozenset({"xyz", "world_xyz", "tcp_xyz", "base_xyz", "vx", "vy"})


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
    """Translate model-facing centimeters and native metres."""

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
                # Existing RoboCasa base feedback packs vx, vy and yaw_rate.
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
        # Examples contain literal coordinates; leaving them in metres would
        # silently teach a target 100x too small. Public schemas retain bounds.
        text = re.sub(r"\s*Example:.*", "", text, flags=re.S)
        text = re.sub(
            r"(?<![\w.])([+-]?(?:\d+\.?\d*|\.\d+))\s*(?:metres?|meters?|m)(?=/s|\b)",
            lambda match: f"{float(match[1]) * 100:g} cm",
            text,
        )
        text = re.sub(r"\b(metres?|meters?)\b", r"centi\1", text)
        return re.sub(r"\bm/s\b", "cm/s", text)

    def schema(self, schema: dict) -> dict:
        """Rename fields and scale numeric bounds at their actual property nodes."""

        def walk(node, linear=False):
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
                elif linear and key in {
                    "minimum",
                    "maximum",
                    "exclusiveMinimum",
                    "exclusiveMaximum",
                    "multipleOf",
                }:
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

    def external_field(name):
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
        # Public schema validation already ran before display pixels were
        # mapped to native resolution. Reapplying its pixel/centimetre bounds
        # to native values would reject valid high-resolution queries.
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

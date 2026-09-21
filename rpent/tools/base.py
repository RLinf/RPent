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

"""Tool declarations derived from Python functions and instance methods."""

from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Annotated, Any, Generic, ParamSpec, get_type_hints

from docstring_parser import DocstringStyle, parse
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import core_schema

ParamsT = ParamSpec("ParamsT")
MAX_TOOL_TEXT_BYTES = 60000


@dataclass
class ToolResult:
    """Tool data, PNG images, and an optional error."""

    data: dict[str, Any] = field(default_factory=dict)
    images: list[bytes] = field(default_factory=list)
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Combine result fields into the public JSON payload."""
        payload = dict(self.data)
        if self.error is not None:
            payload["error"] = self.error
        return payload

    def to_text(self) -> str:
        """Encode the original tool payload without truncating internal state."""
        # ASCII output makes character counts equal to UTF-8 byte counts.
        text = json.dumps(
            self.to_dict(), indent=2, allow_nan=False, ensure_ascii=True, default=str
        )
        if len(text) <= MAX_TOOL_TEXT_BYTES:
            return text
        suffix = "\n[truncated]"
        return text[: MAX_TOOL_TEXT_BYTES - len(suffix)] + suffix


class _ToolJsonSchema(GenerateJsonSchema):
    """Publish tool schemas without Python metadata or implicit defaults.

    Field(json_schema_extra={"default": ...}) explicitly advertises a default.
    Runtime argument validation remains independent of schema presentation.
    """

    def field_title_should_be_set(self, schema: Any) -> bool:
        return False

    def model_schema(self, schema: core_schema.ModelSchema) -> dict[str, Any]:
        result = super().model_schema(schema)
        result.pop("title", None)
        result.pop("additionalProperties", None)
        return result

    def default_schema(self, schema: core_schema.WithDefaultSchema) -> dict[str, Any]:
        return self.generate_inner(schema["schema"])

    def dict_schema(self, schema: core_schema.DictSchema) -> dict[str, Any]:
        result = super().dict_schema(schema)
        if result.get("additionalProperties") is True:
            result.pop("additionalProperties")
        return result

    def union_schema(self, schema: core_schema.UnionSchema) -> dict[str, Any]:
        result = super().union_schema(schema)
        choices = result.get("anyOf", [])
        if choices and all(set(choice) == {"type"} for choice in choices):
            return {"type": [choice["type"] for choice in choices]}
        return result

    def nullable_schema(self, schema: core_schema.NullableSchema) -> dict[str, Any]:
        inner = self.generate_inner(schema["schema"])
        if isinstance(inner.get("type"), str) and not any(
            key in inner for key in ("enum", "const", "allOf", "anyOf", "oneOf", "not")
        ):
            inner["type"] = [inner["type"], "null"]
            return inner
        return {"anyOf": [inner, {"type": "null"}]}


@dataclass(frozen=True)
class Tool(Generic[ParamsT]):
    """A callable tool and the parameter model used by Toolkit to validate it.

    Access through an instance binds the handler to that instance. Direct Python
    calls keep the handler's usual semantics; Toolkit validates model input before
    execution. Neither binding nor validation creates robot resources.
    """

    name: str
    description: str
    args_schema: type[BaseModel]
    handler: Callable[ParamsT, ToolResult]
    readonly: bool = False
    _unbound_method: bool = False

    @property
    def input_schema(self) -> dict[str, Any]:
        """Return JSON Schema for the same parameters that Toolkit validates."""
        return self.args_schema.model_json_schema(schema_generator=_ToolJsonSchema)

    def with_handler(self, handler: Callable[..., ToolResult]) -> Tool:
        """Bind resources or an execution guard while retaining the declaration."""
        return replace(self, handler=handler, _unbound_method=False)

    def __get__(self, instance: Any, owner: type | None = None) -> Tool:
        if instance is None:
            return self
        return replace(
            self,
            handler=self.handler.__get__(instance, owner),
            _unbound_method=False,
        )

    def __call__(self, *args: ParamsT.args, **kwargs: ParamsT.kwargs) -> ToolResult:
        return self.handler(*args, **kwargs)


def tool(
    function: Callable[ParamsT, ToolResult] | None = None,
    /,
    *,
    name: str | None = None,
    readonly: bool = False,
    exclude: tuple[str, ...] = (),
    required: tuple[str, ...] = (),
) -> Any:
    """Declare a tool from a function or an instance method with typed parameters.

    Google-style docstrings supply tool and parameter descriptions. Annotated Field
    constraints become part of the generated schema; advertise defaults explicitly
    with Field(json_schema_extra={"default": ...}). A leading
    self is bound by Python and excluded from model inputs. Use
    @tool(readonly=True) to skip Toolkit's automatic observation capture.

    name overrides the published tool name. exclude names internal resources
    supplied by the caller. required can require a model argument while retaining
    a Python default for internal calls.
    """
    namespace = dict(sys._getframe(1).f_locals)
    if function is None:
        return lambda handler: _declare_tool(
            handler, exclude, required, namespace, tool_name=name, readonly=readonly
        )
    return _declare_tool(
        function, exclude, required, namespace, tool_name=name, readonly=readonly
    )


def _declare_tool(
    function: Callable,
    exclude: tuple[str, ...],
    required: tuple[str, ...],
    namespace: dict,
    *,
    tool_name: str | None,
    readonly: bool,
) -> Tool:
    parameters = dict(inspect.signature(function).parameters)
    unbound_method = next(iter(parameters), None) == "self"
    if unbound_method:
        parameters.pop("self")
    for name in exclude:
        parameters.pop(name)
    for name, parameter in parameters.items():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(f"Tool parameter {name!r} must accept a keyword argument")
        if parameter.annotation is inspect.Parameter.empty:
            raise TypeError(f"Tool parameter {name!r} needs a type annotation")

    # Resolve only public inputs: return annotations may refer to optional imports.
    annotations = get_type_hints(
        SimpleNamespace(
            __annotations__={name: p.annotation for name, p in parameters.items()}
        ),
        globalns=function.__globals__,
        localns=namespace,
        include_extras=True,
    )
    doc = parse(inspect.getdoc(function) or "", style=DocstringStyle.GOOGLE)
    descriptions = {
        parameter.arg_name: parameter.description
        for parameter in doc.params
        if parameter.description is not None
    }
    fields: dict[str, Any] = {}
    for name, parameter in parameters.items():
        annotation = annotations[name]
        if name in descriptions:
            annotation = Annotated[annotation, Field(description=descriptions[name])]
        default = (
            ...
            if name in required or parameter.default is inspect.Parameter.empty
            else parameter.default
        )
        fields[name] = (annotation, default)
    return Tool(
        name=tool_name or function.__name__,
        description="\n\n".join(
            part for part in (doc.short_description, doc.long_description) if part
        ),
        args_schema=create_model(
            f"{function.__name__}Parameters",
            __module__=function.__module__,
            __config__=ConfigDict(extra="forbid", allow_inf_nan=False),
            **fields,
        ),
        handler=function,
        readonly=readonly,
        _unbound_method=unbound_method,
    )


def iter_tools(*owners: Any) -> Iterator[Tool]:
    """Collect decorated tools from modules, classes, or instances in definition order.

    Instance tools are bound to their owner. Inherited methods are included unless
    overridden; properties and other descriptors are not evaluated during discovery.
    """
    for owner in owners:
        cls = owner if isinstance(owner, type) else type(owner)
        members: dict[str, Any] = {}
        for base in reversed(cls.__mro__):
            members.update(vars(base))
        if not isinstance(owner, type):
            members.update(getattr(owner, "__dict__", {}))
        for attribute, value in members.items():
            if isinstance(value, Tool):
                yield getattr(owner, attribute)

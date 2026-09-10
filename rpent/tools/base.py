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

"""Native tool protocol."""

from __future__ import annotations

import inspect
import json
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    ParamSpec,
    TypeVar,
    get_type_hints,
)

import numpy as np
from docstring_parser import DocstringStyle, parse
from pydantic import BaseModel, Field, create_model
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import core_schema

if TYPE_CHECKING:
    from rpent.memory import MemoryManager
    from rpent.session import EnvState

ParamsT = ParamSpec("ParamsT")
RobotT = TypeVar("RobotT")
MAX_TOOL_TEXT_BYTES = 60000


class _ToolJsonSchema(GenerateJsonSchema):
    """Keep the published tool schema independent of Python-only metadata.

    Runtime defaults come from the function signature. Only defaults declared
    with Field(json_schema_extra=...) are advertised, matching existing tools.
    """

    def field_title_should_be_set(self, schema: Any) -> bool:
        return False

    def model_schema(self, schema: core_schema.ModelSchema) -> dict[str, Any]:
        result = super().model_schema(schema)
        result.pop("title", None)
        result.pop("description", None)
        return result

    def default_schema(self, schema: core_schema.WithDefaultSchema) -> dict[str, Any]:
        return self.generate_inner(schema["schema"])

    def nullable_schema(self, schema: core_schema.NullableSchema) -> dict[str, Any]:
        inner = self.generate_inner(schema["schema"])
        if isinstance(inner.get("type"), str) and not any(
            key in inner for key in ("enum", "const", "allOf", "anyOf", "oneOf", "not")
        ):
            if inner["type"] != "null":
                inner["type"] = [inner["type"], "null"]
            return inner
        return {"anyOf": [inner, {"type": "null"}]}


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
        text = json.dumps(self.to_dict(), indent=2, allow_nan=False, ensure_ascii=True)
        if len(text) <= MAX_TOOL_TEXT_BYTES:
            return text
        suffix = "\n[truncated]"
        return text[: MAX_TOOL_TEXT_BYTES - len(suffix)] + suffix


class ToolCancelled(Exception):
    """Raised when a tool reaches a safe cancellation boundary."""


@dataclass(frozen=True)
class ToolContext(Generic[RobotT]):
    """References to this run's resources and this invocation's cancellation."""

    state: EnvState
    memory: MemoryManager
    robot: RobotT
    output_dir: Path
    record_frame: Callable[[np.ndarray], None]
    _cancel_event: threading.Event = field(repr=False)

    def check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise ToolCancelled("Tool call cancelled.")


@dataclass(frozen=True)
class Tool(Generic[ParamsT]):
    """A handler and its generated parameter model, fixed before execution.

    readonly allows shared execution and skips automatic observation capture.
    """

    name: str
    description: str
    args_schema: type[BaseModel]
    handler: Callable[ParamsT, ToolResult]
    readonly: bool = False

    @property
    def input_schema(self) -> dict[str, Any]:
        """Generate the model-facing schema from the same model used for validation."""
        return self.args_schema.model_json_schema(
            schema_generator=_ToolJsonSchema,
            union_format="primitive_type_array",
        )


def _parameter_model(
    handler: Callable,
    descriptions: dict[str, str],
    namespace: dict[str, Any],
) -> type[BaseModel]:
    """Derive validation from public parameters without evaluating injected types."""
    parameters = inspect.signature(handler).parameters
    public = {name: param for name, param in parameters.items() if name != "ctx"}
    annotations = get_type_hints(
        SimpleNamespace(
            __annotations__={name: p.annotation for name, p in public.items()}
        ),
        globalns=handler.__globals__,
        localns=namespace,
        include_extras=True,
    )
    fields: dict[str, Any] = {}
    for name, param in public.items():
        annotation = annotations[name]
        if name in descriptions:
            annotation = Annotated[annotation, Field(description=descriptions[name])]
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, default)
    return create_model(
        f"{handler.__name__}Parameters",
        __module__=handler.__module__,
        **fields,
    )


def readonly(handler: Callable[ParamsT, ToolResult]) -> Callable[ParamsT, ToolResult]:
    """Skip automatic environment observation capture; file writes remain allowed.

    Place this marker below @tool. Readonly calls can execute concurrently.
    """
    setattr(handler, "_rpent_readonly", True)
    return handler


def tool(function: Callable[ParamsT, ToolResult], /) -> Tool[ParamsT]:
    """Declare typed parameters and constraints in the signature, prose in Args.

    The function name becomes the tool name. Google-style docstrings supply the
    tool and parameter descriptions.
    Every handler declares a required keyword-only ctx, injected by the executor
    and excluded from the schema.
    Place @tool above @readonly. By default calls are exclusive; robot tools
    other than finish capture observations afterward. @readonly allows shared
    execution and disables capture.
    """
    # Resolve annotations in factories/tests as well as at module scope, without
    # retaining the caller's frame or namespace in the resulting Tool.
    namespace = dict(sys._getframe(1).f_locals)

    doc = parse(inspect.getdoc(function), style=DocstringStyle.GOOGLE)
    descriptions = {
        param.arg_name: param.description
        for param in doc.params
        if param.description is not None and param.arg_name != "ctx"
    }
    return Tool(
        name=function.__name__,
        description="\n\n".join(
            part for part in (doc.short_description, doc.long_description) if part
        ),
        args_schema=_parameter_model(function, descriptions, namespace),
        handler=function,
        readonly=getattr(function, "_rpent_readonly", False),
    )

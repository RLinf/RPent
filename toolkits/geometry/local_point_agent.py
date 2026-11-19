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

"""Use a local Molmo point service as the sole model of a typed Pydantic AI agent."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, FiniteFloat, PositiveInt
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from rpent.robots.components.molmo_client import MolmoResult


class VisualGrounder(Protocol):
    """Select an original-image column/row point from RGB and a description."""

    def ground(self, image: bytes, query: str) -> MolmoResult:
        """Return a visible point or an absent target."""
        ...


class PointOutput(BaseModel):
    """At most one point measured in the supplied RGB raster."""

    xy: tuple[FiniteFloat, FiniteFloat] | None
    image_size: tuple[PositiveInt, PositiveInt]
    answer: str | None = None


class LocalPointAgent:
    """Translate structured agent I/O to one local native point call without a text planner."""

    def __init__(self, backend: VisualGrounder):
        self._backend = backend

        def respond(messages, info):
            request = next(
                p for p in messages[-1].parts if isinstance(p, UserPromptPart)
            )
            query, image = request.content
            result = backend.ground(image.data, query)
            output = PointOutput(
                xy=result.point_xy if result.found else None,
                image_size=result.image_size,
                answer=result.answer,
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name, output.model_dump(mode="json")
                    )
                ]
            )

        self._agent = Agent(
            FunctionModel(respond, model_name="local-molmo-point"),
            output_type=PointOutput,
            retries=0,
        )

    def verify(
        self, image: bytes, query: str | None, *, isolated: bool = False
    ) -> str | None:
        """Reuse the injected local model; old point-only backends have no identity verdict."""
        verify = getattr(self._backend, "verify", None)
        if verify is None:
            return None
        return verify(image, query, isolated=True) if isolated else verify(image, query)

    def ground(self, image: bytes, query: str) -> MolmoResult:
        result = self._agent.run_sync(
            [query, BinaryContent(data=image, media_type="image/png")]
        ).output
        return MolmoResult(
            found=result.xy is not None,
            point_xy=result.xy,
            image_size=result.image_size,
            answer=result.answer,
        )

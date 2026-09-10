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
"""Transport-independent client for RPent's Molmo grounding service."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import imageio.v2 as imageio
import numpy as np

from rpent.utils.rpc import RpcClient


@dataclass(frozen=True)
class MolmoResult:
    """One pixel Molmo would put the gripper on, if it found the object."""

    found: bool
    point_xy: tuple[float, float] | None = None
    image_size: tuple[int, int] | None = None
    answer: str | None = None


class MolmoClient:
    """Client wrapping the Molmo service over any :class:`RpcClient`."""

    def __init__(self, client: RpcClient, *, timeout_s: float = 180.0) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def ground(
        self,
        image: bytes | bytearray | memoryview | np.ndarray,
        query: str,
    ) -> MolmoResult:
        """Point at ``query`` in ``image``.

        The pixel comes back in the image's own resolution as ``(col, row)``,
        which is the order the model writes it in and the reverse of the
        ``[row, col]`` that SAM3 takes.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("ground requires a non-empty query")

        if isinstance(image, np.ndarray):
            buffer = io.BytesIO()
            imageio.imwrite(buffer, image, format="png")
            image_bytes = buffer.getvalue()
        else:
            image_bytes = bytes(image)

        payload = self._client.call(
            "ground",
            kwargs={
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "query": query.strip(),
            },
            timeout_s=self._timeout_s,
        )
        return self._decode_result(payload)

    @staticmethod
    def _decode_result(payload: Any) -> MolmoResult:
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid Molmo ground response: {payload!r}")
        answer = payload.get("answer")
        point = payload.get("point_xy")
        if point is None:
            return MolmoResult(found=False, answer=answer)
        if not isinstance(point, list) or len(point) != 2:
            raise RuntimeError(f"invalid Molmo point_xy: {point!r}")
        size = payload.get("image_size")
        return MolmoResult(
            found=True,
            point_xy=(float(point[0]), float(point[1])),
            image_size=(
                (int(size[0]), int(size[1]))
                if isinstance(size, list) and len(size) == 2
                else None
            ),
            answer=answer,
        )

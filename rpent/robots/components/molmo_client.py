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
import math
from dataclasses import dataclass
from typing import Any, Literal

import imageio.v2 as imageio
import numpy as np

from rpent.utils.rpc import RpcClient


@dataclass(frozen=True)
class MolmoResult:
    """One visible pixel on the queried object or part, if found."""

    found: bool
    point_xy: tuple[float, float] | None = None
    image_size: tuple[int, int] | None = None
    answer: str | None = None


class MolmoClient:
    """Client wrapping the Molmo service over any :class:`RpcClient`."""

    def __init__(
        self,
        client: RpcClient,
        *,
        timeout_s: float = 180.0,
        point_mode: Literal["manipulation", "geometry"] = "manipulation",
    ) -> None:
        if point_mode not in ("manipulation", "geometry"):
            raise ValueError("point_mode must be manipulation or geometry")
        self._client = client
        self._timeout_s = timeout_s
        self._point_mode = point_mode

    def ground(
        self,
        image: bytes | bytearray | memoryview | np.ndarray,
        query: str,
    ) -> MolmoResult:
        """Point at ``query`` in ``image``.

        The pixel comes back in the image's own resolution as ``(col, row)``,
        which is the order the model writes it in and the reverse of the
        ``[row, col]`` that SAM3 takes. Configure ``point_mode="geometry"``
        for a material point on an exact part; the default retains Flash's
        manipulation-point prompt.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("ground requires a non-empty query")

        if isinstance(image, np.ndarray):
            buffer = io.BytesIO()
            imageio.imwrite(buffer, image, format="png")
            image_bytes = buffer.getvalue()
        else:
            image_bytes = bytes(image)

        kwargs = {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "query": query.strip(),
        }
        # Keep existing Flash requests compatible with the default RPC contract.
        if self._point_mode == "geometry":
            kwargs["point_mode"] = self._point_mode
        payload = self._client.call(
            "molmo.ground",
            kwargs=kwargs,
            timeout_s=self._timeout_s,
        )
        return self._decode_result(payload)

    @staticmethod
    def _decode_result(payload: Any) -> MolmoResult:
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid Molmo ground response: {payload!r}")
        answer = payload.get("answer")
        size = payload.get("image_size")
        image_size = None
        if size is not None:
            if (
                not isinstance(size, list)
                or len(size) != 2
                or any(type(value) is not int or value <= 0 for value in size)
            ):
                raise RuntimeError(f"invalid Molmo image_size: {size!r}")
            image_size = (size[0], size[1])
        point = payload.get("point_xy")
        if point is None:
            return MolmoResult(found=False, image_size=image_size, answer=answer)
        if not isinstance(point, list) or len(point) != 2:
            raise RuntimeError(f"invalid Molmo point_xy: {point!r}")
        try:
            col, row = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid Molmo point_xy: {point!r}") from exc
        if not math.isfinite(col) or not math.isfinite(row):
            raise RuntimeError(f"invalid Molmo point_xy: {point!r}")
        return MolmoResult(
            found=True,
            point_xy=(col, row),
            image_size=image_size,
            answer=answer,
        )

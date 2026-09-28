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

"""Transport-independent client for the point locator service."""

from __future__ import annotations

import base64
import io
import math
from typing import TYPE_CHECKING, Any

from rpent.utils.rpc import RpcClient

if TYPE_CHECKING:
    import numpy as np


def validate_point(value: Any) -> tuple[float, float]:
    """Require two finite numeric coordinates at a model or RPC boundary."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in value
        )
    ):
        raise ValueError(f"invalid locator point: {value!r}")
    return float(value[0]), float(value[1])


class LocatorClient:
    """Locate an object through any RPC transport, independently of the model."""

    def __init__(self, client: RpcClient, *, timeout_s: float = 180.0) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def locate(
        self,
        image: bytes | bytearray | memoryview | np.ndarray,
        query: str,
    ) -> tuple[float, float] | None:
        """Return an original-image pixel ``(col, row)``, or no located target.

        Args:
            image: Encoded image bytes or an RGB image array.
            query: Object or location to point at.

        Returns:
            A pixel in the input image's resolution, or ``None`` for a miss.
            Transport and malformed-response errors propagate to the caller.
        """
        import numpy as np

        if not isinstance(query, str) or not query.strip():
            raise ValueError("locate requires a non-empty query")
        if isinstance(image, np.ndarray):
            import imageio.v2 as imageio

            buffer = io.BytesIO()
            imageio.imwrite(buffer, image, format="png")
            image_bytes = buffer.getvalue()
        else:
            image_bytes = bytes(image)
        payload = self._client.call(
            "locator.locate",
            kwargs={
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "query": query.strip(),
            },
            timeout_s=self._timeout_s,
        )
        if not isinstance(payload, dict) or "point_xy" not in payload:
            raise RuntimeError(f"invalid locator response: {payload!r}")
        point = payload["point_xy"]
        return None if point is None else validate_point(point)

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

"""Bounded Molmo-to-agent fallback shared by both Franka variants."""

from typing import Any, Callable

import numpy as np

from robots.franka.flash.common import GroundingError, localize


def localize_with_fallback(
    toolkit: Any,
    robot: str,
    anchor: dict,
    molmo: Any,
    *,
    arm: str | None,
    agent: Any = None,
    note: Callable[[str], None] = lambda _: None,
) -> np.ndarray:
    """Try Molmo once, then optionally refresh and try the point-only agent once."""
    attempts = []
    query = dict(anchor)
    try:
        for provider, selector in (("molmo", molmo), ("agent", agent)):
            if selector is None:
                raise GroundingError("no fallback selector configured")
            toolkit.raise_if_cancelled()
            try:
                point = localize(toolkit.state, robot, query, selector, arm=arm)
            except GroundingError as exc:
                attempts.append(
                    {"provider": provider, "status": "failed", "reason": str(exc)}
                )
                if provider == "agent" or agent is None:
                    raise
                note(
                    "Molmo localization failed; refreshing observation for the grounding agent."
                )
                query["phrase"] = (
                    anchor["phrase"]
                    + ". A previous selection failed: "
                    + str(exc)[:1000]
                    + ". Select a different visible material point on the SAME named part."
                )
                toolkit.raise_if_cancelled()
                toolkit.refresh_flash_state()
            else:
                attempts.append(
                    {
                        "provider": provider,
                        "status": "valid",
                        "point_xyz": point.tolist(),
                    }
                )
                return point
    finally:
        toolkit.state.save(
            "flash_grounding.json", {"anchor": anchor, "attempts": attempts}
        )
    raise GroundingError("no point selector produced a valid target")

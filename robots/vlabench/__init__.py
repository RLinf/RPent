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

"""VLABench robot extension."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robots.vlabench.agent_toolkit import get_toolkit
    from robots.vlabench.spec import get_robot_spec

__all__ = ["get_robot_spec", "get_toolkit"]


def __getattr__(name: str) -> Any:
    """Load runtime factories only when requested, keeping control imports light."""
    if name == "get_robot_spec":
        from robots.vlabench.spec import get_robot_spec

        return get_robot_spec
    if name == "get_toolkit":
        from robots.vlabench.agent_toolkit import get_toolkit

        return get_toolkit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

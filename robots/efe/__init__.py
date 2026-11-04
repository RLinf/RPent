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

"""EFE standard-mode robot integration.

The task container exposes one episode through ``rcb_client``. RPent treats
that service as the robot: no simulator internals and no privileged state. A
task that declares a reset budget exposes a limited reset; RPent never resets
on its own. ``robots.efe.harbor_agent`` runs this integration inside a Harbor
trial.
"""

from typing import Any

_FACTORY_NAMES = ("get_robot_spec", "get_toolkit")

__all__ = list(_FACTORY_NAMES)


def __getattr__(name: str) -> Any:
    """Load the robot factories on first use.

    The EFE CLI lives in this package, so ``--help`` must not import the robot
    framework.
    """
    if name in _FACTORY_NAMES:
        from robots.efe import robot_spec

        return getattr(robot_spec, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

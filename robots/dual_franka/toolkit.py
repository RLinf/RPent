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

"""Native dual-Franka toolkit with shared Franka session lifecycle."""

from robots.dual_franka import tools
from robots.franka.toolkit import FrankaToolkit


class DualFrankaToolkit(FrankaToolkit):
    """Common native tools plus dual-Franka motion and perception."""

    _robot_tools = tools.DUAL_FRANKA_TOOLS
    _dump_state = staticmethod(tools.dump_state)
    _build_observation = staticmethod(tools.build_observation)

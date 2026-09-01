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

"""Robot implementations loaded by name.

Each subpackage (e.g. :mod:`robots.libero`) bundles the agent-side robot
package (``get_robot_spec`` / ``get_toolkit`` factories, toolkit, prompts,
guides) together with its server-side scripts (``env_server.py`` /
``vla_server.py``). The robot registry in :mod:`rpent.robots.base` resolves a
robot by importing ``robots.<name>``.
"""

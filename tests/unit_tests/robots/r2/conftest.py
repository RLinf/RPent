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

from __future__ import annotations

from pathlib import Path

import pytest

#: A head camera 1 m above base_link on a yaw joint, looking along +x.
TINY_URDF = """<robot name="tiny">
  <link name="ground"/><link name="base_link"/><link name="neck"/><link name="cam"/>
  <joint name="J_World" type="fixed">
    <parent link="ground"/><child link="base_link"/>
  </joint>
  <joint name="J_NeckYaw" type="revolute">
    <parent link="base_link"/><child link="neck"/>
    <origin xyz="0 0 1" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="J_Cam" type="fixed">
    <parent link="neck"/><child link="cam"/>
    <origin xyz="0.1 0 0" rpy="-1.5707963267948966 0 -1.5707963267948966"/>
  </joint>
</robot>
"""


@pytest.fixture
def tiny_urdf(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.urdf"
    path.write_text(TINY_URDF)
    return path

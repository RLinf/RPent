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

"""Shared scene identifiers for the RoboLab episode adapter."""

import os

CAMERAS = ("over_shoulder_left_camera", "wrist_cam")


def require_isaac_eula() -> None:
    """Fail early when headless Isaac has no explicit operator acceptance."""
    if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "Y":
        raise RuntimeError(
            "RoboLab requires OMNI_KIT_ACCEPT_EULA=Y after accepting the NVIDIA Omniverse EULA"
        )

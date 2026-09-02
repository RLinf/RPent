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

from typing import Any

import pytest

from rpent.tools.toolkit import readonly


class FakeSingleArmPrimitives:
    """No-runtime primitive surface shared by LIBERO and RoboCasa tests."""

    instances: list[FakeSingleArmPrimitives] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.reset_calls = 0
        self.recording_started = False
        type(self).instances.append(self)

    def reset(self) -> dict[str, Any]:
        self.reset_calls += 1
        return {"success": True}

    def reset_episode(self, reason: str) -> dict[str, Any]:
        return {"success": True, "reason": reason}

    def start_recording(self) -> None:
        self.recording_started = True

    def recorded_frame_count(self) -> int:
        return 0

    def frame_slice(self, start: int) -> list[Any]:
        del start
        return []

    def stop_recording(self) -> list[Any]:
        return []

    def dump_success_criteria(self) -> str:
        return "offline success criteria"

    @readonly
    def segment(self, **kwargs: Any) -> dict[str, Any]:
        return {"segment": kwargs}

    @staticmethod
    def _operation(name: str, **kwargs: Any) -> dict[str, Any]:
        return {"operation": name, "arguments": kwargs}

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_to", **kwargs)

    def pi0_pick(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("pi0_pick", **kwargs)

    def pi0_doubled(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("pi0_doubled", **kwargs)

    def release(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("release", **kwargs)

    def set_gripper(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("set_gripper", **kwargs)

    def rotate_wrist(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rotate_wrist", **kwargs)

    def rotate_pitch(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rotate_pitch", **kwargs)

    def move_pose(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_pose", **kwargs)

    def move_delta(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_delta", **kwargs)

    def scripted_grasp(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("scripted_grasp", **kwargs)

    def rldx_skill(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rldx_skill", **kwargs)

    def rldx_arm(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("rldx_arm", **kwargs)

    def navigate_to(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("navigate_to", **kwargs)

    def move_base(self, **kwargs: Any) -> dict[str, Any]:
        return self._operation("move_base", **kwargs)


@pytest.fixture
def fake_single_arm_primitives() -> type[FakeSingleArmPrimitives]:
    FakeSingleArmPrimitives.instances.clear()
    return FakeSingleArmPrimitives

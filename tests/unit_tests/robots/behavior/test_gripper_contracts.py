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
from typing import Any

import numpy as np
import pytest

from robots.behavior.rlinf_env import (
    GRIPPER_CLOSE_COMMAND,
    GRIPPER_COMMAND_CONTROL_CYCLES,
    GRIPPER_OPEN_COMMAND,
)
from robots.behavior.schemas import (
    ENV_ACTION_SEGMENTS,
    RAW_PROPRIO_SEGMENTS,
    validate_action_chunk,
)
from tests.unit_tests.robots.behavior._backend_fakes import (
    _FakeOfficialBehaviorEnv,
    _official_backend,
    _visual_check,
)


def test_gripper_close_builds_hold_action_and_target_command(tmp_path: Path) -> None:
    backend, env = _official_backend(tmp_path)

    result = backend.close(hand="left", visual_hand_check=_visual_check("left"))
    chunk = validate_action_chunk(np.stack(env.actions, axis=0))
    first = chunk[0]

    assert result["status"] == "ok"
    assert result["primitive_success"] is True
    assert result["task_success"] is False
    assert result["stop_reason"] == "requested_actions_completed"
    assert result["visual_hand_check"] == _visual_check("left")
    assert result["visual_hand_check_verification"] == "not_verified"
    assert result["action_shape"] == [1, 23]
    assert result["action_chunk_shape"] == [GRIPPER_COMMAND_CONTROL_CYCLES, 23]
    assert chunk.shape == (GRIPPER_COMMAND_CONTROL_CYCLES, 23)
    assert np.allclose(chunk, first)
    assert np.allclose(first[ENV_ACTION_SEGMENTS["base"]], 0.0)
    assert np.allclose(
        first[ENV_ACTION_SEGMENTS["trunk"]],
        env.raw[RAW_PROPRIO_SEGMENTS["trunk"]],
    )
    assert np.allclose(
        first[ENV_ACTION_SEGMENTS["left_arm"]],
        env.raw[RAW_PROPRIO_SEGMENTS["left_arm"]],
    )
    assert np.allclose(
        first[ENV_ACTION_SEGMENTS["right_arm"]],
        env.raw[RAW_PROPRIO_SEGMENTS["right_arm"]],
    )
    assert first[ENV_ACTION_SEGMENTS["left_gripper"]][0] == GRIPPER_CLOSE_COMMAND
    assert first[ENV_ACTION_SEGMENTS["right_gripper"]][0] == GRIPPER_OPEN_COMMAND


def test_gripper_latch_holds_non_target_hand_and_open_echoes_release_check(
    tmp_path: Path,
) -> None:
    backend, env = _official_backend(tmp_path)

    backend.close(hand="left", visual_hand_check=_visual_check("left"))
    env.actions.clear()
    release_check = {
        "camera": "head",
        "frame_id": "release-frame",
        "assessment": "target_visibly_released",
    }
    result = backend.open(
        hand="right",
        visual_hand_check=_visual_check("right"),
        release_visual_check=release_check,
    )
    first = validate_action_chunk(np.stack(env.actions, axis=0))[0]

    assert result["status"] == "ok"
    assert result["release_visual_check"] == release_check
    assert result["release_visual_check_verification"] == "not_verified"
    assert first[ENV_ACTION_SEGMENTS["left_gripper"]][0] == GRIPPER_CLOSE_COMMAND
    assert first[ENV_ACTION_SEGMENTS["right_gripper"]][0] == GRIPPER_OPEN_COMMAND


def test_gripper_task_success_uses_only_official_done_success(tmp_path: Path) -> None:
    terminated_env = _FakeOfficialBehaviorEnv(terminated_on_step=1)
    backend, env = _official_backend(tmp_path / "terminated", terminated_env)

    result = backend.close(hand="left", visual_hand_check=_visual_check("left"))

    assert result["stop_reason"] == "terminated"
    assert result["primitive_success"] is True
    assert result["task_success"] is False
    assert backend.official_success_latched is False
    assert len(env.actions) == 1
    rejected = backend.open(hand="left", visual_hand_check=_visual_check("left"))
    assert rejected["stop_reason"] == "episode_ended"
    assert len(env.actions) == 1

    success_env = _FakeOfficialBehaviorEnv(success_on_step=1)
    success_backend, _success_env = _official_backend(tmp_path / "success", success_env)
    success = success_backend.open(
        hand="right",
        visual_hand_check=_visual_check("right"),
    )

    assert success["stop_reason"] == "official_task_success"
    assert success["task_success"] is True
    assert success["official_success_receipt"]["source"] == 'info["done"]["success"]'


def test_numpy_boolean_cannot_become_official_success(tmp_path: Path) -> None:
    from robots.behavior.terminal_success import (
        make_raw_success_receipt,
        official_task_success,
    )

    info = {"done": {"success": np.bool_(True)}}
    assert official_task_success(info) is False
    assert make_raw_success_receipt(info, env_step=1) is None

    backend, env = _official_backend(tmp_path)
    with pytest.raises(TypeError, match="must be the Python boolean True"):
        backend._note_info(info)

    original_chunk_step = env.env_chunk_step

    def numpy_success(actions: Any) -> tuple:
        observations, rewards, terms, truncs, infos = original_chunk_step(actions)
        infos[-1][0]["done"]["success"] = np.bool_(True)
        return observations, rewards, terms, truncs, infos

    env.env_chunk_step = numpy_success
    with pytest.raises(TypeError, match="must be the Python boolean True"):
        backend.chunk_step(backend._hold_action_from_current_proprio()[None, :])
    assert backend.official_success_latched is False


def test_gripper_execution_error_uses_motion_error_envelope(tmp_path: Path) -> None:
    backend, env = _official_backend(tmp_path)
    assert backend._last_obs is not None
    backend._last_obs["states"][RAW_PROPRIO_SEGMENTS["trunk"]] = np.nan

    result = backend.open(hand="left", visual_hand_check=_visual_check("left"))

    assert result["status"] == "failed"
    assert result["primitive_success"] is False
    assert result["task_success"] is False
    assert result["stop_reason"] == "error"
    assert "NaN or infinity" in result["error"]
    assert env.actions == []


def test_backend_lifecycle_close_still_closes_env_without_primitive_args(
    tmp_path: Path,
) -> None:
    backend, env = _official_backend(tmp_path)

    assert backend.close() == {"status": "ok", "closed": True}
    assert env.closed is True
    assert backend.close() == {
        "status": "ok",
        "closed": True,
        "already_closed": True,
    }


@pytest.mark.parametrize(
    ("ending", "terminated", "success"),
    [
        ("contact", False, False),
        ("terminated", True, False),
        ("official_task_success", False, True),
        ("official_task_success", True, True),
        ("duration_limit", False, False),
    ],
)
def test_press_executes_bounded_hold_actions_and_reports_raw_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ending: str,
    terminated: bool,
    success: bool,
) -> None:
    backend, env = _official_backend(
        tmp_path,
        _FakeOfficialBehaviorEnv(
            terminated_on_step=1 if terminated else None,
            success_on_step=1 if success else None,
        ),
    )

    def state():
        return {
            "control_dt": 0.05,
            "hands": {
                "left": {
                    "position": np.array([0.0, 0.0, len(env.actions) * 0.0005]),
                    "approach_direction": np.array([0.0, 0.0, 1.0]),
                    "joint_positions": env.raw[RAW_PROPRIO_SEGMENTS["left_arm"]],
                    "joint_lower_limits": np.full(7, -3.0),
                    "joint_upper_limits": np.full(7, 3.0),
                    "jacobian": np.eye(6, 7),
                    "contacts": ["button"]
                    if ending == "contact" and env.actions
                    else [],
                }
            },
        }

    monkeypatch.setattr(backend, "_get_motion_state", state)
    hold = backend._hold_action_from_current_proprio()
    result = backend.press(
        hand="left", visual_hand_check=_visual_check("left"), duration_s=0.1
    )
    assert result["stop_reason"] == ending
    assert result["task_success"] is success
    assert result["terminated"] is terminated
    assert result["truncated"] is False
    assert result["primitive_success"] is (
        ending in {"contact", "official_task_success"}
    )
    assert result["executed_steps"] == (2 if ending == "duration_limit" else 1)
    untouched = np.ones(23, dtype=bool)
    untouched[ENV_ACTION_SEGMENTS["left_arm"]] = False
    for action in env.actions:
        assert validate_action_chunk(action[None, :]).shape == (1, 23)
        np.testing.assert_array_equal(action[untouched], hold[untouched])
    assert result["visual_hand_check_verification"] == "not_verified"


@pytest.mark.parametrize("duration", [0, -1, 11, float("nan"), float("inf")])
def test_press_rejects_invalid_duration_before_motion(tmp_path: Path, duration: float):
    backend, env = _official_backend(tmp_path)
    with pytest.raises(ValueError, match="duration_s"):
        backend.press(
            hand="left", visual_hand_check=_visual_check("left"), duration_s=duration
        )
    assert not env.actions

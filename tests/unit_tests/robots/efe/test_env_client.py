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

"""EFE client checks must reject commands before the service executes them."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from robots.efe.env_client import EfeEnvClient

_LAYOUT = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle"]
_LIMITS = {name: [-3.0, 3.0] for name in _LAYOUT} | {"gripper": [0.002, 0.037]}

_SPEC = {
    "protocol": "eai-standard/2.2",
    "control_hz": 500.0,
    "max_chunk_s": 10.0,
    "max_frames_per_call": 3000,
    "cameras": {"table_cam_front": {}, "wrist_cam": {}},
    "default_cameras": ["table_cam_front", "wrist_cam"],
    "action_types": {
        "joint_position": {
            "dim": 6,
            "layout": [*_LAYOUT, "gripper"],
            "limits": _LIMITS,
        },
        "done": {"dim": 0, "layout": []},
        "not_yet": ["ee_pose", "joint_delta"],
    },
}


def _actions(rows: int) -> np.ndarray:
    """Return a valid chunk: zero joints and an open gripper."""
    actions = np.zeros((rows, 6))
    actions[:, 5] = 0.02
    return actions


def _result(**meta):
    return SimpleNamespace(meta={"done": False, "control_steps_chunk": 5, **meta})


class FakeRobot:
    """Record calls so a rejected command can prove the service saw nothing."""

    def __init__(self, spec=None):
        self.spec_data = dict(spec or _SPEC)
        self.base = "http://sim:8000"
        self.submit_calls = []
        self.observe_calls = []
        self.post_calls = []
        self.post_result = SimpleNamespace(
            meta={"episode": 1, "resets_left": 4, "done": False}
        )
        self.status_calls = 0
        self.submit_result = _result()
        self.observe_result = _result(done=False)
        self.status_result = {"done": False, "reason": None}

    def _post(self, path, meta, parts, timeout):
        self.post_calls.append((path, meta, parts))
        return self.post_result

    def spec(self):
        return self.spec_data

    def submit(self, action_type, actions, **kwargs):
        self.submit_calls.append((action_type, np.asarray(actions), kwargs))
        return self.submit_result

    def get_obs(self, **kwargs):
        self.observe_calls.append(kwargs)
        return self.observe_result

    def status(self):
        self.status_calls += 1
        return self.status_result


@pytest.fixture
def robot():
    return FakeRobot()


@pytest.fixture
def client(robot):
    return EfeEnvClient(robot=robot)


def test_observe_requests_current_frames_only(client, robot):
    client.observe(cameras=["wrist_cam"], depth=True)
    assert robot.observe_calls == [
        {"cameras": ["wrist_cam"], "seconds": 0, "depth": True, "video": "frames"}
    ]
    assert client.observe_calls == 1


def test_observe_can_skip_frame_rendering(client, robot):
    client.observe(cameras=[], video="none")

    assert robot.observe_calls == [
        {"cameras": [], "seconds": 0, "depth": None, "video": "none"}
    ]


def test_observe_rejects_unknown_camera(client, robot):
    with pytest.raises(ValueError, match="Unknown cameras"):
        client.observe(cameras=["overhead"])
    assert robot.observe_calls == []


def test_submit_forwards_a_valid_chunk(client, robot):
    actions = _actions(3)
    client.submit("joint_position", actions, dt=0.01)
    (action_type, forwarded, kwargs), *_ = robot.submit_calls
    assert action_type == "joint_position"
    assert forwarded.shape == (3, 6)
    assert kwargs["dt"] == pytest.approx(0.01)
    assert kwargs["video"] == "none"
    assert client.submit_calls == 1
    assert client.control_steps == 5


@pytest.mark.parametrize(
    ("action_type", "actions", "match"),
    [
        ("ee_delta", _actions(1), "Unknown action type"),
        ("joint_position", np.zeros((1, 5)), "needs shape"),
    ],
)
def test_submit_rejects_bad_shapes(client, robot, action_type, actions, match):
    with pytest.raises(ValueError, match=match):
        client.submit(action_type, actions)
    assert robot.submit_calls == []


def test_submit_rejects_non_action_spec_entries(client, robot):
    with pytest.raises(ValueError, match="Unknown action type") as excinfo:
        client.submit("not_yet", [])

    assert "Available: ['done', 'joint_position']" in str(excinfo.value)
    assert robot.submit_calls == []


def test_submit_rejects_non_finite_values(client, robot):
    actions = _actions(2)
    actions[1, 2] = np.nan
    with pytest.raises(ValueError, match="elbow"):
        client.submit("joint_position", actions)
    assert robot.submit_calls == []


def test_submit_rejects_values_outside_the_spec_limits(client, robot):
    actions = _actions(2)
    actions[0, 5] = 0.5
    with pytest.raises(ValueError, match="gripper outside"):
        client.submit("joint_position", actions)
    assert robot.submit_calls == []


@pytest.mark.parametrize(
    ("dt", "rows", "match"),
    [
        (0.003, 2, "whole multiple"),
        (0.6, 2, "at most"),
        (-0.002, 2, "positive"),
        (float("inf"), 2, "positive finite"),
    ],
)
def test_submit_rejects_bad_timing(client, robot, dt, rows, match):
    with pytest.raises(ValueError, match=match):
        client.submit("joint_position", _actions(rows), dt=dt)
    assert robot.submit_calls == []


def test_default_timing_respects_the_chunk_limit(client, robot):
    with pytest.raises(ValueError, match="max_chunk_s"):
        client.submit("joint_position", _actions(5001))
    assert robot.submit_calls == []


def test_done_action_needs_no_timing(client, robot):
    client.submit("done", [])
    action_type, forwarded, kwargs = robot.submit_calls[0]
    assert action_type == "done"
    assert forwarded.size == 0
    assert kwargs["dt"] is None


def test_a_done_result_blocks_later_commands(client, robot):
    robot.submit_result = _result(done=True, reason="success")
    client.submit("joint_position", _actions(1))
    assert client.done is True
    assert client.done_reason == "success"
    with pytest.raises(RuntimeError, match="episode ended"):
        client.submit("joint_position", _actions(1))
    with pytest.raises(RuntimeError, match="episode ended"):
        client.observe()
    assert len(robot.submit_calls) == 1
    assert robot.observe_calls == []


def test_status_reports_the_episode_end(client, robot):
    robot.status_result = {"done": True, "reason": "timeout"}
    client.status()
    assert client.done is True
    assert client.done_reason == "timeout"
    with pytest.raises(RuntimeError, match="timeout"):
        client.observe()


def test_status_keeps_the_last_observation(client, robot):
    robot.observe_result = _result(proprio={"layout": [], "now": []}, step=4)
    client.observe()
    robot.status_result = {"done": False, "reason": None, "step": 4}

    client.status()

    assert client.last_meta["step"] == 4
    assert "proprio" in client.last_meta


def test_a_spec_without_action_types_is_rejected():
    spec = dict(_SPEC)
    spec.pop("action_types")
    with pytest.raises(ValueError, match="action_types"):
        EfeEnvClient(robot=FakeRobot(spec))


def test_an_unbounded_joint_limit_skips_the_check(client, robot):
    """A continuous joint declares ``[null, null]`` instead of omitting the key."""
    spec = copy.deepcopy(_SPEC)
    spec["action_types"]["joint_position"]["limits"]["wrist_angle"] = [None, None]
    client = EfeEnvClient(robot=FakeRobot(spec))
    robot = client._robot
    actions = _actions(2)
    actions[:, 4] = 1.0e3

    client.submit("joint_position", actions)

    assert len(robot.submit_calls) == 1


def test_a_one_sided_joint_limit_still_checks_the_declared_bound(client, robot):
    spec = copy.deepcopy(_SPEC)
    spec["action_types"]["joint_position"]["limits"]["waist"] = [None, 0.5]
    client = EfeEnvClient(robot=FakeRobot(spec))
    robot = client._robot
    actions = _actions(1)
    actions[0, 0] = 0.7

    with pytest.raises(ValueError, match="waist outside"):
        client.submit("joint_position", actions)

    assert robot.submit_calls == []


def test_success_hidden_matches_the_service_spelling():
    """``/spec`` returns a sentence, so the flag is a prefix rather than a value."""
    spec = {**_SPEC, "success_signal": "hidden: the service never tells you..."}

    assert EfeEnvClient(robot=FakeRobot(spec)).success_hidden is True


def test_success_hidden_is_false_for_a_visible_signal():
    spec = {**_SPEC, "success_signal": "visible"}

    assert EfeEnvClient(robot=FakeRobot(spec)).success_hidden is False


def test_an_unsupported_protocol_is_rejected_before_any_call():
    spec = {**_SPEC, "protocol": "eai-standard/2.1"}
    spec.pop("control_hz")
    spec["control_dt"] = 0.02

    with pytest.raises(ValueError, match="Unsupported EFE service protocol"):
        EfeEnvClient(robot=FakeRobot(spec))


def test_reset_revalidates_cameras(robot):
    spec = {**_SPEC, "budget": {"reset": 5}}
    client = EfeEnvClient(robot=FakeRobot(spec))
    robot = client._robot

    with pytest.raises(ValueError, match="Unknown cameras"):
        client.reset(cameras=["ceiling_cam"])

    assert robot.post_calls == []


def test_reset_is_allowed_after_an_attempt_that_ended_early(robot):
    """The budget is for abandoning a failed attempt, not only a live one."""
    spec = {**_SPEC, "budget": {"reset": 5}}
    client = EfeEnvClient(robot=FakeRobot(spec))
    robot = client._robot
    client._record_state({"done": True, "reason": "failed"})

    client.reset()

    assert robot.post_calls[0][0] == "/reset"
    assert client.done is False


def test_reset_is_refused_after_the_trial_succeeded(robot):
    spec = {**_SPEC, "budget": {"reset": 5}}
    client = EfeEnvClient(robot=FakeRobot(spec))
    client._record_state({"done": True, "reason": "success"})

    with pytest.raises(RuntimeError, match="Cannot reset"):
        client.reset()


def test_reset_needs_a_declared_budget(client):
    with pytest.raises(ValueError, match="no reset budget"):
        client.reset()


def test_reset_starts_a_new_episode(robot):
    spec = {**_SPEC, "budget": {"reset": 5}}
    client = EfeEnvClient(robot=FakeRobot(spec))
    robot = client._robot

    meta = client.reset(cameras=["wrist_cam"])

    assert robot.post_calls == [
        ("/reset", {"video": "none", "cameras": ["wrist_cam"]}, [])
    ]
    assert meta["resets_left"] == 4
    assert client.reset_calls == 1
    assert client.done is False


def test_finish_needs_agent_ended_episodes(client):
    with pytest.raises(ValueError, match="agent-ended"):
        client.finish()


def test_finish_ends_a_hidden_success_episode(monkeypatch):
    spec = {**_SPEC, "success_signal": "hidden"}
    robot = FakeRobot(spec)
    client = EfeEnvClient(robot=robot)
    calls = []

    def fake_post(url, payload):
        calls.append((url, payload))
        return {"ok": True, "done": True, "reason": "finished"}

    monkeypatch.setattr(client, "_post_json", fake_post)

    status = client.finish()

    assert calls == [(f"{robot.base}/finish", {})]
    assert status["ok"] is True
    assert client.done is True
    assert client.done_reason == "finished"
    assert client.finish_calls == 1


def test_finish_reports_a_client_without_a_base_url():
    class NoBaseRobot(FakeRobot):
        def __init__(self, spec=None):
            super().__init__(spec)
            self.base = None

    client = EfeEnvClient(robot=NoBaseRobot({**_SPEC, "success_signal": "hidden"}))

    with pytest.raises(RuntimeError, match="no base URL"):
        client.finish()


def test_submit_rejects_unknown_camera(client, robot):
    with pytest.raises(ValueError, match="Unknown cameras"):
        client.submit("joint_position", _actions(1), cameras=["ceiling_cam"])

    assert robot.submit_calls == []

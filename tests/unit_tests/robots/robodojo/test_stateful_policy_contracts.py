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

"""Stateful execution budgets, model feedback and primitive takeover without RPC."""

from types import SimpleNamespace

import pytest

from robots.robodojo.stateful_policy import StatefulPolicyEnv, run_policy
from rpent.tools.toolkit import ToolCancelled
from tests.unit_tests.robots.robodojo.fakes import EpisodePolicy
from tests.unit_tests.robots.robodojo.fakes import FakeEnv as EpisodeEnv


@pytest.mark.parametrize("budgets", [(400,), (3, 5, 2, 31)])
@pytest.mark.parametrize("done_at", [None, 137])
def test_complete_episode_preserves_observations_inference_and_termination(
    budgets, done_at
):
    native, client = EpisodeEnv(400, done_at), EpisodePolicy()
    env = StatefulPolicyEnv(
        native,
        client,
        lambda: None,
    )
    reports = []
    try:
        while not reports or not reports[-1]["episode_ended"]:
            before = native.frame
            env.get_obs()  # read-only tool boundary never advances the timeline
            assert native.frame == before
            reports.append(env.run(budgets[len(reports) % len(budgets)]))
        count = done_at or 400
        assert native.frame == count
        assert sum(r["executed_steps"] for r in reports) == count
        assert client.events == [("reset",)] + [
            event
            for step in range(count)
            for event in (
                [("observe", step), ("infer", step)]
                if step % 8 == 0
                else [("observe", step)]
            )
        ]
        inferences = [i for r in reports for i in r["inferences"]]
        assert [i["observation_step"] for i in inferences] == list(range(0, count, 8))
        assert all(i["info"]["upstream"]["latency_ms"] == 123.5 for i in inferences)
        assert reports[-1]["stop_reason"] == "episode_end"
        assert reports[-1]["pending_actions"] == 0
        assert env.run(4)["executed_steps"] == 0
        assert native.frame == count
    finally:
        env.close()


@pytest.mark.parametrize(
    "failure", [ToolCancelled("cancelled"), RuntimeError("lost observation")]
)
def test_tool_error_preserves_partial_execution_and_discards_actions(failure):
    native, client = EpisodeEnv(), EpisodePolicy()
    env = StatefulPolicyEnv(
        native,
        client,
        lambda: None,
    )
    original = client.update_obs

    def observe(obs, step):
        original(obs, step)
        if step == 2:
            raise failure

    client.update_obs = observe
    try:
        result = run_policy(primitives=SimpleNamespace(env=env), max_steps=8)
        assert result.is_error
        assert result.data["executed_steps"] == 2
        assert result.data["pending_actions"] == 0
        assert len(result.data["inferences"]) == 1
        # A callback exception is a failed feedback delivery, even if its type
        # resembles a scheduler cancellation. Confirmed progress remains known.
        assert result.data["stop_reason"] == "error"
        assert client.events.count(("reset",)) == 1
        with pytest.raises(RuntimeError, match="unavailable"):
            env.run(1)
    finally:
        env.close()


def test_public_primitive_feedback_replaces_awomo_cache_without_duplicate_history():
    from robots.robodojo.primitives import RoboDojoPrimitives

    native, client = EpisodeEnv(), EpisodePolicy()
    env = StatefulPolicyEnv(
        native,
        client,
        lambda: None,
    )
    primitive = RoboDojoPrimitives(env=env, check_cancelled=lambda: None)
    native.get_motion_state = lambda: {"left_eef_pose": [0, 0, 1, 1, 0, 0, 0]}
    native.plan_arm_path = lambda *a: {"status": "Success", "position": [[0.0] * 6]}
    try:
        env.run(2)
        assert len(env._pending) == 6
        assert primitive.hold(steps=2).data["executed_actions"] == 2
        assert not env._pending
        assert (
            primitive.move_delta(arm="left", dxyz=[0, 0, 0.01], substeps=1).data[
                "executed_actions"
            ]
            == 1
        )
        env.run(1)
        assert [e[1] for e in client.events if e[0] == "observe"] == list(range(7))
        assert [e[1] for e in client.events if e[0] == "infer"] == [0, 5]
        assert client.events.count(("reset",)) == 1
    finally:
        env.close()


@pytest.mark.parametrize(
    "failure", [TimeoutError("model result uncertain"), ToolCancelled("lost inference")]
)
def test_model_inference_failure_also_blocks_native_primitives(failure):
    from robots.robodojo.primitives import RoboDojoPrimitives

    native, client = EpisodeEnv(), EpisodePolicy()
    env = StatefulPolicyEnv(
        native,
        client,
        lambda: None,
    )

    def fail():
        raise failure

    client.get_action = fail
    try:
        with pytest.raises(
            RuntimeError if isinstance(failure, ToolCancelled) else TimeoutError
        ):
            env.run(2)
        assert env.last_run["stop_reason"] == "error"
        env.resume_execution()
        with pytest.raises(RuntimeError, match="unavailable"):
            env.run(1)
        result = (
            RoboDojoPrimitives(env=env, check_cancelled=lambda: None).hold(steps=2).data
        )
        assert result["stop_reason"] == "synchronization_lost"
        assert result["executed_actions"] == native.frame == 0
        assert client.events.count(("reset",)) == 1
    finally:
        env.close()
